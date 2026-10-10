// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * chat.deepseek.com client (unofficial web API) — the transport layer of the
 * OpenAI-compatible server.
 *
 * Everything below was verified LIVE (2026-09-13) against a real account:
 *   POST /api/v0/chat/create_pow_challenge + SHA3 WASM solver -> x-ds-pow-response
 *   POST /api/v0/chat_session/create                           -> chat_session_id
 *   POST /api/v0/chat/completion  (SSE patch-stream)           -> text / thinking
 *   POST /api/v0/file/upload_file                              -> file_id (status PENDING!)
 *   GET  /api/v0/file/fetch_files?file_ids=...                 -> file status
 *
 * THE MOST IMPORTANT DISCOVERY (it cost the most time):
 *  1. `ref_file_ids` accepts ONLY files in SUCCESS status. A freshly uploaded file is
 *     PENDING/PARSING and the completion rejects it with biz_code 9 "invalid ref file id".
 *     That is why uploadFile() waits for readiness (polling fetch_files).
 *
 * The session is NOT named (update_title was removed) — one persistent conversation per
 * token is enough.
 *
 * The layers below (PoW, headers, SSE parser) live in battle-tested .ts modules;
 * Node 22.6+ strips types natively.
 */
import { DS_BASE, buildDsHeaders, createChatSession, createPowHeader, envelopeError, isBusyGenerating, isInvalidSessionError, isMutedError, isThrottled, muteUntilMs, parseWebSse } from './webapi.ts'
import { AdapterLlmError, httpErrorCode, parseRetryAfterMs } from './auth.ts'
import { log } from './log.mjs'
import { setTimeout as sleep } from 'node:timers/promises'

export { DS_BASE }

/** The model exposed in /v1/models — the only one this server serves. */
export const MODEL_ID = process.env.MODEL_ID || 'deepseek/deepseek-v4.1-flash'

/**
 * Builds the WebAuth object from the session token.
 *
 * The token does NOT come from .env — it arrives from the client as the API key
 * (Authorization: Bearer <token>). Cookie/fingerprints are optional;
 * the token alone is enough (verified).
 */
export function authFromToken(token, extra = {}) {
  const value = String(token || '').trim()
  if (!value) throw new Error('missing web session token — send it as the API key (Authorization: Bearer <token>)')
  return {
    token: value,
    cookie: extra.cookie ?? '',
    hifDliq: extra.hifDliq ?? '',
    hifLeim: extra.hifLeim ?? '',
    wasmUrl: extra.wasmUrl ?? '',
    userAgent: extra.userAgent ?? '',
    capturedAt: new Date().toISOString(),
  }
}

/** POST JSON to a web endpoint; returns {status, text, json}. */
async function postJson(auth, path, body, signal) {
  let resp
  try {
    resp = await fetch(`${DS_BASE}${path}`, {
      method: 'POST',
      headers: buildDsHeaders(auth),
      body: JSON.stringify(body ?? {}),
      signal,
    })
  } catch (error) {
    throw new AdapterLlmError(`DeepSeek request failed (${path}): ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  let json
  try { json = JSON.parse(text) } catch {}
  return { status: resp.status, text, json, resp }
}

/** GET JSON na endpoint webowy. */
async function getJson(auth, path, signal) {
  let resp
  try {
    resp = await fetch(`${DS_BASE}${path}`, { method: 'GET', headers: buildDsHeaders(auth), signal })
  } catch (error) {
    throw new AdapterLlmError(`DeepSeek request failed (${path}): ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  let json
  try { json = JSON.parse(text) } catch {}
  return { status: resp.status, text, json, resp }
}

/** Does the response look like an anti-bot HTML page instead of JSON. */
function looksLikeHtml(text) {
  return /^\s*<!doctype html|^\s*<html/i.test(String(text || ''))
}

// ── Sessions ───────────────────────────────────────────────────────────────

/** Creates a new chat session. Returns chat_session_id. */
export async function newChatSession(auth, signal) {
  return createChatSession(auth, signal)
}

/**
 * Stops generation on the DeepSeek side (best effort).
 *
 * `stop_stream` is needed when we abandon the stream halfway: the client disconnected,
 * the idle watchdog fired, or the consumer stopped iterating. Without it, generation keeps
 * running on the account (blocking the next request, because the web allows only one at a time).
 *
 * Payload (verified): {chat_session_id, message_id} — `message_id` is the
 * `response_message_id` from the `ready` event/snapshot, NOT the user message id.
 */
export async function stopChatStream(auth, sessionId, messageId) {
  if (!sessionId || messageId === undefined || messageId === null) return false
  try {
    const r = await postJson(
      auth,
      '/api/v0/chat/stop_stream',
      { chat_session_id: sessionId, message_id: messageId },
      AbortSignal.timeout(10_000),
    )
    return !envelopeError(r.json)
  } catch {
    return false
  }
}

/** Deletes a session (best effort — errors are ignored). */
export async function deleteChatSession(auth, sessionId) {
  if (!sessionId) return false
  try {
    const r = await postJson(auth, '/api/v0/chat_session/delete', { chat_session_id: sessionId }, AbortSignal.timeout(10_000))
    return !envelopeError(r.json)
  } catch {
    return false
  }
}



// ── Files / images ──────────────────────────────────────────────────────

const FILE_READY = new Set(['SUCCESS', 'READY', 'PARSED', 'FINISHED'])
const FILE_FAILED = new Set(['FAILED', 'ERROR', 'BLOCKED', 'REJECTED'])

/**
 * Uploads a file (image or document) and returns {fileId, status, modelKind, isImage}.
 *
 * NOTE: the returned file is usually in PENDING status — it is only usable in
 * `ref_file_ids` after waitForFile(). For a ready id, use uploadFileReady().
 */
export async function uploadFile(auth, { data, mediaType, name }, signal) {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data)
  const pow = await createPowHeader(auth, '/api/v0/file/upload_file', signal)
  const headers = { ...buildDsHeaders(auth) }
  // multipart sets its own boundary — content-type must be removed
  delete headers['content-type']
  headers['x-ds-pow-response'] = pow
  const form = new FormData()
  form.append('file', new Blob([bytes], { type: mediaType || 'application/octet-stream' }), name || 'file.bin')
  let resp
  try {
    resp = await fetch(`${DS_BASE}/api/v0/file/upload_file`, { method: 'POST', headers, body: form, signal })
  } catch (error) {
    throw new AdapterLlmError(`DeepSeek upload failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  let json
  try { json = JSON.parse(text) } catch {}
  if (looksLikeHtml(text)) throw new AdapterLlmError('DeepSeek upload hit an anti-bot HTML page', 'AUTH', { status: resp.status })
  if (!resp.ok) {
    throw new AdapterLlmError(`DeepSeek upload failed (HTTP ${resp.status}): ${text.slice(0, 200)}`, httpErrorCode(resp.status), { status: resp.status })
  }
  const biz = envelopeError(json)
  if (biz) {
    // NOTE (measured on a live account): DeepSeek rate-limits UPLOADS separately from
    // generation. Code 7 = "rate limit reached" (also mute/busy) — if we classified it as
    // PROVIDER_ERROR, uploadAttachments would treat the attachment as permanently broken and
    // skip the image/file even though waiting would have been enough. So every throttling
    // variant maps to RATE_LIMIT, letting the layer above retry.
    const throttled = biz.code === 5 || biz.code === 7 || isThrottled(biz.msg) || isMutedError(biz) || isBusyGenerating(biz.msg)
    throw new AdapterLlmError(
      `DeepSeek upload rejected (code ${biz.code}): ${biz.msg}`,
      throttled ? 'RATE_LIMIT' : 'PROVIDER_ERROR',
      { status: resp.status },
    )
  }
  const file = json?.data?.biz_data
  const fileId = file?.id
  if (typeof fileId !== 'string' || !fileId) throw new AdapterLlmError('DeepSeek upload returned no file id', 'MALFORMED_RESPONSE', { status: resp.status })
  return { fileId, status: file.status, modelKind: file.model_kind, isImage: file.is_image === true, name: file.file_name }
}

/** Reads a file status (GET fetch_files?file_ids=...). */
export async function fileStatus(auth, fileId, signal) {
  const r = await getJson(auth, `/api/v0/file/fetch_files?file_ids=${encodeURIComponent(fileId)}`, signal)
  const files = r.json?.data?.biz_data?.files
  const file = Array.isArray(files) ? files.find((f) => f?.id === fileId) ?? files[0] : undefined
  return { status: file?.status, errorCode: file?.error_code, file }
}

/**
 * Waits until the file is ready for use in `ref_file_ids`.
 *
 * This is the step that separates a working image from `biz_code 9 invalid ref file id`.
 */
export async function waitForFile(auth, fileId, { timeoutMs = 90_000, intervalMs = 700, signal, onWait } = {}) {
  const deadline = Date.now() + timeoutMs
  let last = 'PENDING'
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new AdapterLlmError('upload wait aborted', 'ABORTED')
    let snapshot
    try {
      snapshot = await fileStatus(auth, fileId, signal)
    } catch {
      snapshot = { status: undefined }
    }
    last = snapshot.status || last
    if (FILE_READY.has(last)) return { fileId, status: last, file: snapshot.file }
    if (FILE_FAILED.has(last)) {
      throw new AdapterLlmError(`DeepSeek could not process the file (status ${last}${snapshot.errorCode ? `, error ${snapshot.errorCode}` : ''})`, 'PROVIDER_ERROR')
    }
    onWait?.(last)
    await sleep(intervalMs, undefined, { signal }).catch(() => {})
  }
  throw new AdapterLlmError(`DeepSeek file was still ${last} after ${timeoutMs}ms`, 'TIMEOUT')
}

/** Upload + wait for readiness. This is the real entry point for images. */
export async function uploadFileReady(auth, file, options = {}) {
  const uploaded = await uploadFile(auth, file, options.signal)
  if (FILE_READY.has(uploaded.status)) return uploaded
  const ready = await waitForFile(auth, uploaded.fileId, { ...options, timeoutMs: options.timeoutMs ?? 90_000 })
  return { ...uploaded, status: ready.status }
}

// ── Completion (streaming) ──────────────────────────────────────────────

/**
 * Turns a web API error into an error with a code the OpenAI client understands.
 * Returns {status, code, message, retryAfterMs}.
 */
/** Does the message mean the input limit was exceeded (the web returns this as biz_msg). */
function isContextLimitError(message) {
  return /length\s+limit\s+reached|context\s+(?:length|window)|too\s+many\s+tokens|maximum\s+context|input.{0,20}(?:too\s+long|limit)|内容.{0,12}(?:过长|太长)|上下文.{0,12}(?:过长|超出)/i.test(String(message ?? ''))
}

function envelopeFailure(json, status) {
  const biz = envelopeError(json)
  if (!biz) return undefined
  const muted = isMutedError(biz)
  const busy = !muted && isBusyGenerating(biz.msg)
  const throttled = !muted && !busy && isThrottled(biz.msg)
  const until = muteUntilMs(json)
  if (muted) {
    const seconds = until ? Math.max(1, Math.round((until - Date.now()) / 1000)) : undefined
    return { status: 429, code: 'rate_limit_error', message: `DeepSeek account is temporarily muted${seconds ? ` for ~${seconds}s` : ''} (web-side throttle, not a ban).`, retryAfterMs: until ? Math.max(0, until - Date.now()) : 60_000 }
  }
  if (busy) return { status: 429, code: 'rate_limit_error', message: 'DeepSeek Web allows only one generation per account at a time.', retryAfterMs: 5_000 }
  if (throttled) return { status: 429, code: 'rate_limit_error', message: 'DeepSeek is throttling this account (sending too fast).', retryAfterMs: 10_000 }
  if (isInvalidSessionError(biz)) return { status: 409, code: 'invalid_session', message: `DeepSeek chat session is no longer valid (${biz.code}: ${biz.msg}).`, retryAfterMs: 0 }
  if (biz.code === 9) return { status: 400, code: 'invalid_request_error', message: `DeepSeek rejected a referenced file (${biz.msg}). The file must be fully processed (status SUCCESS) before use.` }
  // The web session accumulates messages on its side and after ~40 turns rejects new
  // ones with `message count exceeded` (biz_code 3). The history lives on the client
  // anyway, so the right reaction is to recreate the session — hence mapping this to
  // INVALID_SESSION, which openai.mjs handles through getSession(forceNew).
  if (biz.code === 3 || /message\s+count\s+exceeded/i.test(biz.msg)) {
    return {
      status: 409,
      code: 'invalid_session',
      message: `The DeepSeek web session reached its message limit (${biz.msg}) — recreating it.`,
    }
  }
  if (isContextLimitError(biz.msg)) {
    return {
      status: 400,
      code: 'context_length_exceeded',
      message: `DeepSeek rejected the request: input limit exceeded (${biz.msg}). Shorten the history or raise MAX_PROMPT_CHARS.`,
    }
  }
  return { status: status >= 400 ? status : 502, code: 'provider_error', message: `DeepSeek error ${biz.code}: ${biz.msg}` }
}

/**
 * Shared SSE reader for the stream-generating chat endpoints (`completion` and `continue`).
 *
 * Both endpoints return the same patch-stream format and need identical handling: a PoW
 * header, content-type validation, an idle watchdog, and `stop_stream` when the stream is
 * abandoned early. Only the target path and the JSON body differ.
 *
 * `meta` events ARE forwarded to the caller (they carry the response message id); the
 * transport still remembers the id so it can stop generation on abandonment.
 */
async function* readChatStream(auth, { sessionId, path, body, signal, idleTimeoutMs }) {
  const pow = await createPowHeader(auth, path, signal)
  const headers = {
    ...buildDsHeaders(auth, `${DS_BASE}/a/chat/s/${sessionId}`),
    accept: 'text/event-stream',
    'x-ds-pow-response': pow,
  }

  const controller = new AbortController()
  const composed = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal

  // DeepSeek occasionally answers 202 Accepted with an empty body instead of the SSE
  // stream — a transient "the request was accepted, the stream is not ready yet" state
  // (seen intermittently on the web backend). It is not a real error and not auth-related,
  // so it is retried here with a short backoff instead of failing the whole turn.
  const ACCEPTED_RETRIES = 5
  const ACCEPTED_BACKOFF_MS = [500, 1_000, 2_000, 4_000, 8_000]
  let resp
  for (let attempt = 0; ; attempt += 1) {
    try {
      resp = await fetch(`${DS_BASE}${path}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: composed,
      })
    } catch (error) {
      if (signal?.aborted) throw new AdapterLlmError('request aborted by the client', 'ABORTED', { cause: error })
      throw new AdapterLlmError(`DeepSeek ${path} request failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
    }
    if (resp.status !== 202) break
    // Drain the body so the connection can be reused, then wait before trying again.
    await resp.text().catch(() => '')
    if (attempt >= ACCEPTED_RETRIES - 1) {
      throw new AdapterLlmError(
        `DeepSeek ${path} kept answering HTTP 202 (accepted, no stream) after ${ACCEPTED_RETRIES} attempts`,
        'TRANSPORT',
        { status: 202 },
      )
    }
    const waitMs = ACCEPTED_BACKOFF_MS[Math.min(attempt, ACCEPTED_BACKOFF_MS.length - 1)]
    log.warn('DeepSeek returned HTTP 202 (accepted, no stream) — retrying', {
      path,
      attempt: attempt + 1,
      retries: ACCEPTED_RETRIES,
      wait_ms: waitMs,
    })
    await sleep(waitMs, undefined, { signal }).catch(() => {})
    if (signal?.aborted) throw new AdapterLlmError('request aborted by the client', 'ABORTED')
  }

  const contentType = String(resp.headers.get('content-type') ?? '')
  if (!resp.ok || !contentType.includes('text/event-stream')) {
    const text = await resp.text().catch(() => '')
    let json
    try { json = JSON.parse(text) } catch {}
    if (looksLikeHtml(text)) {
      throw new AdapterLlmError('DeepSeek returned an anti-bot HTML page instead of a stream (session token expired?)', 'AUTH', { status: resp.status })
    }
    const failure = envelopeFailure(json, resp.status)
    const retryAfter = parseRetryAfterMs(resp.headers.get('retry-after'))
    // Codes must be stable: openai.mjs recognises "expired session -> recreate it" by them.
    const failureCode = !failure
      ? httpErrorCode(resp.status)
      : failure.status === 429
        ? 'RATE_LIMIT'
        : failure.status === 401 || failure.status === 403
          ? 'AUTH'
          : failure.status === 409
            ? 'INVALID_SESSION'
            : 'PROVIDER_ERROR'
    throw new AdapterLlmError(
      failure?.message ?? `DeepSeek ${path} failed (HTTP ${resp.status}): ${text.slice(0, 200)}`,
      failureCode,
      {
        status: failure?.status ?? resp.status,
        ...(failure?.retryAfterMs || retryAfter ? { providerRetryAfterMs: failure?.retryAfterMs || retryAfter } : {}),
        cause: new Error(text.slice(0, 300)),
      },
    )
  }

  // Idle watchdog: abort the stream when SSE stays silent for too long.
  let timer = null
  let settled = false
  let fireIdle
  const idle = new Promise((_, reject) => { fireIdle = reject })
  const arm = () => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      if (settled) return
      controller.abort('idle')
      fireIdle(new AdapterLlmError(`DeepSeek stream idle for ${idleTimeoutMs}ms`, 'TIMEOUT'))
    }, idleTimeoutMs)
    timer.unref?.()
  }
  arm()

  let responseMessageId
  let finishedNaturally = false
  try {
    const iterator = parseWebSse(resp.body)[Symbol.asyncIterator]()
    for (;;) {
      const next = await Promise.race([iterator.next(), idle])
      arm()
      if (next.done) {
        finishedNaturally = true
        break
      }
      const event = next.value
      // Remember the response message id (needed by stop_stream/continue), but still
      // forward the event so the caller can track it across rounds.
      if (event.kind === 'meta') responseMessageId = event.messageId
      yield event
    }
  } finally {
    settled = true
    if (timer) clearTimeout(timer)
    try { controller.abort('consumer stopped') } catch {}
    // We abandoned the stream early (client disconnect, idle timeout, or a consumer that
    // stopped iterating) -> tell DeepSeek to stop generating. Best effort: the account
    // allows only one generation at a time, so leaving it running blocks the next request.
    // Fire-and-forget, with its own timeout (the caller signal is already aborted here).
    if (!finishedNaturally && responseMessageId !== undefined) {
      void stopChatStream(auth, sessionId, responseMessageId)
    }
  }
}

/**
 * Starts a new completion for the given prompt (the first round of a turn).
 *
 * @returns async generator of events: {kind:'text'|'thinking'|'status'|'meta'|'finish'|'error'}
 * @throws AdapterLlmError with .failure.code when the request cannot start
 */
export async function* streamCompletion(auth, params) {
  const {
    sessionId,
    prompt,
    refFileIds = [],
    thinkingEnabled = false,
    searchEnabled = false,
    modelType = 'default',
    signal,
    idleTimeoutMs = 120_000,
  } = params

  yield* readChatStream(auth, {
    sessionId,
    path: '/api/v0/chat/completion',
    body: {
      chat_session_id: sessionId,
      parent_message_id: null,
      prompt,
      ref_file_ids: refFileIds,
      thinking_enabled: thinkingEnabled,
      search_enabled: searchEnabled,
      model_type: modelType,
      action: null,
      preempt: false,
    },
    signal,
    idleTimeoutMs,
  })
}

/**
 * NOTE (2026-09-30): `continueChatStream` (`POST /api/v0/chat/continue`) was REMOVED.
 *
 * The PoW challenge is bound to the target path, and DeepSeek's challenge endpoint now
 * rejects that path outright:
 *
 *     POST /api/v0/chat/create_pow_challenge {"target_path":"/api/v0/chat/continue"}
 *     -> {"code":0,"data":{"biz_code":1,"biz_msg":"INVALID_TARGET_PATH"}}
 *
 * (`/chat/completion`, `/chat/regenerate` and `/file/upload_file` are still accepted.)
 * Auto-continue therefore always rebuilds the prompt and goes through `/chat/completion`.
 */

/**
 * Collects the whole stream into {text, thinking, finish, events}.
 * Used by non-streaming mode and internal retries.
 */
export async function collectCompletion(auth, params) {
  let text = ''
  let thinking = ''
  let finish
  for await (const event of streamCompletion(auth, params)) {
    if (event.kind === 'text') text += event.text
    else if (event.kind === 'thinking') thinking += event.text
    else if (event.kind === 'finish') finish = event.reason
    else if (event.kind === 'error') {
      if (event.code === 'RATE_LIMIT') {
        throw new AdapterLlmError(event.message, 'RATE_LIMIT', { providerRetryAfterMs: event.retryAfterMs ?? 5_000 })
      }
      throw new AdapterLlmError(event.message, 'PROVIDER_ERROR')
    }
  }
  return { text, thinking, finish }
}

/** Quick token liveness check (GET users/current). */
export async function verifyToken(auth, signal) {
  const r = await getJson(auth, '/api/v0/users/current', signal)
  if (looksLikeHtml(r.text)) return { ok: false, error: 'anti-bot HTML page' }
  const biz = envelopeError(r.json)
  if (biz) return { ok: false, error: `${biz.code}: ${biz.msg}` }
  if (r.status !== 200) return { ok: false, error: `HTTP ${r.status}` }
  const user = r.json?.data?.biz_data ?? r.json?.data ?? {}
  return { ok: true, user: { id: user?.id, email: user?.email, mobile: user?.mobile_number } }
}

/**
 * Turns OFF model training for this account ("Improve the model for everyone").
 *
 * DeepSeek's web client writes that privacy switch as `training_allowed` on
 * `POST /api/v0/users/update_settings`, and the session token alone authorises it — no
 * cookie, no PoW header (verified live 2026-10-10: HTTP 200, code 0).
 *
 * Used by the home server when `DISABLE_TRAINING=1` (see pool.mjs) and by
 * `tools/disable-training.py` for a one-off batch over an explicit token list.
 */
export async function disableTraining(auth, signal) {
  const r = await postJson(auth, '/api/v0/users/update_settings', { training_allowed: false }, signal)
  if (looksLikeHtml(r.text)) return { ok: false, error: 'anti-bot HTML page' }
  const biz = envelopeError(r.json)
  if (biz) return { ok: false, error: `${biz.code}: ${biz.msg}` }
  // WARNING: the envelope keeps the REAL business error in data.biz_code while code is 0.
  const inner = r.json?.data
  if (inner && typeof inner === 'object' && Number(inner.biz_code) > 0) {
    return { ok: false, error: `${inner.biz_code}: ${inner.biz_msg ?? 'unknown error'}` }
  }
  if (r.status !== 200) return { ok: false, error: `HTTP ${r.status}` }
  return { ok: true }
}
