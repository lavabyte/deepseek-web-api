// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * OpenAI-compatible HTTP server backed by a chat.deepseek.com web session.
 *
 * AUTH — the API key is a COMMA-SEPARATED LIST of chat.deepseek.com session tokens:
 *
 *     Authorization: Bearer tok1,tok2,tok3
 *
 * A session token is Base64 (letters, digits, `+`, `/`, `=`), so a comma never occurs
 * inside one — the split is unambiguous. `tok1,tok2,tok3` is therefore three accounts, not
 * one malformed key. A single token with no comma works exactly as before.
 *
 * Each token is a separate DeepSeek account, and the web endpoint allows only ONE
 * generation per account at a time, so the pool spreads concurrent requests across the
 * listed accounts (see src/pool.mjs). Sessions, rate-limit cooldowns and quotas are keyed
 * by the individual token digest, so everything lines up.
 *
 * Endpoints:
 *   GET  /v1/models               -> model list (one: deepseek/deepseek-v4-flash)
 *   POST /v1/chat/completions     -> chat (stream=true and false)
 *   GET  /health                  -> token-pool state (?verify=1 forces a live re-check)
 *
 * Configuration (.env):
 *   PORT                    listen port (default 8787)
 *   HOST                    interface (default 127.0.0.1)
 *   MODEL_ID                model name exposed in /v1/models
 *   MAX_PROMPT_CHARS        prompt length limit (default 1000000)
 *   AUTO_CONTINUE=0         disables automatically continuing a cut-off reply
 *   DATA_DIR                directory for session state (default ./data)
 *   ACCESS_ALLOWLIST        "all" (default) or comma-separated sha256 token hashes
 *   QUOTA_MAX_REQUESTS      0 (default, unlimited) or requests per window per account
 *   QUOTA_WINDOW_MS         quota window (default 3600000 = 1 hour)
 *   SESSION_TTL_MS          drop unused session state after this (default 30 days)
 *   SESSION_MAX_ENTRIES     hard cap on stored sessions (default 1000)
 *
 * Run: npm start
 */
// MUST be the first import: ESM evaluates dependencies in source order, and every module
// below reads process.env while it initialises. Without this, .env would arrive too late
// (see src/env.mjs).
import './env.mjs'

import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'
import { MODEL_ID } from './deepseek.mjs'
import { log, clientIp, tokenPrefix } from './log.mjs'
import { ensureSession, tokenKey, DATA_DIR } from './session.mjs'
import { runChatCompletion } from './openai.mjs'
import { acquireToken, poolStatus, verifyPool, reviveTokens, reportOutcome } from './pool.mjs'
import { accessControlEnabled, isTokenAllowed, consumeQuota, quotaEnabled } from './access.mjs'
import { muteToken, DEFAULT_COOLDOWN_MS } from './ratelimit.mjs'

const PORT = Number(process.env.PORT || 8787)
const HOST = process.env.HOST || '127.0.0.1'
const CREATED_AT = Math.floor(Date.now() / 1000)
/** How many OTHER accounts a request may be moved to after its account turns out to be dead. */
const POOL_RETRIES = Math.max(0, Number(process.env.POOL_RETRIES || 1))

// ── Helpers ─────────────────────────────────────────────────────────────

/** Security headers applied to every response. */
function securityHeaders() {
  return {
    'x-content-type-options': 'nosniff',
    'referrer-policy': 'no-referrer',
    // The API returns JSON/SSE only; it must never be framed or embedded.
    'x-frame-options': 'DENY',
  }
}

function sendJson(res, status, payload, extraHeaders = {}) {
  const body = JSON.stringify(payload)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    ...securityHeaders(),
    ...extraHeaders,
  })
  res.end(body)
}

function sendError(res, status, message, { type = 'invalid_request_error', code = null, param = null, headers = {} } = {}) {
  sendJson(res, status, { error: { message, type, param, code } }, headers)
}

async function readJsonBody(req, limitBytes = 64 * 1024 * 1024) {
  const chunks = []
  let size = 0
  for await (const chunk of req) {
    size += chunk.length
    if (size > limitBytes) throw Object.assign(new Error(`request body exceeds ${limitBytes} bytes`), { code: 'invalid_request' })
    chunks.push(chunk)
  }
  const text = Buffer.concat(chunks).toString('utf8')
  if (!text.trim()) return {}
  try {
    return JSON.parse(text)
  } catch (error) {
    // A malformed body is the caller's mistake — 400, not a 500 from the server.
    throw Object.assign(new Error(`invalid JSON body: ${error.message}`), { code: 'invalid_request' })
  }
}

/** Internal error code -> HTTP response in OpenAI format. */
function httpStatusFor(error) {
  const code = error?.code || error?.failure?.code
  if (code === 'AUTH' || code === 'MISSING_CREDENTIAL') return 401
  if (code === 'RATE_LIMIT' || code === 'POOL_THROTTLED' || code === 'quota_exceeded') return 429
  if (code === 'TIMEOUT' || code === 'POOL_TIMEOUT') return 504
  if (code === 'ABORTED') return 499
  if (code === 'TRANSPORT') return 502
  if (code === 'EMPTY_RESPONSE') return 503
  if (code === 'INVALID_SESSION') return 409
  // The pool itself is unusable: nothing configured, or every token retired. That is a
  // server-side problem, not the caller's — 503, so clients retry instead of giving up.
  if (code === 'POOL_EMPTY' || code === 'POOL_EXHAUSTED') return 503
  // A malformed body, an oversized one, or an empty `messages` array: the caller sent
  // something wrong, so say 400 rather than letting it look like a server failure.
  if (code === 'invalid_request') return 400
  // Exceeding the context is a CLIENT task error, not a server error — 400, so the client
  // knows it should shorten the history rather than retry.
  if (code === 'CONTEXT_WINDOW_EXCEEDED') return 400
  return 500
}

function openAiErrorType(status) {
  if (status === 401) return 'authentication_error'
  if (status === 429) return 'rate_limit_error'
  if (status === 400) return 'invalid_request_error'
  if (status >= 500) return 'server_error'
  return 'provider_error'
}

function errorMessage(error) {
  const base = error?.message ?? String(error)
  const retry = error?.failure?.providerRetryAfterMs ?? error?.providerRetryAfterMs
  return retry ? `${base} (retry after ~${Math.ceil(retry / 1000)}s)` : base
}

/** Account-wide request budget exceeded — the client's fault, and it must not mute DeepSeek. */
function quotaError(quota) {
  const retryAfterSec = Math.max(1, Math.ceil(quota.retryAfterMs / 1000))
  return Object.assign(
    new Error(`Quota exceeded: ${quota.limit} requests per window on this account. Retry in ~${retryAfterSec}s.`),
    { code: 'quota_exceeded', providerRetryAfterMs: quota.retryAfterMs },
  )
}

/**
 * Raw API key from the request (the `Bearer ` prefix already removed).
 *
 * The key IS the credential — there is no separate API_KEY in .env. `x-api-key` is
 * accepted too, because several OpenAI clients send that header instead.
 */
function rawKeyFromRequest(req) {
  const header = String(req.headers.authorization || '')
  if (header.startsWith('Bearer ')) return header.slice(7).trim()
  if (header.startsWith('bearer ')) return header.slice(7).trim()
  return String(req.headers['x-api-key'] || '').trim()
}

/** Splits the key into individual DeepSeek tokens (Base64, so a comma is never inside one). */
function tokensFromRequest(req) {
  const raw = rawKeyFromRequest(req)
  if (!raw) return []
  const out = []
  const seen = new Set()
  for (const chunk of raw.split(',')) {
    const token = chunk.trim()
    if (!token || seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

/** Model as seen by the client. */
function modelObject() {
  return {
    id: MODEL_ID,
    object: 'model',
    created: CREATED_AT,
    owned_by: 'deepseek-web',
    permission: [],
    root: MODEL_ID,
    parent: null,
    // Informational fields (some clients read them):
    context_window: 1_048_576,
    max_output_tokens: 32_768,
    capabilities: { vision: true, files: true, tools: true, reasoning: true, streaming: true },
  }
}

/** Builds an OpenAI-format response (non-streaming) from the collected events. */
function completionPayload({ id, created, text, thinking, toolCalls, finishReason, usage, model }) {
  const message = { role: 'assistant', content: text || null }
  if (thinking) message.reasoning_content = thinking
  if (toolCalls.length) {
    message.tool_calls = toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }))
  }
  return {
    id,
    object: 'chat.completion',
    created,
    model,
    choices: [{ index: 0, message, logprobs: null, finish_reason: finishReason }],
    usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  }
}

/** Drains a completion generator into a single reply. */
async function collect(generator) {
  let text = ''
  let reasoning = ''
  let finishReason = 'stop'
  let usage
  const toolCalls = []
  for await (const event of generator) {
    if (event.type === 'text') text += event.text
    else if (event.type === 'thinking') reasoning += event.text
    else if (event.type === 'tool_call') toolCalls.push(event)
    else if (event.type === 'usage') usage = event.usage
    else if (event.type === 'finish') finishReason = event.reason
  }
  return { text, reasoning, finishReason, usage, toolCalls }
}

// ── Handlers ────────────────────────────────────────────────────────────

async function handleModels(req, res) {
  sendJson(res, 200, { object: 'list', data: [modelObject()] })
}

/**
 * Pool diagnostics. Public by design: it exposes counts and an 8-char fingerprint per
 * token, never a credential. `?verify=1` forces a live re-check against DeepSeek.
 */
async function handleHealth(req, res, url) {
  const tokens = tokensFromRequest(req)
  if (tokens.length === 0) {
    sendJson(res, 200, {
      ok: true,
      auth: { required: true, source: 'comma-separated API key', hint: 'Authorization: Bearer tok1,tok2,tok3' },
      pool: poolStatus([]),
      model: MODEL_ID,
      uptime_s: Math.round(process.uptime()),
      data_dir: DATA_DIR,
    })
    return
  }

  // Verifying always matches the previous behaviour (a bad key reports 503). It costs one
  // read-only request per token; `?verify=0` skips it and reports the cached state.
  const verify = url?.searchParams.get('verify') !== '0'
  const status = verify
    ? await verifyPool(tokens).catch(() => poolStatus(tokens))
    : poolStatus(tokens)

  sendJson(res, status.usable > 0 ? 200 : 503, {
    ok: status.usable > 0,
    auth: { required: true, source: 'comma-separated API key', tokens_in_key: tokens.length },
    pool: status,
    model: MODEL_ID,
    uptime_s: Math.round(process.uptime()),
    data_dir: DATA_DIR,
  })
}

/** A generation bound to one pool account. */
function startCompletion(lease, { body, messages, thinking, search, signal }) {
  return runChatCompletion({
    auth: lease.auth,
    getSession: async (forceNew) => ensureSession(lease.auth, { forceNew, signal }),
    messages,
    tools: body?.tools,
    toolChoice: body?.tool_choice,
    thinking,
    search,
    signal,
  })
}

async function handleChatCompletions(req, res, body, setPoolEntry) {
  const stream = body?.stream === true
  const messages = Array.isArray(body?.messages) ? body.messages : []
  if (messages.length === 0) {
    sendError(res, 400, 'messages must be a non-empty array', { param: 'messages' })
    return
  }

  const tokens = tokensFromRequest(req)
  if (tokens.length === 0) {
    sendError(res, 401, 'Missing API key — send your chat.deepseek.com session token(s) as the API key', {
      type: 'authentication_error',
      code: 'invalid_api_key',
    })
    return
  }

  // The allowlist (when enabled) is checked PER TOKEN: a key may list several accounts and
  // only some of them may be on the list. An empty result means the caller may not use any
  // of the accounts they presented.
  const allowed = tokens.filter((token) => isTokenAllowed(tokenKey(token)))
  if (allowed.length === 0) {
    sendError(res, 403, 'None of the tokens in this API key are on the access allowlist (ACCESS_ALLOWLIST).', {
      type: 'authentication_error',
      code: 'access_denied',
    })
    return
  }

  const model = String(body.model || MODEL_ID)
  const id = `chatcmpl-${randomUUID().replace(/-/g, '')}`
  const created = Math.floor(Date.now() / 1000)
  const controller = new AbortController()
  // Client disconnected -> abort the stream on the web side (otherwise generation keeps running).
  req.on('aborted', () => controller.abort('client aborted'))
  res.on('close', () => {
    if (!res.writableEnded) controller.abort('client closed')
  })

  // Thinking is ON BY DEFAULT: the web model streams its reasoning as THINK fragments and
  // clients that understand `reasoning_content` get the thoughts as they happen. It can be
  // turned off per request with `reasoning_effort: "none"` or `thinking: false`.
  const thinking = body?.reasoning_effort === 'none' || body?.thinking === false || body?.reasoning === false
    ? false
    : true
  const search = body?.web_search === true

  // ── Non-streaming: nothing has been sent yet, so a dead account can be swapped out ──
  if (!stream) {
    for (let attempt = 0; ; attempt++) {
      let lease
      try {
        lease = await acquireToken(allowed, { signal: controller.signal })
        setPoolEntry(lease.entry)
        const quota = consumeQuota(lease.entry.hash)
        if (!quota.allowed) throw quotaError(quota)

        const reply = await collect(startCompletion(lease, { body, messages, thinking, search, signal: controller.signal }))
        reportOutcome(lease.entry, null)
        sendJson(res, 200, completionPayload({
          id, created, model,
          text: reply.text,
          thinking: reply.reasoning,
          toolCalls: reply.toolCalls,
          finishReason: reply.finishReason,
          usage: reply.usage,
        }))
        return
      } catch (error) {
        reportOutcome(lease?.entry, error)
        // Only a RETIRED account is worth another try — and only if one is left to try.
        if (!lease || lease.entry.state !== 'dead' || attempt >= POOL_RETRIES) throw error
        log.warn('account rejected — retrying on another one', {
          id,
          account_index: lease.entry.index,
          attempt: attempt + 1,
        })
      } finally {
        lease?.release()
      }
    }
  }

  // ── Streaming ─────────────────────────────────────────────────────────
  // The account must be settled BEFORE the SSE head goes out: once the status line is on
  // the wire an error can no longer become an HTTP error, it has to be an SSE event.
  const lease = await acquireToken(allowed, { signal: controller.signal })
  setPoolEntry(lease.entry)
  try {
    const quota = consumeQuota(lease.entry.hash)
    if (!quota.allowed) throw quotaError(quota)

    const generator = startCompletion(lease, { body, messages, thinking, search, signal: controller.signal })

    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    })

    const base = { id, object: 'chat.completion.chunk', created, model }
    const write = (payload) => {
      if (res.writableEnded) return
      res.write(`data: ${JSON.stringify(payload)}\n\n`)
    }
    const chunk = (delta, finishReason = null) => ({ ...base, choices: [{ index: 0, delta, logprobs: null, finish_reason: finishReason }] })

    // First delta with the role + heartbeat, so clients do not consider the connection dead.
    write(chunk({ role: 'assistant', content: '' }))
    const heartbeat = setInterval(() => {
      if (!res.writableEnded) res.write(': keep-alive\n\n')
    }, 15_000)
    heartbeat.unref?.()

    let toolIndex = 0
    let finishReason = 'stop'
    let usage

    try {
      for await (const event of generator) {
        if (event.type === 'text') {
          write(chunk({ content: event.text }))
        } else if (event.type === 'thinking') {
          write(chunk({ reasoning_content: event.text }))
        } else if (event.type === 'tool_call') {
          const index = toolIndex++
          write(chunk({ tool_calls: [{ index, id: event.id, type: 'function', function: { name: event.name, arguments: '' } }] }))
          // Arguments go as a separate delta — that is what OpenAI does too.
          write(chunk({ tool_calls: [{ index, function: { arguments: event.arguments } }] }))
        } else if (event.type === 'usage') {
          usage = event.usage
        } else if (event.type === 'finish') {
          finishReason = event.reason
        }
      }
      write(chunk({}, finishReason))
      write({ ...base, choices: [], usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } })
      reportOutcome(lease.entry, null)
    } catch (error) {
      reportOutcome(lease.entry, error)
      const status = httpStatusFor(error)
      // A provider 429 during streaming must ALSO start the account cooldown — the HTTP status
      // is already sent, so this path bypasses the global handler in the router.
      if ((error?.code ?? error?.failure?.code) === 'RATE_LIMIT') {
        const retryAfterMs = error?.failure?.providerRetryAfterMs ?? error?.providerRetryAfterMs
        const muted = muteToken(lease.entry.hash, retryAfterMs)
        log.warn('rate limit cooldown started', { id, token: lease.entry.prefix, cooldown_ms: Math.round(muted) })
      }
      // The stream has already started — the HTTP code cannot change. Send the error as an event.
      write({ error: { message: errorMessage(error), type: openAiErrorType(status), code: error?.code ?? error?.failure?.code ?? 'provider_error' } })
    } finally {
      clearInterval(heartbeat)
    }
  } catch (error) {
    // Nothing was streamed yet, so the router can still turn this into a proper HTTP error.
    if (!res.headersSent) throw error
    log.error('stream failed after the response had started', { id, message: errorMessage(error) })
  } finally {
    lease.release()
    // Only close an SSE response we actually opened — a pre-stream failure (quota, pool
    // error) must be left alone so the router can reply with a real HTTP status.
    if (res.headersSent && !res.writableEnded) {
      res.write('data: [DONE]\n\n')
      res.end()
    }
  }
}

// ── Routing ─────────────────────────────────────────────────────────────

/**
 * Request handler with production concerns: a request id, a duration log line for every
 * request (including aborted ones), and a single place that turns thrown errors into
 * OpenAI-shaped HTTP responses.
 */
const server = createServer(async (req, res) => {
  const started = Date.now()
  const requestId = randomUUID().slice(0, 12)
  const method = req.method || 'GET'
  let path = '/'
  /** Which pool entry served the request — set once one is acquired, for the log line. */
  let poolEntry = null
  const setPoolEntry = (entry) => { poolEntry = entry }

  // Every response carries the request id, so a client report can be matched to a log line.
  res.setHeader('x-request-id', requestId)
  res.on('finish', () => {
    const status = res.statusCode
    const level = status >= 500 ? 'error' : status >= 400 ? 'warn' : 'info'
    log[level]('http', {
      id: requestId,
      method,
      path,
      status,
      ms: Date.now() - started,
      ip: clientIp(req),
      token: tokenPrefix(rawKeyFromRequest(req).split(',')[0]),
      account_index: poolEntry?.index,
    })
  })

  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`)
    path = url.pathname.replace(/\/+$/, '') || '/'

    if (method === 'OPTIONS') {
      res.writeHead(204, {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET,POST,OPTIONS',
        'access-control-allow-headers': 'authorization,content-type,x-api-key',
        ...securityHeaders(),
      })
      res.end()
      return
    }

    if (method === 'GET' && (path === '/health' || path === '/v1/health')) {
      await handleHealth(req, res, url)
      return
    }
    if (method === 'GET' && (path === '/tokens' || path === '/v1/tokens')) {
      const tokens = tokensFromRequest(req)
      sendJson(res, 200, { object: 'list', ...poolStatus(tokens) })
      return
    }
    if (method === 'POST' && (path === '/tokens/revive' || path === '/v1/tokens/revive')) {
      const tokens = tokensFromRequest(req)
      sendJson(res, 200, { object: 'list', ...(await reviveTokens(tokens)) })
      return
    }
    if (method === 'GET' && (path === '/v1/models' || path === '/models')) {
      await handleModels(req, res)
      return
    }
    if (method === 'POST' && (path === '/v1/chat/completions' || path === '/chat/completions')) {
      const body = await readJsonBody(req)
      await handleChatCompletions(req, res, body, setPoolEntry)
      return
    }
    sendError(res, 404, `Unknown route: ${method} ${path}`, { code: 'not_found' })
  } catch (error) {
    if (res.headersSent) {
      try {
        res.end()
      } catch {}
      return
    }
    const status = httpStatusFor(error)
    log[status >= 500 ? 'error' : 'warn']('request failed', {
      id: requestId,
      method,
      path,
      status,
      code: error?.code ?? error?.failure?.code ?? null,
      message: errorMessage(error),
    })
    // A 429 must tell the client when to come back. Without `retry-after` most SDKs retry
    // immediately and make the account-level throttle worse.
    const retryAfterMs = error?.failure?.providerRetryAfterMs ?? error?.providerRetryAfterMs
    const headers = status === 429
      ? { 'retry-after': String(Math.max(1, Math.ceil((Number(retryAfterMs) || DEFAULT_COOLDOWN_MS) / 1000))) }
      : {}
    // A provider 429 starts the account cooldown: subsequent requests are answered locally
    // (see ratelimit.mjs) so DeepSeek's throttle window can actually expire. `quota_exceeded`
    // is OUR limit and must not mute the provider account.
    if (status === 429 && (error?.code ?? error?.failure?.code) === 'RATE_LIMIT' && poolEntry) {
      const muted = muteToken(poolEntry.hash, retryAfterMs)
      log.warn('rate limit cooldown started', {
        id: requestId,
        token: poolEntry.prefix,
        cooldown_ms: Math.round(muted),
      })
    }
    sendError(res, status, errorMessage(error), {
      type: openAiErrorType(status),
      code: error?.code ?? error?.failure?.code ?? null,
      headers,
    })
  }
})

// ── Production hardening ────────────────────────────────────────────────

// Slow-loris protection: cap how long a client may take to send headers/body. Generation
// itself is not limited here (SSE streams can legitimately run for minutes).
server.headersTimeout = Number(process.env.HEADERS_TIMEOUT_MS || 30_000)
server.requestTimeout = Number(process.env.REQUEST_TIMEOUT_MS || 300_000)
server.keepAliveTimeout = Number(process.env.KEEP_ALIVE_TIMEOUT_MS || 65_000)

server.listen(PORT, HOST, () => {
  log.info('server listening', {
    url: `http://${HOST}:${PORT}`,
    model: MODEL_ID,
    allowlist: accessControlEnabled,
    quota: quotaEnabled,
    dataDir: DATA_DIR,
  })
})

/**
 * Graceful shutdown: stop accepting connections, let in-flight requests finish, then exit.
 * A second signal forces an immediate exit. Kubernetes/Docker send SIGTERM on rollout and
 * kill after a grace period, so this avoids cutting off a streaming response mid-flight.
 */
let shuttingDown = false
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    if (shuttingDown) {
      log.warn('second signal — forcing exit')
      process.exit(1)
    }
    shuttingDown = true
    log.info('shutting down', { signal })
    server.close(() => {
      log.info('shutdown complete')
      process.exit(0)
    })
    setTimeout(() => {
      log.warn('shutdown timed out — forcing exit')
      process.exit(1)
    }, Number(process.env.SHUTDOWN_TIMEOUT_MS || 10_000)).unref()
  })
}

// Last-resort handlers: log and keep serving rather than crashing on a stray rejection.
process.on('unhandledRejection', (reason) => {
  log.error('unhandled rejection', { message: reason?.message ?? String(reason) })
})
process.on('uncaughtException', (error) => {
  log.error('uncaught exception', { message: error?.message ?? String(error), stack: error?.stack })
})
