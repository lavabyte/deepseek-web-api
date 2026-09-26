// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * OpenAI-compatible HTTP server backed by a chat.deepseek.com web session.
 *
 * Endpoints:
 *   GET  /v1/models               -> model list (one: deepseek/deepseek-v4-flash)
 *   POST /v1/chat/completions     -> chat (stream=true and false)
 *   GET  /health                  -> session state + token verification
 *
 * Configuration (.env):
 *   PORT                    listen port (default 8787) — the ONLY required variable
 *   HOST                    interface (default 127.0.0.1)
 *   MODEL_ID                model name exposed in /v1/models
 *   MAX_PROMPT_CHARS        prompt length limit (default 1000000)
 *   AUTO_CONTINUE=0         disables automatically continuing a cut-off reply
 *   DATA_DIR                directory for session state (default ./data)
 *   ACCESS_ALLOWLIST        "all" (default) or comma-separated sha256 token hashes
 *   QUOTA_MAX_REQUESTS      0 (default, unlimited) or requests per window per token
 *   QUOTA_WINDOW_MS         quota window (default 3600000 = 1 hour)
 *   SESSION_TTL_MS          drop unused session state after this (default 30 days)
 *   SESSION_MAX_ENTRIES     hard cap on stored sessions (default 1000)
 *   ACCESS_ALLOWLIST        "all" (default) or comma-separated sha256 token hashes
 *   QUOTA_MAX_REQUESTS      0 (default, unlimited) or requests per window per token
 *   QUOTA_WINDOW_MS         quota window (default 3600000 = 1 hour)
 *   SESSION_TTL_MS          drop unused session state after this (default 30 days)
 *   SESSION_MAX_ENTRIES     hard cap on stored sessions (default 1000)
 *
 * The web token is NOT configured in .env — the client passes it as the API KEY:
 *   Authorization: Bearer <token-from-chat.deepseek.com>
 * Each token gets its own persistent session (state in data/api-sessions.json).
 *
 * Run: npm start
 */
import { createServer } from 'node:http'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { randomUUID } from 'node:crypto'
import { MODEL_ID, authFromToken, verifyToken } from './deepseek.mjs'
import { log, clientIp, tokenPrefix } from './log.mjs'
import {
  ensureSession,
  currentSessionId,
  tokenKey,
  DATA_DIR,
} from './session.mjs'
import { runChatCompletion } from './openai.mjs'
import { createRequestGate } from './gate.ts'
import { accessControlEnabled, isTokenAllowed, consumeQuota, quotaEnabled, quotaStatus } from './access.mjs'
import { mutedFor, muteToken, cooldownStatus, DEFAULT_COOLDOWN_MS } from './ratelimit.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Minimal .env loader (no dependencies) — does not overwrite variables already set. */
function loadEnv() {
  const file = join(ROOT, '.env')
  if (!existsSync(file)) return
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eq = trimmed.indexOf('=')
    if (eq === -1) continue
    const key = trimmed.slice(0, eq).trim()
    let value = trimmed.slice(eq + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    if (!(key in process.env)) process.env[key] = value
  }
}
loadEnv()

const PORT = Number(process.env.PORT || 8787)
const HOST = process.env.HOST || '127.0.0.1'
const CREATED_AT = Math.floor(Date.now() / 1000)

/**
 * Per-token request gates — serializes generations (one at a time), WITHOUT artificial
 * delays.
 *
 * An earlier version forced a random 2-4 s pause between requests and a long break after
 * 15 requests. Those pauses are disabled: the server sends requests immediately, and if
 * DeepSeek replies 429 the backoff (1s, 5s, 5s, 15s, 1min) is handled reactively in
 * openai.mjs. Serialization remains, because the web endpoint only allows ONE generation
 * per ACCOUNT.
 *
 * The gates are keyed by token, not global: the same account must be serialized, but two
 * different accounts may generate in parallel. A single global gate would make user B
 * wait for user A even though their accounts are independent.
 */
const gates = new Map()
function gateFor(tokenHash) {
  let perToken = gates.get(tokenHash)
  if (!perToken) {
    perToken = createRequestGate({
      allowConcurrent: process.env.ALLOW_CONCURRENT === '1',
      minIntervalMs: 0,
      maxIntervalMs: 0,
      longRunThreshold: 0,
    })
    gates.set(tokenHash, perToken)
  }
  return perToken
}

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
    if (size > limitBytes) throw new Error(`request body exceeds ${limitBytes} bytes`)
    chunks.push(chunk)
  }
  const text = Buffer.concat(chunks).toString('utf8')
  if (!text.trim()) return {}
  return JSON.parse(text)
}

/** Internal error code -> HTTP response in OpenAI format. */
function httpStatusFor(error) {
  const code = error?.code || error?.failure?.code
  if (code === 'AUTH' || code === 'MISSING_CREDENTIAL') return 401
  if (code === 'RATE_LIMIT') return 429
  if (code === 'TIMEOUT') return 504
  if (code === 'ABORTED') return 499
  if (code === 'TRANSPORT') return 502
  if (code === 'EMPTY_RESPONSE') return 503
  if (code === 'INVALID_SESSION') return 409
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

/**
 * Extracts the web session token from the request headers.
 *
 * The client's API key IS the chat.deepseek.com session token — there is no separate
 * API_KEY or token in .env. This way the server holds no secret and supports multiple
 * clients (each token = its own account and its own persistent session).
 */
function tokenFromRequest(req) {
  const header = String(req.headers.authorization || '')
  if (header.startsWith('Bearer ')) return header.slice(7).trim()
  if (header.startsWith('bearer ')) return header.slice(7).trim()
  return String(req.headers['x-api-key'] || '').trim()
}

/**
 * Builds the WebAuth for a request, or null when the token is missing.
 *
 * The cookie is optional and also comes from the request (`x-deepseek-cookie`) —
 * the token alone is enough, so normally nobody sends it.
 */
function authFromRequest(req) {
  const token = tokenFromRequest(req)
  if (!token) return null
  try {
    return authFromToken(token, { cookie: String(req.headers['x-deepseek-cookie'] || '').trim() })
  } catch {
    return null
  }
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

// ── Handlers ────────────────────────────────────────────────────────────

async function handleModels(req, res) {
  sendJson(res, 200, { object: 'list', data: [modelObject()] })
}

async function handleHealth(req, res) {
  // Without a token there is nothing to verify — /health is a public probe.
  const reqAuth = authFromRequest(req)
  if (!reqAuth) {
    sendJson(res, 200, {
      ok: true,
      token: { valid: false, error: 'no token — pass it as the API key (Authorization: Bearer <token>)' },
      session: { id: null },
      model: MODEL_ID,
      uptime_s: Math.round(process.uptime()),
    })
    return
  }

  const tokenHash = tokenKey(reqAuth.token)
  const who = await verifyToken(reqAuth).catch((error) => ({ ok: false, error: error?.message ?? String(error) }))
  let sessionId = currentSessionId(tokenHash)
  if (who.ok && !sessionId) {
    try {
      sessionId = await ensureSession(reqAuth)
    } catch {
      /* a missing session is not a critical error — it will be created on the first request */
    }
  }
  sendJson(res, who.ok ? 200 : 503, {
    ok: who.ok,
    token: who.ok ? { valid: true, account: who.user?.email || who.user?.mobile || who.user?.id } : { valid: false, error: who.error },
    session: { id: sessionId ?? null },
    access: { allowlist: accessControlEnabled, allowed: isTokenAllowed(tokenHash) },
    quota: quotaStatus(tokenHash),
    rate_limit: cooldownStatus(tokenHash),
    model: MODEL_ID,
    uptime_s: Math.round(process.uptime()),
  })
}

/** Builds an OpenAI-format SSE listing and streams the deltas. */
async function handleChatCompletions(req, res, body) {
  const stream = body?.stream === true
  const messages = Array.isArray(body?.messages) ? body.messages : []
  if (messages.length === 0) {
    sendError(res, 400, 'messages must be a non-empty array', { param: 'messages' })
    return
  }

  const reqAuth = authFromRequest(req)
  if (!reqAuth) {
    sendError(res, 401, 'Missing API key — pass your chat.deepseek.com session token as the API key', {
      type: 'authentication_error',
      code: 'invalid_api_key',
    })
    return
  }

  const reqTokenHash = tokenKey(reqAuth.token)
  if (!isTokenAllowed(reqTokenHash)) {
    sendError(res, 403, 'This token is not on the access allowlist (ACCESS_ALLOWLIST).', {
      type: 'authentication_error',
      code: 'access_denied',
    })
    return
  }
  const quota = consumeQuota(reqTokenHash)
  if (!quota.allowed) {
    const retryAfterSec = Math.ceil(quota.retryAfterMs / 1000)
    sendError(res, 429, `Quota exceeded: ${quota.limit} requests per window. Retry in ~${retryAfterSec}s.`, {
      type: 'rate_limit_error',
      code: 'quota_exceeded',
    }, { 'retry-after': String(retryAfterSec) })
    return
  }

  // DeepSeek rate-limit cooldown (see ratelimit.mjs). While muted, the request is answered
  // locally with 429 + `retry-after` and NEVER reaches DeepSeek — contacting it would reset
  // the provider's window and keep the account throttled indefinitely.
  const cooldownMs = mutedFor(reqTokenHash)
  if (cooldownMs > 0) {
    const retryAfterSec = Math.max(1, Math.ceil(cooldownMs / 1000))
    sendError(res, 429, `DeepSeek rate limit active. The account is paused; retry in ~${retryAfterSec}s.`, {
      type: 'rate_limit_error',
      code: 'rate_limit_cooldown',
    }, { 'retry-after': String(retryAfterSec) })
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

  // One "acquire" for the whole HTTP request: covers all auto-continue rounds and uploads.
  // This way two requests from this server never generate at the same time.
  let release
  try {
    release = await gateFor(reqTokenHash).acquire('chat', controller.signal)
  } catch (error) {
    if (controller.signal.aborted) return
    throw error
  }

  const generator = runChatCompletion({
    auth: reqAuth,
    getSession: async (forceNew) => ensureSession(reqAuth, { forceNew, signal: controller.signal }),
    messages,
    tools: body?.tools,
    toolChoice: body?.tool_choice,
    thinking,
    search: body?.web_search === true,
    signal: controller.signal,
  })

  if (!stream) {
    let text = ''
    let reasoning = ''
    let finishReason = 'stop'
    let usage
    const toolCalls = []
    try {
      for await (const event of generator) {
        if (event.type === 'text') text += event.text
        else if (event.type === 'thinking') reasoning += event.text
        else if (event.type === 'tool_call') toolCalls.push(event)
        else if (event.type === 'usage') usage = event.usage
        else if (event.type === 'finish') finishReason = event.reason
      }
    } finally {
      release?.()
    }
    sendJson(res, 200, completionPayload({ id, created, text, thinking: reasoning, toolCalls, finishReason, usage, model }))
    return
  }

  // ── Streaming ─────────────────────────────────────────────────────────
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
  } catch (error) {
    const status = httpStatusFor(error)
    // A provider 429 during streaming must ALSO start the account cooldown — the HTTP status
    // is already sent, so this path bypasses the global handler in the router.
    if ((error?.code ?? error?.failure?.code) === 'RATE_LIMIT') {
      const retryAfterMs = error?.failure?.providerRetryAfterMs ?? error?.providerRetryAfterMs
      const muted = muteToken(reqTokenHash, retryAfterMs)
      log.warn('rate limit cooldown started', { id, token: tokenPrefix(reqAuth.token), cooldown_ms: Math.round(muted) })
    }
    // The stream has already started — the HTTP code cannot change. Send the error as an event.
    write({ error: { message: errorMessage(error), type: openAiErrorType(status), code: error?.code ?? error?.failure?.code ?? 'provider_error' } })
  } finally {
    clearInterval(heartbeat)
    release?.()
    if (!res.writableEnded) {
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
      token: tokenPrefix(tokenFromRequest(req)),
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
      await handleHealth(req, res)
      return
    }
    if (method === 'GET' && (path === '/v1/models' || path === '/models')) {
      await handleModels(req, res)
      return
    }
    if (method === 'POST' && (path === '/v1/chat/completions' || path === '/chat/completions')) {
      const body = await readJsonBody(req)
      await handleChatCompletions(req, res, body)
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
    if (status === 429 && (error?.code ?? error?.failure?.code) === 'RATE_LIMIT') {
      const muted = muteToken(tokenKey(tokenFromRequest(req)), retryAfterMs)
      log.warn('rate limit cooldown started', {
        id: requestId,
        token: tokenPrefix(tokenFromRequest(req)),
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