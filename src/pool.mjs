// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Per-request DeepSeek token pool.
 *
 * FORMAT
 *   The client's API key IS a comma-separated list of chat.deepseek.com session tokens:
 *
 *       Authorization: Bearer tok1,tok2,tok3
 *
 *   A session token is Base64 (letters, digits, `+`, `/`, sometimes `=`), so a comma can
 *   never occur inside one — the split is unambiguous. `tok1,tok2,tok3` is therefore three
 *   accounts, not one malformed key. A single token with no comma keeps working exactly as
 *   before.
 *
 * WHY A POOL
 *   The web endpoint allows only ONE generation per account at a time, so a single token
 *   serialises every caller. With several tokens the pool spreads requests across accounts:
 *   pick the LEAST-USED usable account, hold it for the whole HTTP request, release it at
 *   the end. If every listed account is busy the request queues, and it is aborted if the
 *   client goes away.
 *
 * SHARED STATE
 *   Entries are keyed by the token digest and live in a process-global registry, so two
 *   clients that list the same account do not drive it concurrently. Sessions
 *   (data/api-sessions.json), rate-limit cooldowns (data/rate-limits.json) and quotas are
 *   keyed the same way, so all of it lines up.
 *
 * A token is a full account credential. It is never logged (only an 8-char sha256 prefix),
 * never returned by any endpoint, and never written to disk by this module.
 */
// Loads .env before this module reads process.env (see src/env.mjs for why the ordering
// matters). A no-op when the importer already did it.
import './env.mjs'

import { authFromToken, verifyToken as verifyWebToken } from './deepseek.mjs'
import { currentSessionId, tokenKey } from './session.mjs'
import { mutedFor } from './ratelimit.mjs'
import { quotaStatus } from './access.mjs'
import { log, tokenPrefix } from './log.mjs'

/** How long a pre-flight verification may take before it is called a network failure. */
const VERIFY_TIMEOUT_MS = Number(process.env.TOKEN_VERIFY_TIMEOUT_MS || 20_000)
/** How long a request may wait in the queue before giving up (the HTTP timeout still applies). */
const QUEUE_TIMEOUT_MS = Number(process.env.POOL_QUEUE_TIMEOUT_MS || 300_000)
/** Re-check interval while every usable account is busy or cooling down. */
const POLL_MS = 250
/** `ALLOW_CONCURRENT=1` stops the pool from serialising requests per account. */
const ALLOW_CONCURRENT = process.env.ALLOW_CONCURRENT === '1'

/**
 * Splits the API key into individual DeepSeek tokens.
 *
 * Commas are the only separator; whitespace around entries is trimmed, empties and
 * duplicates are dropped. Order is preserved (it decides tie-breaking).
 *
 * @param {string} raw the Authorization value with the `Bearer ` prefix already removed
 * @returns {string[]}
 */
export function parseTokenList(raw) {
  const out = []
  const seen = new Set()
  for (const chunk of String(raw ?? '').split(',')) {
    const token = chunk.trim()
    if (!token) continue
    if (seen.has(token)) {
      log.warn('duplicate token in the API key — ignored', { token: tokenPrefix(token) })
      continue
    }
    seen.add(token)
    out.push(token)
  }
  return out
}

/** @type {Map<string, object>} token digest -> pool entry (shared across requests). */
const registry = new Map()

/** Returns (creating on first sight) the registry entry for one token. */
function entryFor(token) {
  const hash = tokenKey(token)
  let entry = registry.get(hash)
  if (!entry) {
    entry = {
      token,
      hash,
      cookie: '',
      index: registry.size,
      prefix: tokenPrefix(token),
      /** unknown -> ok | dead. `dead` is sticky until revive() or a restart. */
      state: 'unknown',
      account: null,
      lastError: null,
      lastErrorAt: null,
      verifiedAt: null,
      /** `busy` is the exclusive lock; `inflight` counts requests even when not exclusive. */
      busy: false,
      inflight: 0,
      requests: 0,
      errors: 0,
      lastUsedAt: null,
      /** In-flight verification, so N parallel requests do not verify the same token N times. */
      verifying: null,
    }
    registry.set(hash, entry)
  }
  return entry
}

// ── errors ──────────────────────────────────────────────────────────────

function poolError(code, message, extra = {}) {
  return Object.assign(new Error(message), { code, ...extra })
}

function abortError(signal) {
  const reason = signal?.reason
  if (reason instanceof Error) return reason
  return poolError('ABORTED', typeof reason === 'string' ? reason : 'request aborted')
}

// ── state ───────────────────────────────────────────────────────────────

function isMuted(entry) {
  return mutedFor(entry.hash) > 0
}

/** Usable = not dead and not in a DeepSeek cooldown. */
function isUsable(entry) {
  return entry.state !== 'dead' && !isMuted(entry)
}

function markDead(entry, reason) {
  if (!entry || entry.state === 'dead') return
  entry.state = 'dead'
  entry.lastError = String(reason ?? 'unknown error').slice(0, 300)
  entry.lastErrorAt = new Date().toISOString()
  log.warn('token marked dead — dropped from rotation', {
    index: entry.index,
    token: entry.prefix,
    error: entry.lastError,
  })
  wakeAll()
}

function markOk(entry, user) {
  entry.state = 'ok'
  entry.account = user?.email || user?.mobile || user?.id || null
  entry.lastError = null
  entry.verifiedAt = new Date().toISOString()
  log.info('token verified', { index: entry.index, token: entry.prefix, account: entry.account })
  wakeAll()
}

function authFor(entry) {
  return authFromToken(entry.token, { cookie: entry.cookie })
}

/**
 * Reports how a request that used this account ended.
 *
 * ONLY an authentication failure retires the token. A network error, a timeout or a
 * provider 429 says nothing about whether the credential is still good, so those are
 * recorded and otherwise ignored (the cooldown itself is handled by ratelimit.mjs).
 */
export function reportOutcome(entry, error) {
  if (!entry) return
  entry.requests += 1
  entry.lastUsedAt = new Date().toISOString()
  if (!error) return
  // A client that hung up says nothing about the account — do not pollute its stats.
  if ((error?.code ?? error?.failure?.code) === 'ABORTED') return
  entry.errors += 1
  const code = error?.code ?? error?.failure?.code
  const message = errorMessage(error)
  entry.lastError = message
  entry.lastErrorAt = new Date().toISOString()
  if (code === 'AUTH' || code === 'MISSING_CREDENTIAL' || isAuthMessage(message)) {
    markDead(entry, message)
  }
}

function errorMessage(error) {
  const base = error?.message ?? String(error)
  const retry = error?.failure?.providerRetryAfterMs ?? error?.providerRetryAfterMs
  return String(retry ? `${base} (retry after ~${Math.ceil(retry / 1000)}s)` : base).slice(0, 300)
}

/** Auth failures come in a few shapes; anything else (timeout, 5xx) must NOT kill a token. */
function isAuthMessage(message) {
  return /\b401\b|\b403\b|unauthor|not\s*logged|invalid\s*token|token\s*(is\s*)?(expired|invalid)|login\s*required/i
    .test(String(message ?? ''))
}

// ── waiting for a free account ──────────────────────────────────────────

/** Resolvers of requests parked in `acquireToken`, in arrival order. */
const waiters = new Set()

function wakeOne() {
  const next = waiters.values().next()
  if (!next.done) next.value()
}

/** Everyone parked re-evaluates — used when a token dies or is verified. */
function wakeAll() {
  for (const waiter of [...waiters]) waiter()
}

/** Sleeps until woken, until `ms` elapses, or until the client disconnects. */
function waitForTurn(signal, ms) {
  return new Promise((resolve, reject) => {
    let timer = null
    let settled = false

    function cleanup() {
      waiters.delete(wake)
      if (timer) clearTimeout(timer)
      if (signal) signal.removeEventListener('abort', onAbort)
    }
    function settle(fn, value) {
      if (settled) return
      settled = true
      cleanup()
      fn(value)
    }
    function wake() { settle(resolve, undefined) }
    function onAbort() { settle(reject, abortError(signal)) }

    if (signal?.aborted) {
      settle(reject, abortError(signal))
      return
    }
    waiters.add(wake)
    if (signal) signal.addEventListener('abort', onAbort, { once: true })
    if (Number.isFinite(ms)) timer = setTimeout(wake, Math.max(0, ms))
  })
}

/** Least completed requests wins; ties go to the lower index. Keeps the pool balanced. */
function pickLeastUsed(list) {
  let best = list[0]
  for (const entry of list) {
    if (entry.requests < best.requests) best = entry
  }
  return best
}

// ── verification ────────────────────────────────────────────────────────

/**
 * Verifies one account against `GET /api/v0/users/current`.
 *
 * A definite rejection marks the token dead; a transport failure does NOT (the network
 * being down is not evidence that the credential is bad), it just leaves the entry
 * `unknown` and lets the real request decide.
 */
async function verifyEntry(entry, { signal } = {}) {
  if (entry.verifying) return entry.verifying
  entry.verifying = (async () => {
    const controller = new AbortController()
    const onAbort = () => controller.abort(signal?.reason)
    const timer = setTimeout(() => controller.abort('verification timeout'), VERIFY_TIMEOUT_MS)
    if (signal) {
      if (signal.aborted) controller.abort(signal.reason)
      else signal.addEventListener('abort', onAbort, { once: true })
    }
    try {
      const who = await verifyWebToken(authFor(entry), controller.signal)
      if (who?.ok) {
        markOk(entry, who.user)
        return true
      }
      const reason = who?.error ?? 'verification failed'
      if (isAuthMessage(reason)) markDead(entry, reason)
      else entry.lastError = String(reason).slice(0, 300)
      return !isAuthMessage(reason)
    } catch (error) {
      // Timeout / DNS / offline — the token's fate is unknown, so do not retire it.
      entry.lastError = String(error?.message ?? error).slice(0, 300)
      log.warn('token verification inconclusive', { index: entry.index, token: entry.prefix, error: entry.lastError })
      return true
    } finally {
      clearTimeout(timer)
      if (signal) signal.removeEventListener('abort', onAbort)
      entry.verifying = null
    }
  })()
  return entry.verifying
}

// ── acquisition ─────────────────────────────────────────────────────────

/**
 * Takes one account out of the pool for the duration of a request.
 *
 * @param {string[]} tokens the request's token list (already split and de-duplicated)
 * @param {{ signal?: AbortSignal, exclude?: Set<string> }} [options]
 *   `exclude` holds token digests that must NOT be handed out. A caller retrying a failed
 *   request passes the accounts it has already burned, so each attempt lands on a
 *   DIFFERENT account. When every token is excluded the pool is exhausted for that request.
 * @returns {Promise<{entry: object, auth: object, release: () => void}>}
 *   `release()` is idempotent and MUST be called (try/finally) or the account stays locked.
 * @throws POOL_EMPTY     (503) the request listed no tokens
 * @throws POOL_EXHAUSTED (503) every listed token was rejected, or all were excluded
 * @throws POOL_THROTTLED (429) every listed account is in a DeepSeek cooldown
 * @throws POOL_TIMEOUT   (504) nothing became free in time / the client gave up
 */
export async function acquireToken(tokens, { signal, exclude } = {}) {
  const all = (tokens ?? []).map(entryFor)
  if (all.length === 0) {
    throw poolError(
      'POOL_EMPTY',
      'No DeepSeek tokens in the API key. Send them comma-separated: Authorization: Bearer tok1,tok2,tok3',
    )
  }
  // Rotation: never hand back an account this request has already tried and abandoned.
  const entries = exclude?.size ? all.filter((entry) => !exclude.has(entry.hash)) : all
  if (entries.length === 0) {
    throw poolError(
      'POOL_EXHAUSTED',
      'Every DeepSeek token in this API key was tried for this request and failed.',
    )
  }

  const deadline = Date.now() + QUEUE_TIMEOUT_MS

  for (;;) {
    if (signal?.aborted) throw abortError(signal)

    const usable = entries.filter(isUsable)
    if (usable.length === 0) {
      if (entries.every((entry) => entry.state === 'dead')) {
        throw poolError(
          'POOL_EXHAUSTED',
          'Every DeepSeek token in this API key was rejected. Check the tokens and retry.',
        )
      }
      // Nothing usable, but the accounts are only cooling down — say so instead of hanging.
      const remaining = Math.min(...entries.filter((e) => e.state !== 'dead').map((e) => mutedFor(e.hash)))
      throw poolError(
        'POOL_THROTTLED',
        'All DeepSeek accounts in this API key are rate limited right now. Retry after the cooldown.',
        { providerRetryAfterMs: Math.max(1_000, Math.round(remaining)) },
      )
    }

    const free = ALLOW_CONCURRENT ? usable : usable.filter((entry) => !entry.busy)
    if (free.length > 0) {
      const entry = pickLeastUsed(free)
      const lease = commit(entry)
      if (entry.state === 'unknown') {
        const ok = await verifyEntry(entry, { signal })
        if (!ok) {
          lease.release()
          continue // this account just died — try the next one
        }
      }
      return lease
    }

    if (Date.now() >= deadline) {
      throw poolError('POOL_TIMEOUT', `No DeepSeek account became free within ${Math.round(QUEUE_TIMEOUT_MS / 1000)}s.`)
    }
    await waitForTurn(signal, POLL_MS)
  }
}

function commit(entry) {
  if (!ALLOW_CONCURRENT) entry.busy = true
  entry.inflight += 1
  let released = false
  return {
    entry,
    auth: authFor(entry),
    release() {
      if (released) return
      released = true
      entry.inflight = Math.max(0, entry.inflight - 1)
      if (!ALLOW_CONCURRENT) entry.busy = false
      wakeOne()
    },
  }
}

// ── introspection ───────────────────────────────────────────────────────

/** Full pool state for the given token list. Never contains a token or a cookie. */
export function poolStatus(tokens) {
  const entries = (tokens ?? []).map(entryFor)
  return {
    configured: entries.length,
    usable: entries.filter(isUsable).length,
    free: entries.filter((entry) => isUsable(entry) && !entry.busy).length,
    dead: entries.filter((entry) => entry.state === 'dead').length,
    muted: entries.filter(isMuted).length,
    inflight: entries.reduce((sum, entry) => sum + entry.inflight, 0),
    tokens: entries.map((entry) => {
      const remaining = mutedFor(entry.hash)
      return {
        index: entry.index,
        id: entry.prefix,
        state: entry.state,
        account: entry.account,
        busy: entry.busy,
        inflight: entry.inflight,
        requests: entry.requests,
        errors: entry.errors,
        muted: remaining > 0,
        muted_remaining_ms: Math.max(0, Math.round(remaining)),
        verified_at: entry.verifiedAt,
        last_used_at: entry.lastUsedAt,
        last_error: entry.lastError,
        last_error_at: entry.lastErrorAt,
        // The persistent chat session that account is currently bound to (null until the
        // first request) — the two are useful together when a conversation goes wrong.
        session: currentSessionId(entry.hash) ?? null,
        quota: quotaStatus(entry.hash),
      }
    }),
  }
}

/** Re-checks every listed token against DeepSeek. Dead ones that still work come back. */
export async function verifyPool(tokens, { signal } = {}) {
  const entries = (tokens ?? []).map(entryFor)
  await Promise.all(entries.map((entry) => verifyEntry(entry, { signal })))
  return poolStatus(tokens)
}

/**
 * Clears the dead flag so a token is verified again — for a token that was rotated and
 * now works, or after a transient outage retired it.
 */
export async function reviveTokens(tokens, { signal } = {}) {
  const entries = (tokens ?? []).map(entryFor)
  for (const entry of entries) {
    entry.state = 'unknown'
    entry.verifiedAt = null
  }
  wakeAll()
  await Promise.all(entries.map((entry) => verifyEntry(entry, { signal })))
  return poolStatus(tokens)
}
