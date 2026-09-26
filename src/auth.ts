// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 cv-superding (Ding Li)
// Modifications Copyright 2026 deepseek-web-api contributors (see NOTICE).
/**
 * dsh-deepseek-web-login — credential storage and structured errors.
 *
 * Login credentials come from the chat.deepseek.com web client (captured from a
 * browser window or pasted manually): Bearer token + cookie + anti-bot
 * fingerprint headers (x-hif-*) + the PoW WASM URL.
 *
 * Storage has been upgraded from "a single file" to an "**account store**" (see
 * accounts.ts): this file keeps only the types + the access facade —
 * `readAuth()` returns the active account, `writeAuth()` writes and makes it
 * active, `clearAuth()` removes the active account. This way the adapter, the
 * login flow, diagnostics and **every other call site need not change a line**.
 *
 * Location: `${DSH_HOME || ~/.dsh}/web-login/accounts/<id>.json` (plugin-owned,
 * not part of the settings/credentials seam, so sensitive credentials do not
 * land on a generic configuration surface).
 */
import { existsSync } from 'node:fs'
import type { CookieMeta } from './cookies.ts'
import {
  activeAccount,
  clearActiveAccount,
  removeAccount,
  setActiveAccount,
  upsertAccount,
} from './accounts.ts'
import { resolveDshHome } from './paths.ts'

// resolveDshHome moved to paths.ts (to avoid an import cycle between auth and
// accounts). It is re-exported here unchanged so existing external code
// (transport.ts / net-diagnostics.ts) does not have to change.
export { resolveDshHome }

/** A captured set of web-side login credentials. */
export interface WebAuth {
  /** chat.deepseek.com Bearer token (the web client's localStorage userToken). */
  token: string
  /** deepseek.com domain cookie string (name=value; ...). */
  cookie: string
  /** Anti-bot fingerprint headers (captured from browser requests; empty string if absent). */
  hifDliq: string
  hifLeim: string
  /** PoW solver WASM URL (from page assets; falls back to a known default). */
  wasmUrl: string
  /** Browser UA of the login session (completes the request fingerprint). */
  userAgent: string
  /**
   * A snapshot of the browser's real request headers (accept-language /
   * x-client-version etc., which follow the web client version). Reusing it
   * keeps plugin requests consistent with the web client and avoids hardcoded
   * version numbers that break when DeepSeek upgrades.
   */
  extraHeaders?: Record<string, string>
  /** Capture time (ISO). */
  capturedAt: string
  /** Whether it was persisted without server-side verification (fail-open credential). */
  unverified?: boolean
  /** Masked account display info (optional). */
  user?: { id?: string; display?: string }
  /**
   * Cookie-expiry composition recorded at capture time (optional).
   *
   * Its only purpose is to turn "how long can the login state actually last"
   * from **completely unobservable** into **at least half-observable**.
   * ⚠️ Do not read it as the login lifetime — measurement shows real
   * authentication uses the `token`: sending only the token without cookies
   * passes, sending only cookies without the token is rejected outright.
   * So the latest expiry in here is **only an upper bound on the browser side**,
   * not the credential's expiry time.
   *
   * Old records / manually pasted tokens have no such field (the UI says "not recorded").
   */
  cookieMeta?: CookieMeta[]
}

/** The currently active login credential (undefined when no account is selected). */
export function readAuth(): WebAuth | undefined {
  return activeAccount()
}

/**
 * Writes/updates a credential.
 *
 * Semantics: **whatever is written is what will be used next** — every call
 * site (browser capture, manual token paste, login-flow account backfill)
 * means exactly that, so this also makes it the active account. Writing the
 * same account twice **updates the existing record** (deduplicated by
 * serverId / token, see accounts.upsertAccount).
 */
export function writeAuth(auth: WebAuth): void {
  const record = upsertAccount(auth)
  setActiveAccount(record.id)
}

/**
 * Log out: **removes** the active account from the store (and clears the active
 * pointer).
 *
 * When other accounts remain, it deliberately **does not switch automatically**:
 * an automatic switch would make people think "I only logged out, why am I on
 * another account now". The UI says "N more accounts in the store, click
 * Switch to use one", leaving the choice explicit.
 */
export function clearAuth(): void {
  const active = activeAccount()
  if (active) removeAccount(active.id)
  clearActiveAccount()
}

export function hasUsableAuth(auth: WebAuth | undefined): auth is WebAuth {
  return !!auth && typeof auth.token === 'string' && auth.token.length > 8
}

/**
 * Unwraps the token read back from the page (handles both a bare string and the
 * AppKit-wrapped JSON).
 *
 * ⚠️ Two boundaries that must hold (both are measured shapes):
 *  - when logged out, the web client returns `{"value":null,"__version":"0"}` ->
 *    this must yield an **empty string**; never treat the string "null" as a
 *    token (otherwise a garbage token is sent and 40003 appears out of nowhere).
 *  - older web clients stored a bare token string -> return it unchanged.
 */
export function unwrapStoredToken(raw: unknown): string {
  const text = String(raw ?? '').trim()
  if (!text) return ''
  if (text.startsWith('{')) {
    try {
      const parsed = JSON.parse(text)
      return typeof parsed?.value === 'string' ? parsed.value.trim() : ''
    } catch {
      return ''
    }
  }
  // Fallback: the literals "null"/"undefined" are always treated as empty.
  return text === 'null' || text === 'undefined' ? '' : text
}

/** Masks an account identifier (keeps enough to recognise it without leaking the whole value). */
export function maskIdentifier(raw: string): string {
  const value = String(raw || '').trim()
  if (!value) return ''
  const at = value.indexOf('@')
  if (at > 0) {
    const local = value.slice(0, at)
    const keep = Math.min(3, Math.max(1, local.length - 1))
    return `${local.slice(0, keep)}***${value.slice(at)}`
  }
  if (/^\d{6,}$/.test(value)) return `${value.slice(0, 3)}****${value.slice(-4)}`
  if (value.length <= 4) return `${value[0]}***`
  return `${value.slice(0, 3)}***${value.slice(-2)}`
}

/**
 * Adapter-boundary error. It carries `failure` and `code` as own data
 * properties — LlmRuntime.normalizeLlmFailure reads structured failure info
 * through own properties (not instanceof), so a self-contained bundle crossing
 * a module boundary still carries code/status/retryAfter.
 */
export class AdapterLlmError extends Error {
  readonly failure: { message: string; code: string; status?: number; providerRetryAfterMs?: number }
  readonly code: string
  /**
   * The **release time** of an account-level limit (millisecond timestamp);
   * present only for `user is muted`.
   *
   * Why a separate field: `providerRetryAfterMs` is relative (for retry
   * policies), while "remember until when this account is limited" needs an
   * absolute value. Making callers parse the time out of the error text is unreliable.
   */
  readonly mutedUntilMs?: number

  constructor(
    message: string,
    code: string,
    options: { status?: number; providerRetryAfterMs?: number; mutedUntilMs?: number; cause?: unknown } = {},
  ) {
    super(message)
    this.name = 'LlmError'
    this.code = code
    if (options.mutedUntilMs !== undefined) this.mutedUntilMs = options.mutedUntilMs
    this.failure = {
      message,
      code,
      ...(options.status !== undefined ? { status: options.status } : {}),
      ...(options.providerRetryAfterMs !== undefined ? { providerRetryAfterMs: options.providerRetryAfterMs } : {}),
    }
    // Set cause on the prototype chain (Node 22 supports options.cause, but attach manually to be safe).
    if (options.cause !== undefined) (this as any).cause = options.cause
  }
}

/** Maps an HTTP status to a stable error code (aligned with dsh-llm's default retryable table: SERVER/RATE_LIMIT/TIMEOUT/TRANSPORT). */
export function httpErrorCode(status: number): string {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 402) return 'QUOTA'
  if (status >= 500) return 'SERVER'
  return 'PROVIDER_ERROR'
}

/** Parses Retry-After (seconds or an HTTP date) and returns milliseconds. */
export function parseRetryAfterMs(raw: string | null | undefined): number | undefined {
  if (!raw) return undefined
  const text = String(raw).trim()
  if (/^\d+$/.test(text)) return Math.max(1000, Number(text) * 1000)
  const parsed = Date.parse(text)
  if (!Number.isNaN(parsed)) return Math.max(1000, parsed - Date.now())
  return undefined
}

export function existsFile(path: string): boolean {
  try {
    return existsSync(path)
  } catch {
    return false
  }
}