// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Access control and per-user quotas for the HTTP API.
 *
 * The server accepts ANY valid chat.deepseek.com token by default — that is fine on
 * 127.0.0.1, but once it is exposed publicly anyone with a DeepSeek account can use it,
 * and a single caller can exhaust the account throttle for everyone. This module adds
 * two opt-in layers, both disabled by default so local use is unchanged:
 *
 *   1. ALLOWLIST — only the listed tokens may call the API.
 *      ACCESS_ALLOWLIST = "all" (default) | comma-separated sha256 token hashes (32 hex chars)
 *
 *   2. QUOTA — a per-token request budget inside a sliding window.
 *      QUOTA_MAX_REQUESTS = 0 (default, unlimited) | positive integer
 *      QUOTA_WINDOW_MS    = 3_600_000 (default one hour)
 *
 * Hashes are used instead of raw tokens so the allowlist can be committed without
 * leaking a full credential. A hash is produced by `tokenKey()` from session.mjs.
 */

const HASH_RE = /^[0-9a-f]{8,64}$/i

/** Parses ACCESS_ALLOWLIST into a Set of lowercase hashes, or null for "all". */
function parseAllowlist(raw) {
  const value = String(raw ?? '').trim()
  if (!value || value.toLowerCase() === 'all' || value === '*') return null
  const hashes = value
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter((entry) => HASH_RE.test(entry))
  // An allowlist that parsed to nothing but was non-empty is a configuration error:
  // treat it as "deny all" rather than silently allowing everyone.
  return new Set(hashes)
}

const allowlist = parseAllowlist(process.env.ACCESS_ALLOWLIST)
export const accessControlEnabled = allowlist !== null

/**
 * Returns true when the token may use the API.
 * With no allowlist configured, every token is allowed (local default).
 */
export function isTokenAllowed(tokenHash) {
  if (allowlist === null) return true
  return allowlist.has(String(tokenHash ?? '').toLowerCase())
}

// ── Quota ───────────────────────────────────────────────────────────────

const QUOTA_MAX = Math.max(0, Math.floor(Number(process.env.QUOTA_MAX_REQUESTS || 0)))
const QUOTA_WINDOW_MS = Math.max(1_000, Math.floor(Number(process.env.QUOTA_WINDOW_MS || 3_600_000)))
export const quotaEnabled = QUOTA_MAX > 0

/** { [tokenHash]: number[] } — timestamps of accepted requests, oldest first. */
const hits = new Map()
/** Housekeeping: drop token buckets that have been empty for a whole window. */
const lastSweep = { at: 0 }

function sweep(now) {
  if (now - lastSweep.at < QUOTA_WINDOW_MS) return
  lastSweep.at = now
  for (const [key, stamps] of hits) {
    const fresh = stamps.filter((ts) => now - ts < QUOTA_WINDOW_MS)
    if (fresh.length === 0) hits.delete(key)
    else hits.set(key, fresh)
  }
}

/**
 * Consumes one unit of the caller's quota.
 *
 * @returns {{ allowed: true } | { allowed: false, limit: number, retryAfterMs: number }}
 */
export function consumeQuota(tokenHash) {
  if (!quotaEnabled) return { allowed: true }
  const now = Date.now()
  sweep(now)
  const key = String(tokenHash ?? '')
  const stamps = (hits.get(key) ?? []).filter((ts) => now - ts < QUOTA_WINDOW_MS)
  if (stamps.length >= QUOTA_MAX) {
    const oldest = stamps[0]
    const retryAfterMs = Math.max(0, QUOTA_WINDOW_MS - (now - oldest))
    hits.set(key, stamps)
    return { allowed: false, limit: QUOTA_MAX, retryAfterMs }
  }
  stamps.push(now)
  hits.set(key, stamps)
  return { allowed: true }
}

/**
 * Quota state for diagnostics (/health).
 * Returns null when quotas are disabled.
 */
export function quotaStatus(tokenHash) {
  if (!quotaEnabled) return null
  const now = Date.now()
  const stamps = (hits.get(String(tokenHash ?? '')) ?? []).filter((ts) => now - ts < QUOTA_WINDOW_MS)
  return {
    limit: QUOTA_MAX,
    window_ms: QUOTA_WINDOW_MS,
    used: stamps.length,
    remaining: Math.max(0, QUOTA_MAX - stamps.length),
    reset_in_ms: stamps.length ? Math.max(0, QUOTA_WINDOW_MS - (now - stamps[0])) : 0,
  }
}
