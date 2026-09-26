// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Per-account rate-limit cooldown ("mute") for the DeepSeek web backend.
 *
 * WHY THIS EXISTS
 *   DeepSeek throttles at the ACCOUNT level and, empirically, the window appears to
 *   RESET on every new request. Retrying from a ladder (1s, 5s, 15s, …) therefore does
 *   not help — each retry restarts the timer, so the account never leaves the penalty
 *   box and the client sees a string of failures.
 *
 *   The correct behaviour is the opposite: as soon as ONE 429 is seen, stop talking to
 *   DeepSeek entirely for the cooldown window, and answer every incoming request from
 *   this server with a clean 429 + `retry-after` until the window expires. No request
 *   reaches DeepSeek during the pause, so its timer is allowed to run out.
 *
 * SCOPE
 *   Keyed by token hash, not globally: the throttle is per ACCOUNT, so two different
 *   tokens must not block each other. Mirrors the per-token request gates in server.mjs.
 *
 * PERSISTENCE
 *   The mute is written to `data/rate-limits.json` so a server restart during an active
 *   penalty does not forget it (DeepSeek's timer keeps running independently of us).
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { DATA_DIR } from './session.mjs'

const STATE_FILE = join(DATA_DIR, 'rate-limits.json')

/**
 * How long to stay quiet after a 429 when the provider gives no hint, or gives a shorter
 * one than this. Two minutes matches the observed web-side mute window (see README).
 */
export const DEFAULT_COOLDOWN_MS = Number(process.env.RATE_LIMIT_COOLDOWN_MS || 120_000)

function load() {
  try {
    if (!existsSync(STATE_FILE)) return {}
    const parsed = JSON.parse(readFileSync(STATE_FILE, 'utf8'))
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function save() {
  try {
    mkdirSync(DATA_DIR, { recursive: true })
    // Atomic write: a partial JSON file must never be left behind.
    const tmp = `${STATE_FILE}.tmp`
    writeFileSync(tmp, JSON.stringify(store, null, 2) + '\n', 'utf8')
    renameSync(tmp, STATE_FILE)
  } catch {
    /* the cooldown is a best-effort protection; never fail a request because of it */
  }
}

/** { [tokenHash]: { mutedUntilMs, mutedAtMs, reason } } */
let store = load()

function prune(now = Date.now()) {
  let changed = false
  for (const [hash, entry] of Object.entries(store)) {
    if (!entry || Number(entry.mutedUntilMs || 0) <= now) {
      delete store[hash]
      changed = true
    }
  }
  if (changed) save()
}

/**
 * Remaining cooldown for a token, in milliseconds (0 = not muted).
 * Expired entries are dropped lazily so the file does not grow forever.
 */
export function mutedFor(tokenHash, now = Date.now()) {
  const entry = store[tokenHash]
  if (!entry) return 0
  const until = Number(entry.mutedUntilMs || 0)
  if (!Number.isFinite(until) || until <= now) {
    delete store[tokenHash]
    save()
    return 0
  }
  return until - now
}

/** Wall-clock timestamp the cooldown ends at (0 = not muted). */
export function mutedUntil(tokenHash) {
  return mutedFor(tokenHash) > 0 ? Number(store[tokenHash]?.mutedUntilMs || 0) : 0
}

/**
 * Starts (or extends) the cooldown for a token.
 *
 * `retryAfterMs` is the provider's own hint when it gave one; the effective pause is never
 * SHORTER than DEFAULT_COOLDOWN_MS, because a too-eager return to DeepSeek is exactly what
 * keeps the account throttled.
 *
 * @returns the applied cooldown in milliseconds
 */
export function muteToken(tokenHash, retryAfterMs) {
  if (!tokenHash) return 0
  const hint = Number(retryAfterMs)
  const applied = Math.max(DEFAULT_COOLDOWN_MS, Number.isFinite(hint) && hint > 0 ? hint : 0)
  const now = Date.now()
  const previous = Number(store[tokenHash]?.mutedUntilMs || 0)
  // Never shorten an active cooldown — only extend it.
  const until = Math.max(previous, now + applied)
  store[tokenHash] = { mutedUntilMs: until, mutedAtMs: now, reason: 'deepseek_rate_limit' }
  save()
  return until - now
}

/** Clears a cooldown manually (used by /health?clear_rate_limit=1 and by tests). */
export function clearMute(tokenHash) {
  if (!tokenHash || !(tokenHash in store)) return false
  delete store[tokenHash]
  save()
  return true
}

/** Snapshot for /health: how long this token is muted for, if at all. */
export function cooldownStatus(tokenHash) {
  prune()
  const remainingMs = mutedFor(tokenHash)
  return {
    active: remainingMs > 0,
    remaining_ms: Math.max(0, Math.round(remainingMs)),
    until: mutedUntil(tokenHash) || null,
    cooldown_ms: DEFAULT_COOLDOWN_MS,
  }
}