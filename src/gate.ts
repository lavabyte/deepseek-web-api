// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Request gate — limits "web-side requests in flight on the same account".
 *
 * Why it is needed (measured 2026-09-12):
 *  Reconstructing the start/end time of every call from the plugin log, 272 rounds
 *  contained **16 overlapping pairs**, with a very clear signature: one side is the main
 *  answer (hundreds of words, 6-40 s), the other is **only 8-17 words taking 1-3 s**
 *  — that is DSH's **session-title generation** (`options.purpose === 'session-title'`).
 *  In other words, while you were still waiting for the answer, DSH had already sent
 *  another short request to the same account.
 *
 *  The web side allows only ONE generation per account at a time, and concurrent
 *  generation is rejected (`A message is being generated…`). Worse, measurement showed
 *  that concurrent generation from two windows triggered an account-level limit (mute
 *  for 1 day) in under 6 minutes. So "concurrency" is not free throughput — it is a risk
 *  source to actively avoid.
 *
 * Two constraints:
 *  1. `allowConcurrent === false` (default): **serial**, only one call is admitted at a
 *     time, the rest queue (FIFO).
 *  2. `minIntervalMs`: at least this long **between** two calls (measured from the end of
 *     the previous one), pushing the request density down — this is the item that really
 *     matters for anti-throttling.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

/**
 * Recommended call interval: a **random range of 2-4 seconds** (lower / upper bound).
 *
 * Why a range and not a fixed value: a fixed interval has variance ~0, which is
 * statistically a "timer signature"; human operation intervals have variance. The
 * similar project cuckoo-code (never throttled) uses exactly a 2000-4000 ms random range.
 */
export const DEFAULT_MIN_REQUEST_INTERVAL_MS = 2_000
export const DEFAULT_MAX_REQUEST_INTERVAL_MS = 4_000

/**
 * Long-run protection: after this many consecutive requests, force a long break.
 * 0 = disabled.
 *
 * Why it is needed: the interval only controls "how long between two calls", not "how
 * long it has been running without stopping". Measured 2026-09-12 on an SSH plugin
 * development task: ~70 requests in 12 minutes (many were tool-only rounds with no
 * speech); even with a 2-4 s interval there was no pause at all — that shape looks more
 * scripted than the interval size does. That day the account was temporarily limited
 * twice (the second time for 3 days).
 */
export const DEFAULT_LONG_RUN_THRESHOLD = 15
/** Long-break duration range (1-3 minutes). */
export const DEFAULT_LONG_RUN_BREAK_MS: CleanupRange = { min: 60_000, max: 180_000 }
/** Legal range for the long break (30 seconds to 10 minutes). */
export const LONG_RUN_BREAK_BOUNDS_MS: CleanupRange = { min: 30_000, max: 600_000 }
/** Legal range for the long-run threshold (0 = disabled, max 100). */
export const LONG_RUN_THRESHOLD_BOUNDS = { min: 0, max: 100 }

/** If more than this long has passed since the last request, it counts as "rested" and the consecutive counter resets. */
const LONG_RUN_IDLE_RESET_MS = 120_000

/** Recommended interval presets (used by the settings-page shortcut buttons): [min, max]. */
export const INTERVAL_PRESETS = [
  [1_500, 2_500],
  [2_000, 4_000],
  [5_000, 9_000],
] as const
/** Upper bound for the settings-page slider. */
export const MAX_INTERVAL_MS = 30_000

// ── The three session-cleanup ranges ─────────────────────────────────────
//
// Why these three parameters are also "bounds + random": they used to be **fixed values**
// (accumulate 8 / wait 90 seconds / no gap between deletions). A fixed value is itself a
// machine signature — always acting at the 8th item, always waiting 90 seconds, deletion
// requests firing back to back. Real people are not that precise.
//
// The defaults take a range around the old fixed value (mean ≈ old value), so behaviour
// does not jump; it just swaps "precise" for "has variance".
/** A range from which a random value can be drawn (closed interval; unit depends on the field). */
export interface CleanupRange {
  min: number
  max: number
}

/** How many to accumulate: default 6-10 (mean 8, equal to the old default). */
export const DEFAULT_CLEANUP_BATCH: CleanupRange = { min: 6, max: 10 }
/** How long to wait at most from the first queued session: default 60-120 seconds (mean 90 s, equal to the old default). */
export const DEFAULT_CLEANUP_DELAY_MS: CleanupRange = { min: 60_000, max: 120_000 }
/** Gap between two deletions: default 0.8-2.5 seconds.
 *  Newly added — when batch deletion is not accepted by the server it degrades to
 *  "delete one by one", and the old request sequence had **no gap** in between. */
export const DEFAULT_CLEANUP_GAP_MS: CleanupRange = { min: 800, max: 2_500 }

/** Ranges each field may be set to (the settings-page sliders are drawn from this too). */
export const CLEANUP_BATCH_BOUNDS = { min: 1, max: 50 }
export const CLEANUP_DELAY_BOUNDS_MS = { min: 5_000, max: 600_000 }
export const CLEANUP_GAP_BOUNDS_MS = { min: 0, max: 60_000 }

/**
 * Normalises any input into a legal range: non-numbers ignored, clamped to bounds, and
 * **swapped automatically when min/max are inverted**. Returning undefined means "this
 * field is invalid, treat it as absent" (the caller falls back to the default).
 */
export function normalizeCleanupRange(
  value: unknown,
  bounds: { min: number; max: number },
): CleanupRange | undefined {
  if (!value || typeof value !== 'object') return undefined
  const raw = value as { min?: unknown; max?: unknown }
  const lo = Number(raw.min)
  const hi = Number(raw.max)
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined
  const clamp = (n: number) => Math.min(bounds.max, Math.max(bounds.min, Math.floor(n)))
  // A user dragging the two sliders past each other should not error, nor should a
  // "min > max" state survive.
  return { min: clamp(Math.min(lo, hi)), max: clamp(Math.max(lo, hi)) }
}

export interface GateSettings {
  allowConcurrent: boolean
  /** Long-run protection: force a long break after this many consecutive requests (0 = disabled). */
  longRunThreshold: number
  /** Long-break duration range (ms); the built-in default is used when unset. */
  longRunBreakMs?: CleanupRange
  /** Interval lower bound (ms). Equal to the upper bound degrades to a fixed interval. */
  minRequestIntervalMs: number
  /** Interval upper bound (ms). The actual wait is drawn **randomly** from [min, max]. */
  maxRequestIntervalMs: number
  /**
   * Temporary-session cleanup policy. It does not take part in throttling; it just shares
   * the same settings file and settings page for storage. The actual execution is in
   * webapi.ts's createSessionCleaner (the type is widened to a literal to avoid a
   * circular dependency).
   */
  sessionCleanup?: 'immediate' | 'deferred' | 'keep'
  /** deferred: how many to accumulate. The actual threshold is **re-randomised each cleanup round**. */
  cleanupBatch?: CleanupRange
  /** deferred: how long to wait at most from the first queued session (ms). Re-randomised each round. */
  cleanupDelayMs?: CleanupRange
  /** Gap between two deletions (ms). Re-randomised before each deletion. */
  cleanupGapMs?: CleanupRange
}

/** Throttling settings file: `${DSH_HOME || ~/.dsh}/web-login/gate.json` (plugin-owned, same dir as credentials). */
export function gateSettingsPath(): string {
  const home = process.env.DSH_HOME || join(homedir(), '.dsh')
  return join(home, 'web-login', 'gate.json')
}

/**
 * Reads values saved by the settings page. A missing/corrupt file returns undefined
 * (falls back to cordis config). Priority: **settings page (file) > cordis config >
 * built-in default** — the settings page is an explicit user action and must not be
 * overridden by an old value in the config file.
 */
export function readGateSettings(): Partial<GateSettings> | undefined {
  try {
    const file = gateSettingsPath()
    if (!existsSync(file)) return undefined
    const parsed = JSON.parse(readFileSync(file, 'utf8'))
    const out: Partial<GateSettings> = {}
    if (typeof parsed?.allowConcurrent === 'boolean') out.allowConcurrent = parsed.allowConcurrent
    if (Number.isFinite(parsed?.minRequestIntervalMs)) {
      out.minRequestIntervalMs = clampInterval(Number(parsed.minRequestIntervalMs))
    }
    if (Number.isFinite(parsed?.maxRequestIntervalMs)) {
      out.maxRequestIntervalMs = clampInterval(Number(parsed.maxRequestIntervalMs))
    }
    // Old config (0.1.20 stored only min) = fixed-interval semantics: the upper bound follows the lower bound.
    if (out.minRequestIntervalMs !== undefined && out.maxRequestIntervalMs === undefined) {
      out.maxRequestIntervalMs = out.minRequestIntervalMs
    }
    const cleanup = parsed?.sessionCleanup
    if (cleanup === 'immediate' || cleanup === 'deferred' || cleanup === 'keep') {
      out.sessionCleanup = cleanup
    }
    const batch = normalizeCleanupRange(parsed?.cleanupBatch, CLEANUP_BATCH_BOUNDS)
    if (batch) out.cleanupBatch = batch
    const delay = normalizeCleanupRange(parsed?.cleanupDelayMs, CLEANUP_DELAY_BOUNDS_MS)
    if (delay) out.cleanupDelayMs = delay
    const gap = normalizeCleanupRange(parsed?.cleanupGapMs, CLEANUP_GAP_BOUNDS_MS)
    if (gap) out.cleanupGapMs = gap
    if (Number.isFinite(parsed?.longRunThreshold)) {
      const n = Math.round(Number(parsed.longRunThreshold))
      out.longRunThreshold = Math.max(
        LONG_RUN_THRESHOLD_BOUNDS.min,
        Math.min(LONG_RUN_THRESHOLD_BOUNDS.max, n),
      )
    }
    const lrb = normalizeCleanupRange(parsed?.longRunBreakMs, LONG_RUN_BREAK_BOUNDS_MS)
    if (lrb) out.longRunBreakMs = lrb
    return Object.keys(out).length > 0 ? out : undefined
  } catch {
    return undefined
  }
}

export function writeGateSettings(settings: GateSettings): void {
  const file = gateSettingsPath()
  mkdirSync(join(file, '..'), { recursive: true })
  writeFileSync(file, JSON.stringify(settings, null, 2) + '\n', 'utf8')
}

/** Normalises any input into a legal interval: non-number -> default, negative -> 0, over the cap -> the cap. */
export function clampInterval(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_MIN_REQUEST_INTERVAL_MS
  return Math.min(MAX_INTERVAL_MS, Math.max(0, Math.floor(value)))
}

export interface RequestGateOptions {
  /** Allow concurrency on the same account (default false). Enabling restores the high-risk "title and main answer sent at once" behaviour. */
  allowConcurrent?: boolean
  /** Minimum interval between two calls (ms). 0 = unlimited. */
  minIntervalMs?: number
  /** Interval upper bound (ms). If absent and minIntervalMs is not given, the default upper bound is used; if minIntervalMs is given, it takes the same value (keeping the old "fixed interval" semantics). */
  maxIntervalMs?: number
  /** Long-run protection: force a long break after this many consecutive requests (0 = disabled). Defaults to DEFAULT_LONG_RUN_THRESHOLD. */
  longRunThreshold?: number
  /** Long-break duration range (ms). Defaults to DEFAULT_LONG_RUN_BREAK_MS. */
  longRunBreakMs?: CleanupRange
  /** Random source (for unit-test injection). */
  random?: () => number
  logger?: { info?: (msg: string) => void; warn?: (msg: string) => void; debug?: (msg: string) => void }
  /** Convenient for unit-test injection. */
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

export interface RequestGate {
  /** Acquires admission; the returned function must be called once (idempotent) to release and admit the next in the queue. */
  /** signal can cancel while queued/waiting (F11); on cancel it throws AbortError and the gate does not deadlock. */
  acquire(label?: string, signal?: AbortSignal): Promise<() => void>
  /** Current state (for diagnostics/tests). */
  stats(): { running: number; waiting: number; lastFinishedAt: number }
  /** Reads the currently effective throttling settings. */
  settings(): GateSettings
  /** Changes settings at runtime (called after the settings page saves); returns the new values. */
  configure(next: Partial<GateSettings>): GateSettings
}

export function createRequestGate(options: RequestGateOptions = {}): RequestGate {
  // Mutable at runtime (effective immediately after the settings page saves, no restart)
  let allowConcurrent = options.allowConcurrent === true
  let longRunThreshold = options.longRunThreshold ?? DEFAULT_LONG_RUN_THRESHOLD
  let longRunBreakMs: CleanupRange | undefined = options.longRunBreakMs
  /** Consecutive request counter (reset after a long break, or after resting enough). */
  let consecutive = 0

  let minIntervalMs = clampInterval(
    options.minIntervalMs ?? (options.maxIntervalMs !== undefined ? options.maxIntervalMs : DEFAULT_MIN_REQUEST_INTERVAL_MS),
  )
  let maxIntervalMs = clampInterval(
    options.maxIntervalMs ?? (options.minIntervalMs !== undefined ? options.minIntervalMs : DEFAULT_MAX_REQUEST_INTERVAL_MS),
  )
  if (maxIntervalMs < minIntervalMs) maxIntervalMs = minIntervalMs
  const random = options.random ?? Math.random
  const now = options.now ?? (() => Date.now())
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)))
  const logger = options.logger

  /** Queue tail: resolves only after each call finishes, guaranteeing FIFO and "the previous one is not done, so the next is not admitted". */
  let tail: Promise<void> = Promise.resolve()
  let running = 0
  let waiting = 0
  let lastFinishedAt = 0
  /** Whether any call has finished yet — the first call must not be held up by the interval rule. */
  let hasFinished = false

  async function acquire(label = 'call', signal?: AbortSignal): Promise<() => void> {
    let releaseMine!: () => void
    const mine = new Promise<void>((resolve) => {
      releaseMine = resolve
    })
    const prev = tail
    tail = prev.then(() => mine)

    const aborted = (): Error => {
      const error = new Error(`"${label}" was cancelled while waiting in the gate`)
      error.name = 'AbortError'
      return error
    }
    /** Makes the wait interruptible: on abort it rejects immediately, without waiting for the timer/previous request. */
    const waitOrAbort = (inner: Promise<unknown>): Promise<void> => {
      if (!signal) return inner.then(() => undefined)
      if (signal.aborted) return Promise.reject(aborted())
      return new Promise<void>((resolve, reject) => {
        const onAbort = () => {
          signal.removeEventListener('abort', onAbort)
          reject(aborted())
        }
        signal.addEventListener('abort', onAbort, { once: true })
        inner.then(
          () => {
            signal.removeEventListener('abort', onAbort)
            resolve()
          },
          (error) => {
            signal.removeEventListener('abort', onAbort)
            reject(error)
          },
        )
      })
    }

    waiting += 1
    try {
      // Serial: wait for all previous calls to finish. Concurrent mode skips this step (the interval still applies).
      if (!allowConcurrent) {
        if (running > 0 || waiting > 1) {
          logger?.debug?.(`deepseek-web: "${label}" queued (still running ahead: ${running}, waiting: ${waiting - 1})`)
        }
        await waitOrAbort(prev)
      }

    // Long-run protection: first work out "has it been running non-stop for a long time".
    // A long time since the last finish -> the human rested, so the consecutive counter resets.
    if (hasFinished && now() - lastFinishedAt > LONG_RUN_IDLE_RESET_MS) consecutive = 0
    const breakRange = longRunBreakMs ?? DEFAULT_LONG_RUN_BREAK_MS
    const needsBreak = longRunThreshold > 0 && consecutive > 0 && consecutive >= longRunThreshold

    // Interval: drawn **randomly** from [min, max], measured from the previous FINISH time
    // (not the previous start — otherwise a rapid chain of requests after a long answer stays dense).
    // Each wait differs, avoiding a timer signature.
    const gap = needsBreak ? Math.round(breakRange.min + random() * Math.max(0, breakRange.max - breakRange.min)) : nextGap()
    if (hasFinished && (needsBreak || maxIntervalMs > 0)) {
      const waitMs = lastFinishedAt + gap - now()
      if (needsBreak) {
        consecutive = 0
        logger?.info?.(
          `deepseek-web: ${longRunThreshold} consecutive requests — taking a long break of ${Math.round(gap / 1000)}s (long-run guard: steady back-to-back runs look more scripted than spaced-out ones)`,
        )
      }
      if (waitMs > 0) {
        if (!needsBreak) {
          logger?.info?.(
            `deepseek-web: less than ${gap}ms since the last request (window ${minIntervalMs}~${maxIntervalMs})` +
              `; waiting ${Math.round(waitMs)}ms before sending "${label}" (avoids account-level throttling)`,
          )
        }
        await waitOrAbort(sleep(waitMs))
      }
    }

    } catch (error) {
      // On cancel (or a previous error) we must remove ourselves from the queue: if this
      // node's `mine` never resolves, the requests queued behind it are stuck on `tail`
      // forever — the gate deadlocks.
      releaseMine()
      throw error
    } finally {
      waiting -= 1
    }

    running += 1
    let released = false
    return () => {
      if (released) return
      released = true
      running -= 1
      lastFinishedAt = now()
      hasFinished = true
      consecutive += 1
      releaseMine()
    }
  }

  /** The interval actually used this time: random within the range; fixed when min == max. */
  function nextGap(): number {
    if (maxIntervalMs <= minIntervalMs) return minIntervalMs
    return Math.round(minIntervalMs + random() * (maxIntervalMs - minIntervalMs))
  }

  /** The session-cleanup policy is not implemented in this module; it only borrows the settings file for storage (the host reads it and hands it to the cleaner). */
  let cleanupMode: GateSettings['sessionCleanup']
  // The three cleanup ranges (also not part of throttling). They live here so they can be **persisted**:
  // writeGateSettings writes the return value of settings(); if it is not stored, it is lost.
  let cleanupBatch: CleanupRange | undefined
  let cleanupDelayMs: CleanupRange | undefined
  let cleanupGapMs: CleanupRange | undefined

  function settings(): GateSettings {
    return {
      allowConcurrent,
      minRequestIntervalMs: minIntervalMs,
      maxRequestIntervalMs: maxIntervalMs,
      ...(cleanupMode ? { sessionCleanup: cleanupMode } : {}),
      ...(cleanupBatch ? { cleanupBatch } : {}),
      ...(cleanupDelayMs ? { cleanupDelayMs } : {}),
      ...(cleanupGapMs ? { cleanupGapMs } : {}),
      longRunThreshold,
      ...(longRunBreakMs ? { longRunBreakMs } : {}),
    }
  }

  function configure(next: Partial<GateSettings>): GateSettings {
    if (typeof next.allowConcurrent === 'boolean') allowConcurrent = next.allowConcurrent
    if (next.minRequestIntervalMs !== undefined) minIntervalMs = clampInterval(Number(next.minRequestIntervalMs))
    if (next.maxRequestIntervalMs !== undefined) maxIntervalMs = clampInterval(Number(next.maxRequestIntervalMs))
    if (next.sessionCleanup !== undefined) cleanupMode = next.sessionCleanup
    // The three ranges: an invalid input is simply "absent" (no error, and it does not overwrite a valid existing value).
    if (next.cleanupBatch !== undefined) {
      const value = normalizeCleanupRange(next.cleanupBatch, CLEANUP_BATCH_BOUNDS)
      if (value) cleanupBatch = value
    }
    if (next.cleanupDelayMs !== undefined) {
      const value = normalizeCleanupRange(next.cleanupDelayMs, CLEANUP_DELAY_BOUNDS_MS)
      if (value) cleanupDelayMs = value
    }
    if (next.cleanupGapMs !== undefined) {
      const value = normalizeCleanupRange(next.cleanupGapMs, CLEANUP_GAP_BOUNDS_MS)
      if (value) cleanupGapMs = value
    }
    if (next.longRunThreshold !== undefined && Number.isFinite(next.longRunThreshold)) {
      longRunThreshold = Math.max(
        LONG_RUN_THRESHOLD_BOUNDS.min,
        Math.min(LONG_RUN_THRESHOLD_BOUNDS.max, Math.round(next.longRunThreshold)),
      )
    }
    if (next.longRunBreakMs !== undefined) {
      const value = normalizeCleanupRange(next.longRunBreakMs, LONG_RUN_BREAK_BOUNDS_MS)
      if (value) longRunBreakMs = value
    }
    // The two settings-page sliders may be dragged to "max < min"; correct it here (no error, just clamp).
    if (maxIntervalMs < minIntervalMs) maxIntervalMs = minIntervalMs
    logger?.info?.(
      `deepseek-web: request throttling updated — ${allowConcurrent ? 'concurrent (not recommended)' : 'serial'} · ` +
        `gap ${minIntervalMs}~${maxIntervalMs}ms (randomised)` +
        (cleanupBatch ? ` · cleanup after ${cleanupBatch.min}~${cleanupBatch.max} chats` : '') +
        (cleanupDelayMs
          ? ` · max wait ${Math.round(cleanupDelayMs.min / 1000)}~${Math.round(cleanupDelayMs.max / 1000)}s`
          : '') +
        (cleanupGapMs ? ` · delete gap ${cleanupGapMs.min}~${cleanupGapMs.max}ms` : '') +
        ` · long-run guard: ${longRunThreshold > 0 ? `every ${longRunThreshold} requests take a long break of ${Math.round((longRunBreakMs ?? DEFAULT_LONG_RUN_BREAK_MS).min / 1000)}~${Math.round((longRunBreakMs ?? DEFAULT_LONG_RUN_BREAK_MS).max / 1000)}s` : 'off'}`,
    )
    return settings()
  }

  return {
    acquire,
    stats: () => ({ running, waiting, lastFinishedAt }),
    settings,
    configure,
  }
}