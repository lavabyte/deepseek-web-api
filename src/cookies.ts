// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Collection and interpretation of cookie expiry information.
 *
 * ## Why this is worth doing
 *
 * The credentials the plugin stores only hold the `name=value` string — the
 * **expiry time has always been dropped** — so "will the login state expire if
 * it is not used for a long time" had no observable clue in the UI at all.
 *
 * Measured (2026-09-12, read-only, zero quota): DeepSeek's read-only endpoints
 * **do not return set-cookie**, and the token is not rotated, so there is
 * **no way to renew anything**; all we can do is **capture the expiry time**,
 * shrinking "uncertain" by one notch — at least we can answer "which of these
 * cookies were session-level to begin with".
 *
 * A side finding (which decides "what to look at"): actual authentication uses
 * the **`token`**, not a cookie — sending only the token without cookies passes,
 * sending only cookies without the token is rejected outright (40002 Missing
 * Token). So a cookie's expiry is **not** an upper bound on the login state's
 * lifetime; it only tells us which ones the browser will drop first.
 *
 * ## Two sources with different shapes, both must be recognised
 *
 * | Source | Field | Session-level marker |
 * | --- | --- | --- |
 * | CDP `Storage.getCookies` (real Edge/Chrome) | `expires` | `-1` |
 * | Electron `session.cookies.get()` (the plugin's own window) | `expirationDate` | field missing |
 *
 * Both are in **seconds** (Unix epoch); this module normalises to **milliseconds**.
 */

export interface CookieMeta {
  /** Cookie name. */
  name: string
  /** Owning domain (kept verbatim, to help debug domain-matching issues). */
  domain: string
  /** Session-level: expires when the browser closes (the server-side session has its own lifetime, not bound by this). */
  session: boolean
  /** Expiry time of a persistent cookie (**milliseconds** epoch); absent for session-level or unknown. */
  expiresAt?: number
}

/**
 * Normalises a cookie's expiry shape. The two sources use different field
 * names; this unifies them.
 *
 * Order: an explicit `session === true` wins; otherwise look at
 * `expires` / `expirationDate`. Missing, non-numeric, or `<= 0` is **always
 * treated as session-level** — CDP reports session-level as `-1`, and "0" is
 * meaningless in Unix epoch (treating it as valid would compute the year 1970).
 */
export function readCookieExpiry(raw: any): { session: boolean; expiresAt?: number } {
  if (!raw || typeof raw !== 'object') return { session: true }
  if (raw.session === true) return { session: true }
  const seconds = Number(raw.expires ?? raw.expirationDate)
  if (!Number.isFinite(seconds) || seconds <= 0) return { session: true }
  return { session: false, expiresAt: Math.round(seconds * 1000) }
}

/**
 * Picks the target-domain cookies from a cookie array of any origin and
 * converts them to `CookieMeta`.
 *
 * ⚠️ `filter` is supplied by the caller and **must be character-for-character
 * identical to the filter used where the cookie header is assembled**: this
 * metadata is meant to describe exactly "the cookies actually attached to the
 * request". The two capture paths originally used different filters (the real
 * browser one used `deepseek`, the Electron one used `deepseek.com`); this is
 * deliberately not forced into one — unifying them would change the request
 * header content, and that change cannot be caught by a unit test.
 */
export function pickCookieMeta(
  cookies: readonly any[],
  filter: (domain: unknown) => boolean,
): CookieMeta[] {
  const out: CookieMeta[] = []
  for (const raw of cookies ?? []) {
    const name = typeof raw?.name === 'string' ? raw.name : ''
    if (!name) continue
    const domain = String(raw?.domain ?? '')
    if (!filter(domain)) continue
    const { session, expiresAt } = readCookieExpiry(raw)
    out.push({ name, domain, session, ...(expiresAt !== undefined ? { expiresAt } : {}) })
  }
  return out
}

/**
 * Normalises an "already normalised" `CookieMeta` array (used when reading
 * account records from disk). Malformed entries are dropped rather than
 * throwing — one bad record must not make the whole account store unreadable.
 */
export function normalizeCookieMetaList(raw: unknown): CookieMeta[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const out: CookieMeta[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const name = typeof (item as any).name === 'string' ? (item as any).name : ''
    if (!name) continue
    const expiresRaw = Number((item as any).expiresAt)
    const hasExpiry = Number.isFinite(expiresRaw) && expiresRaw > 0
    const session = (item as any).session === true || !hasExpiry
    out.push({
      name,
      domain: typeof (item as any).domain === 'string' ? (item as any).domain : '',
      session,
      ...(session ? {} : { expiresAt: Math.round(expiresRaw) }),
    })
  }
  return out.length > 0 ? out : undefined
}

export interface CookieLifeSummary {
  total: number
  sessionCount: number
  persistentCount: number
  /** The **latest**-expiring persistent cookie — answers "how long can the browser side hold out". */
  latest?: { name: string; expiresAt: number; daysLeft: number }
}

/**
 * Aggregated result. **Returns `undefined` when there is no record** (rather
 * than an all-zero object) — the caller must distinguish "no expiry info was
 * captured" (old record / manually pasted token) from "captured, all
 * session-level"; the UI copy for these two cases is completely different.
 */
export function summarizeCookieLife(
  metas: readonly CookieMeta[] | undefined,
  now: number = Date.now(),
): CookieLifeSummary | undefined {
  const list = metas ?? []
  if (list.length === 0) return undefined
  const persistent = list.filter((item) => !item.session && Number.isFinite(item.expiresAt))
  let latest: CookieLifeSummary['latest']
  for (const item of persistent) {
    const expiresAt = item.expiresAt as number
    if (!latest || expiresAt > latest.expiresAt) {
      latest = { name: item.name, expiresAt, daysLeft: (expiresAt - now) / 86_400_000 }
    }
  }
  return {
    total: list.length,
    sessionCount: list.length - persistent.length,
    persistentCount: persistent.length,
    ...(latest ? { latest } : {}),
  }
}

/** Colloquial rendering of the remaining time. */
export function describeRemaining(daysLeft: number): string {
  if (daysLeft <= 0) return 'expired'
  if (daysLeft >= 1) return `${Math.floor(daysLeft)} days left`
  const hours = Math.max(1, Math.round(daysLeft * 24))
  return `${hours} hours left`
}

/**
 * One sentence describing the cookie lifetime composition (shown directly in
 * the UI).
 *
 * The three cases are written separately because their meaning is completely
 * different:
 *  - no record -> old record / manually pasted token; a fresh login will fill it in;
 *  - all session-level -> the browser side has no expiry at all, so watching how long it lasts is meaningless;
 *  - persistent present -> give the remaining time of the latest one (**only an
 *    upper bound on the browser side, not the login lifetime**).
 */
export function describeCookieLife(
  summary: CookieLifeSummary | undefined,
  now: number = Date.now(),
): string {
  if (!summary) return 'not recorded (a fresh login will fill this in)'
  const parts = [`${summary.total} cookies`]
  parts.push(summary.sessionCount > 0 ? `${summary.sessionCount} session-only` : 'none session-only')
  if (summary.persistentCount > 0) parts.push(`${summary.persistentCount} persistent`)
  if (summary.latest) {
    // Note: `summary.latest.daysLeft` is computed at the moment this summary was
    // built. If the caller displays it much later, pass `now` in to recompute.
    const daysLeft = (summary.latest.expiresAt - now) / 86_400_000
    parts.push(`${summary.latest.name} ${describeRemaining(daysLeft)}`)
  }
  return parts.join(' · ')
}