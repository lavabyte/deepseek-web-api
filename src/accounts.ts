// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 cv-superding (Ding Li)
// Modifications Copyright 2026 deepseek-web-api contributors (see NOTICE).
/**
 * Account store — upgrades "one account" to "a store of accounts, one-click switch".
 *
 * ## Why this exists (2026-09-12, inspired by workbuddy-switch)
 *
 * Originally there was only one credential file (`deepseek-auth.json`); switching
 * accounts cost: **log out -> clear the browser partition -> log in again -> wait for it
 * to be captured**, and the original account was unreachable in the meantime. The
 * reference project treats an "account" as a first-class citizen (account cards, status,
 * near-expiry highlighting, import/export); the applicable parts are brought over here:
 * **multiple accounts coexisting + one-click switch + import/export**.
 *
 * Directory layout:
 * ```
 * <DSH_HOME>/web-login/
 *   ├── accounts.json            # index: { activeId } (order and display info live in the account files)
 *   ├── accounts/acc_xxxx.json   # one per account (WebAuth + metadata), atomic write + 0600
 *   └── deepseek-auth.json       # legacy single-account file (used only for the first migration)
 * ```
 *
 * ## ⚠️ Risk notice (the user must see it, not just read it in the docs)
 *
 * The account store makes "switching accounts" easy, and **using multiple accounts in
 * rotation to evade a per-account limit has a cost**:
 *
 *  1. The same provider will **link** multiple accounts (same device, same IP, same
 *     fingerprint, similar behaviour patterns). Once judged as "one person's alt
 *     accounts", the response is usually heavier than single-account overuse, and it can
 *     affect **all** linked accounts.
 *  2. Therefore this plugin **only offers manual switching** and deliberately **does no
 *     automatic rotation** — a real person does not switch accounts every few minutes to
 *     keep sending messages, and automatic switching is a very strong machine-behaviour
 *     signature that **directly conflicts** with the plugin's efforts to reduce machine
 *     identifiability in transport/interval/session cleanup.
 *  3. Every file in the account store contains **fully usable credentials** (token +
 *     cookie). Exported backup files are plaintext too — sharing one is giving the
 *     account away.
 *
 * In other words: this feature aims at "**making it easier to switch between your own
 * several normal accounts**" (e.g. work/personal), **not** at "evading the limit by
 * rotation".
 */
import { normalizeCookieMetaList } from './cookies.ts'
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, chmodSync } from 'node:fs'
import { join } from 'node:path'
import { randomUUID } from 'node:crypto'
import { legacyAuthFilePath, webLoginDir } from './paths.ts'
import type { WebAuth } from './auth.ts'

/** Account-store index. */
interface AccountsIndex {
  version: number
  /** The active account id; absent means "no account selected". */
  activeId?: string
}

/**
 * An account = credentials (all WebAuth fields) + metadata.
 *
 * Deliberately **flat** (rather than a `{ meta, auth }` nesting): this way it
 * naturally satisfies `WebAuth`, `readAuth()` can return it directly as credentials,
 * and the adapter/login flow need not change a line.
 */
export interface AccountRecord extends WebAuth {
  /** Stable id (`acc_` + 8 hex chars), unique in the store, independent of the token (the token may refresh). */
  id: string
  /** A user-editable label (e.g. "work"). When empty the UI shows the masked account. */
  label?: string
  /**
   * The DeepSeek server-side user id (from `users/current`'s `data.id`).
   * Purpose: **update instead of inserting when the same account is captured again**,
   * avoiding a pile of duplicates in the store.
   */
  serverId?: string
  /** Time of the last successful active liveness probe (ISO). See `src/probe.ts`. */
  lastVerifiedAt?: string
  /** Last active-probe failure (reason kept, to tell expiry from a network problem at a glance). */
  lastVerifyError?: { at: string; message: string }
  /**
   * Observed account-level limit (from a **rejected generation request**, not a probe).
   *
   * Correction (measured 2026-09-12): this used to say "during a limit, users/current
   * still returns 200, so a probe cannot detect it" — the 200 is correct, but the
   * **response body carries `chat: { is_muted, mute_until }`**, i.e. a probe actually
   * can detect it; it just is not wired up yet. This state is still only learned from
   * **failed generation** (the mute_until in the failure envelope, see webapi.ts's muteUntilMs).
   */
  limit?: { untilMs: number; observedAt: string }
}

const INDEX_VERSION = 1

export function accountsDir(): string {
  return join(webLoginDir(), 'accounts')
}

export function accountsIndexPath(): string {
  return join(webLoginDir(), 'accounts.json')
}

/**
 * Blocks ids that "escape the account directory".
 *
 * ⚠️ F01 (2026-09-12 audit): the old implementation did `join(accountsDir(), `${id}.json`)`
 * **with no validation**; `importAccounts` treats the **id in a backup file as the primary
 * key verbatim**, and the HTTP route also consumes the caller-supplied id directly. So an
 * id of `../../../../Users/me/evil` could read, write, and delete JSON outside the
 * account directory.
 *
 * Only "dangerous characters" are blocked here, without demanding a format: historical
 * ids come in more than one shape (acc_ prefix + 8/16 hex, possibly others during
 * migration), and tightening to a whitelist would break old accounts. The only thing
 * that must be blocked is what can traverse paths.
 */
function assertSafeAccountId(id: string): string {
  const text = String(id ?? '')
  if (!text || text.includes('\0') || text === '.' || text === '..' || /[/\\]/.test(text)) {
    throw new Error(`invalid account id (path separator or relative path segment): ${JSON.stringify(text)}`)
  }
  return text
}

export function accountFilePath(id: string): string {
  return join(accountsDir(), `${assertSafeAccountId(id)}.json`)
}

/** New account id. A random id rather than a token hash: the token refreshes, the id must not follow. */
export function newAccountId(): string {
  return `acc_${randomUUID().replace(/-/g, '').slice(0, 8)}`
}

/** Atomic write (temp file + replace), tightening permissions to 0600 on non-Windows. */
function writeJsonAtomic(file: string, value: unknown): void {
  mkdirSync(join(file, '..'), { recursive: true })
  const tmp = `${file}.tmp-${process.pid}`
  // The temp file is 0600 from creation: it holds **fully usable credentials**. The old
  // approach wrote it with default permissions (subject to umask, commonly 0644) and
  // chmod'd afterwards, leaving an exposure window.
  writeFileSync(tmp, JSON.stringify(value, null, 2), { encoding: 'utf8', mode: 0o600 })
  // ⚠️ F02 (2026-09-12 audit): the old approach was "rmSync the target, then rename over it" —
  // if rename failed (disk full / in use / permissions), **the original file was already
  // gone** and the credentials were lost. On Windows `fs.renameSync` can overwrite an
  // existing target directly anyway (measured win32 + Node 22: renameSync overwrites
  // successfully, the target gets the new content, the temp file disappears), so that
  // rmSync was both unnecessary and the only data-loss risk. After removing it:
  // rename fails -> remove the temp file -> the original is intact.
  try {
    renameSync(tmp, file)
  } catch (error) {
    try {
      rmSync(tmp, { force: true })
    } catch {}
    throw error
  }
  if (process.platform !== 'win32') {
    try {
      chmodSync(file, 0o600)
    } catch {}
  }
}

function readJson<T>(file: string): T | undefined {
  try {
    if (!existsSync(file)) return undefined
    return JSON.parse(readFileSync(file, 'utf8')) as T
  } catch {
    return undefined
  }
}

function readIndex(): AccountsIndex {
  const parsed = readJson<AccountsIndex>(accountsIndexPath())
  return {
    version: INDEX_VERSION,
    ...(typeof parsed?.activeId === 'string' && parsed.activeId ? { activeId: parsed.activeId } : {}),
  }
}

function writeIndex(index: AccountsIndex): void {
  writeJsonAtomic(accountsIndexPath(), { version: INDEX_VERSION, ...(index.activeId ? { activeId: index.activeId } : {}) })
}

/** Normalises any object into an AccountRecord (missing fields get defaults; invalid credentials return undefined). */
function normalizeRecord(raw: any, fallbackId?: string): AccountRecord | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const token = typeof raw.token === 'string' ? raw.token : ''
  if (!token) return undefined
  return {
    id: typeof raw.id === 'string' && raw.id ? raw.id : (fallbackId ?? newAccountId()),
    token,
    cookie: typeof raw.cookie === 'string' ? raw.cookie : '',
    hifDliq: typeof raw.hifDliq === 'string' ? raw.hifDliq : '',
    hifLeim: typeof raw.hifLeim === 'string' ? raw.hifLeim : '',
    wasmUrl: typeof raw.wasmUrl === 'string' ? raw.wasmUrl : '',
    userAgent: typeof raw.userAgent === 'string' ? raw.userAgent : '',
    ...(raw.extraHeaders && typeof raw.extraHeaders === 'object' ? { extraHeaders: raw.extraHeaders } : {}),
    capturedAt: typeof raw.capturedAt === 'string' ? raw.capturedAt : '',
    ...(raw.unverified === true ? { unverified: true } : {}),
    ...(raw.user && typeof raw.user === 'object' ? { user: raw.user } : {}),
    // Cookie expiry composition: malformed entries are dropped in cookies.ts and do not make the whole record unreadable.
    ...(() => {
      const meta = normalizeCookieMetaList(raw.cookieMeta)
      return meta ? { cookieMeta: meta } : {}
    })(),
    ...(typeof raw.label === 'string' && raw.label ? { label: raw.label } : {}),
    ...(typeof raw.serverId === 'string' && raw.serverId ? { serverId: raw.serverId } : {}),
    ...(typeof raw.lastVerifiedAt === 'string' ? { lastVerifiedAt: raw.lastVerifiedAt } : {}),
    ...(raw.lastVerifyError && typeof raw.lastVerifyError?.at === 'string'
      ? { lastVerifyError: { at: raw.lastVerifyError.at, message: String(raw.lastVerifyError.message ?? '') } }
      : {}),
    ...(raw.limit && Number.isFinite(raw.limit?.untilMs)
      ? { limit: { untilMs: Number(raw.limit.untilMs), observedAt: String(raw.limit.observedAt ?? '') } }
      : {}),
  }
}

/** All accounts in the store, newest capture first. */
export function listAccounts(): AccountRecord[] {
  let names: string[] = []
  try {
    names = readdirSync(accountsDir()).filter((name) => name.endsWith('.json') && !name.includes('.tmp-'))
  } catch {
    return []
  }
  const records: AccountRecord[] = []
  for (const name of names) {
    const id = name.replace(/\.json$/, '')
    const record = normalizeRecord(readJson(accountFilePath(id)), id)
    if (record) records.push(record)
  }
  records.sort((a, b) => String(b.capturedAt).localeCompare(String(a.capturedAt)))
  return records
}

export function readAccount(id: string): AccountRecord | undefined {
  if (!id) return undefined
  return normalizeRecord(readJson(accountFilePath(id)), id)
}

export function saveAccount(record: AccountRecord): void {
  writeJsonAtomic(accountFilePath(record.id), record)
}

export function activeAccountId(): string | undefined {
  const { activeId } = readIndex()
  if (!activeId) return undefined
  // The account the index points at may have been removed — treat it as "not selected"
  // rather than handing the caller a dangling id. The id itself may be invalid (the index
  // file was edited externally) -> likewise "not selected", rather than letting an
  // exception blow through the caller.
  try {
    return existsSync(accountFilePath(activeId)) ? activeId : undefined
  } catch {
    return undefined
  }
}

/** The currently active account (undefined when none). */
export function activeAccount(): AccountRecord | undefined {
  const id = activeAccountId()
  return id ? readAccount(id) : undefined
}

export function setActiveAccount(id: string): boolean {
  if (!existsSync(accountFilePath(id))) return false
  writeIndex({ activeId: id })
  return true
}

export function clearActiveAccount(): void {
  writeIndex({})
}

export function updateAccount(id: string, patch: Partial<AccountRecord>): AccountRecord | undefined {
  const current = readAccount(id)
  if (!current) return undefined
  const next = normalizeRecord({ ...current, ...patch, id }, id)
  if (!next) return undefined
  saveAccount(next)
  return next
}

/**
 * Removes an account from the store (**deletes the credential file**).
 *
 * Why not follow other reversible operations with "rename and keep": what is stored here
 * is a **fully usable credential**, and the semantics of "log out / remove" is "this
 * credential must no longer be on disk" — keeping a plaintext `.removed-<time>` backup
 * would make "logged out" a lie (a security regression). Protection against accidental
 * deletion comes from two things: a **confirmation** in the UI, and the account-store
 * **export backup**.
 */
export function removeAccount(id: string): boolean {
  const file = accountFilePath(id)
  if (!existsSync(file)) return false
  try {
    rmSync(file, { force: true })
  } catch {
    return false
  }
  if (readIndex().activeId === id) clearActiveAccount()
  return true
}

/**
 * Writes/updates an account's credentials (both login capture and manual token paste go here).
 *
 * Deduplication order:
 *  1. has `serverId` and the store already has the same `serverId` -> **update that record** (same account recaptured);
 *  2. otherwise a record with the exact same token -> update (the case where serverId is not yet available);
 *  3. otherwise -> insert.
 *
 * ⚠️ **Credential fields always take the values passed this time; no merging**: the caller
 * (e.g. the login flow) uses `writeAuth({ ...auth, unverified: true })` to mean "this time
 * verification did not succeed"; if the old record's fields were reused, this `unverified`
 * would stick forever and could never be cleared. Only metadata (label / probe time /
 * limit state) needs to survive across calls, so only those fields are inherited.
 */
export function upsertAccount(auth: WebAuth, patch: Partial<AccountRecord> = {}): AccountRecord {
  // serverId may arrive via `patch` or be placed directly into `auth` by the caller (the
  // login flow likes `writeAuth({ ...auth, user })`) — both must be recognised, or
  // deduplication silently fails and the store piles up duplicates.
  const incoming = auth as Partial<AccountRecord>
  const serverId = patch.serverId ?? incoming.serverId

  const all = listAccounts()
  const existing =
    (serverId ? all.find((item) => item.serverId && item.serverId === serverId) : undefined) ??
    all.find((item) => item.token === auth.token)
  const id = patch.id ?? existing?.id ?? newAccountId()

  const carried: Partial<AccountRecord> = {}
  for (const key of ['label', 'serverId', 'lastVerifiedAt', 'lastVerifyError', 'limit'] as const) {
    const value = (patch as any)[key] ?? (incoming as any)[key] ?? (existing as any)?.[key]
    if (value !== undefined) (carried as any)[key] = value
  }

  const record = normalizeRecord({ ...auth, ...carried, id }, id)!
  saveAccount(record)
  return record
}

/**
 * Packages export data (including plaintext credentials — the caller must explain the risk to the user).
 *
 * ⚠️ This data now has two outlets with different security postures:
 *   1. `exportAccountsToFile()` + `POST /accounts/export`: **credentials never leave the
 *      host**; the host writes to disk itself and returns only the path. Always kept, it
 *      is the fallback path.
 *   2. `POST /accounts/export-json`: hands the content to the UI, which pops a system
 *      "Save as" to write it. To let the user choose the save location, this path is
 *      unavoidable (see the comment on that route in index.ts).
 */
export function exportAccounts(): { version: number; exportedAt: string; warning: string; accounts: AccountRecord[] } {
  return {
    version: INDEX_VERSION,
    exportedAt: new Date().toISOString(),
    warning: 'This file contains full login credentials (token + cookie) — as good as the account itself. Do not share it or commit it.',
    accounts: listAccounts(),
  }
}

/**
 * Exports to a **file inside the plugin directory** and returns the path
 * (`<web-login>/exports/accounts-<timestamp>.json`).
 *
 * This is the fallback path: used when the UI cannot get a system "Save as" (the host did
 * not inject File System Access, or the dialog was refused by the platform), guaranteeing
 * export never breaks. Its advantage is that plaintext credentials do not enter the HTTP
 * response body — only the **path** is returned to the UI.
 */
export function exportAccountsToFile(): { path: string; count: number } {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const file = join(webLoginDir(), 'exports', `accounts-${stamp}.json`)
  writeJsonAtomic(file, exportAccounts())
  return { path: file, count: listAccounts().length }
}

/** Import (validate + dedupe + backfill id). Returns the inserted/updated counts. */
export function importAccounts(payload: unknown): { imported: number; updated: number; skipped: number } {
  const list: any[] = Array.isArray(payload)
    ? payload
    : Array.isArray((payload as any)?.accounts)
      ? (payload as any).accounts
      : []
  let imported = 0
  let updated = 0
  let skipped = 0
  for (const raw of list) {
    const candidate = normalizeRecord(raw)
    if (!candidate) {
      skipped += 1
      continue
    }
    const before = listAccounts()
    const matched =
      (candidate.serverId ? before.find((item) => item.serverId && item.serverId === candidate.serverId) : undefined) ??
      before.find((item) => item.token === candidate.token)
    if (matched) {
      // Keep the existing local metadata (label, probe time, limit state), replace only the credentials.
      updateAccount(matched.id, { ...candidate, id: matched.id, label: candidate.label ?? matched.label })
      updated += 1
    } else {
      saveAccount(candidate)
      imported += 1
    }
  }
  // When there is not a single account, make the first imported one active (otherwise after
  // importing it says "no account selected", which is baffling).
  if (!readIndex().activeId) {
    const first = listAccounts()[0]
    if (first) setActiveAccount(first.id)
  }
  return { imported, updated, skipped }
}

/**
 * One-time migration: move the pre-0.1.25 single-account file into the store.
 *
 * The old file is **renamed for the record** (not deleted); after success the migration
 * does not run again. Returns the migrated account (undefined when none).
 */
export function migrateLegacyAuth(): AccountRecord | undefined {
  const legacy = legacyAuthFilePath()
  const record = normalizeRecord(readJson(legacy))
  if (!record) return undefined
  // Already in the store (same token) -> do not re-import.
  const existing = listAccounts().find((item) => item.token === record.token)
  const saved = existing ?? record
  if (!existing) saveAccount(record)
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  try {
    renameSync(legacy, `${legacy}.migrated-${stamp}`)
  } catch {}
  if (!activeAccountId()) setActiveAccount(saved.id)
  return saved
}

/** The migration runs only once, when "the store is empty AND the legacy file exists". */
export function migrateLegacyAuthIfNeeded(): AccountRecord | undefined {
  try {
    if (listAccounts().length > 0) return undefined
    if (!existsSync(legacyAuthFilePath())) return undefined
    return migrateLegacyAuth()
  } catch {
    return undefined
  }
}

/** Account display name: label first, then the masked account, then the id. */
export function accountTitle(record: AccountRecord, mask: (raw: string) => string): string {
  if (record.label) return record.label
  const display = record.user?.display || record.user?.id || ''
  // When no name at all is available (just captured, not verified yet), do not show the
  // **internal id** (`acc_cd8e05ec`) as the name — seeing a hex string makes users think
  // it is a bug (measured feedback). Say "unidentified" and keep a short suffix so several
  // unidentified accounts remain distinguishable.
  if (!display) return `Unidentified account (${record.id.replace(/^acc_/, '').slice(0, 8)})`
  return mask(display)
}

/** Store file footprint (for the UI's "account store usage", and to spot abnormal growth). */
export function accountsFootprint(): { count: number; bytes: number } {
  let bytes = 0
  let count = 0
  try {
    for (const name of readdirSync(accountsDir())) {
      if (!name.endsWith('.json') || name.includes('.tmp-')) continue
      count += 1
      try {
        bytes += statSync(join(accountsDir(), name)).size
      } catch {}
    }
  } catch {}
  return { count, bytes }
}