// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Persistent, single web session — SEPARATE FOR EACH TOKEN.
 *
 * DESIGN (user requirement):
 *   There is EXACTLY ONE chat session on chat.deepseek.com (with no name assigned).
 *   Every request to our OpenAI-compatible server carries the FULL history, and we
 *   insert it as ONE message (the first) into that session — instead of appending
 *   one turn per request. This way:
 *     - history lives on the client side (like OpenAI), not in the web chat,
 *     - context is not duplicated (the web does not attach history with parent_message_id: null),
 *     - the DeepSeek panel shows one conversation instead of hundreds of junk sessions
 *       (the previous plugin created and deleted hundreds of sessions a day — exactly
 *       the bot signature we want to avoid).
 *
 * TOKEN KEYING:
 *   The session token now comes from the client (as the API key), so state is kept
 *   per token — the SHA-256 digest of the token is the key in data/api-sessions.json.
 *   This way two different tokens do not steal each other's session id, and a server
 *   restart does not create a new session.
 */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deleteChatSession, newChatSession } from './deepseek.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
export const DATA_DIR = process.env.DATA_DIR || join(ROOT, 'data')
const STATE_FILE = join(DATA_DIR, 'api-sessions.json')

/** How long an unused session entry is kept before pruning (default 30 days). */
const SESSION_TTL_MS = Number(process.env.SESSION_TTL_MS || 30 * 24 * 60 * 60 * 1000)
/** Hard cap on stored entries; the most recently used survive (default 1000). */
const SESSION_MAX_ENTRIES = Number(process.env.SESSION_MAX_ENTRIES || 1000)

/** The token itself is never logged — only its digest is used as the state key. */
export function tokenKey(token) {
  return createHash('sha256').update(String(token || '')).digest('hex').slice(0, 32)
}

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
  mkdirSync(DATA_DIR, { recursive: true })
  // atomic write: never leave a partial JSON file behind
  const tmp = `${STATE_FILE}.tmp`
  writeFileSync(tmp, JSON.stringify(store, null, 2) + '\n', 'utf8')
  renameSync(tmp, STATE_FILE)
}

/** { [tokenHash]: { sessionId, createdAt } } */
let store = load()
/** Serializes concurrent requests — one session handles one request at a time. */
let chain = Promise.resolve()

/** Queues session operations (FIFO). */
export function withLock(fn) {
  const run = chain.then(fn, fn)
  chain = run.then(() => undefined, () => undefined)
  return run
}

function entryFor(tokenHash) {
  const entry = store[tokenHash]
  return entry && typeof entry === 'object' ? entry : {}
}

function entryTimestamp(entry) {
  const parsed = Date.parse(String(entry?.lastUsedAt || entry?.createdAt || ''))
  return Number.isFinite(parsed) ? parsed : 0
}

function setEntry(tokenHash, entry) {
  // Stamp every live entry so housekeeping can tell how stale it is. An empty entry
  // (`forgetSession`) carries no session and is stamped only when a session is stored.
  const stamped = entry && entry.sessionId ? { ...entry, lastUsedAt: new Date().toISOString() } : entry
  store = { ...store, [tokenHash]: stamped }
  save()
}

/**
 * Drops expired and excess session entries.
 *
 * `data/api-sessions.json` grows with every token that ever called the server, and
 * nothing used to remove entries — a long-lived or public deployment would grow the file
 * without bound. Entries not used for SESSION_TTL_MS are dropped; if more than
 * SESSION_MAX_ENTRIES remain, the most recently used are kept. Legacy entries without a
 * timestamp are treated as recent and kept.
 *
 * @returns {{ removed: number, remaining: number }}
 */
export function pruneSessions() {
  const now = Date.now()
  let kept = Object.entries(store).filter(([, entry]) => {
    const ts = entryTimestamp(entry)
    return ts === 0 || now - ts <= SESSION_TTL_MS
  })
  if (kept.length > SESSION_MAX_ENTRIES) {
    kept = kept.sort((a, b) => entryTimestamp(b[1]) - entryTimestamp(a[1])).slice(0, SESSION_MAX_ENTRIES)
  }
  const next = Object.fromEntries(kept)
  const removed = Object.keys(store).length - Object.keys(next).length
  if (removed > 0) {
    store = next
    save()
  }
  return { removed, remaining: Object.keys(store).length }
}

export function currentSessionId(tokenHash) {
  return entryFor(tokenHash).sessionId
}

/**
 * Returns the session id for the given token, creating it if needed.
 * `forceNew` discards the current one (e.g. when the server replies "invalid chat session id").
 */
export async function ensureSession(auth, { forceNew = false, signal } = {}) {
  const tokenHash = tokenKey(auth?.token)
  return withLock(async () => {
    const current = entryFor(tokenHash)
    if (forceNew && current.sessionId) {
      await deleteChatSession(auth, current.sessionId)
      setEntry(tokenHash, {})
    }
    const after = entryFor(tokenHash)
    if (after.sessionId) return after.sessionId
    const sessionId = await newChatSession(auth, signal)
    setEntry(tokenHash, {
      sessionId,
      createdAt: new Date().toISOString(),
    })
    return sessionId
  })
}

/** Forgets the stored session for the given token (e.g. on account switch). */
export function forgetSession(tokenHash) {
  if (tokenHash) setEntry(tokenHash, {})
}

// Prune once at startup so a state file that grew before housekeeping existed is
// cleaned up on the first boot. No-op (and no write) when nothing is stale.
pruneSessions()