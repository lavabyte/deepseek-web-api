// SPDX-License-Identifier: MIT
// Copyright (c) 2026 DeepSeek Web API contributors
/**
 * Self-update: keep this checkout in sync with the GitHub repository.
 *
 * WHY IT IS CAREFUL
 *   The home edition is a FORK of the official tree, not a copy: `src/server.mjs`,
 *   `.env.example` and `README.md` are deliberately different, and `tools/` exists only
 *   there. An updater that mirrors the repository would destroy that.
 *
 *   The state file therefore records TWO hashes per file — the upstream blob and the
 *   local content that was in sync with it:
 *
 *     { upstream: <git blob sha>, local: <git blob sha|null>, state: 'synced'|'forked' }
 *
 *   A file is replaced only when ALL of these hold:
 *     1. `state` is 'synced' (it was not already a local fork),
 *     2. the local content still equals the recorded `local` (nobody edited it since),
 *     3. `local === upstream` (the recorded local copy was in sync with upstream), and
 *     4. the repository now has a different blob than the recorded `upstream`.
 *
 *   Everything else is KEPT and reported:
 *     - `forked`  — local content differs from upstream and stays that way (deliberate fork
 *                   or a later local edit); it is no longer auto-updated;
 *     - `conflicts` — a NEW upstream file whose path already exists locally.
 *
 *   The very first run only records a baseline and writes NOTHING, so an install that
 *   already diverges cannot lose a file to a first sync.
 *
 * HOW
 *   The GitHub git-tree API yields every blob's SHA-1 in ONE request; only files that
 *   actually changed are downloaded, and each download is verified against the tree SHA
 *   before it is written. Zero npm dependencies (global fetch).
 *
 * CONFIG (.env)
 *   AUTO_UPDATE=1                    master switch (default 1 = on)
 *   AUTO_UPDATE_REPO                 default lavabyte/deepseek-web-api
 *   AUTO_UPDATE_REF                  default main
 *   AUTO_UPDATE_INTERVAL_MS          default 21600000 (6 h)
 *   AUTO_UPDATE_INITIAL_DELAY_MS     default 10000 (first check after boot)
 *   AUTO_UPDATE_SKIP                 extra comma-separated paths to never touch
 *   AUTO_UPDATE_RESTART=1            restart after applying (default: on under systemd)
 *   GITHUB_TOKEN                     optional, raises the API rate limit
 */
import { createHash } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve, sep } from 'node:path'
import { ROOT } from './env.mjs'
import { DATA_DIR } from './session.mjs'
import { log } from './log.mjs'

/** Default master switch: ON. Only an explicit `AUTO_UPDATE=0` turns it off. */
const ENABLED = String(process.env.AUTO_UPDATE ?? '1') !== '0'
const REPO = String(process.env.AUTO_UPDATE_REPO || 'lavabyte/deepseek-web-api').trim()
const REF = String(process.env.AUTO_UPDATE_REF || 'main').trim()
const INTERVAL_MS = Math.max(60_000, Number(process.env.AUTO_UPDATE_INTERVAL_MS || 6 * 3_600_000))
const INITIAL_DELAY_MS = Math.max(0, Number(process.env.AUTO_UPDATE_INITIAL_DELAY_MS || 10_000))
const REQUEST_TIMEOUT_MS = Math.max(5_000, Number(process.env.AUTO_UPDATE_TIMEOUT_MS || 30_000))
/** Extra paths this install never touches, on top of the built-in list. */
const SKIP = new Set(
  String(process.env.AUTO_UPDATE_SKIP || '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean),
)
/**
 * Restart after applying an update. Under systemd `INVOCATION_ID` is set and
 * `Restart=always` brings the service straight back, so this defaults to on there and
 * off for a hand-run `npm start` (where exiting would simply stop the server).
 */
const RESTART = String(process.env.AUTO_UPDATE_RESTART ?? (process.env.INVOCATION_ID ? '1' : '0')) === '1'

const STATE_FILE = join(DATA_DIR, 'update-state.json')
/** Never written, whatever the repository says. */
const PROTECTED_FILES = new Set(['.env', 'tests/.token'])
const PROTECTED_PREFIXES = ['data/', 'node_modules/', '.git/', '.probe/']

/** Last run summary, exposed through /health. */
let lastResult = {
  enabled: ENABLED,
  repo: `${REPO}@${REF}`,
  checked_at: null,
  commit: null,
  updated: 0,
  forked: 0,
  conflicts: 0,
  error: null,
}

/** Read-only view of the last check, for the health endpoint. */
export function updateStatus() {
  return { ...lastResult }
}

/** Is an automatic restart configured? Used by server.mjs. */
export function autoUpdateRestartEnabled() {
  return RESTART
}

/** Git blob SHA-1 — the same identifier the tree API reports for a file. */
function gitBlobSha(content) {
  const buffer = Buffer.isBuffer(content) ? content : Buffer.from(content)
  return createHash('sha1').update(`blob ${buffer.length}\0`).update(buffer).digest('hex')
}

function readState() {
  try {
    if (!existsSync(STATE_FILE)) return null
    const parsed = JSON.parse(readFileSync(STATE_FILE, 'utf8'))
    if (!parsed || typeof parsed !== 'object' || !parsed.files || typeof parsed.files !== 'object') return null
    return parsed
  } catch (error) {
    log.warn('auto-update: state file unreadable, starting over', { error: String(error?.message ?? error) })
    return null
  }
}

function writeState(state) {
  mkdirSync(DATA_DIR, { recursive: true })
  const tmp = `${STATE_FILE}.tmp`
  writeFileSync(tmp, `${JSON.stringify(state, null, 2)}\n`)
  renameSync(tmp, STATE_FILE)
}

function isProtected(path) {
  if (PROTECTED_FILES.has(path)) return true
  if (SKIP.has(path)) return true
  return PROTECTED_PREFIXES.some((prefix) => path.startsWith(prefix))
}

/** Resolves a repository-relative path inside `base`, refusing anything that escapes it. */
function safeJoin(base, path) {
  if (!path || path.startsWith('/') || path.includes('..')) return null
  const abs = resolve(base, path)
  if (abs !== base && !abs.startsWith(base + sep)) return null
  return abs
}

/** Blob SHA of a regular file, or null when it is missing or not a regular file. */
function localBlobSha(abs) {
  try {
    if (!existsSync(abs) || !statSync(abs).isFile()) return null
    return gitBlobSha(readFileSync(abs))
  } catch {
    return null
  }
}

function githubHeaders(accept) {
  const headers = { accept, 'user-agent': 'deepseek-web-api-auto-update' }
  if (process.env.GITHUB_TOKEN) headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  return headers
}

/** One request for every file in the tree, with its blob SHA. */
async function fetchTree(signal) {
  const url = `https://api.github.com/repos/${REPO}/git/trees/${encodeURIComponent(REF)}?recursive=1`
  const res = await fetch(url, {
    headers: githubHeaders('application/vnd.github+json'),
    signal: signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  })
  const text = await res.text()
  if (!res.ok) throw new Error(`tree request failed (HTTP ${res.status}): ${text.slice(0, 160)}`)
  let payload
  try {
    payload = JSON.parse(text)
  } catch {
    throw new Error(`tree response was not JSON: ${text.slice(0, 160)}`)
  }
  if (!payload || !Array.isArray(payload.tree)) throw new Error('tree response had no `tree` array')
  const files = payload.tree.filter((entry) => entry?.type === 'blob' && typeof entry.path === 'string')
  return { commit: payload.sha, files, truncated: payload.truncated === true }
}

/** Downloads one file and verifies it against the tree's blob SHA. */
async function fetchBlob(entry, commit, signal) {
  const url = `https://raw.githubusercontent.com/${REPO}/${commit}/${entry.path.split('/').map(encodeURIComponent).join('/')}`
  const res = await fetch(url, {
    headers: githubHeaders('application/vnd.github.raw'),
    signal: signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  })
  if (!res.ok) throw new Error(`raw request failed (HTTP ${res.status})`)
  const buffer = Buffer.from(await res.arrayBuffer())
  const actual = gitBlobSha(buffer)
  if (actual !== entry.sha) {
    throw new Error(`integrity check failed (expected ${entry.sha.slice(0, 10)}, got ${actual.slice(0, 10)})`)
  }
  return buffer
}

/**
 * Writes a file through a temporary name so a reader never sees a half-written file.
 * `mode` comes from the git tree entry ("100755" for executables): without it `start.sh`
 * would come back without its executable bit and refuse to run.
 */
function writeAtomic(abs, buffer, mode) {
  mkdirSync(dirname(abs), { recursive: true })
  const tmp = `${abs}.update-${process.pid}.tmp`
  writeFileSync(tmp, buffer)
  if (mode === '100755' || mode === '100775') chmodSync(tmp, 0o755)
  renameSync(tmp, abs)
}

/**
 * Compares the checkout with the repository and (unless `apply` is false) writes the
 * files that are safe to write. Never throws — a network or API problem is reported in
 * the returned summary, because a failed update must not affect serving.
 */
export async function runAutoUpdate({ apply = true, signal } = {}) {
  const summary = {
    ok: false,
    commit: null,
    baseline: false,
    updated: [],
    forked: [],
    conflicts: [],
    skipped: [],
    errors: [],
    /** Files recorded as forks (not updatable) — counted, not only discovered this run. */
    forkedTotal: 0,
    checked_at: new Date().toISOString(),
  }
  try {
    const { commit, files, truncated } = await fetchTree(signal)
    summary.commit = commit
    if (truncated) summary.errors.push('repository tree was truncated — some files were not considered')

    const state = readState()
    // No state yet (or an empty one) = baseline: record only, write nothing.
    const baseline = !state || Object.keys(state.files).length === 0
    summary.baseline = baseline
    const recorded = state?.files ?? {}
    const nextFiles = { ...recorded }
    const now = new Date().toISOString()

    for (const entry of files) {
      const path = entry.path
      if (isProtected(path)) {
        summary.skipped.push({ path, reason: 'protected' })
        continue
      }
      const abs = safeJoin(ROOT, path)
      if (!abs) {
        summary.skipped.push({ path, reason: 'unsafe path' })
        continue
      }

      const record = recorded[path]
      const localSha = localBlobSha(abs)

      if (baseline) {
        // Assume every local difference is intentional; only record what is here.
        nextFiles[path] = {
          upstream: entry.sha,
          local: localSha,
          state: localSha && localSha !== entry.sha ? 'forked' : 'synced',
          at: now,
        }
        continue
      }

      if (!record) {
        // A file the repository has and this install has never recorded.
        if (localSha !== null) {
          // Byte-identical to upstream: adopt it as synced. This is how the updater starts
          // managing a file that predates it (and how it can ever update itself).
          if (localSha === entry.sha) {
            nextFiles[path] = { upstream: entry.sha, local: localSha, state: 'synced', at: now }
            continue
          }
          summary.conflicts.push({ path, reason: 'new upstream file collides with a local file' })
          nextFiles[path] = { upstream: entry.sha, local: localSha, state: 'forked', at: now }
          continue
        }
        if (!apply) {
          summary.skipped.push({ path, reason: 'would create' })
          continue
        }
        try {
          const buffer = await fetchBlob(entry, commit, signal)
          writeAtomic(abs, buffer, entry.mode)
          nextFiles[path] = { upstream: entry.sha, local: gitBlobSha(buffer), state: 'synced', at: now }
          summary.updated.push(path)
          log.info('auto-update: file created', { path })
        } catch (error) {
          summary.errors.push(`${path}: ${String(error?.message ?? error)}`)
        }
        continue
      }

      // Local content moved since we last recorded it: a human edit (or deletion).
      if (localSha !== record.local) {
        summary.forked.push({ path, reason: localSha === null ? 'deleted locally' : 'edited locally' })
        nextFiles[path] = { upstream: entry.sha, local: localSha, state: 'forked', at: now }
        continue
      }

      // Already a fork: keep it. If its content now equals upstream exactly (the edit was
      // pushed, or reverted), start tracking it again.
      if (record.state === 'forked') {
        if (localSha !== null && localSha === entry.sha) {
          nextFiles[path] = { upstream: entry.sha, local: localSha, state: 'synced', at: now }
        } else {
          nextFiles[path] = { upstream: entry.sha, local: record.local, state: 'forked', at: record.at }
        }
        continue
      }

      // Synced and untouched.
      if (entry.sha === record.upstream) {
        nextFiles[path] = record
        continue
      }
      if (localSha === null) {
        summary.forked.push({ path, reason: 'deleted locally' })
        nextFiles[path] = { upstream: entry.sha, local: null, state: 'forked', at: now }
        continue
      }
      if (!apply) {
        summary.skipped.push({ path, reason: 'would update' })
        continue
      }
      try {
        const buffer = await fetchBlob(entry, commit, signal)
        writeAtomic(abs, buffer, entry.mode)
        nextFiles[path] = { upstream: entry.sha, local: gitBlobSha(buffer), state: 'synced', at: now }
        summary.updated.push(path)
        log.info('auto-update: file updated', { path })
      } catch (error) {
        summary.errors.push(`${path}: ${String(error?.message ?? error)}`)
      }
    }

    // Files that vanished upstream are LEFT ALONE on purpose: deleting a user's file
    // because a branch dropped it is worse than carrying a stale copy.
    for (const path of Object.keys(recorded)) {
      if (!files.some((entry) => entry.path === path)) {
        summary.skipped.push({ path, reason: 'gone upstream (kept)' })
        nextFiles[path] = recorded[path]
      }
    }

    // Count every fork on record, not only the ones noticed this run: /health must show the
    // real number of files the updater will never touch.
    summary.forkedTotal = Object.values(nextFiles).filter((item) => item?.state === 'forked').length

    if (apply) {
      writeState({ repo: REPO, ref: REF, commit, checked_at: now, files: nextFiles })
    }
    summary.ok = true
  } catch (error) {
    summary.errors.push(String(error?.message ?? error))
  }

  lastResult = {
    enabled: ENABLED,
    repo: `${REPO}@${REF}`,
    checked_at: summary.checked_at,
    commit: summary.commit,
    updated: summary.updated.length,
    forked: summary.forkedTotal,
    conflicts: summary.conflicts.length,
    error: summary.errors[0] ?? null,
  }
  return summary
}

/**
 * Starts the periodic check. Called once from server.mjs after `listen`; the first check
 * waits a moment so it never competes with the boot work.
 *
 * @param {{ onApplied?: (result: object) => void }} [options]
 *   `onApplied` runs only when files were written AND a restart is configured.
 */
export function startAutoUpdate({ onApplied } = {}) {
  if (!ENABLED) {
    log.info('auto-update disabled', { hint: 'AUTO_UPDATE=1 to enable' })
    return
  }

  let running = false
  const tick = async () => {
    if (running) return
    running = true
    try {
      const result = await runAutoUpdate({ apply: true })
      if (result.baseline) {
        log.info('auto-update: baseline recorded (no files written)', { commit: result.commit })
      } else if (result.updated.length > 0) {
        log.warn('auto-update: files replaced', { commit: result.commit, files: result.updated.join(', ') })
      }
      if (result.forked.length > 0) {
        log.warn('auto-update: locally modified files kept', {
          files: result.forked.map((item) => item.path).join(', '),
        })
      }
      if (result.conflicts.length > 0) {
        log.warn('auto-update: new upstream files collide with local files', {
          files: result.conflicts.map((item) => item.path).join(', '),
        })
      }
      if (result.errors.length > 0) {
        log.warn('auto-update: completed with errors', { errors: result.errors.slice(0, 3).join(' | ') })
      }
      if (result.updated.length > 0 && RESTART && typeof onApplied === 'function') {
        onApplied(result)
      }
    } catch (error) {
      log.warn('auto-update failed', { error: String(error?.message ?? error) })
    } finally {
      running = false
    }
  }

  const initial = setTimeout(tick, INITIAL_DELAY_MS)
  initial.unref?.()
  const interval = setInterval(tick, INTERVAL_MS)
  interval.unref?.()
  log.info('auto-update enabled', {
    repo: `${REPO}@${REF}`,
    interval_s: Math.round(INTERVAL_MS / 1000),
    restart: RESTART,
  })
}
