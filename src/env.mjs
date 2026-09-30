// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Minimal .env loader (zero dependencies).
 *
 * WHY THIS IS A SEPARATE MODULE
 *   `import` declarations are hoisted: every imported module is fully evaluated BEFORE the
 *   importing module's body runs. When the loader lived inside server.mjs it therefore
 *   executed too late — session.mjs (DATA_DIR, SESSION_TTL_MS), ratelimit.mjs
 *   (RATE_LIMIT_COOLDOWN_MS), access.mjs (QUOTA_*), log.mjs (LOG_LEVEL) and deepseek.mjs
 *   (MODEL_ID) had all already read process.env while it was still empty, so those settings
 *   silently came from the defaults and .env only ever reached code that read process.env
 *   later. ESM evaluates dependencies in source order, so server.mjs imports THIS module
 *   first and the bug disappears.
 *
 * Rules:
 *   - a variable that already exists in the real environment WINS (.env is a fallback),
 *   - blank lines and lines starting with `#` are ignored,
 *   - `KEY="value"` and `KEY='value'` have the quotes stripped,
 *   - a BOM and CRLF line endings are tolerated (.env is often edited in Notepad).
 */
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Project root — `src/..`. */
export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

let loaded = false

/** Parses .env text into a plain object (no side effects — used by the tests). */
export function parseEnv(text) {
  const out = {}
  for (const rawLine of String(text).split(/\r?\n/)) {
    let line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const eq = line.indexOf('=')
    if (eq === -1) continue
    const key = line.slice(0, eq).trim()
    let value = line.slice(eq + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"') && value.length > 1)
      || (value.startsWith("'") && value.endsWith("'") && value.length > 1)) {
      value = value.slice(1, -1)
    }
    if (key) out[key] = value
  }
  return out
}

/**
 * Reads `.env` into process.env. Idempotent: a second call (from a test, or from a module
 * that only needs the env to be there) does nothing.
 */
export function loadEnv(file = join(ROOT, '.env')) {
  if (loaded) return process.env
  loaded = true
  let text
  try {
    if (!existsSync(file)) return process.env
    text = readFileSync(file, 'utf8')
  } catch {
    return process.env
  }
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1)
  for (const [key, value] of Object.entries(parseEnv(text))) {
    if (!(key in process.env)) process.env[key] = value
  }
  return process.env
}

// Runs on import — this IS the module's whole job.
loadEnv()
