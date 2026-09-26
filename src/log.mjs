// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Minimal structured logger (zero dependencies).
 *
 * Production needs machine-readable logs: one JSON object per line, so a log collector
 * can filter by level, status, duration or request id without regex-parsing prose. When
 * stdout is a TTY the same records are printed as readable single lines instead.
 *
 *   LOG_LEVEL   debug | info | warn | error   (default: info)
 *   LOG_FORMAT  json | text                   (default: json when not a TTY, text when TTY)
 *
 * SECURITY: tokens are never logged. Callers pass a short hash prefix (see tokenPrefix)
 * when they want to correlate requests without exposing a credential.
 */
import { createHash } from 'node:crypto'

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 }
const LEVEL = LEVELS[String(process.env.LOG_LEVEL || 'info').toLowerCase()] ?? LEVELS.info
const FORMAT = String(
  process.env.LOG_FORMAT || (process.stdout.isTTY ? 'text' : 'json'),
).toLowerCase()

/** Short, stable, non-reversible token fingerprint for log correlation. */
export function tokenPrefix(token) {
  if (!token) return undefined
  return createHash('sha256').update(String(token)).digest('hex').slice(0, 8)
}

function write(level, message, fields) {
  if (LEVELS[level] < LEVEL) return
  const record = { t: new Date().toISOString(), level, msg: message, ...fields }
  const line =
    FORMAT === 'json'
      ? JSON.stringify(record)
      : `${record.t} ${level.toUpperCase().padEnd(5)} ${message}` +
        (Object.keys(fields).length ? ` ${JSON.stringify(fields)}` : '')
  const stream = level === 'error' || level === 'warn' ? process.stderr : process.stdout
  stream.write(line + '\n')
}

export const log = {
  debug: (message, fields = {}) => write('debug', message, fields),
  info: (message, fields = {}) => write('info', message, fields),
  warn: (message, fields = {}) => write('warn', message, fields),
  error: (message, fields = {}) => write('error', message, fields),
}

/**
 * Resolves the client IP for logging.
 *
 * Behind a reverse proxy `req.socket.remoteAddress` is the proxy, which makes every
 * request look identical. With TRUST_PROXY=1 the first X-Forwarded-For hop is used
 * instead. It is opt-in because a client can spoof that header when there is no proxy.
 */
export function clientIp(req) {
  if (process.env.TRUST_PROXY === '1') {
    const forwarded = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim()
    if (forwarded) return forwarded
    const real = String(req.headers['x-real-ip'] || '').trim()
    if (real) return real
  }
  return req.socket?.remoteAddress || undefined
}
