// SPDX-License-Identifier: MIT
// Copyright (c) 2026 DeepSeek Web API contributors
/**
 * Credential resolution: WHO supplies the DeepSeek session tokens.
 *
 * Two run modes, chosen entirely from .env — the code path is otherwise identical, which
 * is what lets the home edition stop being a separate source tree:
 *
 *   1. CLIENT MODE (no DEEPSEEK_TOKENS)
 *      The caller's API key IS the credential: a comma-separated list of chat.deepseek.com
 *      session tokens. The server stores no secret. This is the original behaviour.
 *
 *          Authorization: Bearer tok1,tok2,tok3
 *
 *   2. SERVER MODE (DEEPSEEK_TOKENS has at least one entry)
 *      The operator puts the pool in .env and every client shares it:
 *
 *          DEEPSEEK_TOKENS=tok1,tok2,tok3
 *          API_KEY=                       # empty = anyone who can reach the port
 *          API_KEY=sk-my-private-key      # or: require this key from every client
 *
 *      An empty API_KEY is the LAN / Tailscale case (the port itself is the boundary); a
 *      non-empty API_KEY is compared in constant time and a mismatch is a 401.
 *
 * Mode is decided by whether DEEPSEEK_TOKENS parses to at least one token. Verification of
 * those tokens stays LAZY (the pool checks each on first use) so a boot never blocks on
 * DeepSeek; a pool where every token turns out dead reports POOL_EXHAUSTED (503) instead of
 * silently falling back to client-supplied tokens, which would be a security surprise.
 *
 * A token is never logged and never returned by an endpoint.
 */
// Loads .env before this module reads process.env (see src/env.mjs for why the ordering
// matters). A no-op when the importer already did it.
import './env.mjs'

import { timingSafeEqual } from 'node:crypto'
import { parseTokenList } from './pool.mjs'

/** Comma-separated session tokens the OPERATOR configured; empty = client mode. */
// DEEPSEEK_TOKENS is canonical; HOME_TOKENS is accepted as a migration alias so an
// existing home-edition .env keeps working while that tree is retired.
export const SERVER_TOKENS = parseTokenList(process.env.DEEPSEEK_TOKENS || process.env.HOME_TOKENS || '')

/** `server` when a pool is configured in .env, otherwise `client`. */
export const authMode = SERVER_TOKENS.length > 0 ? 'server' : 'client'

/** Key every client must present in server mode. Empty = this server asks for no key. */
const API_KEY = String(process.env.API_KEY || '').trim()

/**
 * Raw API key from the request (the `Bearer ` prefix already removed).
 * `x-api-key` is accepted too, because several OpenAI clients send that header instead.
 */
export function rawKeyFromRequest(req) {
  const header = String(req.headers.authorization || '')
  if (header.startsWith('Bearer ')) return header.slice(7).trim()
  if (header.startsWith('bearer ')) return header.slice(7).trim()
  return String(req.headers['x-api-key'] || '').trim()
}

/** Constant-time comparison; the length check short-circuits and is not itself a secret. */
function safeEqual(a, b) {
  const left = Buffer.from(String(a))
  const right = Buffer.from(String(b))
  if (left.length !== right.length) return false
  return timingSafeEqual(left, right)
}

/**
 * Decides whether a client may talk to the API at all.
 *  - client mode: always ok here — the tokens themselves are what the pool validates;
 *  - server mode: only the API_KEY check, or nothing at all when API_KEY is empty.
 */
export function checkAuth(req) {
  if (authMode === 'client' || !API_KEY) return { ok: true }
  const provided = rawKeyFromRequest(req)
  if (!provided) {
    return {
      ok: false,
      status: 401,
      code: 'invalid_api_key',
      message: 'Missing API key — send Authorization: Bearer <API_KEY>.',
    }
  }
  if (!safeEqual(provided, API_KEY)) {
    return { ok: false, status: 401, code: 'invalid_api_key', message: 'Invalid API key.' }
  }
  return { ok: true }
}

/**
 * The DeepSeek tokens this request may use.
 *  - server mode: the operator's pool (identical for every client);
 *  - client mode: the tokens embedded in the caller's key.
 */
export function resolveTokens(req) {
  if (authMode === 'server') return SERVER_TOKENS
  return parseTokenList(rawKeyFromRequest(req))
}

/** Public description of the auth setup, for /health and the boot log. Never a credential. */
export function authInfo() {
  if (authMode === 'server') {
    const required = Boolean(API_KEY)
    return {
      mode: 'server',
      required,
      source: required ? 'API_KEY (.env)' : 'none (server-side pool)',
      api_key_required: required,
      tokens_in_pool: SERVER_TOKENS.length,
    }
  }
  return {
    mode: 'client',
    required: true,
    source: 'comma-separated API key',
    api_key_required: true,
    tokens_in_pool: null,
  }
}
