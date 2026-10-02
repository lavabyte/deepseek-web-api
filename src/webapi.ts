// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * DeepSeek web (chat.deepseek.com) API client:
 * PoW SHA3 WASM solving + chat_session lifecycle + /chat/completion SSE stream parsing.
 *
 * Protocol basis (cross-verified against several active reverse-engineered implementations, 2026):
 *   POST /api/v0/chat/create_pow_challenge  {target_path} → data.biz_data.challenge
 *   POST /api/v0/chat_session/create        {}            → data.biz_data.chat_session.id
 *   POST /api/v0/chat_session/delete        {chat_session_id}
 *   POST /api/v0/chat/completion            {chat_session_id, parent_message_id:null, prompt,
 *                                            ref_file_ids:[], thinking_enabled, search_enabled,
 *                                            model_type, action:null, preempt:false}
 *   Request headers: Authorization: Bearer <token>, Cookie, x-hif-*, x-ds-pow-response
 *   The SSE payload is a patch stream:
 *     {"v":{"response":{...}}}                  full snapshot (fragments / content)
 *     {"p":"response/fragments","o":"APPEND","v":{type,content}}
 *     {"p":"response/fragments/-1/content","v":"…"}
 *     {"p":"response/thinking_content","v":"..."} old format: thinking, direct
 *     {"p":"response/content","v":"..."}          old format: body, direct
 *     {"v":"..."} / {"o":"APPEND","v":"..."}        continuation of the previous path
 *     {"p":"response/status","v":"FINISHED"}    status
 */
import type { WebAuth } from './auth.ts'
import { AdapterLlmError, httpErrorCode, parseRetryAfterMs } from './auth.ts'
// The default ranges come from gate.ts -- the settings-page slider bounds and these defaults MUST be
// the same set, or what the UI shows and what actually runs are two different things.
import {
  DEFAULT_CLEANUP_BATCH,
  DEFAULT_CLEANUP_DELAY_MS,
  DEFAULT_CLEANUP_GAP_MS,
  type CleanupRange,
} from './gate.ts'

export const DS_BASE = 'https://chat.deepseek.com'

/** Known default address of the PoW solver WASM (fallback when page-asset capture fails). */
export const DEFAULT_WASM_URL = 'https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm'

/** Browser UA fallback (used when capture fails). */
export const FALLBACK_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'

export interface DsHeaders {
  [key: string]: string
}

/**
 * Assembles the headers for one web request.
 * Prefers the real browser headers captured at login (extraHeaders), then overrides
 * authorization/cookie/fingerprint with the latest session; user-agent uses the browser value
 * (the web endpoint requires a browser fingerprint). DSH attribution is declared explicitly via
 * the `x-deepseek-harness` header.
 */
export function buildDsHeaders(auth: WebAuth, referer?: string): DsHeaders {
  const headers: DsHeaders = {
    'user-agent': auth.userAgent || FALLBACK_UA,
    accept: 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'content-type': 'application/json',
    origin: DS_BASE,
    referer: referer || `${DS_BASE}/`,
    'x-client-platform': 'web',
    'x-client-version': '2.0.0',
    'x-app-version': '2.0.0',
    ...(auth.extraHeaders ?? {}),
  }
  headers.authorization = `Bearer ${auth.token}`
  headers['content-type'] = 'application/json'
  headers.origin = DS_BASE
  headers.referer = referer || `${DS_BASE}/`
  headers['user-agent'] = auth.userAgent || headers['user-agent'] || FALLBACK_UA
  headers['x-deepseek-harness'] = 'deepseek-harness (+https://github.com/deepseek-ai/deepseek-harness); provider=deepseek-web'
  // These two headers are generated/captured per request by this plugin; never reuse stale snapshot values
  delete headers['x-ds-pow-response']
  if (auth.cookie) headers.cookie = auth.cookie
  else delete headers.cookie
  if (auth.hifDliq) headers['x-hif-dliq'] = auth.hifDliq
  if (auth.hifLeim) headers['x-hif-leim'] = auth.hifLeim
  return headers
}

// -- Response envelope (the web endpoint often returns failure as HTTP 200 + a business error code) --

/** The web endpoint's unified envelope: code===0 means success; when non-zero, msg is a user-facing diagnostic. */
export function envelopeError(json: any): { code: number; msg: string } | undefined {
  if (!json || typeof json !== 'object') return undefined
  const code = (json as any).code
  if (typeof code === 'number' && code !== 0) {
    return { code, msg: String((json as any).msg ?? (json as any).message ?? 'unknown error') }
  }
  // WARNING: the web endpoint puts the REAL business error in data.biz_code while the outer code is
  // still 0. Recognising only the outer code (measured 2026-09-11) downgraded a clearly explained
  // server error into the unintelligible and NON-RETRYABLE
  //   `non-streaming response (content-type: application/json): {...}` + MALFORMED_RESPONSE.
  // Once biz_code is recognised, both the real cause and targeted recovery (recreate the session
  // and retry) become possible.
  const bizCode = (json as any).data?.biz_code
  if (typeof bizCode === 'number' && bizCode !== 0) {
    const bizMsg = (json as any).data?.biz_msg
    const text = bizMsg === undefined || bizMsg === null || bizMsg === '' ? 'unknown error' : String(bizMsg)
    return { code: bizCode, msg: text }
  }
  return undefined
}

/**
 * Account-temporarily-limited detection (measured 2026-09-11):
 *   {"code":0,"data":{"biz_code":5,"biz_msg":"user is muted",
 *                     "biz_data":{"is_muted":1,"mute_until":1789173841.894}}}
 * This is a SERVER-SIDE LIMIT ON THE ACCOUNT (the free web endpoint silently throttling
 * high-frequency automated calls), not a plugin bug: the session is valid and session creation
 * succeeds; only completion is rejected. The release time must be told to the user explicitly,
 * and there must be NO BUSY RETRY -- otherwise every round fires a pointless request and may
 * extend the limit.
 */
export function isMutedError(biz: { code?: number; msg?: string } | undefined): boolean {
  return biz?.code === 5 || /user\s+is\s+muted|account\s+is\s+muted/i.test(String(biz?.msg ?? ''))
}

/** Reads the limit-release time from the response envelope (ms); undefined when unavailable. */
export function muteUntilMs(json: any): number | undefined {
  const raw = (json as any)?.data?.biz_data?.mute_until
  const seconds = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isFinite(seconds) || seconds <= 0) return undefined
  return Math.round(seconds * 1000)
}

/** User-readable message for a limited account (with the release time). */
function mutedMessage(untilMs: number | undefined): string {
  if (untilMs === undefined) {
    return 'DeepSeek Web has temporarily restricted this account (user is muted) and gave no release time. Every web-model call will fail until it lifts — wait it out, or switch to an official API key.'
  }
  const when = new Date(untilMs).toLocaleString('zh-CN', { hour12: false })
  const minutes = Math.max(1, Math.round((untilMs - Date.now()) / 60_000))
  return (
    `DeepSeek Web has temporarily restricted this account (user is muted): expected ${when} — about ${minutes} minutes from now.` +
    'Every web-model call fails until then (the session itself is valid and creating chats still works — only sending is refused). ' +
    'Wait it out, or switch to an official API key. The free web endpoint throttles heavy automated use silently, and a conversation that just ran many tool steps is the likeliest to hit it.'
  )
}

/**
 * Concurrent-generation rejection: "only one generation per account at a time" (measured
 * 2026-09-11: two DSH windows sharing one web account; while one was generating, the other's
 * request got `A message is being generated, please try again later.`).
 * This is NOT a ban (a ban is `user is muted`), but it cannot succeed immediately either --
 * classify it as a retryable RATE_LIMIT and let dsh-llm-retry resend later, rather than failing
 * the whole round.
 */
export function isBusyGenerating(message: string): boolean {
  return /being generated|try again later|请稍后再试|稍后再试|正在生成/i.test(String(message ?? ''))
}

/**
 * Consecutive-throttle state: after one limit hit, back off for longer; do not keep hitting the
 * throttle window. (Measured 2026-09-11 afternoon: one account was limited repeatedly and all 5
 * retries landed inside the window -> the round failed.)
 */
/**
 * The fetch implementation currently in use.
 *
 * Defaults to Node's global fetch (undici). The host may inject **Electron's `net.fetch`** --
 * which goes through Chromium's native network stack and brings a TLS / HTTP2 fingerprint
 * consistent with a real browser. Why it matters: measured Node fetch vs Chrome fingerprints
 * differ STRUCTURALLY (JA4 h1 vs h2, Node has no GREASE, 55 ciphers vs 15, completely different
 * extension sets).
 *
 * Note: in Electron's utility process, `require('electron')` exposes only `net` and
 * `systemPreferences` (measured 2026-09-12), so the host can only inject net.fetch and has no
 * other network-related capability.
 */
let injectedFetch: typeof fetch | undefined

/**
 * The fetch actually used for requests -- deliberately resolved FRESH EACH TIME
 * (`injectedFetch ?? fetch`) instead of freezing the global fetch at module load.
 *
 * Reason (hit in practice on 2026-09-12): freezing it breaks "replace globalThis.fetch AFTER
 * the module loads" -- which is exactly how the unit tests stub it, so requests bypassed the
 * stub and REALLY HIT THE LIVE ENDPOINT (returning an INVALID_TOKEN while the test appeared to
 * verify error classification but was actually making a network call).
 */
function activeFetch(input: any, init?: any): Promise<Response> {
  return (injectedFetch ?? fetch)(input, init)
}

/** Injects a fetch implementation; pass undefined to restore Node's global fetch. */
export function setFetchImpl(impl?: typeof fetch): void {
  injectedFetch = impl
}

/**
 * The fetch currently in effect (diagnostics/update checks use it so SIDE REQUESTS share the
 * same transport as web requests). Note it already resolves fresh each time; use it directly.
 */
export function currentFetch(input: any, init?: any): Promise<Response> {
  return activeFetch(input, init)
}

/** Whether the injected implementation or Node's native one is in use (diagnostics). */
export function fetchImplKind(): 'injected' | 'node' {
  return injectedFetch ? 'injected' : 'node'
}

let throttleStreak = 0
let lastThrottleAt = 0

/** Next throttle backoff (ms): starts at 20 s, doubles each time, capped at 90 s, plus 0-30% jitter. */
export function throttleBackoffMs(now: number = Date.now()): number {
  // More than 5 minutes without a limit hit means the window has passed; restart the counter
  if (now - lastThrottleAt > 5 * 60_000) throttleStreak = 0
  const base = Math.min(20_000 * 2 ** throttleStreak, 90_000)
  const jitter = Math.round(base * 0.3 * Math.random())
  return base + jitter
}

/** Records a throttle; returns the backoff to apply this time (ms). */
function noteThrottled(now: number = Date.now()): number {
  if (now - lastThrottleAt > 5 * 60_000) throttleStreak = 0
  throttleStreak += 1
  lastThrottleAt = now
  return throttleBackoffMs(now)
}

/**
 * Account-level throttling: "sending too frequently".
 *
 * Measured 2026-09-11 16:11 (an SSE error event, not an HTTP 429):
 *   `消息发送过于频繁，请稍后重试`
 * WARNING: note it differs from the concurrent rejection by ONE CHARACTER: the concurrent one
 * says "please try again later", the throttle one says "please retry later". Previously only the
 * former matched, so this one fell through to PROVIDER_ERROR (NON-RETRYABLE) -> the whole round
 * failed and the user had to click Continue manually.
 *
 * It is also not the same as `user is muted` (which has an explicit release time): throttling is
 * short-lived, and a long enough backoff gets past it. Give 20 s (the concurrent one only gets
 * 5 s): hitting it more often makes the limit more likely to be extended.
 */
export function isThrottled(message: string): boolean {
  return /过于频繁|太频繁|操作频繁|too\s+many\s+requests|rate\s*limit|稍后重试|限流/i.test(
    String(message ?? ''),
  )
}

/**
 * Invalid-session detection: the server expresses "this chat_session_id does not exist / is
 * invalid" via biz_msg. Trigger scenarios (measured): the session was deleted before the request
 * went out (the old version scheduled deletion 1.5 s after creating the session, while PoW
 * solving + connecting can exceed 1.5 s), or the server reclaimed an idle session.
 * Such failures ARE TRANSPARENTLY RECOVERABLE: every plugin call uses a fresh session and does
 * not depend on server-side history -> just resend on another session.
 */
export function isInvalidSessionError(biz: { code?: number; msg?: string } | undefined): boolean {
  return /invalid\s+chat\s+session|chat\s+session\s+(?:not\s+found|expired|invalid)|chat_session_id[^\p{L}]{0,4}(?:无效|不存在|已过期|非法)|会话.{0,8}(?:无效|不存在|已过期)/iu.test(
    String(biz?.msg ?? ''),
  )
}

/** Business error code -> stable error code (40003/40001: authorization failure). */
function bizErrorCode(code: number): string {
  if (code === 40003 || code === 40001) return 'AUTH'
  if (code === 429) return 'RATE_LIMIT'
  return 'PROVIDER_ERROR'
}

function bizErrorMessage(code: number, msg: string): string {
  if (code === 40003 || code === 40001) {
    return `DeepSeek Web authorization failed: ${msg} — the session is expired or invalid; paste a fresh token in Settings → DeepSeek Web`
  }
  return `DeepSeek Web error (code ${code}）：${msg}`
}

// -- PoW solving ------------------------------------------

interface PoWChallenge {
  algorithm: string
  challenge: string
  salt: string
  difficulty: string | number
  expire_at: string | number
  signature: string
}

let wasmModuleCache: { url: string; promise: Promise<WebAssembly.Module> } | null = null
/** Verified-working / discovered WASM addresses (cached by the exact value recorded on the credential, to avoid probing on every request). */
let resolvedWasmUrl: { key: string; url: string } | null = null

async function isReachable(url: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const resp = await activeFetch(url, { method: 'GET', headers: { range: 'bytes=0-0' }, signal: signal ?? AbortSignal.timeout(10_000) })
    return resp.ok || resp.status === 206
  } catch {
    return false
  }
}

/** Discovers the current build's sha3 wasm address from the web homepage/JS chunks (the hash changes per release). */
async function discoverWasmUrl(signal?: AbortSignal): Promise<string | undefined> {
  try {
    const html = await (await activeFetch(`${DS_BASE}/`, { signal: signal ?? AbortSignal.timeout(15_000) })).text()
    const direct = html.match(/https?:\/\/[^"'\s]*sha3[_a-z0-9.]*\.wasm/i)
    if (direct) return direct[0]
    const scripts = [...html.matchAll(/(?:src|href)="([^"]+\.js)"/g)].map((match) => match[1]).slice(0, 8)
    for (const src of scripts) {
      const url = src.startsWith('http') ? src : new URL(src, `${DS_BASE}/`).href
      try {
        const js = await (await activeFetch(url, { signal: AbortSignal.timeout(15_000) })).text()
        const found = js.match(/[^"'\s]*sha3[_a-z0-9.]*\.wasm/i)
        if (found) return found[0].startsWith('http') ? found[0] : new URL(found[0], url).href
      } catch {}
    }
  } catch {}
  return undefined
}

/**
 * Resolves a usable PoW WASM address: credential value -> known default -> page discovery.
 * The result is cached once per exact credential value, so probing does not happen on every request.
 */
export async function resolveWasmUrl(auth: WebAuth, signal?: AbortSignal): Promise<string> {
  const key = auth.wasmUrl || ''
  if (resolvedWasmUrl?.key === key) return resolvedWasmUrl.url
  // Whitelist the credential's address first: drop it (and warn) if invalid; never send a request with it
  const fromAuth = checkedWasmUrl(auth.wasmUrl)
  if (auth.wasmUrl && !fromAuth) {
    lastWasmUrlRejection = auth.wasmUrl
  }
  const candidates = [fromAuth, checkedWasmUrl(DEFAULT_WASM_URL)].filter((url): url is string => !!url)
  for (const url of candidates) {
    if (await isReachable(url, signal)) {
      resolvedWasmUrl = { key, url }
      return url
    }
  }
  const discovered = checkedWasmUrl(await discoverWasmUrl(signal))
  if (discovered) {
    resolvedWasmUrl = { key, url: discovered }
    return discovered
  }
  // Nothing usable: fail with a clear 'address invalid / not found' error rather than falling back to an unvalidated address
  return fromAuth ?? checkedWasmUrl(DEFAULT_WASM_URL) ?? DEFAULT_WASM_URL
}

/**
 * F12 (2026-09-12 audit): whitelist validation of the PoW WASM address.
 *
 * Why needed: `auth.wasmUrl` comes mainly from IMPORTED ACCOUNT BACKUPS and can be crafted to
 * any address (the audit reproduced intranet / cloud metadata / file: protocol targets).
 * Electron's net.fetch supports more protocols than Node fetch, so the latter's protocol
 * restrictions cannot be treated as a universal boundary.
 *
 * Why limit to deepseek.com rather than pinning a single host: the default address carries a
 * content hash (sha3_wasm_bg.7b9ca65ddd.wasm) that breaks the moment the vendor changes it;
 * and `wasmUrl` is in fact NOT captured from the browser (it is always empty in browser-login),
 * so page discovery (discoverWasmUrl) is the only fallback path. Keep discovery capability, and
 * only whitelist 'may this be used', blocking SSRF while keeping a way forward.
 */
const MAX_WASM_BYTES = 8 * 1024 * 1024

/** Most recent credential wasmUrl rejected by the whitelist (for diagnostics/tests; this module has no logger, so it keeps state instead of logging). */
export let lastWasmUrlRejection: string | undefined

/** Returns the normalised address when legal, otherwise undefined (the caller falls back and warns). */
export function checkedWasmUrl(raw: unknown): string | undefined {
  if (typeof raw !== 'string' || !raw) return undefined
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return undefined
  }
  if (url.protocol !== 'https:') return undefined
  if (url.username || url.password) return undefined
  if (url.port && url.port !== '443') return undefined
  const host = url.hostname.toLowerCase()
  // Blocks 169.254.169.254 / localhost / intranet / any third party, while tolerating a future CDN change
  if (host !== 'deepseek.com' && !host.endsWith('.deepseek.com')) return undefined
  if (!url.pathname.toLowerCase().endsWith('.wasm')) return undefined
  return url.href
}

async function loadWasmModule(wasmUrl: string): Promise<WebAssembly.Module> {
  if (wasmModuleCache?.url === wasmUrl) return wasmModuleCache.promise
  const promise = (async () => {
    const resp = await activeFetch(wasmUrl, { signal: AbortSignal.timeout(15_000) })
    if (!resp.ok) throw new Error(`PoW WASM fetch failed (HTTP ${resp.status})`)
    // Size cap: the sha3 wasm used for PoW is only tens of KB; 8MB is generous and still blocks 'download hundreds of MB then compile'.
    // Read to bytes first, then validate, to avoid feeding an oversized response straight into WebAssembly.compile.
    const declared = Number(resp.headers?.get?.('content-length') ?? Number.NaN)
    if (Number.isFinite(declared) && declared > MAX_WASM_BYTES) {
      throw new Error(`PoW WASM has an unexpected size (${declared} bytes, limit ${MAX_WASM_BYTES}): refusing to load`)
    }
    const bytes = new Uint8Array(await resp.arrayBuffer())
    if (bytes.byteLength > MAX_WASM_BYTES) {
      throw new Error(`PoW WASM has an unexpected size (${bytes.byteLength} bytes, limit ${MAX_WASM_BYTES}): refusing to load`)
    }
    return WebAssembly.compile(bytes)
  })()
  wasmModuleCache = { url: wasmUrl, promise }
  promise.catch(() => {
    if (wasmModuleCache?.url === wasmUrl) wasmModuleCache = null
  })
  return promise
}

/**
 * Calls DeepSeek's sha3_wasm_bg to solve the PoW.
 * wasm_solve(retptr, challengePtr, challengeLen, prefixPtr, prefixLen, difficulty)；
 * prefix = `${salt}_${expire_at}_`; returns the float64 answer (rounded).
 */
async function solvePoW(challenge: PoWChallenge, wasmUrl: string): Promise<number> {
  const module = await loadWasmModule(wasmUrl)
  const instance = await WebAssembly.instantiate(module, { wbg: {} })
  const e = instance.exports as any
  if (typeof e.wasm_solve !== 'function' || typeof e.__wbindgen_export_0 !== 'function' || !e.memory) {
    throw new Error('PoW WASM exports missing (wasm_solve / __wbindgen_export_0 / memory)')
  }
  const encoder = new TextEncoder()
  const cBytes = encoder.encode(challenge.challenge)
  const pBytes = encoder.encode(`${challenge.salt}_${challenge.expire_at}_`)
  const cP = e.__wbindgen_export_0(cBytes.length, 1) >>> 0
  const pP = e.__wbindgen_export_0(pBytes.length, 1) >>> 0
  new Uint8Array(e.memory.buffer).set(cBytes, cP)
  new Uint8Array(e.memory.buffer).set(pBytes, pP)
  const sp = e.__wbindgen_add_to_stack_pointer(-16)
  e.wasm_solve(sp, cP, cBytes.length, pP, pBytes.length, Number(challenge.difficulty))
  const dv = new DataView(e.memory.buffer)
  const code = dv.getInt32(sp, true)
  const answer = dv.getFloat64(sp + 8, true)
  e.__wbindgen_add_to_stack_pointer(16)
  if (code === 0 || !Number.isFinite(answer) || answer <= 0) throw new Error(`PoW solve failed (code=${code})`)
  return Math.floor(answer)
}

/** Obtains the PoW response header value for one completion request (base64 JSON).
 *
 * The retry wrapper is `createPowHeader` below: this only makes ONE attempt.
 */
async function requestPowHeader(auth: WebAuth, targetPath: string, signal?: AbortSignal): Promise<string> {
  let resp: Response
  try {
    resp = await activeFetch(`${DS_BASE}/api/v0/chat/create_pow_challenge`, {
      method: 'POST',
      headers: buildDsHeaders(auth),
      body: JSON.stringify({ target_path: targetPath }),
      signal,
    })
  } catch (error: any) {
    throw new AdapterLlmError(`DeepSeek PoW challenge request failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  if (!resp.ok) {
    const retryAfter = parseRetryAfterMs(resp.headers.get('retry-after'))
    throw new AdapterLlmError(
      `DeepSeek PoW challenge failed (HTTP ${resp.status})${text ? `: ${text.slice(0, 160)}` : ''}`,
      httpErrorCode(resp.status),
      { status: resp.status, ...(retryAfter !== undefined ? { providerRetryAfterMs: retryAfter } : {}) },
    )
  }
  let json: any
  try {
    json = JSON.parse(text)
  } catch {
    // HTTP 200, but the body is not JSON. Real cases:
    //   - an anti-bot / gateway page (the session expired or the request was challenged),
    //   - an empty body (an edge node dropped the response),
    //   - a truncated body (the connection broke midway).
    // The fragment is included so the log says WHICH case it is, and it is classified as
    // transient (TRANSPORT) -- the challenge is cheap and side-effect free, so a momentary
    // blip should not kill the whole chat round.
    const body = String(text ?? '').trim()
    const looksHtml =
      /^(?:<!doctype\s+html|<html|<\?xml)/i.test(body) ||
      /captcha|cloudflare|anti-?bot|人机验证|安全验证/i.test(body)
    const detail = body.length === 0 ? 'empty body' : `unexpected body: ${body.slice(0, 160)}`
    throw new AdapterLlmError(
      looksHtml
        ? 'DeepSeek PoW challenge returned an HTML page instead of JSON (anti-bot check or expired session)'
        : `DeepSeek PoW challenge returned non-JSON (${detail})`,
      // Both variants are transient at this layer: an HTTP 200 whose body is not JSON is a
      // gateway / anti-bot hiccup, NOT a definitive auth failure. `createPowHeader` retries
      // this code, and if the HTML keeps coming back the user still sees the actionable
      // "anti-bot / expired session" message. (Was `AUTH` for HTML, which no layer retried.)
      'MALFORMED_RESPONSE',
      { status: resp.status },
    )
  }
  const biz = envelopeError(json)
  if (biz) {
    throw new AdapterLlmError(bizErrorMessage(biz.code, biz.msg), bizErrorCode(biz.code), { status: resp.status })
  }
  const challenge: PoWChallenge | undefined = json?.data?.biz_data?.challenge
  if (!challenge?.challenge || !challenge?.salt || !challenge?.signature) {
    throw new AdapterLlmError(
      'DeepSeek PoW challenge is missing fields (the session may have expired, or a human check is required)',
      'MALFORMED_RESPONSE',
      { status: resp.status },
    )
  }
  const wasmUrl = await resolveWasmUrl(auth, signal)
  const answer = await solvePoW(challenge, wasmUrl)
  const payload = JSON.stringify({
    algorithm: challenge.algorithm,
    challenge: challenge.challenge,
    salt: challenge.salt,
    answer,
    signature: challenge.signature,
    target_path: targetPath,
  })
  return Buffer.from(payload).toString('base64')
}

/**
 * PoW challenge with retries.
 *
 * WARNING: 2026-09-18 (user report: repeated "DeepSeek PoW challenge returned non-JSON"):
 * the challenge endpoint can return HTTP 200 with a body that is not JSON (an anti-bot page,
 * an empty body, a truncated response). The old version then threw `MALFORMED_RESPONSE`,
 * which no layer retried -> the whole chat round failed hard.
 *
 * 2026-10-02 (user report: "returned non-JSON (empty body)" still appears occasionally):
 * the endpoint intermittently answers 200 with a COMPLETELY EMPTY body. That is an edge /
 * gateway hiccup, not a client error, and four closely-spaced attempts could all land
 * inside the same blip. The ladder is therefore longer, backs off further, and is jittered
 * so parallel requests do not retry in lockstep against the same unhealthy edge node.
 *
 * The challenge is cheap and side-effect free, so retrying is safe; once attempts are
 * exhausted the error propagates.
 *
 * Only non-transient errors are NOT retried (e.g. `AUTH` from HTTP 401/403 -- an expired
 * session needs a new token, not another attempt).
 */
export async function createPowHeader(auth: WebAuth, targetPath: string, signal?: AbortSignal): Promise<string> {
  // ~15 s total across 8 attempts: enough to ride out an edge-node blip, short enough that
  // a genuinely broken endpoint still fails while the caller is watching.
  const BACKOFF_MS = [250, 500, 1_000, 2_000, 3_000, 4_000, 5_000]
  let lastError: unknown
  for (let attempt = 0; attempt <= BACKOFF_MS.length; attempt += 1) {
    try {
      return await requestPowHeader(auth, targetPath, signal)
    } catch (error: any) {
      const code = error?.code ?? error?.failure?.code
      const retryable = code === 'TRANSPORT' || code === 'MALFORMED_RESPONSE'
      if (!retryable || attempt === BACKOFF_MS.length || signal?.aborted) throw error
      lastError = error
      // Jitter (0-250 ms) so concurrent requests do not hammer the same edge node in step.
      const waitMs = BACKOFF_MS[attempt] + Math.floor(Math.random() * 250)
      // NOTE: deliberately NOT unref'd. The retry delay is part of the operation, so it
      // must keep the event loop alive — an unref'd timer can be skipped when the process
      // is otherwise idle, which made a standalone call hang (observed 2026-09-18).
      await new Promise<void>((resolve) => setTimeout(resolve, waitMs))
    }
  }
  throw lastError
}

// -- File upload (image input) ----------------------------
//
// Measured (2026-09): the web endpoint does not take images as 'native multimodal input'; the
// page uploads the image as a file (`/api/v0/file/upload_file`; the PoW scenario is that path),
// gets `data.biz_data.id` (shaped like file-xxxx, model_kind=VISION), and references it in the
// completion request via `ref_file_ids`. Verified: after uploading a 'left red, right blue' PNG,
// the model answered accurately 'left red, right=blue'.

export interface UploadedFile {
  fileId: string
  name?: string
}

/** Uploads an image and returns the file_id. `data` is the raw encoded bytes (png/jpeg/webp/gif). */
export async function uploadImageFile(
  auth: WebAuth,
  input: { data: Uint8Array; mediaType: string; name?: string },
  signal?: AbortSignal,
): Promise<UploadedFile> {
  const targetPath = '/api/v0/file/upload_file'
  const powHeader = await createPowHeader(auth, targetPath, signal)
  const headers: DsHeaders = { ...buildDsHeaders(auth) }
  // multipart sets the boundary via FormData; content-type must be removed
  delete headers['content-type']
  headers['x-ds-pow-response'] = powHeader

  const form = new FormData()
  const bytes = input.data instanceof Uint8Array ? input.data : new Uint8Array(input.data as any)
  form.append('file', new Blob([bytes], { type: input.mediaType || 'image/png' }), input.name || 'image.png')

  let resp: Response
  try {
    resp = await activeFetch(`${DS_BASE}${targetPath}`, { method: 'POST', headers, body: form, signal })
  } catch (error: any) {
    throw new AdapterLlmError(`DeepSeek image upload failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  let json: any
  try {
    json = JSON.parse(text)
  } catch {
    json = undefined
  }
  if (!resp.ok) {
    throw new AdapterLlmError(
      `DeepSeek image upload failed (HTTP ${resp.status})${text ? `: ${text.slice(0, 160)}` : ''}`,
      httpErrorCode(resp.status),
      { status: resp.status },
    )
  }
  const biz = envelopeError(json)
  if (biz) throw new AdapterLlmError(`DeepSeek image upload rejected (code ${biz.code}）：${biz.msg}`, bizErrorCode(biz.code), { status: resp.status })
  const fileId = json?.data?.biz_data?.id ?? json?.data?.id
  if (typeof fileId !== 'string' || !fileId) {
    throw new AdapterLlmError('DeepSeek image upload returned no file_id', 'MALFORMED_RESPONSE', { status: resp.status })
  }
  return { fileId, ...(input.name ? { name: input.name } : {}) }
}

// -- Session ----------------------------------------------

/** Creates a web chat session and returns the chat_session_id. */
export async function createChatSession(auth: WebAuth, signal?: AbortSignal): Promise<string> {
  let resp: Response
  try {
    resp = await activeFetch(`${DS_BASE}/api/v0/chat_session/create`, {
      method: 'POST',
      headers: buildDsHeaders(auth),
      body: '{}',
      signal,
    })
  } catch (error: any) {
    throw new AdapterLlmError(`DeepSeek session create failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  }
  const text = await resp.text()
  if (!resp.ok) {
    const retryAfter = parseRetryAfterMs(resp.headers.get('retry-after'))
    throw new AdapterLlmError(
      `DeepSeek session create failed (HTTP ${resp.status})${text ? `: ${text.slice(0, 160)}` : ''}`,
      httpErrorCode(resp.status),
      { status: resp.status, ...(retryAfter !== undefined ? { providerRetryAfterMs: retryAfter } : {}) },
    )
  }
  let json: any
  try {
    json = JSON.parse(text)
  } catch {
    throw new AdapterLlmError('DeepSeek session create returned non-JSON', 'MALFORMED_RESPONSE', { status: resp.status })
  }
  const biz = envelopeError(json)
  if (biz) {
    throw new AdapterLlmError(bizErrorMessage(biz.code, biz.msg), bizErrorCode(biz.code), { status: resp.status })
  }
  const id = json?.data?.biz_data?.chat_session?.id || json?.data?.biz_data?.id
  if (typeof id !== 'string' || !id) {
    throw new AdapterLlmError('DeepSeek session create missing id', 'MALFORMED_RESPONSE', { status: resp.status })
  }
  return id
}

/** Best-effort deletion of a web session (avoids polluting the user's web chat list). Fails silently. */
// -- Session cleanup: suppressing the 'create one and immediately delete one every round' machine signature --
//
// Background: one model call sends 4 requests -- create session -> get PoW -> completion -> delete
// session. Of those, 'create a temporary session every round and delete it right after' is one of
// the strongest machine-behaviour signatures (a real person would never do it).
//
// Why sessions cannot simply be reused: DSH hands us the FULL history every time, and the web
// session is STATEFUL -- reusing it would show the server two copies of the context ('history +
// full prompt') and blow the window quickly. So only the DELETION side can be optimised.
//
//   immediate = old behaviour: delete 1.5 s after the call ends (one DELETE per round)
//   deferred  = default: clean up once batchSize sessions accumulate, or once delayMs has passed
//                since the first was queued; on cleanup, FIRST TRY ONE BATCH DELETE request (if the
//                server supports it, N sessions cost 1 request), otherwise fall back to deleting
//                one by one and never try batching again (only one wasted attempt).
//   keep      = never delete: fewest requests, but leaves temporary sessions on the web side.

export type SessionCleanupMode = 'immediate' | 'deferred' | 'keep'

export interface SessionCleanupPolicy {
  mode: SessionCleanupMode
  /** Current effective value: the one drawn for THIS round when ranges are set, otherwise the fixed value (ms). */
  delayMs: number
  /** Current effective value: how many accumulated sessions trigger immediate cleanup. */
  batchSize: number
  /** Current effective deletion gap (ms) -- how long to pause between two adjacent delete requests. */
  gapMs: number
  /**
   * Three optional range pairs. When given, a fresh value is drawn 'per round / per deletion';
   * when omitted, fixed-value semantics are kept -- so old callers (passing a hard batchSize /
   * delayMs) behave exactly as before.
   *
   * Why they exist: A FIXED VALUE IS ITSELF A MACHINE SIGNATURE -- always acting at the 8th item,
   * always waiting exactly 90 seconds, delete requests fired back to back (0 gap). Real people
   * are not that precise.
   */
  batchRange?: CleanupRange
  delayRange?: CleanupRange
  gapRange?: CleanupRange
}

export const DEFAULT_SESSION_CLEANUP: SessionCleanupPolicy = {
  mode: 'deferred',
  // The mean lands on the old defaults (90s / 8 items / 1.5s), so behaviour does not jump after the upgrade -- it just gains variance
  delayMs: Math.round((DEFAULT_CLEANUP_DELAY_MS.min + DEFAULT_CLEANUP_DELAY_MS.max) / 2),
  batchSize: Math.round((DEFAULT_CLEANUP_BATCH.min + DEFAULT_CLEANUP_BATCH.max) / 2),
  gapMs: Math.round((DEFAULT_CLEANUP_GAP_MS.min + DEFAULT_CLEANUP_GAP_MS.max) / 2),
  batchRange: DEFAULT_CLEANUP_BATCH,
  delayRange: DEFAULT_CLEANUP_DELAY_MS,
  gapRange: DEFAULT_CLEANUP_GAP_MS,
}

/**
 * Maximum number of session ids packed into a single request.
 *
 * Why it exists: the queue can build up a lot when 'cleanup is slow' (e.g. you come back two
 * hours later and one flush has to delete dozens). 'Deleting a whole batch in one request' is
 * exactly what users worry about -- so beyond this number it is split into several requests,
 * pausing a random interval between them.
 */
const MAX_IDS_PER_REQUEST = 20

export interface SessionCleanerOptions {
  policy?: Partial<SessionCleanupPolicy>
  logger?: { info?: (msg: string) => void; debug?: (msg: string) => void }
  /** For unit-test injection. */
  fetchImpl?: typeof fetch
  setTimeoutImpl?: (fn: () => void, ms: number) => any
  clearTimeoutImpl?: (t: any) => void
  /** Random source. Injecting a deterministic sequence in tests gives repeatable values. Defaults to Math.random. */
  randomImpl?: () => number
}

export interface SessionCleaner {
  schedule(auth: WebAuth, sessionId: string): void
  /** Flushes the queue immediately (for tests / unload). */
  flush(): Promise<void>
  pendingCount(): number
  policy(): SessionCleanupPolicy
  /** Changes the policy at runtime (called after the settings page saves); returns the new values. */
  configure(next: Partial<SessionCleanupPolicy>): SessionCleanupPolicy
}

export function createSessionCleaner(options: SessionCleanerOptions = {}): SessionCleaner {
  const policy: SessionCleanupPolicy = {
    mode: options.policy?.mode ?? DEFAULT_SESSION_CLEANUP.mode,
    delayMs: Math.max(0, Math.floor(options.policy?.delayMs ?? DEFAULT_SESSION_CLEANUP.delayMs)),
    batchSize: Math.max(1, Math.floor(options.policy?.batchSize ?? DEFAULT_SESSION_CLEANUP.batchSize)),
    // No range given (old caller / old config) -> gap is 0, i.e. NO EXTRA GAP; keeps the old behaviour.
    // Only an explicitly configured gapRange enables 'pause between deletions'.
    gapMs: Math.max(
      0,
      Math.floor(options.policy?.gapMs ?? (options.policy?.gapRange ? DEFAULT_SESSION_CLEANUP.gapMs : 0)),
    ),
    // Ranges are OPTIONAL: when an old caller passes only a hard batchSize / delayMs, this keeps
    // 'no range' = fixed-value semantics (otherwise their 3 would be overridden by the default
    // range 6-10 and the tests and old behaviour would all break).
    ...(options.policy?.batchRange ? { batchRange: options.policy.batchRange } : {}),
    ...(options.policy?.delayRange ? { delayRange: options.policy.delayRange } : {}),
    ...(options.policy?.gapRange ? { gapRange: options.policy.gapRange } : {}),
  }
  /** Gives default delay/batch by mode when the policy switches (immediate uses the old parameters). */
  function applyModeDefaults(): void {
    if (policy.mode === 'immediate') {
      policy.delayMs = 1_500
      policy.batchSize = 1
    } else if (policy.mode === 'deferred' && policy.batchSize <= 1) {
      // Switching back from 'keep / immediate' to 'deferred' draws a fresh set of random values (not the hardcoded defaults)
      policy.delayMs = policy.delayRange
        ? pickInt(policy.delayRange)
        : DEFAULT_SESSION_CLEANUP.delayMs
      policy.batchSize = policy.batchRange
        ? pickInt(policy.batchRange)
        : DEFAULT_SESSION_CLEANUP.batchSize
    }
  }
  const doFetch = options.fetchImpl ?? fetch
  const setT = options.setTimeoutImpl ?? ((fn: () => void, ms: number) => setTimeout(fn, ms))
  const clearT = options.clearTimeoutImpl ?? ((t: any) => clearTimeout(t))
  const logger = options.logger

  let queue: { auth: WebAuth; sessionId: string }[] = []
  let timer: any
  /** Set once the server is found to reject batch deletion -- after that, always delete one by one; no more wasted requests. */
  let batchUnsupported = false

  const random = options.randomImpl ?? Math.random

  /** Integer in [min, max] (closed interval). `random` is injectable, so tests are repeatable. */
  function pickInt(range: CleanupRange): number {
    const lo = Math.min(range.min, range.max)
    const hi = Math.max(range.min, range.max)
    if (hi <= lo) return lo
    // An injected fake random source may return 1; clamp to stay in range
    return Math.min(hi, lo + Math.floor(random() * (hi - lo + 1)))
  }

  /**
   * Drawn anew when a new cleanup round starts (the queue goes from empty to non-empty): how many
   * to accumulate this round, and how long to wait at most.
   *
   * Why draw per ROUND rather than every time: the threshold and wait time must stay stable within
   * a round, otherwise 'accumulate 6-10' degrades into 'seemingly always triggering'. A fresh set
   * each round keeps variance without losing rhythm.
   */
  function rollCycle(): void {
    if (policy.mode !== 'deferred') return
    if (policy.batchRange) policy.batchSize = Math.max(1, pickInt(policy.batchRange))
    if (policy.delayRange) policy.delayMs = Math.max(0, pickInt(policy.delayRange))
  }

  /** Draws a gap before each delete request (and records the current value for the settings page). */
  function rollGap(): number {
    policy.gapMs = policy.gapRange ? Math.max(0, pickInt(policy.gapRange)) : Math.max(0, policy.gapMs)
    return policy.gapMs
  }

  /** Sleeps using the injected timer (in tests this is 'wait for the fake timer to fire'). */
  function sleep(ms: number): Promise<void> {
    if (!(ms > 0)) return Promise.resolve()
    return new Promise<void>((resolve) => {
      const handle = setT(() => resolve(), ms)
      ;(handle as any)?.unref?.()
    })
  }

  /** Sets up the 'cleanup when due' timer once (leaves it alone if already set). */
  function armTimer(): void {
    if (timer !== undefined) return
    if (policy.mode === 'keep') return
    timer = setT(() => {
      void flush()
    }, Math.max(0, policy.delayMs))
    ;(timer as any)?.unref?.()
  }

  async function deleteOne(auth: WebAuth, sessionId: string): Promise<void> {
    try {
      await doFetch(`${DS_BASE}/api/v0/chat_session/delete`, {
        method: 'POST',
        headers: buildDsHeaders(auth),
        body: JSON.stringify({ chat_session_id: sessionId }),
        signal: AbortSignal.timeout(10_000),
      })
    } catch {
      /* a cleanup failure does not affect the main flow */
    }
  }

  /**
   * Deletes one batch: prefer a single batch request; if the server does not accept it, DELETE ONE BY ONE.
   *
   * When deleting one by one, a random gap is paused between two requests -- this used to be FIRED
   * BACK TO BACK (a batch of 20 meant 20 consecutive requests), which is the most script-like part.
   */
  async function deleteChunk(batch: { auth: WebAuth; sessionId: string }[]): Promise<void> {
    if (batch.length === 0) return
    // WARNING: F07 (2026-09-12 audit): a batch delete sends ONLY ONE credential (an HTTP request has
    // only one Authorization header), so if the batch mixes sessions from different accounts it is
    // 'deleting B's sessions with A's credential' -- at best the server rejects the whole batch, at
    // worst resp.ok is taken as total success (the old code returned on ok without verifying each id
    // was actually deleted). When accounts are mixed it degrades to one-by-one deletion, which uses
    // each session's own auth.
    const firstToken = batch[0].auth?.token
    const sameAccount = batch.every((item) => item.auth?.token === firstToken)
    if (batch.length > 1 && !batchUnsupported && sameAccount) {
      try {
        const resp = await doFetch(`${DS_BASE}/api/v0/chat_session/delete`, {
          method: 'POST',
          headers: buildDsHeaders(batch[0].auth),
          body: JSON.stringify({ chat_session_ids: batch.map((b) => b.sessionId) }),
          signal: AbortSignal.timeout(15_000),
        })
        let ok = resp.ok
        if (ok) {
          const text = await resp.text().catch(() => '')
          try {
            const json = text ? JSON.parse(text) : undefined
            if (json && envelopeError(json)) ok = false
          } catch {
            ok = false
          }
        }
        if (ok) {
          logger?.debug?.(`deepseek-web: cleaned up ${batch.length} temporary chats in a single request`)
          return
        }
        batchUnsupported = true
        logger?.debug?.('deepseek-web: the server rejected batch chat deletion; deleting one by one from now on')
      } catch {
        // a network error != unsupported; still worth trying next time
      }
    }

    for (let i = 0; i < batch.length; i += 1) {
      if (i > 0) await sleep(rollGap())
      await deleteOne(batch[i].auth, batch[i].sessionId)
    }
    logger?.debug?.(`deepseek-web: cleaned up ${batch.length} temporary chats`)
  }

  /** The real cleanup worker. Sharded: even a large queue is not deleted all at once (see MAX_IDS_PER_REQUEST). */
  async function doFlush(): Promise<void> {
    if (timer !== undefined) {
      clearT(timer)
      timer = undefined
    }
    const batch = queue
    queue = []
    try {
      if (batch.length === 0) return
      for (let i = 0; i < batch.length; i += MAX_IDS_PER_REQUEST) {
        if (i > 0) await sleep(rollGap())
        await deleteChunk(batch.slice(i, i + MAX_IDS_PER_REQUEST))
      }
    } catch (error: any) {
      // a cleanup failure does not affect the main flow
      logger?.debug?.(`deepseek-web: chat cleanup error (ignored): ${error?.message ?? error}`)
    } finally {
      // new sessions may have accumulated during cleanup: re-arm the timer, or they sit in the queue until the next enqueue
      if (queue.length > 0) armTimer()
    }
  }

  /**
   * Flushes the queue immediately.
   *
   * SERIALISED: if the previous cleanup has not finished, this one waits behind it -- otherwise the
   * delete requests of two flushes interleave, exactly the 'fired back to back' we want to avoid.
   * A queued flush takes the queue only when its turn comes, so it picks up sessions accumulated
   * in the meantime.
   */
  let chain: Promise<void> = Promise.resolve()
  function flush(): Promise<void> {
    chain = chain.then(doFlush, doFlush)
    return chain
  }

  function schedule(auth: WebAuth, sessionId: string): void {
    if (policy.mode === 'keep') return
    // Queue goes from empty to non-empty = a new round begins -> redraw this round's threshold and max wait
    if (queue.length === 0) rollCycle()
    queue.push({ auth, sessionId })
    // Only 'deferred' cleans up as soon as the threshold is reached. 'immediate' always goes through
    // the delay -- keeping the old behaviour: delete a while after the call ends, avoiding the too-tight
    // rhythm of 'a DELETE immediately after the stream ends'.
    if (policy.mode === 'deferred' && queue.length >= policy.batchSize) {
      void flush()
      return
    }
    armTimer()
  }

  function configure(next: Partial<SessionCleanupPolicy>): SessionCleanupPolicy {
    const modeChanged = next.mode !== undefined && next.mode !== policy.mode
    if (next.mode !== undefined) policy.mode = next.mode
    if (next.delayMs !== undefined) policy.delayMs = Math.max(0, Math.floor(next.delayMs))
    if (next.batchSize !== undefined) policy.batchSize = Math.max(1, Math.floor(next.batchSize))
    if (next.gapMs !== undefined) policy.gapMs = Math.max(0, Math.floor(next.gapMs))
    // Range: replace it when given; leave it as-is when not ('omitted = no randomness' semantics must not be lost)
    for (const key of ['batchRange', 'delayRange', 'gapRange'] as const) {
      const value = next[key]
      if (value && Number.isFinite(value.min) && Number.isFinite(value.max)) {
        policy[key] = { min: Math.floor(Math.min(value.min, value.max)), max: Math.floor(Math.max(value.min, value.max)) }
      }
    }
    if (modeChanged) applyModeDefaults()
    if (policy.mode === 'keep') void flush() // when switching to 'never delete', clear what is already queued to avoid leftovers
    logger?.info?.(
      `deepseek-web: chat cleanup policy updated — ${policy.mode}` +
        (policy.mode === 'deferred'
          ? ` (clean up after ${policy.batchSize} chats or ${Math.round(policy.delayMs / 1000)}s` +
            (policy.gapRange ? `; when batch delete is unsupported, delete one by one with a ${policy.gapMs}ms` : '') +
            '）'
          : ''),
    )
    return { ...policy }
  }

  return {
    schedule,
    flush,
    pendingCount: () => queue.length,
    policy: () => ({ ...policy }),
    configure,
  }
}

/** Default cleaner (immediate semantics, compatible with old callers). */
const defaultCleaner = createSessionCleaner({
  policy: { mode: 'immediate', delayMs: 1_500, batchSize: 1 },
})

export function scheduleDeleteSession(auth: WebAuth, sessionId: string): void {
  defaultCleaner.schedule(auth, sessionId)
}

/** Validates the session: prefers users/current, falls back to a PoW challenge probe when the endpoint does not exist. */
/**
 * Picks a DISPLAYABLE account identifier from the `users/current` user object.
 *
 * Two pitfalls that must be remembered (both hit in practice, 2026-09-12):
 *
 *  1. DO NOT chain them with `??`. For an account with no email set, the endpoint returns
 *     `email: ""`, and an empty string is NOT nullish -- `"" ?? x` yields `""`, which blocks the
 *     whole fallback chain on the spot, so display is always empty and the UI has to fall back to
 *     showing the internal id (`acc_cd8e05ec`). So pick by 'HAS CONTENT', skipping undefined /
 *     null / whitespace.
 *
 *  2. THE FIELD NAME MUST MATCH THE RESPONSE. The phone number is `mobile_number` (not `mobile`),
 *     and the server already returns a MASKED form (e.g. `183******78`), so it can be shown directly.
 *
 * Measured response shape (only relevant fields listed):
 *   { id, token, email: "", mobile_number: "183******78", area_code: "+86", chat: {...} }
 */
export function pickUserDisplay(user: any): string {
  const candidates = [
    user?.email,
    user?.mobile_number,
    user?.mobile,
    user?.phone,
    user?.username,
    user?.nickname,
    user?.name,
  ]
  for (const value of candidates) {
    if (value === undefined || value === null) continue
    const text = String(value).trim()
    if (text) return text
  }
  return ''
}

/**
 * Decides whether the SHAPE of the `users/current` response body is trustworthy.
 *
 * WARNING: F09 (2026-09-12 audit): the old code set json to undefined when `resp.json()` threw,
 * and `envelopeError(undefined)` returns undefined, so it went straight to `ok: true` and returned
 * an EMPTY user({}). In other words: an anti-bot page / WAF block page / empty response -- which
 * are also HTTP 200 -- would be treated as 'validation passed'.
 *
 * The consequence is very real: the probe says 'passed' and the account looks fine, so the
 * 're-login required' button added in 0.1.31 never fires; nothing was confirmed, yet it reports
 * success. The cost of an occasional failure on a read-only, zero-quota request is one retry --
 * far cheaper than a false success.
 *
 * Extracted as a pure function so it can be unit-tested (validateAuth makes network requests, so this branch cannot be tested).
 */
export function classifyAuthEnvelope(json: unknown): { ok: true } | { ok: false; error: string } {
  if (!json || typeof json !== 'object' || Array.isArray(json)) {
    return { ok: false, error: 'the users/current response is not a JSON object (probably an anti-bot page or a gateway block)' }
  }
  const bizError = envelopeError(json)
  if (bizError) return { ok: false, error: bizError.msg }
  // Shape fallback: neither data nor code means this is not a business envelope we recognise.
  if ((json as any).data === undefined && (json as any).code === undefined) {
    return { ok: false, error: 'the users/current response has neither data nor code (unexpected shape)' }
  }
  return { ok: true }
}

export async function validateAuth(
  auth: WebAuth,
  signal?: AbortSignal,
): Promise<{ ok: boolean; user?: { id?: string; display?: string }; error?: string }> {
  try {
    const resp = await activeFetch(`${DS_BASE}/api/v0/users/current`, { headers: buildDsHeaders(auth), signal })
    if (resp.ok) {
      let json: any
      try {
        json = await resp.json()
      } catch {
        json = undefined
      }
      // shape validation (pure function; see the comment on classifyAuthEnvelope)
      const verdict = classifyAuthEnvelope(json)
      if (!verdict.ok) return verdict
      const payload = json?.data?.biz_data ?? json?.data
      const user = payload?.user ?? payload ?? {}
      const display = pickUserDisplay(user)
      return {
        ok: true,
        user: {
          ...(user?.id !== undefined ? { id: String(user.id) } : {}),
          ...(display ? { display } : {}),
        },
      }
    }
    if (resp.status === 404) {
      await createPowHeader(auth, '/api/v0/chat/completion', signal)
      return { ok: true }
    }
    return { ok: false, error: `users/current HTTP ${resp.status}` }
  } catch (error: any) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

// -- SSE stream parsing -----------------------------------

export type WebStreamEvent =
  | { kind: 'thinking'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'status'; value: string }
  | { kind: 'meta'; messageId: number | string }
  | { kind: 'finish'; reason?: string }
  | {
      kind: 'error'
      message: string
      raw?: string
      /** Semantic classification (e.g. concurrent generation -> RATE_LIMIT); the caller decides whether to retry. */
      code?: string
      retryAfterMs?: number
      /** RATE_LIMIT sub-type: concurrent preemption (wait for the other side to finish) vs account throttling (wait for the limit to lift) -- the message and backoff differ. */
      rateLimitKind?: 'concurrent' | 'throttled'
    }

interface Fragment {
  /** Server-side fragment id, when present. Used to dedupe snapshot vs BATCH delivery. */
  id?: number | string
  type: string
  content: string
  emitted: number
}

function isReasoningType(type: string): boolean {
  const t = type.toUpperCase()
  return t === 'THINK' || t === 'REASONING' || t === 'THINKING'
}

/** Splits a byte stream into lines (SSE frames are separated by \n). */
async function* iterateLines(body: any): AsyncGenerator<string> {
  const decoder = new TextDecoder()
  let buffer = ''
  const drain = function* (): Generator<string> {
    let idx: number
    while ((idx = buffer.indexOf('\n')) !== -1) {
      yield buffer.slice(0, idx).replace(/\r$/, '')
      buffer = buffer.slice(idx + 1)
    }
  }
  if (typeof body?.getReader === 'function') {
    const reader = body.getReader()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        yield* drain()
      }
    } finally {
      try {
        reader.releaseLock?.()
      } catch {}
    }
  } else if (body?.[Symbol.asyncIterator]) {
    for await (const chunk of body) {
      buffer += decoder.decode(chunk, { stream: true })
      yield* drain()
    }
  }
  if (buffer.length > 0) yield buffer.replace(/\r$/, '')
}

/**
 * State machine for parsing the web completion payload (unit-testable: handle returns events to yield).
 *
 * WARNING: the correctness model (a 2026-09 incident fix):
 *   Real incident: the model answered a full paragraph, yet DSH showed only 1-3 character
 *   fragments like '，' '不上' '了一圈', together with EMPTY_RESPONSE retries. Root cause: the old
 *   implementation de-duplicated with a MONOTONICALLY INCREASING emitted counter, while a snapshot
 *   can reset the derived text to something shorter -- once the counter was inflated it stayed high,
 *   and text was only emitted when its length exceeded it, so everything before was lost and only
 *   the remainder's tail survived; an empty remainder then triggered a retry.
 *
 *   The rules now:
 *     1) INCREMENTAL EVENTS DRIVE EMISSION (fragment APPEND / -1/content / thinking_content / content / bare v)
 *     2) SNAPSHOTS ONLY RECONCILE: top up only when the candidate text is a STRICT EXTENSION of what
 *        was already emitted; shorter (stale snapshot) or divergent (server reorder/rollback) is
 *        ignored outright, and emitted content is never reset
 *     3) a snapshot can never make emitted content smaller -> no lost characters and no false EMPTY_RESPONSE
 */
export function createSseState() {
  const fragments: Fragment[] = []
  /** fragments-derived text (only a candidate for snapshot reconciliation). */
  let fragmentsText = ''
  let fragmentsThinking = ''
  /** Derived text for the direct format (only a candidate for snapshot reconciliation). */
  let directText = ''
  let directThinking = ''
  /** The canonical emitted stream (monotonically increasing). */
  let outText = ''
  let outThinking = ''
  let divergences = 0
  let sink: 'fragments' | 'thinking' | 'content' | null = null
  /**
   * Which channel the last emitted delta belonged to. Used as a fallback when a
   * continuation arrives with no fragment to attach to — before 2026-09-19 it always
   * went to text, so a thinking continuation whose fragment had been dropped leaked
   * into the visible answer.
   */
  let lastKind: 'text' | 'thinking' = 'text'
  let pendingFinish: string | undefined
  let sawData = false
  /** Response message id, needed by /chat/stop_stream to halt server-side generation. */
  let responseMessageId: number | string | undefined
  let metaEmitted = false

  const emit = (out: WebStreamEvent[], kind: 'text' | 'thinking', delta: string): void => {
    if (!delta) return
    if (kind === 'text') outText += delta
    else outThinking += delta
    lastKind = kind
    out.push({ kind, text: delta })
  }
  const emitText = (out: WebStreamEvent[], delta: string): void => emit(out, 'text', delta)
  const emitThinking = (out: WebStreamEvent[], delta: string): void => emit(out, 'thinking', delta)

  /** Snapshot reconciliation: top up only when the candidate is a strict extension; ignore stale/divergent (better to miss one snapshot than emit garbage or lose characters). */
  const reconcile = (out: WebStreamEvent[], kind: 'text' | 'thinking', candidate: string): void => {
    const current = kind === 'text' ? outText : outThinking
    if (!candidate || candidate === current) return
    if (candidate.startsWith(current)) {
      emit(out, kind, candidate.slice(current.length))
      return
    }
    if (current.startsWith(candidate)) return // stale (shorter) snapshot
    divergences += 1 // divergent: ignore
  }

  /** Rebuilds the fragments-derived text (used when a snapshot overwrites). */
  const rebuildFragmentText = (): void => {
    fragmentsText = ''
    fragmentsThinking = ''
    for (const fragment of fragments) {
      if (isReasoningType(fragment.type)) fragmentsThinking += fragment.content
      else fragmentsText += fragment.content
    }
  }
  /** Snapshot: full table replacement + reconciliation (does not emit directly). */
  const replaceFragments = (list: any[]): void => {
    const incoming: Fragment[] = []
    for (const f of list) {
      if (f && typeof f === 'object' && typeof f.content === 'string') {
        const id = f.id ?? f.fragment_id
        incoming.push({
          ...(id !== undefined && id !== null ? { id } : {}),
          type: String(f.type ?? 'RESPONSE'),
          content: f.content,
          emitted: 0,
        })
      }
    }
    // ⚠️ 2026-09-19 (user report: a whole thinking block appeared as normal text):
    // a snapshot that carries NO usable fragments must not wipe the fragments we are in
    // the middle of. Otherwise the next `response/fragments/-1/content` finds no last
    // fragment and falls back to TEXT — even though the continuation is thinking.
    // Adopt the snapshot only when it actually has data, or before anything was emitted.
    if (incoming.length === 0 && fragments.length > 0 && (outText.length > 0 || outThinking.length > 0)) {
      return
    }
    fragments.length = 0
    fragments.push(...incoming)
    rebuildFragmentText()
    sink = fragments.length > 0 ? 'fragments' : null
  }
  /**
   * Adds a fragment, or reconciles it against one already known by `id`.
   *
   * WARNING: 2026-09-19 (user report: "the code cuts off a chunk, so entire thoughts are sent
   * as text"): DeepSeek delivers the SAME fragment twice — once in a full snapshot and once
   * in a `{"p":"response","o":"BATCH"}` envelope. Deduping by id prevents a double emit,
   * and treating a longer same-id payload as an extension prevents a lost delta.
   */
  const upsertFragment = (f: any, out: WebStreamEvent[]): void => {
    if (!f || typeof f !== 'object' || typeof f.content !== 'string') return
    const id = f.id ?? f.fragment_id
    const existing =
      id !== undefined && id !== null ? fragments.find((x) => x.id === id) : undefined
    if (existing) {
      // Known fragment: emit only the extension, if the server grew it.
      if (f.content.length > existing.content.length && f.content.startsWith(existing.content)) {
        const delta = f.content.slice(existing.content.length)
        existing.content = f.content
        if (isReasoningType(existing.type)) {
          fragmentsThinking += delta
          emitThinking(out, delta)
        } else {
          fragmentsText += delta
          emitText(out, delta)
        }
      }
      return
    }
    const fragment: Fragment = {
      ...(id !== undefined && id !== null ? { id } : {}),
      type: String(f.type ?? 'RESPONSE'),
      content: f.content,
      emitted: 0,
    }
    fragments.push(fragment)
    if (isReasoningType(fragment.type)) {
      fragmentsThinking += fragment.content
      emitThinking(out, fragment.content)
    } else {
      fragmentsText += fragment.content
      emitText(out, fragment.content)
    }
  }
  /** Incremental: append a fragment (its content is new -> emitted directly). */
  const appendFragments = (incoming: any, out: WebStreamEvent[]): void => {
    const list = Array.isArray(incoming) ? incoming : incoming !== undefined ? [incoming] : []
    for (const f of list) upsertFragment(f, out)
    sink = fragments.length > 0 ? 'fragments' : null
  }
  /** Incremental: continue writing the last fragment. */
  const appendToLastFragment = (text: string, out: WebStreamEvent[]): void => {
    const fragment = fragments[fragments.length - 1]
    if (!fragment) {
      // WARNING: 2026-09-19 (user report: "the code cuts off a chunk, so entire thoughts are sent
      // as text"): with no last fragment there is nothing to attach to, and the old
      // fallback ALWAYS went to text — so a thinking continuation whose fragment had been
      // lost leaked into the visible answer. Follow the channel of the last emitted delta.
      if (lastKind === 'thinking') {
        directThinking += text
        emitThinking(out, text)
      } else {
        directText += text
        emitText(out, text)
      }
      return
    }
    fragment.content += text
    if (isReasoningType(fragment.type)) {
      fragmentsThinking += text
      emitThinking(out, text)
    } else {
      fragmentsText += text
      emitText(out, text)
    }
  }
  /** Incremental: a bare continuation belongs to the current sink. */
  const appendSink = (text: string, out: WebStreamEvent[]): void => {
    if (sink === 'thinking') {
      directThinking += text
      emitThinking(out, text)
    } else if (sink === 'content') {
      directText += text
      emitText(out, text)
    } else if (sink === 'fragments') {
      appendToLastFragment(text, out)
    }
  }

  return {
    /** Payload handling (increments emit directly; snapshots only reconcile). */
    handlePayload(d: any, eventName?: string): WebStreamEvent[] {
      const out: WebStreamEvent[] = []
      sawData = true
      // 0) Capture the response message id once. It arrives first in the `ready`
      //    event (`{"response_message_id":2,...}`) and is also present in the full
      //    snapshot (`v.response.message_id`). /chat/stop_stream needs it to abort
      //    generation, so emit it as a `meta` event for the transport to remember.
      const metaId = d && typeof d === 'object' ? (d.response_message_id ?? d.v?.response?.message_id) : undefined
      if (metaId !== undefined && metaId !== null && !metaEmitted) {
        metaEmitted = true
        responseMessageId = metaId
        out.push({ kind: 'meta', messageId: metaId })
      }
      // 1) full response snapshot
      if (d && typeof d === 'object' && d.v && typeof d.v === 'object' && d.v.response && typeof d.v.response === 'object') {
        const response = d.v.response
        if (Array.isArray(response.fragments)) {
          replaceFragments(response.fragments)
          // fragments win when present; otherwise use content
          if (fragments.length > 0) {
            reconcile(out, 'thinking', fragmentsThinking)
            reconcile(out, 'text', fragmentsText)
          }
        }
        if (typeof response.content === 'string' && response.content.length > 0) {
          // A snapshot's `content` is only the direct-format answer. When fragment data is
          // present it is authoritative, so do NOT let a late/empty `content` hijack the
          // sink away from an in-progress thinking fragment (the 2026-09-19 leak).
          if (fragments.length === 0) {
            directText = response.content
            sink = 'content'
            reconcile(out, 'text', directText)
          }
        }
        if (response.finish_reason !== undefined && response.finish_reason !== null) {
          pendingFinish = String(response.finish_reason)
        }
        return out
      }
      // 2) model error event: classify semantically (concurrent generation -> retryable RATE_LIMIT); the caller decides retry vs error
      if (d && typeof d === 'object' && d.type === 'error') {
        const message = typeof d.content === 'string' ? d.content : typeof d.message === 'string' ? d.message : 'model error'
        const event: WebStreamEvent & { raw?: string } = {
          kind: 'error',
          message,
          ...(d.finish_reason !== undefined ? { raw: String(d.finish_reason) } : {}),
        }
        if (isBusyGenerating(message)) {
          event.code = 'RATE_LIMIT'
          event.retryAfterMs = 5_000
          event.rateLimitKind = 'concurrent'
        } else if (isThrottled(message)) {
          // account-level throttling ('sending too frequently, retry later'): back off progressively on repeated hits; do not keep hitting it
          event.code = 'RATE_LIMIT'
          event.retryAfterMs = noteThrottled()
          event.rateLimitKind = 'throttled'
        }
        out.push(event)
        return out
      }
      // 3) named SSE events (title ignored; toast treated as an informational error)
      if (eventName === 'toast') {
        const message = d && typeof d === 'object' ? (d.content ?? d.message ?? JSON.stringify(d)) : String(d)
        const full = `DeepSeek toast: ${String(message).slice(0, 200)}`
        const event: WebStreamEvent = { kind: 'error', message: full }
        if (isBusyGenerating(full)) {
          event.code = 'RATE_LIMIT'
          event.retryAfterMs = 5_000
          event.rateLimitKind = 'concurrent'
        } else if (isThrottled(full)) {
          event.code = 'RATE_LIMIT'
          event.retryAfterMs = noteThrottled()
          event.rateLimitKind = 'throttled'
        }
        out.push(event)
        return out
      }
      if (eventName === 'title') return out
      // 4) top-level finish_reason
      if (d && typeof d === 'object' && d.finish_reason !== undefined && d.finish_reason !== null) {
        pendingFinish = String(d.finish_reason)
        return out
      }
      const path: string | undefined = d?.p
      const value = d?.v
      if (typeof path === 'string') {
        switch (path) {
          case 'response/fragments':
            appendFragments(value, out)
            return out
          case 'response/fragments/-1/content': {
            if (typeof value === 'string') {
              appendToLastFragment(value, out)
              sink = 'fragments'
            }
            return out
          }
          case 'response/thinking_content':
            if (typeof value === 'string') {
              directThinking += value
              emitThinking(out, value)
              sink = 'thinking'
            }
            return out
          case 'response/content':
            if (typeof value === 'string') {
              directText += value
              emitText(out, value)
              sink = 'content'
            }
            return out
          case 'response/finish_reason':
            if (typeof value === 'string') pendingFinish = value
            return out
          case 'response/status':
            if (typeof value === 'string') {
              out.push({ kind: 'status', value })
              if (value === 'FINISHED') pendingFinish = pendingFinish ?? 'FINISHED'
            }
            return out
          case 'response': {
            // BATCH envelope: `{"p":"response","o":"BATCH","v":[{p,v},...]}`.
            //
            // ⚠️ 2026-09-19 (user report: a whole thinking block leaked into the answer):
            // the inner ops carry paths RELATIVE to `response/` and usually have NO `o`
            // field, e.g. `{"p":"fragments","v":[{type:"THINK",...}]}`. The old code only
            // accepted `o === 'APPEND'`, so the THINK fragment was dropped, `fragments`
            // stayed empty, and every following `-1/content` fell back to TEXT.
            if (Array.isArray(value)) {
              for (const op of value) {
                if (!op || typeof op !== 'object') continue
                const subPath = op.p
                const subValue = op.v
                if (subPath === 'fragments') {
                  appendFragments(subValue, out)
                } else if (subPath === 'fragments/-1/content') {
                  if (typeof subValue === 'string') {
                    appendToLastFragment(subValue, out)
                    sink = 'fragments'
                  }
                } else if (subPath === 'thinking_content') {
                  if (typeof subValue === 'string') {
                    directThinking += subValue
                    emitThinking(out, subValue)
                    sink = 'thinking'
                  }
                } else if (subPath === 'content') {
                  if (typeof subValue === 'string') {
                    directText += subValue
                    emitText(out, subValue)
                    sink = 'content'
                  }
                } else if (subPath === 'status') {
                  if (typeof subValue === 'string') {
                    out.push({ kind: 'status', value: subValue })
                    if (subValue === 'FINISHED') pendingFinish = pendingFinish ?? 'FINISHED'
                  }
                } else if (subPath === 'finish_reason') {
                  if (typeof subValue === 'string') pendingFinish = subValue
                }
                // `has_pending_fragment` and anything else: ignore.
              }
            }
            return out
          }
          default:
            return out
        }
      }
      // 5) a continuation with no path: belongs to the current sink
      if (typeof value === 'string' && value.length > 0) appendSink(value, out)
      return out
    },
    /** Public entry point (payloads already emitted increments directly; this only does fallback reconciliation). */
    handle(d: any, eventName?: string): WebStreamEvent[] {
      return this.handlePayload(d, eventName)
    },
    /** End of stream: produce finish (if data was actually received). */
    finish(): WebStreamEvent[] {
      return sawData ? [{ kind: 'finish', reason: pendingFinish }] : []
    },
    /** Diagnostics: emitted body/thinking length and snapshot divergence count (for tests and troubleshooting). */
    stats(): { text: string; thinking: string; divergences: number } {
      return { text: outText, thinking: outThinking, divergences }
    },
  }
}

/** Parses the /chat/completion SSE byte stream, producing incremental text/thinking events. */
export async function* parseWebSse(body: any): AsyncGenerator<WebStreamEvent> {
  const state = createSseState()
  let eventName = ''
  /**
   * F13 (2026-09-12 audit): the SSE spec allows MULTIPLE `data:` lines in one event, which must be
   * joined with `\n` and parsed as a whole once complete. The old implementation ran `JSON.parse`
   * per line, so once the server split one JSON across several lines (or the payload itself
   * contained newlines) every line failed to parse and was silently dropped by `catch { continue }`
   * -- showing up as 'the stream suddenly broke / a chunk is missing' with no error at all.
   */
  let dataLines: string[] = []
  const flushData = (): { events: WebStreamEvent[]; done: boolean } => {
    if (dataLines.length === 0) return { events: [], done: false }
    const data = dataLines.join('\n').trim()
    dataLines = []
    if (data.length === 0) return { events: [], done: false }
    if (data === '[DONE]') return { events: Array.from(state.finish()), done: true }
    let parsed: any
    try {
      parsed = JSON.parse(data)
    } catch {
      return { events: [], done: false }
    }
    return { events: Array.from(state.handle(parsed, eventName)), done: false }
  }

  for await (const line of iterateLines(body)) {
    if (line.length === 0) {
      // blank line = end of event
      const flushed = flushData()
      for (const event of flushed.events) yield event
      if (flushed.done) return
      eventName = ''
      continue
    }
    if (line.startsWith(':')) continue
    if (line.startsWith('event:')) {
      // a new event name appearing = the previous event ended (some implementations omit the blank line; close it here too)
      const flushed = flushData()
      for (const event of flushed.events) yield event
      if (flushed.done) return
      eventName = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
      continue
    }
  }
  // end of stream: the tail may have no blank line; accumulated data must not be dropped
  const tail = flushData()
  for (const event of tail.events) yield event
  if (tail.done) return
  for (const event of state.finish()) yield event
}

// -- Completion request -----------------------------------

// -- Session reuse (added 2026-09-12 after a measured determination) --

/**
 * Maximum requests sent on one reused web session before switching to a new one; 0 = disable reuse (back to 'one session per request').
 *
 * Why reuse is possible (MEASURED, not inferred):
 * every completion sends `parent_message_id: null` -> every message is a ROOT message in the
 * session with no parent chain, and the server's context walk over the message tree reaches
 * nothing. Determination experiment (2026-09-12): in the same session, first send 'remember the
 * code ZC-7391-KX, reply only OK' -> got `OK`; then ask 'what is the code' -> answered 'I do not
 * know'. This proves same-session history does NOT enter the context. (The 0.1.21 comment
 * claiming 'reuse would double the context' was unmeasured inference, disproven by this experiment.)
 *
 * Benefit: measured 2026-09-12, 182 web sessions were created in one day (peak 74/hour, densest
 * 8/minute), because every DSH round = create a session and delete one after use -- no real person
 * creates and deletes conversations like that. After reuse, session creation drops to 'rounds / N'.
 */
export const DEFAULT_SESSION_REUSE_TURNS = 20

/** Reuse slot: the session currently reusable for an account. Key is the credential digest (never logged, never a plaintext key). */
let reuseSlot: { key: string; sessionId: string; turns: number } | undefined

/** Credential digest: only used to tell 'is this the same account'. Not a security measure, never logged. */
function accountKey(auth: WebAuth): string {
  const raw = `${auth?.token ?? ''}|${auth?.cookie ?? ''}`
  let hash = 2166136261
  for (let i = 0; i < raw.length; i += 1) {
    hash ^= raw.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(36)
}

interface SessionLease {
  sessionId: string
  reused: boolean
  /** Old sessions invalidated by rotation (the caller is responsible for reclaiming them). */
  retired?: string
}

async function leaseSession(
  auth: WebAuth,
  signal: AbortSignal,
  transport: CompletionTransport,
  maxTurns: number,
): Promise<SessionLease> {
  const key = accountKey(auth)
  const limit = Number.isFinite(maxTurns) ? Math.max(0, Math.floor(maxTurns)) : DEFAULT_SESSION_REUSE_TURNS
  if (limit > 0 && reuseSlot && reuseSlot.key === key && reuseSlot.turns < limit) {
    reuseSlot.turns += 1
    return { sessionId: reuseSlot.sessionId, reused: true }
  }
  // Account switch / rotation limit reached: retire the old session (on an account switch, do not reclaim someone else's session and delete it by mistake)
  const retired = reuseSlot && reuseSlot.key === key ? reuseSlot.sessionId : undefined
  const sessionId = await transport.createSession(auth, signal)
  reuseSlot = limit > 0 ? { key, sessionId, turns: 1 } : undefined
  return { sessionId, reused: false, ...(retired ? { retired } : {}) }
}

/** Removes a session from the reuse slot (called on session invalidation / request failure; a new one is created next time). */
export function retireSession(sessionId?: string): void {
  if (!sessionId || (reuseSlot && reuseSlot.sessionId === sessionId)) reuseSlot = undefined
}

/** For tests only: clears the reuse slots. */
export function resetSessionReuse(): void {
  reuseSlot = undefined
}

export interface CompletionParams {
  prompt: string
  thinkingEnabled: boolean
  searchEnabled?: boolean
  modelType: 'default' | 'expert' | 'vision'
  /** The uploaded file's file_id (image input: referenced in the request so the model can see it). */
  refFileIds?: readonly string[]
  signal?: AbortSignal
  idleTimeoutMs?: number
  /** Reuse round limit for one session (0 = one session per request, deleted after use). */
  sessionReuseTurns?: number
  onDeleteSession?: (sessionId: string) => void
}

/**
 * Injectable transport layer for sessions/requests (the real implementation by default).
 * Extracted so unit tests can deterministically reproduce the two paths 'session invalid ->
 * recreate and retry' and 'deletion timing', without hitting the network for real (these two
 * are exactly where problems kept recurring).
 */
export interface CompletionTransport {
  createSession: (auth: WebAuth, signal?: AbortSignal) => Promise<string>
  powHeader: (auth: WebAuth, targetPath: string, signal?: AbortSignal) => Promise<string>
}

const defaultTransport: CompletionTransport = { createSession: createChatSession, powHeader: createPowHeader }

/**
 * Opens one completion request (create session + PoW + send) and returns the usable session and response.
 *
 * Non-SSE responses (a business error envelope wrapped in HTTP 200) are adjudicated here uniformly:
 *  - session invalid (invalid chat session id) -> TRANSPARENTLY RETRY ONCE with a new session (the user does not notice);
 *  - other business errors -> thrown by business code (AUTH / RATE_LIMIT / PROVIDER_ERROR...).
 */
async function openCompletion(
  auth: WebAuth,
  params: CompletionParams,
  signal: AbortSignal,
  transport: CompletionTransport,
): Promise<{ sessionId: string; resp: Response }> {
  let lastFailure: AdapterLlmError | undefined
  for (let attempt = 0; attempt < 2; attempt++) {
    const lease = await leaseSession(
      auth,
      signal,
      transport,
      params.sessionReuseTurns ?? DEFAULT_SESSION_REUSE_TURNS,
    )
    const sessionId = lease.sessionId
    // Only sessions invalidated by ROTATION are reclaimed here; the current session is kept for the next request to reuse
    if (lease.retired) params.onDeleteSession?.(lease.retired)
    let resp: Response
    try {
      resp = await activeFetch(`${DS_BASE}/api/v0/chat/completion`, {
        method: 'POST',
        headers: {
          ...buildDsHeaders(auth, `${DS_BASE}/a/chat/s/${sessionId}`),
          accept: 'text/event-stream',
          'x-ds-pow-response': await transport.powHeader(auth, '/api/v0/chat/completion', signal),
        },
        body: JSON.stringify({
          chat_session_id: sessionId,
          parent_message_id: null,
          prompt: params.prompt,
          ref_file_ids: params.refFileIds ?? [],
          thinking_enabled: params.thinkingEnabled,
          search_enabled: params.searchEnabled ?? false,
          model_type: params.modelType,
          action: null,
          preempt: false,
        }),
        signal,
      })
    } catch (error: any) {
      if (params.signal?.aborted) throw new AdapterLlmError('DeepSeek web request aborted by caller', 'ABORTED', { cause: error })
      throw new AdapterLlmError(`DeepSeek web request failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
    }

    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      const code = httpErrorCode(resp.status)
      const retryAfter = parseRetryAfterMs(resp.headers.get('retry-after'))
      const hint =
        code === 'AUTH'
          ? ' — the web session may have expired; paste a fresh token in Settings → DeepSeek Web'
          : code === 'RATE_LIMIT'
            ? ' — web-side throttling on the free tier; just retry later'
            : ''
      retireSession(sessionId) // discard on failure; use a new session next time
      params.onDeleteSession?.(sessionId)
      throw new AdapterLlmError(
        `DeepSeek web completion failed (HTTP ${resp.status})${text ? `: ${text.slice(0, 200)}` : ''}${hint}`,
        code,
        { status: resp.status, ...(retryAfter !== undefined ? { providerRetryAfterMs: retryAfter } : {}), cause: new Error(text) },
      )
    }
    if (!resp.body) {
      retireSession(sessionId)
      params.onDeleteSession?.(sessionId)
      throw new AdapterLlmError('DeepSeek web completion returned no body', 'EMPTY_RESPONSE')
    }

    // HTTP 200 may also be a 'business error envelope' or an HTML challenge page -- any non-SSE is treated as an error first
    const contentType = String(resp.headers.get('content-type') ?? '')
    if (contentType.includes('text/event-stream')) return { sessionId, resp }

    const text = await resp.text().catch(() => '')
    let parsed: any
    try {
      parsed = JSON.parse(text)
    } catch {}
    const biz = envelopeError(parsed)
    const muted = isMutedError(biz)
    const busy = !muted && !!biz && isBusyGenerating(biz.msg)
    const untilMs = muteUntilMs(parsed)
    const failure = biz
      ? new AdapterLlmError(
          muted
            ? mutedMessage(untilMs)
            : busy
              ? 'DeepSeek Web allows only one generation per account at a time (another window or tab is generating with the same account). This step retries automatically; if you need web models in two places, give one of them a different provider or account.'
              : bizErrorMessage(biz.code, biz.msg),
          muted || busy ? 'RATE_LIMIT' : isInvalidSessionError(biz) ? 'TRANSPORT' : bizErrorCode(biz.code),
          {
            status: resp.status,
            // release time far exceeds the retry policy's ceiling -> dsh-llm-retry gives up immediately (rather than busy-firing requests)
            ...(muted && untilMs !== undefined ? { providerRetryAfterMs: Math.max(0, untilMs - Date.now()) } : {}),
            // the absolute value is carried separately: the host records it on the account and shows a countdown in the settings page
            ...(muted && untilMs !== undefined ? { mutedUntilMs: untilMs } : {}),
            ...(busy ? { providerRetryAfterMs: 5_000 } : {}),
          },
        )
      : new AdapterLlmError(
          `DeepSeek Web returned a non-streaming response (content-type: ${contentType || 'unknown'}）：${text.slice(0, 200)}`,
          'MALFORMED_RESPONSE',
          { status: resp.status },
        )
    retireSession(sessionId) // this session is dead; reclaim it now rather than leave garbage
    params.onDeleteSession?.(sessionId)
    if (attempt === 0 && biz && isInvalidSessionError(biz)) {
      lastFailure = failure
      continue
    }
    throw failure
  }
  throw lastFailure ?? new AdapterLlmError('DeepSeek Web could not establish a usable chat', 'PROVIDER_ERROR')
}

/**
 * Issues one web completion request and streams events; the session is best-effort deleted AFTER THE STREAM ENDS.
 *
 * WARNING: deletion timing is the easiest thing to get wrong in this module (measured failure
 * 2026-09-11): the old implementation called `onDeleteSession` IMMEDIATELY AFTER creating the
 * session, and internally it 'deletes after 1.5 s', so the session could be deleted by itself
 * BEFORE the completion request went out -- if PoW solving + connecting exceeded 1.5 s, the
 * server replied with an invalid session. More insidious is 'the session vanishes halfway through
 * generation', where the server may cut the stream outright -- showing up as a reply that stops
 * mid-sentence and an incomplete tool call (exactly the class of truncation we have been chasing).
 * Deletion now happens only in finally (normal end, error or caller abort all reach it), so the
 * session is alive for the whole request.
 */
export async function* streamWebCompletion(
  auth: WebAuth,
  params: CompletionParams,
  transport: CompletionTransport = defaultTransport,
): AsyncGenerator<WebStreamEvent> {
  const idle = params.idleTimeoutMs ?? 120_000
  const controller = new AbortController()
  const signal = params.signal ? AbortSignal.any([params.signal, controller.signal]) : controller.signal

  const { sessionId, resp } = await openCompletion(auth, params, signal, transport)

  // Idle watchdog: an SSE event gap exceeding idle is a timeout
  let timer: ReturnType<typeof setTimeout> | null = null
  let settled = false
  let fireIdle: (error: unknown) => void = () => {}
  const idlePromise = new Promise<never>((_, reject) => {
    fireIdle = reject
  })
  const armIdle = (): void => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      if (settled) return
      controller.abort('idle timeout')
      fireIdle(new AdapterLlmError(`DeepSeek web stream idle timeout after ${idle}ms`, 'TIMEOUT'))
    }, idle)
    ;(timer as any).unref?.()
  }
  armIdle()

  try {
    const iterator = parseWebSse(resp.body)[Symbol.asyncIterator]()
    while (true) {
      const result = await Promise.race([iterator.next(), idlePromise])
      armIdle()
      if (result.done) break
      yield result.value
    }
  } catch (error: any) {
    if (error instanceof AdapterLlmError) throw error
    if (params.signal?.aborted) throw new AdapterLlmError('DeepSeek web stream aborted by caller', 'ABORTED', { cause: error })
    throw new AdapterLlmError(`DeepSeek web stream failed: ${error?.message ?? error}`, 'TRANSPORT', { cause: error })
  } finally {
    settled = true
    if (timer) clearTimeout(timer)
    try {
      controller.abort('stream consumer stopped')
    } catch {}
    // Session reclamation is queued AFTER THE STREAM ENDS (normal end / error / caller abort all reach here).
    // Deleting early would make the session vanish mid-generation -- see the incident note at the top of streamWebCompletion.
    // In reuse mode the current session is NOT deleted (it is left for the next request); only when reuse is off is it reclaimed here.
    if ((params.sessionReuseTurns ?? DEFAULT_SESSION_REUSE_TURNS) === 0) {
      params.onDeleteSession?.(sessionId)
    }
  }
}
