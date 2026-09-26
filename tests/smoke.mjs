/**
 * Full server smoke test: starts the server on a random port and calls it like an OpenAI client.
 *
 * Checks:
 *   1. GET  /v1/models                        -> exactly one model, expected id
 *   2. POST /v1/chat/completions (non-stream) -> reply + usage
 *   3. POST /v1/chat/completions (stream)     -> SSE deltas + [DONE]
 *   4. the same persistent session after both -> the history is NOT duplicated on the web side
 *   5. image (data URL)                       -> the model sees the content
 *   6. text file (base64)                     -> the model reads the content
 *   7. tool_calls (JSON protocol)             -> the call is parsed correctly
 *
 * Run:
 *   DEEPSEEK_SESSION_TOKEN='<token-from-chat.deepseek.com>' node tests/smoke.mjs
 *
 * The token is this server's API KEY — it travels in the Authorization: Bearer header.
 * It can also be placed in tests/.token (gitignored).
 */
import { spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as sleep } from 'node:timers/promises'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = Number(process.env.SMOKE_PORT || 8899)
const BASE = `http://127.0.0.1:${PORT}`

/** Token = API key. Order: env, then tests/.token. */
function readToken() {
  const fromEnv = (process.env.DEEPSEEK_SESSION_TOKEN || '').trim()
  if (fromEnv) return fromEnv
  const file = join(ROOT, 'tests', '.token')
  if (existsSync(file)) return readFileSync(file, 'utf8').trim()
  return ''
}

const TOKEN = readToken()
if (!TOKEN) {
  console.error(
    'No token. Set it in the environment or in tests/.token:\n' +
      "  DEEPSEEK_SESSION_TOKEN='<token>' node tests/smoke.mjs\n" +
      '  # or: echo "<token>" > tests/.token',
  )
  process.exit(1)
}

/** Authorization headers for every call to our server. */
const AUTH = { authorization: `Bearer ${TOKEN}` }

let failures = 0
const check = (label, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

// ── 128x128 PNG, solid red, as a data URL ───────────────────────────────
// Note: tiny 1x1 PNGs are interpreted inconsistently (the first version of this test got
// a "white" answer). A 128x128 solid color is unambiguous for the model.
import zlib from 'node:zlib'
const crcTable = (() => { const t = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0 } return t })()
const crc32 = (b) => { let c = 0xffffffff; for (const x of b) c = crcTable[(c ^ x) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0 }
const pngChunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const tb = Buffer.from(type, 'ascii'); const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([tb, data]))); return Buffer.concat([len, tb, data, crc]) }
function solidPng(size, [r, g, b]) {
  const raw = Buffer.alloc((size * 3 + 1) * size)
  for (let y = 0; y < size; y++) {
    raw[y * (size * 3 + 1)] = 0
    for (let x = 0; x < size; x++) {
      const o = y * (size * 3 + 1) + 1 + x * 3
      raw[o] = r; raw[o + 1] = g; raw[o + 2] = b
    }
  }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4); ihdr[8] = 8; ihdr[9] = 2
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk('IHDR', ihdr), pngChunk('IDAT', zlib.deflateSync(raw)), pngChunk('IEND', Buffer.alloc(0))])
}
const RED_PNG_DATA_URL = 'data:image/png;base64,' + solidPng(128, [255, 0, 0]).toString('base64')

const child = spawn(process.execPath, [join(ROOT, 'src', 'server.mjs')], {
  cwd: ROOT,
  env: { ...process.env, PORT: String(PORT), HOST: '127.0.0.1' },
  stdio: ['ignore', 'pipe', 'pipe'],
})
let serverLog = ''
child.stdout.on('data', (d) => { serverLog += d.toString() })
child.stderr.on('data', (d) => { serverLog += d.toString() })

const shutdown = () => {
  if (!child.killed) child.kill('SIGTERM')
}
process.on('exit', shutdown)

async function waitForServer(timeoutMs = 25_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${BASE}/v1/models`)
      if (r.ok) return true
    } catch {}
    await sleep(300)
  }
  return false
}

/** Parses the SSE from a chat.completions reply in OpenAI format. */
async function readSse(res) {
  const text = await res.text()
  const chunks = []
  let done = false
  for (const block of text.split('\n\n')) {
    const line = block.split('\n').find((l) => l.startsWith('data: '))
    if (!line) continue
    const payload = line.slice(6)
    if (payload === '[DONE]') { done = true; continue }
    try { chunks.push(JSON.parse(payload)) } catch {}
  }
  let content = ''
  let reasoning = ''
  const toolCalls = []
  let finishReason = null
  let usage = null
  for (const c of chunks) {
    if (c.error) throw new Error(`stream error: ${c.error.message}`)
    if (!c.choices || c.choices.length === 0) { if (c.usage) usage = c.usage; continue }
    const choice = c.choices[0]
    if (choice.delta?.content) content += choice.delta.content
    if (choice.delta?.reasoning_content) reasoning += choice.delta.reasoning_content
    for (const tc of choice.delta?.tool_calls ?? []) {
      const index = tc.index ?? 0
      toolCalls[index] ??= { id: '', name: '', arguments: '' }
      if (tc.id) toolCalls[index].id = tc.id
      if (tc.function?.name) toolCalls[index].name += tc.function.name
      if (tc.function?.arguments) toolCalls[index].arguments += tc.function.arguments
    }
    if (choice.finish_reason) finishReason = choice.finish_reason
  }
  return { content, reasoning, toolCalls: toolCalls.filter(Boolean), finishReason, usage, done, chunks }
}

async function chat(body) {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...AUTH },
    body: JSON.stringify(body),
  })
  if (body.stream) return { status: res.status, ...(await readSse(res)) }
  const json = await res.json().catch(() => null)
  return { status: res.status, json }
}

try {
  const up = await waitForServer()
  if (!up) {
    console.error('The server did not start within 25 s. Log:\n' + serverLog)
    process.exit(1)
  }
  console.log('server started\n')

  // ── 1. /v1/models ─────────────────────────────────────────────────────
  const models = await (await fetch(`${BASE}/v1/models`)).json()
  check('GET /v1/models returns a list with one model', models?.object === 'list' && models.data?.length === 1, JSON.stringify(models?.data?.map((m) => m.id)))
  check('model id is deepseek/deepseek-v4-flash', models?.data?.[0]?.id === 'deepseek/deepseek-v4-flash', models?.data?.[0]?.id)

  // ── 2. non-streaming ──────────────────────────────────────────────────
  const plain = await chat({
    model: 'deepseek/deepseek-v4-flash',
    messages: [{ role: 'user', content: 'Answer with exactly one word: PONG' }],
  })
  check('non-stream: HTTP 200', plain.status === 200, String(plain.status))
  check('non-stream: object is chat.completion', plain.json?.object === 'chat.completion', plain.json?.object)
  check('non-stream: role is assistant', plain.json?.choices?.[0]?.message?.role === 'assistant')
  check('non-stream: non-empty reply', (plain.json?.choices?.[0]?.message?.content ?? '').length > 0, JSON.stringify(plain.json?.choices?.[0]?.message?.content?.slice(0, 60)))
  check('non-stream: usage present', Number.isFinite(plain.json?.usage?.total_tokens), JSON.stringify(plain.json?.usage))

  // ── 3. streaming ──────────────────────────────────────────────────────
  const streamed = await chat({
    model: 'deepseek/deepseek-v4-flash',
    stream: true,
    messages: [{ role: 'user', content: 'Count from 1 to 3, digits only.' }],
  })
  check('stream: HTTP 200', streamed.status === 200)
  check('stream: [DONE] at the end', streamed.done === true)
  check('stream: some text deltas', streamed.content.length > 0, JSON.stringify(streamed.content.slice(0, 60)))
  check('stream: finish_reason present', streamed.finishReason !== null, String(streamed.finishReason))
  check('stream: usage present', Number.isFinite(streamed.usage?.total_tokens), JSON.stringify(streamed.usage))

  // ── 4. one session (unnamed) ──────────────────────────────────────────
  // /health requires a token (without it, it returns ok:true but without account verification).
  const health = await (await fetch(`${BASE}/health`, { headers: AUTH })).json()
  check('health: token valid', health?.ok === true, JSON.stringify(health?.token))
  check('health: session has no name (we do not set a title)', !('title' in (health?.session ?? {})) && !('titleApplied' in (health?.session ?? {})), JSON.stringify(health?.session))
  check('health: session id stable', typeof health?.session?.id === 'string' && health.session.id.length > 0, health?.session?.id)

  // ── 5. image ──────────────────────────────────────────────────────────
  const vision = await chat({
    model: 'deepseek/deepseek-v4-flash',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What color is the attached image? Answer with one word.' },
          { type: 'image_url', image_url: { url: RED_PNG_DATA_URL } },
        ],
      },
    ],
  })
  const visionText = vision.json?.choices?.[0]?.message?.content ?? ''
  check('image: the reply mentions red', /czerwon|red|красн/i.test(visionText), JSON.stringify(visionText.slice(0, 80)))

  // ── 6. text file ──────────────────────────────────────────────────────
  const docBody = 'Report. MARKER-SMOKE: FILE-9182. End.'
  const fileCall = await chat({
    model: 'deepseek/deepseek-v4-flash',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is the MARKER-SMOKE value in the attached file? Give the value only.' },
          { type: 'file', file: { filename: 'report.txt', file_data: Buffer.from(docBody, 'utf8').toString('base64'), media_type: 'text/plain' } },
        ],
      },
    ],
  })
  const fileText = fileCall.json?.choices?.[0]?.message?.content ?? ''
  check('file: the model read the content', /FILE-9182/.test(fileText), JSON.stringify(fileText.slice(0, 80)))

  // ── 7. tools ──────────────────────────────────────────────────────────
  const toolCall = await chat({
    model: 'deepseek/deepseek-v4-flash',
    messages: [{ role: 'user', content: 'What is the weather in Warsaw? Use the tool.' }],
    tools: [
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Returns the current weather for the given city.',
          parameters: {
            type: 'object',
            properties: { city: { type: 'string', description: 'City name' } },
            required: ['city'],
          },
        },
      },
    ],
    tool_choice: 'required',
  })
  const calls = toolCall.json?.choices?.[0]?.message?.tool_calls ?? []
  check('tools: the model returned tool_calls', calls.length > 0, JSON.stringify(calls).slice(0, 160))
  check('tools: tool name correct', calls[0]?.function?.name === 'get_weather', calls[0]?.function?.name)
  let parsedArgs = null
  try { parsedArgs = JSON.parse(calls[0]?.function?.arguments ?? '') } catch {}
  check('tools: arguments are valid JSON with city', typeof parsedArgs?.city === 'string', JSON.stringify(parsedArgs))
  check('tools: finish_reason = tool_calls', toolCall.json?.choices?.[0]?.finish_reason === 'tool_calls', toolCall.json?.choices?.[0]?.finish_reason)

  // ── 8. session still the same (no request created a new one) ──────────
  const health2 = await (await fetch(`${BASE}/health`, { headers: AUTH })).json()
  check('persistent session unchanged after 6 requests', health2?.session?.id === health?.session?.id, `${health?.session?.id} -> ${health2?.session?.id}`)

  // ── 9. API key = web token ────────────────────────────────────────────
  // Without a token, chat must refuse (401), while the model list stays public.
  const noAuth = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'deepseek/deepseek-v4-flash', messages: [{ role: 'user', content: 'test' }] }),
  })
  check('missing API key -> 401', noAuth.status === 401, String(noAuth.status))
  const badAuth = await fetch(`${BASE}/health`, { headers: { authorization: 'Bearer not-a-real-token' } })
  const badAuthJson = await badAuth.json().catch(() => null)
  check('bad token -> health 503', badAuth.status === 503, `${badAuth.status} ${JSON.stringify(badAuthJson?.token)}`)
  check('another token does not steal the session', badAuthJson?.session?.id === null, JSON.stringify(badAuthJson?.session))
} catch (error) {
  check('smoke without exceptions', false, error?.stack ?? String(error))
  if (serverLog) console.error('\n--- server log ---\n' + serverLog.slice(-3000))
} finally {
  shutdown()
  await sleep(400)
}

console.log(`\n${failures === 0 ? 'ALL OK' : `${failures} FAILURES`}`)
if (failures !== 0 && serverLog) console.error('\n--- server log (tail) ---\n' + serverLog.slice(-2000))
process.exit(failures === 0 ? 0 : 1)