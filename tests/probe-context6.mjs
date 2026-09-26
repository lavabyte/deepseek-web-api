/**
 * Context-limit measurement — with a FRESH session for every measurement.
 *
 * Why earlier results were inconsistent: all measurements were loaded into the SAME
 * persistent "API" session (that is how the server works). After a dozen huge documents
 * the session started returning `DeepSeek error 3: message count exceeded` — meaning the
 * limit was NOT about prompt size, but about the number of messages accumulated in the
 * session. That poisoned the measurement: 3.2M chars "failed" even though it passed in a
 * clean session.
 *
 * Now every size gets a FRESH session, which we delete after the measurement. This way we
 * measure only the input-size limit.
 *
 * Run: node tests/probe-context6.mjs
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as sleep } from 'node:timers/promises'
import { authFromToken, deleteChatSession, newChatSession, streamCompletion } from '../src/deepseek.mjs'
import { estimateTokens } from '../src/openai.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const TOKEN = (process.env.DEEPSEEK_SESSION_TOKEN || '').trim()
if (!TOKEN) {
  console.error('No token. Set the variable: export DEEPSEEK_SESSION_TOKEN=<token-from-chat.deepseek.com>')
  process.exit(1)
}
const auth = authFromToken(TOKEN)
const OUT = join(ROOT, 'data', process.env.OUT_NAME || 'ctx-clean.txt')

const FILLER = process.env.FILLER || 'This is a technical document filler sentence. '
const MARK = process.env.MARK || 'KOD-CZYSTY-9001'
const SIZES = (process.env.SIZES || '1000000,2000000,2600000,3000000,3200000,3300000,3350000,3400000')
  .split(',')
  .map((v) => Number(v.trim()))
  .filter((v) => Number.isFinite(v) && v > 0)

const lines = []
function say(text) {
  lines.push(text)
  try {
    mkdirSync(dirname(OUT), { recursive: true })
    writeFileSync(OUT, lines.join('\n') + '\n', 'utf8')
  } catch {}
  console.log(text)
}

function buildDoc(targetChars) {
  const marker = `ACCESS CODE: ${MARK}. `
  const fillerChars = Math.max(0, targetChars - marker.length)
  const half = Math.floor(fillerChars / 2)
  const a = FILLER.repeat(Math.ceil(half / FILLER.length)).slice(0, half)
  const b = FILLER.repeat(Math.ceil((fillerChars - half) / FILLER.length)).slice(0, fillerChars - half)
  return a + marker + b
}

/** One request in a FRESH session; always deletes the session at the end. */
async function askFresh(prompt) {
  const sessionId = await newChatSession(auth)
  let text = ''
  let error = null
  let statuses = []
  const started = Date.now()
  try {
    for await (const event of streamCompletion(auth, {
      sessionId,
      prompt,
      thinkingEnabled: false,
      idleTimeoutMs: 300_000,
    })) {
      if (event.kind === 'text') text += event.text
      else if (event.kind === 'status') statuses.push(event.value)
      else if (event.kind === 'error') error = event.message
    }
  } catch (err) {
    error = `${err?.code ?? err?.failure?.code ?? 'ERR'}: ${err?.message ?? err}`
  } finally {
    await deleteChatSession(auth, sessionId).catch(() => {})
  }
  return { text, error, statuses, ms: Date.now() - started }
}

/** Control request in a fresh session — distinguishes "size limit" from "sick account". */
async function control() {
  for (let i = 0; i < 3; i += 1) {
    if (i > 0) await sleep(5_000)
    const r = await askFresh('Answer with one short word.')
    if (!r.error && r.text.trim().length > 0) return true
  }
  return false
}

say('CONTEXT LIMIT — fresh session for every measurement')
say('filler: ' + JSON.stringify(FILLER.slice(0, 44)))
say('code ' + MARK + ' at ~50% of the document')
say('='.repeat(90))
say('')

if (!(await control())) {
  say('startup control: FAIL — the account does not respond, aborting')
  process.exit(1)
}
say('startup control: OK')
say('')

let largestOk = 0
let firstFail = null
for (const size of SIZES) {
  const doc = buildDoc(size)
  const prompt = doc + '\n\nQuestion: what ACCESS CODE was given in the document? Answer ONLY with the code or NONE.'
  const r = await askFresh(prompt)
  const found = r.text.includes(MARK)

  say('--- document ' + doc.length + ' / prompt ' + prompt.length + ' chars (~' + estimateTokens(prompt) + ' tok est.) ---')
  say('  time    : ' + (r.ms / 1000).toFixed(1) + 's')
  say('  error   : ' + (r.error ?? '(none)'))
  say('  code    : ' + (found ? 'FOUND' : 'no'))
  say('  reply   : ' + JSON.stringify(r.text.slice(0, 90)))

  if (found) {
    largestOk = Math.max(largestOk, doc.length)
  } else if (firstFail === null) {
    firstFail = { size: doc.length, error: r.error, text: r.text.slice(0, 150) }
  }
  say('')
  await sleep(5_000)
}

say('='.repeat(90))
say('Largest document read in full: ' + (largestOk || '(none)') + ' chars')
if (firstFail) {
  say('First failure: ' + firstFail.size + ' chars')
  say('  error : ' + (firstFail.error ?? '(none — empty reply)'))
}
say('done')