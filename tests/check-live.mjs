/**
 * Live-account test: upload -> waitForFile(SUCCESS) -> completion with ref_file_ids.
 * Checks an image AND a text file (whether the model reads the document content).
 *
 * Run: node tests/check-live.mjs
 */
import { authFromToken, deleteChatSession, newChatSession, uploadFileReady, collectCompletion, verifyToken } from '../src/deepseek.mjs'

function envToken() {
  const token = (process.env.DEEPSEEK_SESSION_TOKEN || '').trim()
  if (!token) {
    console.error('No token. Set the variable: export DEEPSEEK_SESSION_TOKEN=<token-from-chat.deepseek.com>')
    process.exit(1)
  }
  return token
}

const auth = authFromToken(envToken())

const who = await verifyToken(auth)
console.log('token:', who.ok ? `OK (${who.user?.email || who.user?.mobile || who.user?.id})` : `FAIL — ${who.error}`)
if (!who.ok) process.exit(1)

// 128x128 PNG: left half red, right half blue (no external file dependency)
import zlib from 'node:zlib'
const crcTable = (() => { const t = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0 } return t })()
const crc32 = (b) => { let c = 0xffffffff; for (const x of b) c = crcTable[(c ^ x) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0 }
const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const tb = Buffer.from(type, 'ascii'); const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([tb, data]))); return Buffer.concat([len, tb, data, crc]) }
function makePng(w, h, pixel) {
  const raw = Buffer.alloc((w * 3 + 1) * h)
  for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; for (let x = 0; x < w; x++) { const [r, g, b] = pixel(x, y); const o = y * (w * 3 + 1) + 1 + x * 3; raw[o] = r; raw[o + 1] = g; raw[o + 2] = b } }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))])
}

let failures = 0
const check = (label, ok, detail) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`); if (!ok) failures += 1 }

const cleanup = []
try {
  // ── 1. image: upload + wait + completion ──────────────────────────────
  const png = makePng(128, 128, (x) => (x < 64 ? [255, 0, 0] : [0, 0, 255]))
  const t0 = Date.now()
  const up = await uploadFileReady(auth, { data: new Uint8Array(png), mediaType: 'image/png', name: 'redblue.png' })
  console.log(`\nimage upload: ${up.fileId} status=${up.status} (${Date.now() - t0}ms)`)
  check('image ready for use (status SUCCESS)', up.status === 'SUCCESS', up.status)

  const sid = await newChatSession(auth); cleanup.push(sid)
  const r = await collectCompletion(auth, { sessionId: sid, prompt: 'What do you see in the attached image? Answer briefly: left=X, right=Y.', refFileIds: [up.fileId] })
  console.log(`   reply: ${JSON.stringify(r.text.slice(0, 160))}`)
  check('model sees the image (red/blue)', /czerwon|red/i.test(r.text) && /niebiesk|blue/i.test(r.text), r.text.slice(0, 80))

  // ── 2. text file ──────────────────────────────────────────────────────
  const txt = Buffer.from('Service report.\nDOCUMENT-MARKER: FILE-7731\nStatus: OK\n', 'utf8')
  const t1 = Date.now()
  const upTxt = await uploadFileReady(auth, { data: new Uint8Array(txt), mediaType: 'text/plain', name: 'report.txt' })
  console.log(`\nfile upload: ${upTxt.fileId} status=${upTxt.status} (${Date.now() - t1}ms)`)
  check('text file ready for use', upTxt.status === 'SUCCESS', upTxt.status)

  const sid2 = await newChatSession(auth); cleanup.push(sid2)
  const r2 = await collectCompletion(auth, { sessionId: sid2, prompt: 'What is the document marker in the attached file? Give the exact value.', refFileIds: [upTxt.fileId] })
  console.log(`   reply: ${JSON.stringify(r2.text.slice(0, 160))}`)
  check('model reads the text file content', /FILE-7731/.test(r2.text), r2.text.slice(0, 100))

  // ── 3. thinking ───────────────────────────────────────────────────────
  const sid3 = await newChatSession(auth); cleanup.push(sid3)
  const r3 = await collectCompletion(auth, { sessionId: sid3, prompt: 'Compute 17*3. Give only the result.', thinkingEnabled: true })
  console.log(`\nthinking: ${r3.thinking.length} chars, reply=${JSON.stringify(r3.text.slice(0, 60))}`)
  check('thinking_enabled produces a separate thought stream', r3.thinking.length > 0, `${r3.thinking.length} chars`)
  check('reply is correct (51)', /51/.test(r3.text), r3.text.slice(0, 40))
} finally {
  for (const sid of cleanup) await deleteChatSession(auth, sid)
}

console.log(`\n${failures === 0 ? 'ALL OK' : `${failures} FAILURES`}`)
process.exit(failures === 0 ? 0 : 1)