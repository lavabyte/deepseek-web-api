# DeepSeek Web API

### Notice
This entire project was vide-coded with DeepSeek V4.1 Flash
<br><br><br>

An **OpenAI-API-compatible** HTTP server that uses a `chat.deepseek.com` web session
instead of an official API key. It exposes one model:

```
deepseek/deepseek-v4.1-flash
```

It works with any OpenAI client (LangChain, LlamaIndex, Cline, Continue, curl, the
Python/JS SDKs).

## Quick start

Requirement: **Node.js 22.6+** — nothing else. Zero npm dependencies, no build step
(Node strips TypeScript types natively), no compilation.

**Linux / macOS**

```bash
cd deepseek-web-api
./start.sh          # checks the Node version and starts the server
```

**Windows**

```cmd
cd deepseek-web-api
start.cmd           # the same; double-clicking also works
```

Without the script (plain Node or npm):

```bash
npm start           # if you have npm
node src/server.mjs # or directly
```

The server listens on `http://127.0.0.1:8787`. Pass your `chat.deepseek.com` session
token(s) as the API key:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer <token-from-chat.deepseek.com>" \
  -H "content-type: application/json" \
  -d '{"model":"deepseek/deepseek-v4.1-flash","messages":[{"role":"user","content":"Hi"}]}'
```

Environment variables are optional — every one has a working default. Want to change
them? Copy `.env.example` to `.env` and edit.

## API key and token format

The API key **is** a `chat.deepseek.com` session token (or a comma-separated list of
them). There is no separate key in `.env`, so the server holds no secret.

### Where to get a token

1. Log in at <https://chat.deepseek.com>.
2. Open DevTools (**F12**) → **Application** → **Local Storage** →
   `https://chat.deepseek.com`.
3. Find the **`userToken`** entry. Its value is a JSON object:

   ```json
   {
       "value": "<token>",
       "__version": "0"
   }
   ```

4. Right-click the entry → **Copy value**, or copy the `<token>` string from inside
   `"value"` directly.

A token is a full account credential. Treat it like a password; do not commit it or share
it.

### Getting a second token (multiple accounts)

If you list several tokens in one API key, do **not** use the same browser profile and log
out / log in to switch accounts — logging out invalidates the previous token. Instead open
a **separate incognito / private window** for each account and log in there:

```
normal window       → account 1 → userToken 1
incognito window A  → account 2 → userToken 2
incognito window B  → account 3 → userToken 3
```

Each window keeps its own session, so all tokens stay valid at the same time. Then join
them with commas into one API key — see [Format](#format) below.

### Format

```
Authorization: Bearer <token>
```

| Key | Accounts used |
|---|---|
| `tok1` | one |
| `tok1,tok2,tok3` | three |
| `tok1, tok1` | one (duplicates are dropped) |

A session token is Base64 — letters, digits, `+`, `/`, and sometimes `=`. A comma never
occurs inside one, so the split is unambiguous: `tok1,tok2,tok3` is three accounts, not
one malformed key. Whitespace around entries is trimmed. `x-api-key: <tokens>` is
accepted as an alternative to the `Authorization` header.

### Why list several tokens

DeepSeek's web endpoint allows only **one generation per account at a time**. With one
token every request from every client serialises behind that account. With several, the
pool spreads requests across the accounts:

- the **least-used** usable account serves the next request,
- an account is held for the whole HTTP request and released at the end,
- if every account is busy the request queues, and it is aborted when the client goes away,
- an account in a 429 cooldown is skipped — if none are free, the caller gets `429` with
  `retry-after` instead of hanging,
- a token DeepSeek rejects (`AUTH`) is marked dead and dropped from rotation,
- each token is verified once on first use, so an expired entry fails fast on that account
  instead of poisoning every request.

**Rotation on failure.** If a request fails, it is automatically retried on the **next**
account instead of the same unlucky one (`POOL_ATTEMPTS`, default 3). Once every account in
the key has been tried and the whole pass still fails, the last error is returned rather
than looping forever. For streaming, rotation only happens **before** the first SSE event —
after the response head is sent the HTTP status can no longer change, so a later error is
delivered as an SSE error event instead.

Sessions, rate-limit cooldowns and quotas are keyed by the **individual** token digest, so
two accounts listed in one key never collide. A single token with no comma behaves exactly
as before.

Check the state of the accounts in a key:

```bash
curl http://127.0.0.1:8787/health \
  -H "Authorization: Bearer tok1,tok2,tok3"
```

`/health` reports `configured`, `usable`, `dead`, `muted` and one entry per token (8-char
fingerprint, account, request counts, cooldown, session id) — never the token itself.
`?verify=0` skips the live re-check; `POST /tokens/revive` clears the dead flag and
verifies again.

## How it works

### One persistent session

On the DeepSeek side there is **exactly one** conversation. The server does not create a
new session per request and **does not name it** (`update_title` was removed).

Every HTTP request carries the **full history** (like OpenAI), and the server inserts it
as **one message** into that session with `parent_message_id: null`. The history lives on
the client side, not in the web chat — which means:

- context is not duplicated (the web does not attach history when `parent_message_id: null`),
- the DeepSeek panel shows one conversation instead of hundreds of junk sessions,
- restarting the server does not create a new session (state lives in `data/api-sessions.json`).

### Images and files

The web API has no native multimodal input. An image/file is uploaded
(`POST /api/v0/file/upload_file`) and then **referenced** through `ref_file_ids` in the
completion request.

A key detail that is easy to miss: **a freshly uploaded file is in `PENDING` status**, and
the completion rejects it with `biz_code 9 invalid ref file id`. You have to wait until it
becomes `SUCCESS` — the server polls `GET /api/v0/file/fetch_files?file_ids=...` every
~700 ms (typical wait: 2-4 s). `uploadFileReady()` does this, so from the client's point of
view it is transparent.

Supported input formats:

| Form | Example |
|---|---|
| Data URL | `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` |
| http(s) URL | `{"type":"image_url","image_url":{"url":"https://.../photo.jpg"}}` |
| Base64 file | `{"type":"file","file":{"filename":"a.txt","file_data":"<base64>","media_type":"text/plain"}}` |
| File from URL | `{"type":"file","file":{"file_url":"https://.../report.pdf"}}` |

Verified on a live account: an image ("left=red, right=blue") and a text file (the model
read a marker embedded in the document).

### Tools (function calling)

The web API has **no** `tools` field and no native function calling. The server does what
the battle-tested DSH plugin does: it injects a tool-call protocol into the prompt and
stream-parses the model's reply, converting it into OpenAI-format `tool_calls`.

Beyond the happy path, the parser handles:

- streaming — a call may arrive split across several SSE deltas,
- XML/DSML variants, which the model sometimes emits instead of JSON,
- repair of common JSON mistakes (unescaped quotes, missing brackets),
- rejection of "transcript echoes" — the model can replay the history instead of calling a tool.

### Account-limit protection

The server **does not insert artificial pauses** between requests — it sends them
immediately. DeepSeek's limits are handled **reactively**, only when they actually occur:

- **serial** — one generation at a time (the web only allows one per account),
- **429 pause** — when DeepSeek replies 429, the account gets a **2-minute cooldown**.
  During that time the server answers locally with `429` + `retry-after` and **sends
  nothing to DeepSeek** — contacting it resets the throttle window and extends the block.
  The pause is per account (token); state lives in `data/rate-limits.json` and `/health`
  shows `rate_limit: { active, remaining_ms, until }`.

In addition, if the server replies `invalid chat session id` or `message count exceeded`,
the stored session is recreated and the request is retried — the client does not notice.

## Measured context limit

Measured on a live account (2026-09-13), with a fresh session for every measurement — a
document with a unique code at ~50% of its length, a question at the end:

| Script | Passes | Breaks |
|---|---|---|
| Latin | 3 300 000 chars | 3 350 000 chars |
| Chinese | 2 200 000 chars | 2 500 000 chars |
| Mixed | 2 000 000 chars | 2 200 000 chars |

Three different character counts, but converted to tokens they all come to **~800-830k
tokens** — and that is the real ceiling. The model advertises a 1M context; in practice it
fails at ~80% of that. The server-side `input_character_limit = 2621440` is not a hard
boundary either (Latin text passes it comfortably).

Past the limit DeepSeek returns `Length limit reached. Please start a new chat.` — the
server maps this to **HTTP 400 `context_length_exceeded`**, so the client knows it should
shorten the history rather than retry.

`MAX_PROMPT_CHARS` (default 1 000 000) counts **characters**, not tokens. This is safe for
both scripts: 1M characters is ~250k tokens in Latin or ~350k in Chinese — comfortably
below the measured ceiling.

Raw measurement reports: `docs/context-limit-latin.txt`, `docs/context-limit-cjk.txt`.
The measurement is reproducible: `node tests/probe-context6.mjs` (a fresh session per size —
see below why that is necessary).

### Session message-count limit

The web session accumulates messages on its side. After a few dozen turns it starts
rejecting **every** request with `DeepSeek error 3: message count exceeded` — regardless of
prompt size. The server recognises this code, recreates the session and retries the request
(as long as nothing has been sent to the client yet), so the client does not notice.

This was a real bug in the first version: the persistent session returned 500 on everything
after ~40 turns.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | Model list (one entry) |
| POST | `/v1/chat/completions` | Chat; `stream: true` and `false` |
| GET | `/health` | Token-pool state (`?verify=0` skips the live check) |
| GET | `/tokens` | Token-pool state as JSON |
| POST | `/tokens/revive` | Clear the dead flag and verify the key's tokens again |

Aliases without `/v1` (`/models`, `/chat/completions`, `/tokens`) also work.

### Access control, quotas and state

See [API key and token format](#api-key-and-token-format) for how a key maps to accounts.

**Access control** (`ACCESS_ALLOWLIST`, disabled by default) — a list of allowed tokens as
sha256 digests. Without it, any valid token passes. Checked **per token**: a key may list
several accounts and only some of them may be allowed.

**Quotas** (`QUOTA_MAX_REQUESTS`, `QUOTA_WINDOW_MS`, disabled by default) — a per-token
request limit within a time window; exceeding it returns `429` with a `retry-after` header.

**State cleanup** — `data/api-sessions.json` is pruned at startup: entries unused for
longer than `SESSION_TTL_MS` (default 30 days) are removed, and the number of entries is
capped at `SESSION_MAX_ENTRIES` (default 1000).

### Request parameters

| Field | Meaning |
|---|---|
| `messages` | Required. `content` may be a string or a list of parts |
| `stream` | `true` → SSE with `data: ...` and `[DONE]` |
| `tools` / `tool_choice` | The tool protocol described above |
| `reasoning_effort` | `"none"` disables thinking; any other value enables it |
| `thinking` / `reasoning` | `true` enables the `reasoning_content` stream |
| `web_search` | `true` enables search on the DeepSeek side |
| `model` | Ignored except for the echo — there is one model |

`usage` is **estimated** (the web API does not return token counters).

## Tests

```bash
npm run check:live   # token, image and file upload, waiting for SUCCESS, thinking
npm run smoke        # the whole server end-to-end (requires a working token)
```

`npm run smoke` starts the server on port 8899 and calls it like an OpenAI client: models,
non-stream, stream, one session, image, file, tools.

## Structure

| File | Role |
|---|---|
| `src/server.mjs` | HTTP server, routing, OpenAI SSE format |
| `src/openai.mjs` | OpenAI ↔ DeepSeek mapping (prompt, attachments, tools) |
| `src/deepseek.mjs` | Web API client (PoW, sessions, upload, streaming) |
| `src/session.mjs` | Persistent session + state persistence + cleanup (TTL, entry cap) |
| `src/pool.mjs` | Per-request token pool (comma-separated key, account selection, dead-token eviction) |
| `src/access.mjs` | Access control (allowlist) and per-token quotas |
| `src/env.mjs` | Minimal `.env` loader, imported first so every module sees the environment |
| `src/log.mjs` | Structured logs (JSON), request id, IP behind a proxy |
| `src/webapi.ts`, `src/protocol.ts`, `src/auth.ts`, `src/gate.ts` | Battle-tested core (PoW WASM, SSE parser, tool protocol, throttling) |

The `.ts` files are used directly — Node 22.6+ strips types natively. There is no build
step and no `node_modules` dependency.

## Environment variables

The full list with descriptions is in `.env.example`. The most important: `PORT` and
`MAX_PROMPT_CHARS` (default 1 000 000 characters).

The session token is NOT an environment variable — the client passes it as
`Authorization: Bearer <token>` (the API key), optionally several tokens separated by
commas. This way the server stores no secret and every client can use their own DeepSeek
account(s).

## License

**MIT** — see [LICENSE](LICENSE). You may use, modify and host this code, including
commercially; the only condition is to keep the authorship notice.

## Disclaimer

This is an unofficial project, not affiliated with DeepSeek. It uses **private** web
endpoints, which may violate the service's terms of use and result in an account being
limited or blocked. The session token is a full credential — do not commit `.env` and do
not share it.

Protocol-research acknowledgements: [Fly143/deepseek-free-api](https://github.com/Fly143/deepseek-free-api),
[LLM-Red-Team/deepseek-free-api](https://github.com/LLM-Red-Team/deepseek-free-api),
[ForgetMeAI/FreeDeepseekAPI](https://github.com/ForgetMeAI/FreeDeepseekAPI).