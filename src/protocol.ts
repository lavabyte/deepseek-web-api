// SPDX-License-Identifier: MIT
// Copyright (c) 2026 deepseek-web-api contributors
/**
 * Prompt-protocol layer:
 *  1) Serialises the DSH message vocabulary (system / user / assistant / tool-result /
 *     reasoning / tool-call) into a single prompt the web endpoint accepts (the web API
 *     only has a `prompt` string, no `tools` field).
 *  2) Tool-call bridge: the web model has no native function calling, so we use a
 *     "protocol + stream parsing" approach — the instructions tell the model to emit only
 *     {"tool_calls":[{"name":…,"arguments":{…}}]}, and this module does a hold-back scan
 *     over the streamed text: on a match it becomes a tool-call block, otherwise the text
 *     is passed through unchanged.
 */
import { randomUUID } from 'node:crypto'

export interface ToolSchemaLike {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface ToolCallRequest {
  id: string
  name: string
  /** Raw JSON string (DSH ToolCallBlock.arguments semantics). */
  arguments: string
}

/** Output of a single push/flush. */
export interface FilterOutput {
  text: string
  calls: ToolCallRequest[]
  /**
   * The captured protocol block **could not be parsed** (raw = the original text,
   * mode = the marker family).
   * ⚠️ Why this field exists: such a remnant must NEVER be emitted as body text — it is
   * not what the model meant to say, and the Web GUI's markdown renderer turns it into
   * garbage (measured 2026-09: leaked `$ErrorActionPreference='…'` was rendered as a KaTeX
   * inline formula -> the user saw "one character per line + curly quotes"). The caller
   * uses it to decide between "retry" and "show a plain-language hint".
   *
   * `reason` distinguishes failure shapes for diagnostics (measured logs keep only the
   * first 400 chars, so the broken part later in the payload is invisible):
   *  - `unbalanced`: the block is not balanced / not fully received — usually the stream was
   *    cut by the server's 60 s cap, not a model mistake;
   *  - `unparsable`: the block is complete, but malformed (missing bracket, unescaped quote, …);
   *  - `echo`      : the payload wraps a transcript echo (`[Tool Result for …]` etc.) — the
   *    model is **replaying history**, not calling anything. These MUST be dropped: a
   *    8152-char payload with 15 "calls" was captured, all history playback; running it
   *    would re-execute old commands;
   *  - `oversize`  : the capture cap was exceeded, giving up.
   */
  rejected?: { raw: string; mode: 'json' | 'xml'; reason?: 'unbalanced' | 'unparsable' | 'oversize' | 'echo' }
}

/**
 * Per-tool description cap.
 *
 * Raised from 400 to 3200 on 2026-09-12. Reason: DSH actually ships 61 tools, 17 of which
 * have descriptions over 400 chars, and **what got cut was exactly the most important
 * part** — of `pwsh`'s 3010 chars, 2610 explained "a sandbox denial (file access denied)
 * is a policy decision, not a command bug — do not retry a different way", "when a named
 * pipe is unavailable, `stdio:'pipe'` spawn reports EPERM — likewise do not switch
 * approach", "in a read-only sandbox .NET static calls / Add-Type / COM / reflection
 * fail" — i.e. guidance for **what to do on error**; `workflow`'s 2500 chars are the
 * agent() / pipeline() / parallel() hook signatures. Cutting those leaves the model
 * guessing on every error — measured: those two tools were the most frequent failures in
 * rejected.jsonl.
 */
const MAX_DESCRIPTION_CHARS = 3_200

/**
 * Total budget for the tool catalogue (one section).
 *
 * Raised from 24_000 to 56_000 on 2026-09-12. Reason: measured DSH ships 61 tools which
 * need **50,942 chars** without truncating descriptions, while the old budget fit only 35.
 * Worse, **truncation happened alphabetically** (tools are sorted by name), so `write`(w),
 * `web_search`, `web_fetch`, `subagent`, `todo_write`, `skill`, `read_image`, and every
 * `ssh_*`/`sftp_*` were cut, while rarely used `db_tx_rollback`, `db_list_connections`
 * survived. 56_000 leaves about a tenth of headroom (enough for a few more medium tools);
 * beyond that, the "list the names" fallback below kicks in. It will not blow up the
 * context: DeepSeek's web context is 1M, and our maxChars is 120k.
 */
const MAX_TOOLS_SECTION_CHARS = 56_000
const HOLD_BACK_CHARS = 24
const MAX_CAPTURE_CHARS = 1024 * 1024

/**
 * Maximum share of the prompt taken by the head (system + tool protocol + tool catalogue).
 * 0.62 was set in 0.1.33: measured with 61 tools the head is ~63.5k chars, and
 * 0.45 x 120k = 54k does not fit. The transcript still has ~56k chars left — history can
 * be truncated, tool definitions cannot.
 */
const HEAD_RATIO = 0.62
/** Fixed overhead when assembling the protocol section (newlines, `---` separators, omission markers, etc.). */
const PROTOCOL_SLACK_CHARS = 96

export const TOOL_PROTOCOL_INSTRUCTIONS = `# Tool Calling Protocol

You call tools using DeepSeek's native DSML markup. Use this format — it is what you were trained on and what the runner parses.

When you need a tool, output the DSML block as RAW text, with no other text before or after it:

<|DSML|calls>
<|DSML|invoke name="<tool-name>">
<|DSML|parameter name="<argument-name>" string="true">value</|DSML|parameter>
</|DSML|invoke>
</|DSML|calls>

Rules:
1. The tool name goes in the name="..." attribute of <|DSML|invoke>. Each argument is one <|DSML|parameter name="..." string="true">value</|DSML|parameter> child. One invoke per call; a batch (several invoke blocks inside one <|DSML|calls>) is allowed, but prefer exactly one.
2. Emit the markup as RAW text. Do NOT wrap it in a markdown code fence and do NOT escape it into JSON. A fenced block is read as an example to display, not a call to run — it will be shown to the user as text and silently NOT executed.
3. Stop immediately after the closing </|DSML|calls>. The runner executes the call(s) and returns the results as the next message.
4. Never fabricate, guess, or simulate tool output — always wait for the real result.
5. When no tool is needed, answer normally in plain text and do NOT emit any DSML.
6. Values go verbatim inside the parameter element. If a value contains a line break or the sequence ]]>, wrap it in CDATA: <|DSML|parameter name="command"><![CDATA[line1
line2]]></|DSML|parameter>. Prefer keeping commands on ONE line with ; separators so no CDATA is needed.
7. Do NOT use JSON for tool calls, and do NOT use any other XML/HTML tags. The DSML block above is the ONLY accepted format.
8. Always answer in the same language the user writes in (these instructions are English only for precision; the DSML itself is language-neutral).
9. NEVER reproduce the transcript. Do not restate previous turns, "[Tool Result ...]" blocks, tool output, or the current prompt. Emit ONLY the calls you want to run right now.
10. Keep each batch SMALL — at most 3 calls, and prefer exactly 1. If you need more, send them in successive steps.
11. Each call must be able to run on its own: no shared shell variables across calls, no dependence on another call in the same batch.
`

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 3)}...`
}

/** Renders the tool catalogue (with JSON Schema). */
export function buildToolSection(
  tools: readonly ToolSchemaLike[] | undefined,
  maxChars: number = MAX_TOOLS_SECTION_CHARS,
): string {
  if (!tools || tools.length === 0) return ''
  const parts: string[] = ['', '## Available tools']
  // The budget is the smaller of the "built-in cap" and the "allowance given by the caller".
  // The caller (serializePrompt) computes a dynamic allowance from the remaining space — see the F18 comment below.
  let budget = Math.max(0, Math.min(MAX_TOOLS_SECTION_CHARS, maxChars))
  for (let index = 0; index < tools.length; index += 1) {
    const tool = tools[index]
    let schemaText = ''
    try {
      schemaText = JSON.stringify(tool.parameters ?? {})
    } catch {
      schemaText = '{}'
    }
    const block = [
      '',
      `### ${tool.name}`,
      truncate(String(tool.description ?? '').replace(/\s+/g, ' ').trim(), MAX_DESCRIPTION_CHARS),
      `Parameters (JSON Schema): ${schemaText}`,
    ].join('\n')
    if (budget - block.length < 0) {
      // Budget exhausted. **The remaining tool names MUST be listed**: the old form had
      // only "(remaining tools omitted for length)", so the model did not even know which
      // tools exist and had to guess names and parameters — measured: 26 of 61 tools
      // silently disappeared this way. It also explicitly tells the model not to guess
      // parameters, but to ask the user instead.
      const rest = tools
        .slice(index)
        .map((item) => String(item?.name ?? ''))
        .filter(Boolean)
      parts.push(
        `\n(⚠️ The following ${rest.length} tools are NOT described above (omitted for length): ` +
          `${rest.join(', ')}. If you need one of them, ask the user for its exact parameters — ` +
          'do NOT guess them.)',
      )
      break
    }
    budget -= block.length
    parts.push(block)
  }
  return parts.join('\n')
}

function flattenText(blocks: readonly any[] | undefined, out: string[] = []): string[] {
  for (const block of blocks ?? []) {
    if (!block || typeof block !== 'object') continue
    if (block.type === 'text' && typeof block.text === 'string') out.push(block.text)
    else if (block.type === 'tool-result' && Array.isArray(block.content)) flattenText(block.content, out)
  }
  return out
}

function countImages(blocks: readonly any[] | undefined): number {
  let count = 0
  for (const block of blocks ?? []) {
    if (!block || typeof block !== 'object') continue
    if (block.type === 'image') count += 1
    else if (block.type === 'tool-result' && Array.isArray(block.content)) count += countImages(block.content)
  }
  return count
}

/** Collects image-attachment references from the messages in order of appearance (including images embedded in tool-results). */
export function collectImageRefs(messages: readonly any[] | undefined): any[] {
  const refs: any[] = []
  const walk = (blocks: readonly any[] | undefined): void => {
    for (const block of blocks ?? []) {
      if (!block || typeof block !== 'object') continue
      if (block.type === 'image' && block.attachment) refs.push(block.attachment)
      else if (block.type === 'tool-result' && Array.isArray(block.content)) walk(block.content)
    }
  }
  for (const message of messages ?? []) {
    if (!message || typeof message !== 'object') continue
    walk(Array.isArray(message.content) ? message.content : undefined)
  }
  return refs
}

/**
 * Renders tool calls as a DSML block — the markup DeepSeek is natively trained on.
 *
 * Used in the conversation history so the model sees its own past calls in the SAME
 * format it is asked to emit. Imitation is the strongest signal: showing JSON here while
 * demanding DSML in the instructions would fight itself (2026-09-22, user request).
 */
export function renderToolCallsDsml(
  calls: readonly { name?: string; arguments?: unknown }[],
): string {
  const lines: string[] = ['<|DSML|calls>']
  for (const call of calls) {
    const name = String(call?.name ?? '')
    if (!name) continue
    lines.push(`<|DSML|invoke name="${name}">`)
    let args: Record<string, unknown> = {}
    const raw = call?.arguments
    if (raw && typeof raw === 'object') args = raw as Record<string, unknown>
    else if (typeof raw === 'string') {
      try {
        const parsed = JSON.parse(raw)
        args = parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : { _raw: raw }
      } catch {
        args = { _raw: raw }
      }
    }
    for (const [key, value] of Object.entries(args)) {
      const text = typeof value === 'string' ? value : JSON.stringify(value)
      // CDATA whenever the raw value could break the element (newline, `<`, `&`).
      const body = /[\n\r<&]/.test(text) ? `<![CDATA[${text}]]>` : text
      lines.push(`<|DSML|parameter name="${key}" string="true">${body}</|DSML|parameter>`)
    }
    lines.push('</|DSML|invoke>')
  }
  lines.push('</|DSML|calls>')
  return lines.join('\n')
}

/** Renders the tool-call blocks of an assistant message back to DSML (so history teaches the format). */
function renderToolCalls(blocks: readonly any[]): string | null {
  const calls = (blocks ?? []).filter((block) => block?.type === 'tool-call')
  if (calls.length === 0) return null
  return renderToolCallsDsml(
    calls.map((call) => {
      let args: unknown = {}
      try {
        args = call.arguments ? JSON.parse(call.arguments) : {}
      } catch {
        args = { _raw: String(call.arguments ?? '') }
      }
      return { name: String(call.name ?? ''), arguments: args }
    }),
  )
}

/** Middle truncation: keep the start (task/protocol) and the end (latest turn), counting the omission marker against the budget. */
function truncateMiddle(text: string, maxChars: number, tailRatio = 0.7): string {
  if (text.length <= maxChars) return text
  const RESERVE = 64 // marker reservation ("...[N chars omitted]..." is far smaller than this)
  const budget = Math.max(0, maxChars - RESERVE)
  const tail = Math.floor(budget * tailRatio)
  const head = Math.max(0, budget - tail)
  const dropped = text.length - head - tail
  const marker = `\n\n...[${dropped} chars omitted]...\n\n`
  return `${text.slice(0, head)}${marker}${text.slice(text.length - tail)}`
}

export interface SerializeOptions {
  system?: string
  messages: readonly any[]
  tools?: readonly ToolSchemaLike[]
  maxChars?: number
}

/**
 * Serialises into a single web-endpoint prompt.
 * Structure: system -> tool protocol and catalogue -> conversation transcript (User:/Assistant:/[Tool Result]).
 */
export function serializePrompt(options: SerializeOptions): string {
  const maxChars = options.maxChars ?? 120_000
  const system = String(options.system ?? '').trim()
  // ⚠️ F18 (2026-09-12 audit): **compute the tool-catalogue budget from the remaining
  // space**, rather than "render it full, then truncate the head afterwards". The old
  // approach rendered the tool catalogue up to 56k chars, then found the head exceeded
  // `maxChars * ratio`, so `truncateMiddle` cut a chunk out of the **middle** — leaving a
  // **truncated JSON Schema**: the model guesses parameters from half a definition, which
  // is worse than "just do not list this tool"; and it triggers more easily the smaller
  // maxChars is. Now it is reversed: subtract the fixed cost of system + protocol
  // instructions first, and what remains is the allowance for the tool catalogue. If it
  // does not fit, buildToolSection's own fallback kicks in (listing the omitted tool
  // names), so the head is always complete.
  const toolBudget = Math.max(
    0,
    Math.floor(maxChars * HEAD_RATIO) - system.length - TOOL_PROTOCOL_INSTRUCTIONS.length - PROTOCOL_SLACK_CHARS,
  )
  const toolSection = buildToolSection(options.tools, toolBudget)
  const protocol = toolSection ? `\n\n${TOOL_PROTOCOL_INSTRUCTIONS}${toolSection}` : ''

  const lines: string[] = []
  for (const message of options.messages ?? []) {
    if (!message || typeof message !== 'object') continue
    const blocks: any[] = Array.isArray(message.content) ? message.content : []
    if (message.role === 'system') {
      const text = flattenText(blocks).join('')
      if (text.trim()) lines.push(`[System]\n${text}`)
      continue
    }
    if (message.role === 'assistant') {
      const text = flattenText(blocks).join('')
      const renderedCalls = renderToolCalls(blocks)
      if (renderedCalls) lines.push(`Assistant: ${renderedCalls}`)
      else if (text.trim()) lines.push(`Assistant: ${text}`)
      continue
    }
    // user role: may be plain text, or carry tool-result blocks
    const toolResults = blocks.filter((block) => block?.type === 'tool-result')
    const text = flattenText(blocks.filter((block) => block?.type !== 'tool-result')).join('')
    const images = countImages(blocks)
    if (text.trim() || (toolResults.length === 0 && images === 0) || images > 0) {
      // The image itself is uploaded by the caller and attached to the request via
      // ref_file_ids; here we only place a locatable placeholder marker so the model knows
      // which image number belongs to which message (order matches the uploadedImages
      // collection order).
      const imageNote = images > 0 ? `\n${Array.from({ length: images }, () => '[image attached]').join(' ')}` : ''
      lines.push(`User: ${text}${imageNote}`)
    }
    for (const result of toolResults) {
      const body = flattenText(result.content).join('') || '(no output)'
      const errorMark = result.isError ? ' [ERROR]' : ''
      lines.push(`[Tool Result${errorMark} for ${String(result.toolCallId ?? '')}]\n${body}`)
    }
  }

  const transcript = lines.join('\n\n')
  const head = system ? `${system}${protocol}` : protocol.trim()
  const merged = transcript ? `${head}\n\n---\n\n${transcript}` : head

  if (merged.length <= maxChars) return merged
  // Over-long: budget the system+protocol separately, truncate the middle of the transcript.
  // The head (system + tool protocol + tool catalogue) **must stay complete**: if the tool
  // catalogue is cut in the middle, what remains is a truncated JSON Schema — more misleading
  // than "not listing the tool at all" (the model guesses parameters from half a definition).
  // That is why the head ratio was raised from 0.45 to 0.62: measured 2026-09-12 with 61 tools,
  // the head is ~63.5k chars, and 0.45 x 120k = 54k does not fit and would be cut by
  // truncateMiddle. The transcript still has ~56k chars; when exceeded it is middle-truncated as
  // before (history can be cut, tool definitions cannot).
  // With the dynamic budget above, the head should not exceed its share. This is a defensive
  // fallback: if it somehow does, it **does not throw** (that would fail the whole request and
  // the user would get nothing) and middle-truncates as before — but tests should treat this
  // case as a failure.
  const headBudget = Math.min(head.length, Math.floor(maxChars * HEAD_RATIO))
  const boundedHead = head.length <= headBudget ? head : truncateMiddle(head, headBudget, 0.85)
  const transcriptBudget = Math.max(1_000, maxChars - boundedHead.length - 8)
  const boundedTranscript = truncateMiddle(transcript, transcriptBudget, 0.7)
  return `${boundedHead}\n\n---\n\n${boundedTranscript}`
}

// ── Streaming tool-call filter ────────────────────────────

/** Complete JSON call marker: {"tool_calls": or {"tool_call": (whitespace allowed). */
const MARKER_RE = /\{\s*"tool_calls?"\s*:/
/**
 * XML-style call markers (measured: in thinking mode the model occasionally switches to this
 * set, shaped like <tool_calls><invoke name="read"><parameter name="file_path">...); the
 * DSML prefix and the `dsml-` hyphen variant are also accepted.
 *
 * WARNING: a measured leak sample (2026-09-10) — the real source of the garbage — had the
 * model writing the DSML prefix with REPEATED FULLWIDTH PIPES and the wrapper tag name
 * degraded to `calls`. The old form tolerated only a single pipe, so the marker was never
 * recognised, never entered capture state, and reached the body text, where the GUI rendered
 * it as garbage. Pipes are now tolerated with `+` (fullwidth/halfwidth mixed) and `calls` is
 * included in the wrapper tag names.
 */
const DSML_PREFIX = '(?:[|｜]+\\s*DSML\\s*[|｜]+\\s*)?'
const WRAPPER_NAMES = 'tool_calls|tool_call|function_calls|calls'
const XML_STARTER_RE = new RegExp(`<\\s*${DSML_PREFIX}(?:dsml-)?(${WRAPPER_NAMES}|invoke)\\b`, 'i')
/** Code-fence tail (the model often wraps the call block in ```). */
const FENCE_TAIL_RE = /\n?[ \t]*```[a-zA-Z0-9]*[ \t]*\n?$/
const FENCE_HEAD_RE = /^[ \t]*\n?```[ \t]*\n?/

/**
 * Markdown code-context detection.
 *
 * ⚠️ Measured 2026-09-14 (user report): when the model **explains the protocol in its
 * reply**, it writes the call example as markdown code — inline `` `{"tool_calls":…}` ``,
 * an indented code block (4 spaces), or a fenced block with a language tag ```json … ```.
 * The old logic entered capture state merely because "a marker appeared", so these
 * **examples** were treated as real calls: the parser extracted a call from the example,
 * classified the rest of the example as `unbalanced`/`unparsable`, and the whole round was
 * discarded and retried — the user saw "a normal response, yet a tool-call JSON error".
 *
 * A marker inside a code block/inline code is a **reference**, not a call. The tool protocol
 * requires the model to "output only the JSON object, with no other text before or after it"
 * (see TOOL_PROTOCOL_INSTRUCTIONS), so a real call is **never inside a code context**. Hence:
 * if a marker falls inside a code context, it is always emitted as ordinary body text.
 *
 * Detection (only "is the marker start inside a code context" needs to be located):
 *  - inline code: odd number of unescaped backticks before the start on the same line;
 *  - fenced block: odd number of paired fences before the start (counted at line start ``` / ~~~);
 *  - indented block: the start's line begins with 4 spaces or 1 tab (and the line is not empty).
 */
export function isInMarkdownCodeContext(text: string, index: number): boolean {
  if (index <= 0) return false
  const before = text.slice(0, index)
  const lineStart = before.lastIndexOf('\n') + 1
  const linePrefix = before.slice(lineStart)

  // 1) Inline code: odd number of unescaped backticks before the start on this line -> inside `` `…` ``.
  let ticks = 0
  for (let i = 0; i < linePrefix.length; i++) {
    if (linePrefix[i] !== '`') continue
    if (linePrefix[i - 1] === '\\') continue
    ticks += 1
  }
  if (ticks % 2 === 1) return true

  // 2) Indented code block: this line starts with 4 spaces / 1 tab.
  if (/^( {4}|\t)/.test(linePrefix)) return true

  // 3) Fenced code block: an odd number of paired fence toggles in the **complete lines** before the start.
  let open = false
  for (const line of before.split('\n')) {
    if (/^[ \t]*(```|~~~)/.test(line)) open = !open
  }
  if (open) return true

  // 4) DSML-delimited span — JSON markers only. `|DSML|` / `｜DSML｜` is DeepSeek's private
  //    marker notation; when it wraps an example, that example is a quote, not a call.
  //    ⚠️ 2026-09-18 (user report): a `{"tool_calls"` inside `|DSML|…|DSML|` was still
  //    captured as a real call and corrupted the round.
  //
  //    Two deliberate restrictions keep genuine calls capturable:
  //      - only a `{` marker is checked (real calls in DSML use the XML `<|DSML|calls>` form,
  //        and that marker is never treated as "inside a quote" by this rule);
  //      - a delimiter glued to `<` (`<|DSML|calls>`) is NOT counted — only a standalone
  //        `|DSML|` quote marker is, via the negative lookbehind.
  if (text[index] === '{') {
    //    `(?<![<\/])` excludes the DSML marker that is part of a real tag (`<|DSML|calls>`
    //    or `</|DSML|calls>`) — only a standalone quote delimiter counts.
    const dsmlDelimiters = before.match(/(?<![<\/])[|｜]+\s*DSML\s*[|｜]+/gi)
    if (dsmlDelimiters !== null && dsmlDelimiters.length % 2 === 1) return true
  }
  return false
}

/**
 * Markdown code context, tracked incrementally across released chunks.
 *
 * ⚠️ 2026-09-19: the tool filter now streams prose the moment it cannot begin a marker, so
 * by the time a marker arrives the opening backtick / fence has already left `pending`.
 * `isInMarkdownCodeContext` therefore cannot see it any more — the context must be carried
 * forward as state (otherwise a documented example `{"tool_calls":…}` inside a code block
 * is executed as a real call, which was the pre-streaming behaviour).
 */
export interface MarkdownCodeState {
  /** Inside a fenced block (``` / ~~~ at line start). */
  fenceOpen: boolean
  /** Backticks seen on the current line (odd = inside inline code). */
  inlineTicks: number
  /** Start of the current (possibly incomplete) line, capped — used for indentation + fences. */
  lineStart: string
  /** Inside an odd `|DSML|…|DSML|` quote span. */
  dsmlOpen: boolean
}

export function initialMarkdownState(): MarkdownCodeState {
  return { fenceOpen: false, inlineTicks: 0, lineStart: '', dsmlOpen: false }
}

/** Advances the tracked markdown state over text that has been released to the client. */
export function advanceMarkdownState(text: string, state: MarkdownCodeState): MarkdownCodeState {
  let { fenceOpen, inlineTicks, dsmlOpen } = state
  let lineStart = state.lineStart
  let lineFrom = 0
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '\n') {
      if (/^[ \t]*(```|~~~)/.test(lineStart + text.slice(lineFrom, i))) fenceOpen = !fenceOpen
      lineStart = ''
      lineFrom = i + 1
      inlineTicks = 0
    } else if (ch === '`' && text[i - 1] !== '\\') {
      inlineTicks += 1
    }
  }
  lineStart = (lineStart + text.slice(lineFrom)).slice(0, 64)
  // Odd number of standalone DSML delimiters => inside a quoted span.
  const delims = text.match(/(?<![<\/])[|｜]+\s*DSML\s*[|｜]+/gi)
  if (delims && delims.length % 2 === 1) dsmlOpen = !dsmlOpen
  return { fenceOpen, inlineTicks, lineStart, dsmlOpen }
}

/** Is a position inside markdown code context, given the tracked state plus a local prefix? */
export function isInTrackedCodeContext(state: MarkdownCodeState, prefix: string, nextChar: string): boolean {
  const local = advanceMarkdownState(prefix, state)
  if (local.fenceOpen) return true
  if (local.inlineTicks % 2 === 1) return true
  if (/^( {4}|\t)/.test(local.lineStart)) return true
  if (nextChar === '{' && local.dsmlOpen) return true
  return false
}

/**
 * Open/close tag prefixes (lenient form). Strict and lenient parsing **must share the
 * same set**, otherwise you get "findXmlToolCallEnd recognises the close while
 * parseXmlToolCalls does not recognise invoke" -> the whole block degrades into a
 * body-text leak. Covers: `< invoke` (whitespace after the tag name), single/repeated-pipe
 * DSML prefixes (including fullwidth), `<invoke>`.
 */
const TAG_OPEN_PREFIX = `<\\s*${DSML_PREFIX}(?:dsml-)?`
const TAG_CLOSE_PREFIX = `<\\/\\s*${DSML_PREFIX}(?:dsml-)?`
const XML_CLOSE_NAMES = `parameter|invoke|${WRAPPER_NAMES}`

/**
 * Normalises DSML noise into standard tags.
 * Pipes may be **repeated or fullwidth** (the measured sample had double fullwidth pipes),
 * and any following whitespace is consumed too, so the tag name sits directly after `<`
 * (`<` + prefix + ` ` + `invoke` -> `<invoke`).
 */
function normalizeDsml(text: string): string {
  return text
    .replace(new RegExp(`<(/?)${DSML_PREFIX}`, 'gi'), '<$1')
    .replace(/<\s*dsml-/gi, '<')
    .replace(/<\/\s*dsml-/gi, '</')
}

/**
 * JSON call-marker prefix (used for cross-chunk hold-back decisions).
 * ⚠️ 2026-09 incident: real chunking splits the marker into `{"tool` + `_calls":[{"name":…`.
 * The old comparison concatenated an extra quote (`{'{"' + body}`, while body already
 * contains the leading quote -> `{""tool`), so "the tail is a potential prefix" was always
 * false -> half a marker was emitted as body text and could never be reassembled -> the
 * whole JSON leaked into the body. See partialMarkerSuffixLength for the fix.
 */
const JSON_MARKER_STARTERS = ['{"tool_calls"', '{"tool_call"']

/** XML marker prefixes (used for cross-chunk hold-back decisions). `calls` is the degraded wrapper name seen in measurements. */
const XML_MARKER_STARTERS = ['<tool_calls', '<tool_call', '<function_calls', '<calls', '<invoke', '<dsml-tool_calls', '<dsml-invoke']

/**
 * Decides whether the end of `text` is a (possible) marker prefix — i.e. whether to hold back.
 * @returns the number of trailing characters to keep in the buffer (0 = keep nothing)
 */
function partialMarkerSuffixLength(text: string): number {
  const LIMIT = 32
  const from = Math.max(0, text.length - LIMIT)
  const raw = text.slice(from)
  // Candidate start: the last `{` or `<` (located on the **original** slice, so the held length aligns with the source)
  const braceAt = raw.lastIndexOf('{')
  const angleAt = raw.lastIndexOf('<')
  const startAt = Math.max(braceAt, angleAt)
  if (startAt === -1) return 0
  const held = raw.length - startAt
  const normalized = normalizeDsml(raw.slice(startAt))

  if (normalized.startsWith('{')) {
    if (MARKER_RE.test(normalized)) return 0 // already a complete marker, hand it to the capture logic
    const body = normalized.replace(/^\{\s*/, '').replace(/\s+/g, '')
    // body already contains the leading quote (e.g. `"tool`) -> compare against the starter as `{` + body
    const ok = JSON_MARKER_STARTERS.some((starter) => starter.startsWith(`{${body}`))
    return ok ? held : 0
  }
  if (normalized.startsWith('<')) {
    if (XML_STARTER_RE.test(normalized)) return 0 // already a complete marker
    const lower = normalized.toLowerCase().replace(/\s+/g, '')
    if (XML_MARKER_STARTERS.some((starter) => starter.startsWith(lower))) return held
    // ⚠️ A DSML prefix may have arrived **only halfway**: `<|DSML` is still missing the final
    // pipe, so normalizeDsml (which requires "pipe DSML pipe") does not recognise it and the
    // hold check above fails. This was the measured leak path on 2026-09-12: while `pending`
    // sat on half a prefix the hold check returned 0, the half marker was emitted as body text,
    // and the next few characters completed it into `<calls>` — garbage in the visible
    // reply. Fallback: strip the pipes and `dsml` from the candidate and see whether it is a
    // prefix of some starter.
    const loose = lower.replace(/[|｜]|dsml/g, '')
    if (/^<\/?[a-z_]*$/.test(loose) && XML_MARKER_STARTERS.some((starter) => starter.startsWith(loose))) {
      return held
    }
    return 0
  }
  return 0
}

/**
 * Returns the **balanced JSON prefix text starting from the beginning** of the buffer
 * (everything except trailing punctuation).
 *
 * See ToolCallStreamFilter.flush(): when the capture buffer's tail is broken but its first half
 * is itself complete JSON, the first half is salvaged as body text instead of the whole buffer
 * being marked `unbalanced` and discarded. Returning undefined means the buffer has no balanced
 * prefix at all.
 */
export function balancedJsonTextPrefix(text: string): string | undefined {
  const trimmed = text.replace(FENCE_HEAD_RE, '')
  const firstBrace = trimmed.indexOf('{')
  if (firstBrace === -1) return undefined
  const from = trimmed.slice(firstBrace)
  const balanced = extractBalancedJson(from)
  if (!balanced) return undefined
  // prefix = everything before the first `{` + the balanced span itself
  return trimmed.slice(0, firstBrace) + balanced.json
}

/** Extracts a balanced JSON object starting at index 0; returns null when incomplete. */
export function extractBalancedJson(text: string): { json: string; end: number } | null {
  if (text[0] !== '{') return null
  let depth = 0
  let inString = false
  let escape = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (escape) {
      escape = false
      continue
    }
    if (ch === '\\' && inString) {
      escape = true
      continue
    }
    if (ch === '"') {
      inString = !inString
      continue
    }
    if (inString) continue
    if (ch === '{') depth += 1
    else if (ch === '}') {
      depth -= 1
      if (depth === 0) return { json: text.slice(0, i + 1), end: i + 1 }
    }
  }
  return null
}

/** Reads an XML attribute value (supports double quotes / single quotes / bare value). */
function readAttr(attrs: string, name: string): string | undefined {
  const re = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>/]+))`, 'i')
  const match = re.exec(attrs)
  if (!match) return undefined
  return match[1] ?? match[2] ?? match[3]
}

/**
 * Lenient JSON parsing. Measured case: the model writes a Windows path as `"D:\apps\DSH"`
 * (a single backslash, an illegal escape), JSON.parse throws outright -> the tool call fails
 * to parse and the whole block is emitted to the user as body text. First try the text as-is;
 * on failure, repair: turn illegal escapes into literal backslashes, escape raw newlines
 * inside strings, drop trailing commas.
 */
export function parseJsonLenient(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {}
  for (const candidate of jsonRepairCandidates(text)) {
    try {
      const parsed = JSON.parse(candidate)
      if (parsed !== undefined) return parsed
    } catch {}
  }
  return undefined
}

/**
 * Repair candidates, tried in order (only when parsing as-is failed). The order matters:
 *
 * First run the "trailing path backslash" heuristic (in `"…\app.asar\"` the `\"` is an escaped
 * quote -> the string does not terminate; this must be fixed **before** splitting strings, or
 * the whole string range is wrong), then the string-level repair.
 *
 * The smart string-level rule: if a string contains an **illegal escape** (e.g. `\A`), the model
 * wrote backslashes literally without escaping — then every backslash in that string is treated
 * literally; otherwise the `\r` in `\resources` would be read by JSON as a carriage return and
 * the path silently corrupted (measured user sample #2). If a string has no illegal escape,
 * only conservative repair is done (keeping legal escapes such as `\\`, `\"`).
 */
/**
 * Repairs "**an unescaped double quote inside a string value**" — the most frequent breakage
 * measured, and the culprit behind "it just stopped".
 *
 * Measured (2026-09-10 23:27:13, deepseek-reasoner): a command is naturally written
 *   `Get-ChildItem "$env:USERPROFILE\.dsh" | Select-Object Name`
 * The model stuffed the quotes from that string **verbatim** into a JSON string ->
 * `Expected ',' or '}' after property value` -> the whole call was discarded -> that round had no
 * tool call -> the agent loop considered the turn finished normally -> the symptom the user saw
 * was "it stopped mid-sentence".
 *
 * Criterion (stable for JSON syntax): when a double quote is found inside a string, skip
 * whitespace and look at one character — only if it is still `,` `}` `]` (or end of text) does the
 * string really end; otherwise that quote is a literal quote inside the content.
 *
 * ⚠️ A colon must be treated **positionally**: a `"` followed by `:` is structural only in the
 * "key position". Treating a value's `"` + `:` as an end misjudges commands with embedded JSON,
 * e.g. the `"a"` in `node -e "const o={"a":1}"` would be taken as the string's end -> everything
 * after is misaligned -> the call is discarded anyway (I hit exactly this in my first version).
 * So this tracks "were we in key position when the string started" (the previous structural char
 * is `{` / `,` / `[`).
 */
export function escapeInnerQuotes(text: string): string {
  let out = ''
  let inString = false
  let keyPosition = false
  let lastStructural = ''
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (!inString) {
      if (ch === '"') {
        inString = true
        keyPosition = lastStructural === '{' || lastStructural === ',' || lastStructural === '['
        out += ch
        continue
      }
      if (!' \t\n\r'.includes(ch)) lastStructural = ch
      out += ch
      continue
    }
    if (ch === '\\') {
      out += ch + (text[i + 1] ?? '')
      i += 1
      continue
    }
    if (ch === '"') {
      let j = i + 1
      while (j < text.length && ' \t\n\r'.includes(text[j])) j++
      const next = text[j]
      // A colon is structural only in key position; a quote + colon inside a value is content (the norm for embedded JSON)
      const isStructural =
        next === ',' || next === '}' || next === ']' || next === undefined || (next === ':' && keyPosition)
      if (isStructural) {
        inString = false
        lastStructural = next === undefined ? '' : next
        out += ch
      } else {
        out += '\\"'
      }
      continue
    }
    out += ch
  }
  return out
}

/**
 * ⚠️ Deliberately **does not offer** "more aggressive guess" candidates (e.g. treating every
 * `"` followed by `}` as content). Tried it: the result was a disaster — the closing quote of an
 * outer key got escaped too and the whole payload was mangled; and even when parsing luckily
 * succeeded, it could hand over a **corrupted command** and actually run it.
 * Nested quotes (`node -e "console.log({"k":"v"})"`) cannot in principle be disambiguated with
 * a single-character lookahead — the correct handling for such an extreme case is
 * **reject + retry** (after a retry the model usually switches to simpler syntax), not guessing.
 * Better to reject than to hand over a broken command.
 */
export function* jsonRepairCandidates(text: string): Generator<string> {
  const pathTail = (value: string): string => value.replace(/([A-Za-z]:[^"]*?)\\"(?=[,}\]\s])/g, '$1\\\\"')
  // Tried in order: as-is -> escape unescaped quotes -> structural bracket repair; each also
  // runs string-level repair. Every candidate is validated by JSON.parse; the first that
  // succeeds wins.
  for (const base of [text, ...structuralRepairCandidates(text)]) {
    for (const variant of [base, escapeInnerQuotes(base)]) {
      yield repairJsonText(pathTail(variant), { mode: 'smart' })
      yield repairJsonText(variant, { mode: 'smart' })
      yield repairJsonText(pathTail(variant), { mode: 'conservative' })
      yield repairJsonText(variant, { mode: 'conservative' })
    }
  }
}

/**
 * Structural repair candidates: the tool_calls JSON the model writes often has bracket-structure errors
 * (missing closers, array/object closers out of order).
 *
 * Measured shapes covered:
 *  - each call object is missing one `}` (2026-09 incident #4: a batch of 3 calls, one missing each)
 *  - arguments written as an array with `]`/`}` out of order (2026-09-11 incident #5:
 *    `{"tool_calls":[{"name":"pwsh","arguments":[{…}}]}`  <- wrote `}` before closing the args array)
 *  - the outer object is missing its closing `}`
 *
 * Approach (rebuildToolCallJson): stack-guided reordering — when a closer does not match,
 * **insert the missing container closer** to make it match. Only brackets are inserted; string
 * content is never rewritten. Together with parseToolCallJson's "arguments array -> take the
 * single element" unwrapping, such calls can be fully recovered and executed.
 */
export function* structuralRepairCandidates(text: string): Generator<string> {
  const marker = /^\s*\{\s*"tool_calls?"\s*:\s*\[/.exec(text)
  if (!marker) return
  const rebuilt = rebuildToolCallJson(text)
  if (rebuilt && rebuilt !== text) yield rebuilt
}

/**
 * Stack-guided tool_calls JSON reordering (only after strict parsing fails; inserts brackets only,
 * never rewrites string content).
 *
 * Rules:
 *  1) a normal open/close pair matches -> emit as-is and pop the stack;
 *  2) a closer does not match the stack top -> insert the closing sequence that makes it match
 *     (with a cap), then close normally;
 *  3) a `,` appears at the element level of the tool_calls array while the stack top is an
 *     unclosed call object -> insert `}` first (measured shape: every element of a batch call
 *     is missing one `}`);
 *  4) at the end, fill in the remaining closers from the stack.
 *
 * WARNING: safety gate: if the scan ends still inside a string (the typical signature of a
 * stream cut by the server's 60 s cap) -> return null. Filling brackets then would produce a
 * truncated command and actually run it — better to reject (-> retry) than to run half a command.
 */
export function rebuildToolCallJson(text: string): string | null {
  if (!/^\s*\{\s*"tool_calls?"\s*:\s*\[/.test(text)) return null
  let out = ''
  const stack: string[] = []
  let inString = false
  let escape = false
  let insertions = 0
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      out += ch
      if (escape) escape = false
      else if (ch === '\\') escape = true
      else if (ch === '"') inString = false
      continue
    }
    if (ch === '"') {
      inString = true
      out += ch
      continue
    }
    if (ch === '{' || ch === '[') {
      stack.push(ch)
      out += ch
      continue
    }
    if (ch === '}' || ch === ']') {
      const want = ch === '}' ? '{' : '['
      while (stack.length > 0 && stack[stack.length - 1] !== want) {
        if (insertions >= 8) return null
        out += stack[stack.length - 1] === '{' ? '}' : ']'
        stack.pop()
        insertions += 1
      }
      if (stack.length === 0) return null
      stack.pop()
      out += ch
      continue
    }
    if (ch === ',') {
      // `,` at the element level of the tool_calls array (only one call-object layer above the
      // array) with an unclosed stack top -> the model forgot this element's `}`; insert it
      // (measured: every element of a batch call is missing one `}`).
      // More than one layer above the array = the comma is legitimately inside the element; leave it.
      // WARNING: a lookahead is also required: the key-separating comma inside a call object
      // (between "name" and "arguments") is at the same depth here, but what follows is
      // `"arguments"` rather than `{"name"` — without the lookahead every call object breaks.
      const bracketIndex = stack.indexOf('[')
      if (
        bracketIndex === 1 &&
        stack.length - bracketIndex - 1 === 1 &&
        stack[stack.length - 1] === '{' &&
        /^\s*\{\s*"name"\s*:/.test(text.slice(i + 1))
      ) {
        out += '}'
        stack.pop()
        insertions += 1
      }
      out += ch
      continue
    }
    out += ch
  }
  if (inString) return null // safety gate (see above)
  if (insertions > 8) return null
  while (stack.length > 0) {
    out += stack[stack.length - 1] === '{' ? '}' : ']'
    stack.pop()
  }
  return out
}

/**
 * Repairs common JSON syntax problems.
 * @param options.mode - `smart` (default): when an illegal escape appears inside a string,
 *   treat every backslash in that string literally (the norm when the model writes paths
 *   verbatim; avoids `\r`/`\n`/`\t` being taken as escapes);
 *   `conservative`: only fill in illegal escapes, keep the rest as-is.
 */
export function repairJsonText(text: string, options: { mode?: 'smart' | 'conservative' } = {}): string {
  const mode = options.mode ?? 'smart'
  let out = ''
  let inString = false
  let buf = ''
  const flushString = (): void => {
    const raw = buf
    // smart: the whole string is verbatim backslashes -> all literal; otherwise only fill illegal escapes
    const body =
      mode === 'smart' && hasInvalidEscape(raw)
        ? literalizeBackslashes(raw) // treat the whole string literally (the norm for verbatim paths)
        : escapeInvalidEscapes(raw)
    out += `"${body}"`
    buf = ''
  }
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (!inString) {
      if (ch === '"') {
        inString = true
        buf = ''
        continue
      }
      out += ch
      continue
    }
    if (ch === '\\') {
      const next = text[i + 1]
      if (next === undefined) {
        buf += '\\\\'
        continue
      }
      buf += ch + next
      i += 1
      continue
    }
    if (ch === '"') {
      flushString()
      inString = false
      continue
    }
    if (ch === '\n') {
      buf += '\\n'
      continue
    }
    if (ch === '\r') {
      buf += '\\r'
      continue
    }
    if (ch === '\t') {
      buf += '\\t'
      continue
    }
    buf += ch
  }
  if (inString) flushString()
  // drop trailing commas at the end of objects/arrays
  return out.replace(/,(\s*[}\]])/g, '$1')
}

/** Is there an illegal escape in the string (used to tell whether unescaped backslashes were written verbatim). */
function hasInvalidEscape(body: string): boolean {
  for (let i = 0; i < body.length; i++) {
    if (body[i] !== '\\') continue
    const next = body[i + 1]
    if (next === undefined) return true
    if (!'"\\/bfnrtu'.includes(next)) return true
    i += 1
  }
  return false
}

/**
 * Re-escapes a string the model wrote verbatim, using literal semantics.
 * Processed character by character to avoid regex double-escaping: `\\` stays one literal
 * backslash, `\"` stays an escaped quote, and every other single backslash becomes `\\`.
 */
function literalizeBackslashes(raw: string): string {
  let out = ''
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]
    if (ch !== '\\') {
      out += ch
      continue
    }
    const next = raw[i + 1]
    if (next === '\\') {
      out += '\\\\'
      i += 1
      continue
    }
    if (next === '"') {
      out += '\\"'
      i += 1
      continue
    }
    out += '\\\\'
  }
  return out
}

/** Fills only illegal escapes in as literal backslashes; legal escapes are kept as-is. */
function escapeInvalidEscapes(body: string): string {
  let out = ''
  for (let i = 0; i < body.length; i++) {
    const ch = body[i]
    if (ch !== '\\') {
      out += ch
      continue
    }
    const next = body[i + 1]
    if (next === undefined) {
      out += '\\\\'
      continue
    }
    if ('"\\/bfnrtu'.includes(next)) {
      out += ch + next
      i += 1
      continue
    }
    out += '\\\\'
  }
  return out
}

/** Strips the CDATA wrapper and parses the value as JSON (falls back to a string when parsing fails). */
function parseParameterValue(raw: string): unknown {
  let text = raw.trim()
  const cdata = /^<!\[CDATA\[([\s\S]*?)\]\]>$/.exec(text)
  if (cdata) text = cdata[1]
  if (text === '') return ''
  const parsed = parseJsonLenient(text)
  return parsed === undefined ? text : parsed
}

/**
 * Parses an XML/DSML-style tool-call block (whole text, possibly containing several invokes).
 * Supports: `tool_calls`/`function_calls` wrappers, a bare `invoke`, the DSML prefix,
 * CDATA values, attributes in any order, and fence wrapping.
 */
export function parseXmlToolCalls(block: string): ToolCallRequest[] | null {
  const text = normalizeDsml(block).replace(FENCE_HEAD_RE, '').replace(/```\s*$/, '')
  const invokeRe = new RegExp(`${TAG_OPEN_PREFIX}invoke\\b([^>]*)>([\\s\\S]*?)${TAG_CLOSE_PREFIX}invoke\\s*>`, 'gi')
  const calls: ToolCallRequest[] = []
  let invoke: RegExpExecArray | null
  while ((invoke = invokeRe.exec(text)) !== null) {
    const name = readAttr(invoke[1], 'name')
    if (!name) continue
    const body = invoke[2]
    const args: Record<string, unknown> = {}
    let sawParam = false
    const paramRe = new RegExp(`${TAG_OPEN_PREFIX}parameter\\b([^>]*)>([\\s\\S]*?)${TAG_CLOSE_PREFIX}parameter\\s*>`, 'gi')
    let param: RegExpExecArray | null
    while ((param = paramRe.exec(body)) !== null) {
      const key = readAttr(param[1], 'name')
      if (!key) continue
      sawParam = true
      args[key] = parseParameterValue(param[2])
    }
    if (!sawParam) {
      // no parameter child elements: try the body as JSON arguments, otherwise keep it as _raw
      const inner = body.trim()
      if (inner) {
        try {
          const parsed = JSON.parse(inner)
          if (parsed && typeof parsed === 'object') Object.assign(args, parsed as Record<string, unknown>)
          else args._raw = parsed
        } catch {
          args._raw = inner
        }
      }
    }
    calls.push({
      id: `call_${randomUUID().replace(/-/g, '').slice(0, 20)}`,
      name,
      arguments: JSON.stringify(args),
    })
  }
  if (calls.length > 0) return calls
  return salvageXmlToolCalls(text)
}

/**
 * Lenient salvage: the last net for when the XML call block is incompletely terminated.
 *
 * Measured leak sample (2026-09 — exactly the source of the one-character-per-line garbage):
 *   `<tool_calls><invoke name="pwsh"><parameter name="command">…</parameter>`
 * — the parameter values are complete, but the inner invoke closer is missing (common when the
 * stream is cut by the server cap). Strict parsing then does not recognise the invoke (its
 * regex requires the closer), so the whole block is emitted to the user as body text, and the
 * Web GUI renders the `$...$` inside the command as KaTeX — the user sees garbage. (Note: a
 * missing OUTER tool_calls closer is handled by strict parsing already; it is not a leak source.)
 *
 * Approach: rely on no closing tags at all; split into segments by the invoke open tag -> next
 * open tag or block end and take the values. WARNING: a fallback only when strict parsing failed
 * completely, so it never preempts the normal path. Better a possible truncation than a leak —
 * a truncated call is corrected by the model itself on the next round.
 */
function salvageXmlToolCalls(text: string): ToolCallRequest[] | null {
  const invokeStartRe = new RegExp(`${TAG_OPEN_PREFIX}invoke\\b([^>]*)>`, 'gi')
  const starts: { index: number; attrs: string }[] = []
  let match: RegExpExecArray | null
  while ((match = invokeStartRe.exec(text)) !== null) starts.push({ index: match.index, attrs: match[1] })
  if (starts.length === 0) return null

  const calls: ToolCallRequest[] = []
  for (let i = 0; i < starts.length; i++) {
    const name = readAttr(starts[i].attrs, 'name')
    if (!name) continue
    const bodyStart = starts[i].index + starts[i].attrs.length
    // Segment = up to the next invoke open tag; we cannot split on closing tags, because they may be missing entirely.
    const nextStart = starts[i + 1]?.index ?? text.length
    const body = text.slice(bodyStart, nextStart)
    calls.push({
      id: `call_${randomUUID().replace(/-/g, '').slice(0, 20)}`,
      name,
      arguments: JSON.stringify(salvageXmlParameters(body)),
    })
  }
  return calls.length > 0 ? calls : null
}

/** Extracts parameters from a broken invoke body: split on open tags, take each value up to the next open tag or segment end. */
function salvageXmlParameters(body: string): Record<string, unknown> {
  const args: Record<string, unknown> = {}
  const paramStartRe = new RegExp(`${TAG_OPEN_PREFIX}parameter\\b([^>]*)>`, 'gi')
  const found: { start: number; end: number; key: string }[] = []
  let match: RegExpExecArray | null
  while ((match = paramStartRe.exec(body)) !== null) {
    const key = readAttr(match[1], 'name')
    if (key) found.push({ start: match.index, end: paramStartRe.lastIndex, key })
  }
  for (let i = 0; i < found.length; i++) {
    // value = after this parameter's open tag -> before the next parameter open tag (or segment end)
    // Split on open-tag positions rather than on the matching closer: the closing tags may be missing entirely.
    const valueEnd = found[i + 1] ? found[i + 1].start : body.length
    // leftover closing tags for parameter/invoke/tool_calls at the end of a value are stripped
    args[found[i].key] = parseParameterValue(stripXmlClosers(body.slice(found[i].end, valueEnd)))
  }
  if (found.length === 0) {
    const inner = stripXmlClosers(body).trim()
    if (inner) {
      const parsed = parseJsonLenient(inner)
      if (parsed && typeof parsed === 'object') Object.assign(args, parsed as Record<string, unknown>)
      else args._raw = inner
    }
  }
  return args
}

/** Strips leftover closing tags and whitespace from the end of a value. */
function stripXmlClosers(value: string): string {
  const re = new RegExp(`(?:\\s*${TAG_CLOSE_PREFIX}(?:${XML_CLOSE_NAMES})\\s*>)+\\s*$`, 'i')
  return value.replace(re, '')
}

/**
 * Decides whether a captured protocol block is **really an attempt to call a tool** (rather
 * than body text that merely mentions words like `<invoke>`).
 * Only used to decide "discard or emit" on parse failure:
 *  - looks like a call -> discard + warn (never leak as garbage; hand it to the upper layer to retry)
 *  - does not look like a call -> emit as ordinary body text (never swallow the model's reply)
 */
function looksLikeToolCallBlock(mode: 'json' | 'xml', raw: string): boolean {
  if (mode === 'json') return MARKER_RE.test(raw)
  const text = normalizeDsml(raw)
  // An invoke/parameter open tag with a name attribute = a genuine call attempt.
  // Note: the system prompt, when explaining the protocol, writes only `<invoke>` /
  // `<parameter>` (no name=), so it is never misjudged.
  return (
    new RegExp(`${TAG_OPEN_PREFIX}invoke\\b[^>]*\\bname\\s*=`, 'i').test(text) ||
    new RegExp(`${TAG_OPEN_PREFIX}parameter\\b[^>]*\\bname\\s*=`, 'i').test(text)
  )
}

/**
 * Classifies the failure shape for diagnostics — logs keep only the first 400 chars, so the
 * broken part later in the payload is invisible; "not fully received" and "fully received but
 * structurally wrong" MUST be told apart, otherwise you are forever guessing.
 *
 *  - `unbalanced`: the block is not balanced / not fully received — usually the stream was cut
 *    by the server's 60 s cap, not a model mistake;
 *  - `unparsable`: the block is complete, but malformed (missing bracket, unescaped quote, wrong shape);
 *  - `echo`      : the payload wraps a transcript echo (the model is replaying history, not calling).
 */
function classifyFailure(mode: 'json' | 'xml', raw: string): 'unbalanced' | 'unparsable' | 'echo' {
  // Echo is checked first: a captured 8152-char payload had 15 "calls" whose command strings
  // wrapped `[Tool Result for call_…]` / `[Truncated]` — that is history playback; running it
  // would re-execute old commands.
  if (/\[\s*Tool Result\b/i.test(raw)) return 'echo'
  if (mode === 'json') return extractBalancedJson(raw.replace(FENCE_HEAD_RE, '')) ? 'unparsable' : 'unbalanced'
  return findXmlToolCallEnd(raw) === -1 ? 'unbalanced' : 'unparsable'
}

/** Turns parsed JSON into tool-call requests; returns null for a non-protocol shape. */
export function parseToolCallJson(json: string): ToolCallRequest[] | null {
  const parsed: any = parseJsonLenient(json)
  if (!parsed || typeof parsed !== 'object') return null
  const raw = Array.isArray(parsed.tool_calls)
    ? parsed.tool_calls
    : parsed.tool_call && typeof parsed.tool_call === 'object'
      ? [parsed.tool_call]
      : null
  if (!raw) return null
  const calls: ToolCallRequest[] = []
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const name = typeof entry.name === 'string' ? entry.name : typeof entry.tool === 'string' ? entry.tool : ''
    if (!name) continue
    let args = entry.arguments ?? entry.parameters ?? entry.args ?? {}
    // The model occasionally writes arguments as an **array** (measured 2026-09-11:
    // arguments:[{…}], while the spec is an object). When there is exactly one object element,
    // take it — otherwise the arguments are serialized as "[{…}]" and the tool gets garbage.
    if (Array.isArray(args) && args.length === 1 && args[0] && typeof args[0] === 'object' && !Array.isArray(args[0])) {
      args = args[0]
    }
    if (typeof args === 'string') {
      // already a string: parse it and pass it through as-is (DSH arguments semantics are a raw JSON string), otherwise wrap it
      const reparsed = parseJsonLenient(args)
      if (reparsed === undefined) args = JSON.stringify({ _raw: args })
    } else {
      try {
        args = JSON.stringify(args ?? {})
      } catch {
        args = '{}'
      }
    }
    calls.push({ id: `call_${randomUUID().replace(/-/g, '').slice(0, 20)}`, name, arguments: String(args) })
  }
  return calls.length > 0 ? calls : null
}

/**
 * End-of-stream JSON salvage (one step more forgiving than parseToolCallJson).
 *
 * The capture buffer may carry leftover characters after the capture (fence, body text), in
 * which case parsing the whole thing necessarily fails — but the balanced prefix itself is a
 * good call: take the prefix and parse it; do not throw away a whole batch of recoverable calls.
 * Note: an unbalanced truncation is still rejected by structuralRepairCandidates' safety gate.
 */
function parseSalvagedToolCallJson(buffer: string): ToolCallRequest[] | null {
  const text = buffer.replace(FENCE_HEAD_RE, '')
  const direct = parseToolCallJson(text)
  if (direct) return direct
  const balanced = extractBalancedJson(text)
  if (balanced && balanced.end < text.length) return parseToolCallJson(balanced.json)
  return null
}

/**
 * Finds the end position of an XML call block in the capture buffer (including the end tag).
 * - wrapper form (`tool_calls` / `function_calls`): find the matching closing tag
 * - bare `invoke`: after finding the closer, keep absorbing the invoke blocks that follow (same batch)
 * Returns -1 when not fully received yet (keep waiting on the stream).
 */
export function findXmlToolCallEnd(buffer: string): number {
  const text = buffer
  const wrapper = new RegExp(`<\\s*${DSML_PREFIX}(?:dsml-)?(${WRAPPER_NAMES})\\b`, 'i').exec(text)
  const startsWithWrapper = wrapper !== null && wrapper.index === 0
  const isInvokeStart = (value: string): boolean =>
    new RegExp(`^\\s*<\\s*${DSML_PREFIX}(?:dsml-)?invoke\\b`, 'i').test(value)

  if (startsWithWrapper) {
    const tag = wrapper![1].toLowerCase()
    const closeRe = new RegExp(`<\\/\\s*${DSML_PREFIX}(?:dsml-)?${tag}\\s*>`, 'i')
    const match = closeRe.exec(text)
    return match ? match.index + match[0].length : -1
  }
  if (!isInvokeStart(text)) {
    // may be an XML prefix other than the JSON marker (e.g. only a partial invoke received so far)
    return -1
  }
  let cursor = 0
  for (;;) {
    const slice = text.slice(cursor)
    if (!isInvokeStart(slice)) return cursor > 0 ? cursor : -1
    const closeRe = /<\/\s*(?:\|\s*DSML\s*\|\s*)?(?:dsml-)?invoke\s*>/i
    const match = closeRe.exec(slice)
    if (!match) return -1
    cursor += match.index + match[0].length
    const rest = text.slice(cursor)
    // if the next non-whitespace content is a new invoke, keep absorbing it
    if (isInvokeStart(rest)) continue
    // WARNING: measured leak (2026-09-12): the model often DROPS the wrapper open tag and leaves
    // only the closing tag, so the body text ends up with a stray DSML calls closer (sometimes
    // degraded to a bare `calls` closer). An orphaned closer belongs to this batch and must not
    // reach the screen — once fully received, absorb it too. Reproduced with consecutive bare
    // DSML invoke tags followed by a DSML calls closer; the old code emitted it as body text.
    const strayClose = new RegExp(
      `^\\s*<\\/\\s*${DSML_PREFIX}(?:dsml-)?(?:${WRAPPER_NAMES})\\s*>`,
      'i',
    )
    const stray = strayClose.exec(rest)
    if (stray) return cursor + stray[0].length
    // fragments that look like they are still being received (`<`, `</`, a partial DSML prefix,
    // a partial `calls` closer ...) -> keep waiting; do not emit as body text yet. If it truly
    // never completes, flush() goes through parseXmlToolCalls and nothing leaks.
    const tail = rest.trim()
    if (tail.startsWith('<') && /^<\/?\s*[|｜]?\s*[A-Za-z]{0,14}$/.test(tail)) return -1
    return cursor
  }
}

/**
 * Strips stray DSML tool-call fragments.
 *
 * ⚠️ 2026-09-18 (user request): this used to also remove bare HTML-ish tags
 * (`</tool_calls>`, `</invoke>`, `</parameter>`, a lone `voke>` …). Those rules fired on
 * ordinary prose — a sentence, a table cell, a code sample — and cut the reply short.
 * DSML (`|DSML|` / `｜DSML｜` / `dsml-`) is DeepSeek's private notation that cannot appear
 * in a normal answer, so it is the ONLY form removed here. A real call block is consumed
 * by `ToolCallStreamFilter`; a bare tag that leaks is far less damaging than silently
 * truncating the answer.
 */
export function stripStrayToolMarkup(text: string): string {
  if (!text) return text
  if (!/DSML/i.test(text)) return text
  return text.replace(
    // Open or close, with the `|DSML|` / `｜DSML｜` prefix or the `dsml-` hyphen variant.
    new RegExp(
      `<\\/?\\s*(?:(?:[|｜]+\\s*DSML\\s*[|｜]+\\s*)|dsml-)(?:dsml-)?(?:${WRAPPER_NAMES}|invoke|parameter)\\s*>`,
      'gi',
    ),
    '',
  )
}

/**
 * Streaming tool-call filter.
 * - ordinary body text: passed through immediately (only a small tail is held back to watch for
 *   a call marker split across chunks)
 * - on a call marker (JSON or XML): enter capture state; once complete, turn it into a tool-call
 *   request and never leak the marker itself
 * - parse failure: emit the captured content as ordinary body text (degraded but visible; never
 *   silently drop content)
 * - text after the call is handled as ordinary body text again (including fence-tail cleanup)
 */
export class ToolCallStreamFilter {
  private pending = ''
  private capture: { mode: 'json' | 'xml'; buffer: string } | null = null
  private abandoned: { raw: string; mode: 'json' | 'xml' } | null = null
  /** Markdown code context carried across released chunks (see `advanceMarkdownState`). */
  private markdown: MarkdownCodeState = initialMarkdownState()
  /**
   * Whether this round has already emitted at least one **valid** tool_call.
   *
   * WARNING: user-measured 2026-09-14: the model emits one complete, correct call, then starts a
   * second one and the stream is cut in the middle of it. The old logic classified that tail as
   * `unbalanced` -> `rejected`, and the upper layer RE-RAN THE WHOLE ROUND — while the first call
   * had already been handed to the client and will be executed -> the same command runs twice.
   *
   * Correct semantics: once at least one valid call has been delivered, the round is a success;
   * a trailing fragment is extra model output (drop it) and must never trigger a whole-round retry.
   */
  private emittedCall = false
  private readonly knownTools?: ReadonlySet<string>

  constructor(knownTools?: ReadonlySet<string>) {
    this.knownTools = knownTools
  }

  /** Appends released text and advances the tracked markdown code context. */
  private emit(out: FilterOutput, text: string): void {
    if (!text) return
    out.text += text
    this.markdown = advanceMarkdownState(text, this.markdown)
  }

  /**
   * Is the tool marker at `index` inside a markdown code context?
   *
   * The filter releases prose immediately, so the opening backtick/fence is usually already
   * gone from `pending` — the context is reconstructed from the incrementally tracked state.
   */
  private inCodeContext(index: number): boolean {
    return isInTrackedCodeContext(this.markdown, this.pending.slice(0, index), this.pending[index] ?? '')
  }

  push(text: string): FilterOutput {
    const out: FilterOutput = { text: '', calls: [] }
    if (text) {
      if (this.capture) this.capture.buffer += text
      else this.pending += text
    }
    this.drain(out)
    return out
  }

  flush(): FilterOutput {
    const out: FilterOutput = { text: '', calls: [] }
    if (this.capture) {
      // not fully received when the stream ends: first try lenient parsing (escape repair +
      // structural bracket filling + missing-closer salvage).
      const captured = this.capture
      const calls =
        captured.mode === 'xml' ? parseXmlToolCalls(captured.buffer) : parseSalvagedToolCallJson(captured.buffer)
      if (calls) {
        this.emittedCall = true
        out.calls.push(...calls)
      }
      // WARNING: user-measured: the model often emits a COMPLETE body of text with a broken
      // protocol fragment at the end (e.g. a stray `{` after the answer, or a truncated second
      // call). The old logic classified the whole buffer as `unbalanced` -> the round was
      // discarded and retried, even though the model's reply was usually valid.
      // The handling now has two steps:
      //   a) emit the already-balanced PREFIX of the buffer as body text (that is the model's
      //      real answer);
      //   b) mark it rejected only if the remaining tail really looks like a call attempt.
      // This way a broken protocol fragment never pollutes the body, and the body is never lost.
      else if (!this.emittedCall && looksLikeToolCallBlock(captured.mode, captured.buffer)) {
        const salvaged = captured.mode === 'json' ? balancedJsonTextPrefix(captured.buffer) : undefined
        if (salvaged) out.text += stripStrayToolMarkup(salvaged)
        const remainder = salvaged ? captured.buffer.slice(salvaged.length) : captured.buffer
        // the remainder is only whitespace/punctuation -> not a call attempt at all; do not
        // report rejected (that would spuriously trigger a retry).
        if (remainder.trim()) {
          this.abandoned ??= { raw: remainder, mode: captured.mode, reason: classifyFailure(captured.mode, remainder) }
        }
      }
      // a valid call was already delivered -> the tail fragment is just extra model output;
      // drop it without error (otherwise the command would run twice).
      else if (!this.emittedCall) out.text += stripStrayToolMarkup(captured.buffer)
      // not a call (the body merely mentions words like `invoke`) -> emit as usual,
      // WARNING: but strip fragments first: a degraded block of a DSML calls wrapper plus a
      // closer was measured leaking into the body from here (2026-09-12).
      else out.text += stripStrayToolMarkup(captured.buffer)
      this.capture = null
    }
    // end of stream: strip orphan fragments before flushing the held tail -- they escape into
    // the body right here
    out.text += stripStrayToolMarkup(this.pending)
    this.pending = ''
    if (this.abandoned) out.rejected = this.abandoned
    return out
  }

  private drain(out: FilterOutput): void {
    for (;;) {
      if (this.capture) {
        const captured = this.capture
        if (captured.mode === 'xml') {
          const end = findXmlToolCallEnd(captured.buffer)
          if (end === -1) {
            if (captured.buffer.length > MAX_CAPTURE_CHARS) {
              // oversize and still not complete: drop it if it is a call (never emit garbage);
              // if the body merely mentions it, emit as usual
              if (!this.emittedCall && looksLikeToolCallBlock('xml', captured.buffer))
                this.abandoned ??= { raw: captured.buffer, mode: 'xml', reason: 'oversize' }
              else if (!this.emittedCall) out.text += stripStrayToolMarkup(captured.buffer)
              this.capture = null
              continue
            }
            return
          }
          const block = captured.buffer.slice(0, end)
          const calls = parseXmlToolCalls(block)
          if (calls) {
            this.emittedCall = true
            out.calls.push(...calls)
          } else if (!this.emittedCall && looksLikeToolCallBlock('xml', block)) this.abandoned ??= { raw: block, mode: 'xml', reason: 'unparsable' }
          else if (!this.emittedCall) out.text += stripStrayToolMarkup(block)
          this.capture = null
          this.pending = captured.buffer.slice(end).replace(FENCE_HEAD_RE, '') + this.pending
          continue
        }
        const balanced = extractBalancedJson(captured.buffer)
        if (!balanced) {
          if (captured.buffer.length > MAX_CAPTURE_CHARS) {
            // over the cap and still unbalanced: give up, but do NOT emit as body text (that is
            // garbage, not an answer). A valid call was already delivered -> do not report
            // rejected (a whole-round retry would duplicate execution).
            if (!this.emittedCall) this.abandoned ??= { raw: captured.buffer, mode: 'json', reason: 'oversize' }
            this.capture = null
            continue
          }
          return
        }
        const calls = parseToolCallJson(balanced.json)
        if (calls) {
          // unknown tool names are emitted as usual: the runner produces an 'unknown tool'
          // result and the model can correct itself.
          this.emittedCall = true
          out.calls.push(...calls)
          this.capture = null
          this.pending = captured.buffer.slice(balanced.end).replace(FENCE_HEAD_RE, '') + this.pending
          continue
        }
        // wrong shape: reaching capture state means MARKER_RE matched (`tool_calls` JSON), so
        // this is a BROKEN CALL rather than body text -> drop + warn (leaking it into the body
        // is the real source of garbage). But if a valid call was already delivered this round,
        // the fragment is just extra output -> drop without error (avoid duplicate execution).
        const head = captured.buffer.slice(0, balanced.end)
        if (!this.emittedCall && looksLikeToolCallBlock('json', head)) this.abandoned ??= { raw: head, mode: 'json', reason: 'unparsable' }
        else if (!this.emittedCall) out.text += head
        this.capture = null
        this.pending = captured.buffer.slice(balanced.end) + this.pending
        continue
      }

      // Find the earliest complete marker (JSON or XML). Search repeatedly: the body may first
      // contain an EXAMPLE marker inside a code block (which must be emitted as body text), and
      // the real call marker only after it -- taking only the first would mistake the example
      // for a call.
      let jsonMarker = MARKER_RE.exec(this.pending)
      let xmlMarker = XML_STARTER_RE.exec(this.pending)
      for (;;) {
        const jsonIndex = jsonMarker?.index ?? -1
        const xmlIndex = xmlMarker?.index ?? -1
        const useXml = xmlIndex !== -1 && (jsonIndex === -1 || xmlIndex < jsonIndex)
        const index = useXml ? xmlIndex : jsonIndex
        if (index === -1) break
        // the marker falls inside a markdown code context (inline code / indented block / fence)
        // -> it is an EXAMPLE, not a call: emit it with its context as body text, then keep
        // searching after it for the real call marker.
        if (this.inCodeContext(index)) {
          const skipTo = index + 1
          this.emit(out, this.pending.slice(0, skipTo))
          this.pending = this.pending.slice(skipTo)
          jsonMarker = MARKER_RE.exec(this.pending)
          xmlMarker = XML_STARTER_RE.exec(this.pending)
          continue
        }
        let head = this.pending.slice(0, index)
        const fence = FENCE_TAIL_RE.exec(head)
        if (fence) head = head.slice(0, fence.index)
        this.emit(out, head)
        this.capture = { mode: useXml ? 'xml' : 'json', buffer: this.pending.slice(index) }
        this.pending = ''
        break
      }
      if (this.capture) continue
      // no real call marker (possibly only a code-block example) -> take the ordinary body path below.
      //
      // ⚠️ 2026-09-19 (user report: streaming felt far slower than the web UI): hold back
      // ONLY a possible partial-marker suffix, never the whole buffer. The old early return
      // `pending.length <= HOLD_BACK_CHARS` added a fixed ~24-char latency to every reply:
      // until 24 chars accumulated, NOTHING was released. When the tail cannot begin a
      // marker, it is prose and must stream immediately.
      const hold = partialMarkerSuffixLength(this.pending)
      if (hold > 0) {
        this.emit(out, this.pending.slice(0, this.pending.length - hold))
        this.pending = this.pending.slice(this.pending.length - hold)
        return
      }
      this.emit(out, stripStrayToolMarkup(this.pending))
      this.pending = ''
      return
    }
  }
}

// -- Transcript echo guard ---------------------------------

/**
 * Strips the 'system markers' the model imitates (the ds_system / system forms with an
 * environment-details shape).
 *
 * Measured (deepseek-web): the model emits strings of fake system markers in its body text,
 * IMITATING the protocol format it has seen. This is the same class of problem as the
 * transcript echo, but the shape is an XML tag rather than a `[Tool Result]` line, so it is
 * handled in its own layer. Not stripped inside fenced code blocks (a normal answer may
 * discuss these markers).
 *
 * ## Tag list (only ones with FIELD EVIDENCE; no wildcards)
 *
 * WARNING: deliberately NO wildcard like `<[a-z_]+>`: a user's normal answer may be a document
 * discussing these tags, and a wildcard would eat them too. Every name added needs field
 * evidence plus an exhaustive search.
 *
 * | Tag | Evidence |
 * | --- | --- |
 * | `ds_system` | 2026-09-11: 13 occurrences in one body text, with call ids fabricated by
 *   incrementing letters (1a2b3c -> 4d5e6f -> 7a8b9c ...) |
 * | `system` | same batch of evidence |
 * | `ide_result_status` | 2026-09-12 (session `15ac4c56`): the marker appeared in the body
 *   text. WARNING: the string is NOT found in DSH's `app.asar` (0 hits), in any installed
 *   plugin (0 hits), or anywhere under `~/.dsh` (0 hits), and in the session log it appears
 *   ONLY in the model's output field (none in 212 tool/result entries, user messages or
 *   system messages) -> judged to be FABRICATED BY THE MODEL, not provided by DSH |
 * | `budget:token_budget` | 2026-09-16 (user-measured): the marker appeared at the end of
 *   the body text. The model is imitating some agent scaffold's budget marker -- likewise it
 *   appears only in model output |
 *
 * WARNING: `budget:token_budget` contains a namespace colon: in the regex the colon is a
 * literal and `\b` falls between the trailing `t` and `>`, so both closed and unclosed forms
 * match.
 *
 * @returns the stripped text; `stripped` = whether at least one marker was removed (for logs/alerts).
 */
const IMITATED_MARKER_TAGS = ['ds_system', 'system', 'ide_result_status', 'budget:token_budget'] as const

/** Whether any imitated marker's start appears (fast exit; avoids a per-line loop over every body chunk). */
function hasImitatedMarker(text: string): boolean {
  return IMITATED_MARKER_TAGS.some((tag) => text.includes(`<${tag}`))
}

/**
 * Closed form `<tag ...>...</tag>`: a BACKREFERENCE requires the same name at both ends,
 * so a mismatch like `<a>...</b>` is not treated as a pair and swallowed along with its content.
 */
const CLOSED_MARKER_RE = new RegExp(
  `<(${IMITATED_MARKER_TAGS.join('|')})\\b[^>]*>[\\s\\S]*?</\\1>`,
  'g',
)
// NOTE: an unclosed imitated marker is intentionally NOT stripped (2026-09-18).
// The old `OPEN_MARKER_RE` deleted everything from an opening tag to the end of input,
// which truncated the reply whenever the tag was merely mentioned in prose.

/**
 * Strips an environment-details-shaped block.
 *
 * WARNING: user-measured 2026-09-18 (my own previous reply was truncated): the earlier
 * implementation used one whole-text regex that treated ANY appearance of the opening marker
 * as the start of an unclosed block and deleted everything after it. As a result, merely
 * MENTIONING the tag (a table cell, inline code, an ordinary sentence) made the rest of the
 * body disappear.
 *
 * The real client scaffolding is ALWAYS on its own line, so this scans line by line: the open
 * tag must be at line start (whitespace allowed) and not inside a fenced code block to count as
 * a block. Inline mentions and table references are always body text.
 *
 * An unclosed open tag = the stream was cut -> drop everything after it (same rule as the other
 * imitated markers). A closing tag alone on a line is cleaned too, but an inline one is kept.
 */
export class SystemMarkerFilter {
  private pending = ''
  private inFence = false
  /**
   * Lines accumulated after a `<environment_details>` opening tag, waiting for its close.
   *
   * ⚠️ 2026-09-18 (user request): only a FULL block (open AND close) is removed. An open
   * tag with no matching close is NOT a block — it is buffered here and, if the stream ends
   * first, emitted verbatim. The earlier design dropped everything after an open tag, which
   * truncated a reply whenever the model merely mentioned the tag.
   */
  private block: string[] | null = null
  private hits = 0
  /** How long a `<`-leading tail is held before it is ruled out as a tag (real tags are short). */
  private static readonly MAX_MARKER_LOOKAHEAD = 256

  /** Processes complete lines; keeps a trailing partial line for the next push. */
  push(text: string): { text: string; stripped: boolean } {
    this.pending += text
    let out = ''
    for (;;) {
      const nl = this.pending.indexOf('\n')
      if (nl === -1) break
      const line = this.pending.slice(0, nl + 1)
      this.pending = this.pending.slice(nl + 1)
      out += this.consumeLine(line)
    }
    // ⚠️ 2026-09-19 (user report: streaming felt far slower than the web UI): this filter is
    // line-oriented and used to hold ALL text until a newline. A long paragraph with no
    // newline produced zero output until it ended — the client saw nothing, then a burst.
    //
    // A marker can only begin at `<`. So while no block is open, everything BEFORE the
    // first `<` is guaranteed to be prose and is streamed immediately. From the `<` onward
    // we hold only a bounded amount; if it grows past MAX_MARKER_LOOKAHEAD without forming
    // a block, it is prose too (a real tag is short) and is released.
    if (this.block === null && !this.inFence) {
      const lt = this.pending.indexOf('<')
      if (lt === -1) {
        // No '<' anywhere: nothing here can start a marker. Keep only a tiny tail so a '<'
        // split across two chunks is still seen.
        if (this.pending.length > 4) {
          out += this.pending.slice(0, this.pending.length - 4)
          this.pending = this.pending.slice(this.pending.length - 4)
        }
      } else if (lt > 0) {
        out += this.pending.slice(0, lt)
        this.pending = this.pending.slice(lt)
      }
      // `pending` now starts with '<' (or is a tiny tail). Hold it only up to the lookahead
      // bound, after which it cannot be a tag.
      if (this.pending.length > SystemMarkerFilter.MAX_MARKER_LOOKAHEAD) {
        out += this.pending
        this.pending = ''
      }
    }
    return { text: out, stripped: this.hits > 0 }
  }

  flush(): { text: string; stripped: boolean } {
    let out = ''
    // A block that never closed is NOT a full block -> emit it as ordinary text.
    if (this.block !== null) {
      out += this.block.join('')
      this.block = null
    }
    const rest = this.pending
    this.pending = ''
    if (rest) out += this.consumeLine(rest)
    return { text: out, stripped: this.hits > 0 }
  }

  private consumeLine(line: string): string {
    const trimmed = line.trim()
    // Fences are tracked even inside a pending block, so the block still ends correctly.
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      this.inFence = !this.inFence
      return this.appendToBlock(line)
    }
    if (this.inFence) return this.appendToBlock(line)

    // Inside a pending block: keep buffering until the close arrives.
    if (this.block !== null) {
      const close = /<\/environment_details\s*>/i.exec(line)
      if (close) {
        this.hits += 1
        this.block = null
        // Keep whatever follows the closing tag on the same line.
        return line.slice(close.index + close[0].length)
      }
      this.block.push(line)
      return ''
    }

    // Opening tag at line start. A close on the SAME line makes it a full block right away.
    if (/^[ \t]*<environment_details[^>]*>/i.test(line)) {
      const close = /<\/environment_details\s*>/i.exec(line)
      if (close) {
        this.hits += 1
        return line.slice(close.index + close[0].length)
      }
      // No close yet: buffer it. It only counts as a block if the close eventually arrives.
      this.block = [line]
      return ''
    }

    if (!hasImitatedMarker(line)) return line
    // Imitated markers: remove ONLY a complete, correctly paired block. An unclosed tag is
    // left as plain text — a partial marker is far less damaging than a truncated answer.
    return line.replace(CLOSED_MARKER_RE, () => {
      this.hits += 1
      return ''
    })
  }

  /** Appends a line to a pending block, or returns it unchanged when no block is open. */
  private appendToBlock(line: string): string {
    if (this.block !== null) {
      this.block.push(line)
      return ''
    }
    return line
  }

  /** How many markers/blocks were stripped (for logging). */
  get count(): number {
    return this.hits
  }
}

/** One-shot wrapper around `SystemMarkerFilter` for non-streaming paths. */
export function stripSystemMarkers(text: string): { text: string; stripped: boolean } {
  if (!text) return { text, stripped: false }
  const filter = new SystemMarkerFilter()
  const pushed = filter.push(text)
  const flushed = filter.flush()
  return { text: pushed.text + flushed.text, stripped: filter.count > 0 }
}

// -- Web-side disclaimer stripping -------------------------

/**
 * The disclaimer DeepSeek's web endpoint appends automatically at the END OF EVERY REPLY (not
 * part of the model's answer).
 *
 * Two language variants measured (they follow the interface language):
 *   Chinese: the `AI-generated, for reference only` disclaimer (23 chars; matched literally below)
 *   English: `This response is AI-generated, for reference only.` (50 chars)
 * It arrives as SSE increments, sometimes split into tiny chunks like ` AI`, ` gen`, `,`, `content`.
 *
 * Why it must be stripped:
 *   - it sits between two replies (the auto-continue seam is right after it) and the user
 *     thinks the model suddenly inserted a sentence;
 *   - it ends with a terminator like the Chinese closing word or `.` -> `looksMidSentence` can
 *     be permanently true -> EVERY round is misjudged as 'cut mid-sentence', which triggers
 *     auto-continue forever (the continuation appends the disclaimer again, which is judged as
 *     truncated again ...).
 *
 * WARNING: 2026-09-16: the English interface appends an English disclaimer too -- previously
 * only the Chinese one matched, so the English disclaimer leaked into the end of the body.
 */
/**
 * Streaming patterns — the trailing period is REQUIRED.
 *
 * Why not make it optional here: `BoilerplateFilter` matches the moment the pattern
 * completes. With an optional `\.?` the 49-char body matched one character early (before
 * the period arrived), the body was stripped, and the period then leaked into the reply as
 * a stray "." (reproduced 2026-09-16). Requiring the period anchors the match to the real
 * end of the disclaimer, so the whole thing is only stripped once it is complete.
 */
const WEB_DISCLAIMER_PATTERNS: readonly RegExp[] = [
  // Chinese: spacing around `AI` may vary.
  /本回答由\s*AI\s*生成，\s*内容仅供参考，\s*请仔细甄别/g,
  // English: response / answer / content subjects; `AI generated` without a hyphen also matches.
  /This\s+(?:response|answer|content)\s+is\s+AI[-\s]?generated,?\s*for\s+reference\s+only\./gi,
]

/**
 * One-shot patterns — the trailing period MAY be missing.
 *
 * Only safe for `stripWebDisclaimer`, which sees the whole text at once: there is no
 * "arrives later" risk, so a period-less variant can be stripped too. The streaming
 * filter must NOT use these (see above).
 */
const WEB_DISCLAIMER_LOOSE: readonly RegExp[] = [
  /本回答由\s*AI\s*生成，\s*内容仅供参考，\s*请仔细甄别/g,
  /This\s+(?:response|answer|content)\s+is\s+AI[-\s]?generated,?\s*for\s+reference\s+only\.?/gi,
]

/**
 * Streaming hold-back window = the longest disclaimer length - 1.
 *
 * Why 'always hold the last N chars' rather than 'hold only the disclaimer prefix': the
 * disclaimer is split by SSE into arbitrary small chunks, and once the split point falls in the
 * middle of it, holding only the prefix releases the first half, which can never be reassembled
 * when the second half arrives (measured leak on 2026-09-11). A constant window guarantees the
 * complete disclaimer is still in the buffer. Use the longest variant (English, 50 chars) minus 1.
 */
const WEB_DISCLAIMER_MAX_LEN = 50

/**
 * Index of a possible disclaimer start in `text`, or -1 when none can be forming.
 *
 * A disclaimer always begins with the Chinese character or `This` (case-insensitive), so any text
 * begin one is safe to stream immediately instead of waiting for the full 49-char window.
 * The return value also covers a trailing PARTIAL prefix (`Th`, `Thi`, `This`) so a start
 * split across chunks is not released prematurely (2026-09-19).
 */
function partialDisclaimerStart(text: string): number {
  let idx = -1
  const cjk = text.lastIndexOf('本')
  if (cjk !== -1) idx = cjk
  const re = /[Tt]his/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > 0 && /[A-Za-z]/.test(text[m.index - 1])) continue // not a word start
    if (m.index > idx) idx = m.index
  }
  // Trailing partial prefix of `this`, e.g. `xxTh` -> hold from `Th`.
  for (let k = Math.min(4, text.length); k >= 1; k -= 1) {
    const tail = text.slice(text.length - k)
    if (/^[Tt]/.test(tail) && 'this'.startsWith(tail.toLowerCase())) {
      const at = text.length - k
      if (at > idx) idx = at
      break
    }
  }
  return idx
}

/** Finds the earliest disclaimer in `text`; returns `{index, length}` or null. */
function findWebDisclaimer(text: string): { index: number; length: number } | null {
  let best: { index: number; length: number } | null = null
  for (const re of WEB_DISCLAIMER_PATTERNS) {
    re.lastIndex = 0
    const match = re.exec(text)
    if (!match) continue
    if (!best || match.index < best.index) best = { index: match.index, length: match[0].length }
  }
  return best
}

/**
 * One-shot web-disclaimer stripping (non-streaming).
 *
 * End-of-round leftovers must pass through it again: `BoilerplateFilter`'s streaming hold-back
 * only covers the text it RECEIVED, while the last <=49 chars it held back never went through
 * it -- the disclaimer could leak entirely from the tail (in session `6c0dbc47` it was a
 * standalone text block with a single delta, right after a tool call).
 */
export function stripWebDisclaimer(text: string): { text: string; stripped: boolean } {
  let out = text
  // Loose patterns: the whole text is visible here, so a period-less variant is safe to
  // strip as well (the streaming filter must not do this — see WEB_DISCLAIMER_PATTERNS).
  for (const re of WEB_DISCLAIMER_LOOSE) {
    // `replace` resets lastIndex itself for a `g` regex; no manual handling needed.
    out = out.replace(re, '')
  }
  return { text: out, stripped: out !== text }
}

/**
 * End-of-round drain: flush the tails held by the three buffers in TRUE ORDER, plus a final
 * pass of the two cleanups that only run before display.
 *
 * WARNING: why not simply concatenate the three flush results:
 *   - the drain order must be the REVERSE of the pipeline (the deeper a layer, the earlier the
 *     text it held) -- otherwise the last few pieces come out in the wrong order;
 *   - the chars held by the shallower layer (the filter) never passed through the
 *     'strip disclaimer' layer, and the disclaimer loves to sit in the last few chars;
 *   - likewise, fake system markers may hide entirely in the tail.
 * At end of round there is no further input, so a one-shot replace is safe here; no streaming
 * hold-back is needed.
 */
export function drainTextPipeline(
  filter: ToolCallStreamFilter,
  boilerplate: BoilerplateFilter,
  guard: TranscriptEchoGuard,
  systemMarkers?: SystemMarkerFilter,
): {
  text: string
  echoed: boolean
  disclaimers: number
  calls: FilterOutput['calls']
  rejected: FilterOutput['rejected']
} {
  // Flush order is the REVERSE of the pipeline: the deeper a layer sits, the earlier the
  // text it held back, so its tail must be emitted first. Pipeline order is
  // filter -> boilerplate -> guard -> systemMarkers, hence the reverse below.
  const tailSystem = systemMarkers?.flush()
  const tailGuarded = guard.flush()
  const tailBoiled = boilerplate.flush()
  const tail = filter.flush()
  const raw =
    (tailSystem?.text ?? '') + tailGuarded.text + tailBoiled.text + tail.text
  const dedisclaimered = stripWebDisclaimer(raw)
  // The shallower layers' tails never passed through the system-marker filter, so a
  // one-shot pass over the whole remainder is still required. It also removes any stray
  // tool-call closing tags (`</tool_calls>` etc.) held back by the filter's own hold-back.
  const stripped = stripStrayToolMarkup(dedisclaimered.text)
  const cleaned = stripSystemMarkers(stripped)
  return {
    text: cleaned.text,
    echoed: tailGuarded.echoed,
    disclaimers: boilerplate.count + (dedisclaimered.stripped ? 1 : 0),
    calls: tail.calls,
    rejected: tail.rejected,
  }
}

/**
 * Streaming web-disclaimer stripping.
 *
 * Called per chunk: a hit drops the whole disclaimer.
 *
 * WARNING: the hold-back strategy MUST be 'always hold the last |disclaimer|-1 chars', not
 * 'hold only the prefix': the disclaimer is split by SSE into arbitrary small chunks (measured
 * ones include ` AI`, ` gen`, `,`, `content`), and once the split point falls in the middle, the
 * first half is no longer a 'prefix' -- holding only the prefix would release it, and the second
 * half can never be reassembled when it arrives (measured leak on 2026-09-11). Holding 22 chars
 * costs 22 chars of display latency, invisible to the eye.
 */
export class BoilerplateFilter {
  private pending = ''
  private hits = 0
  private stripped = false
  private readonly holdChars = WEB_DISCLAIMER_MAX_LEN - 1

  push(text: string): { text: string; stripped: boolean } {
    this.pending += text
    let out = ''
    for (;;) {
      const hit = findWebDisclaimer(this.pending)
      if (hit) {
        out += this.pending.slice(0, hit.index)
        this.pending = this.pending.slice(hit.index + hit.length)
        this.hits += 1
        this.stripped = true
        continue
      }
      // Prefix-aware hold: the disclaimer always starts with the Chinese character or `This`, so
      // cannot begin one streams immediately instead of waiting for a fixed 49-char window.
      // (2026-09-19: this removed a constant ~49 chars of latency on every delta.)
      const len = this.pending.length
      const start = partialDisclaimerStart(this.pending)
      const keepFrom = start === -1 ? len : Math.max(start, len - Math.min(len, this.holdChars))
      out += this.pending.slice(0, keepFrom)
      this.pending = this.pending.slice(keepFrom)
      return { text: out, stripped: this.stripped }
    }
  }

  flush(): { text: string; stripped: boolean } {
    const rest = this.pending
    this.pending = ''
    return { text: rest, stripped: this.stripped }
  }

  /** How many disclaimers this stream stripped (for the record). */
  get count(): number {
    return this.hits
  }
}

/**
 * Transcript format markers -- the line-start markers `serializePrompt` writes into the prompt.
 *
 * The model IMITATES the transcript format from the prompt and emits tool results / system
 * markers as its answer. This is a SECOND, INDEPENDENT leak source from the 'tool-call marker
 * leak': `ToolCallStreamFilter` only guards against the latter.
 *
 * Measured (2026-09-10, deepseek-web / deepseek-reasoner): the body text contained
 *   `[Tool Result for call_xxx]` + real tool output + `[status: running]`
 * as well as strings of `User: ...` / `Assistant: ...` transcript lines.
 *
 * WARNING: 2026-09-18 (user-measured: one reply was cut in half): the old implementation also
 * matched `[Tool Result...]` / `[Truncated]` ANYWHERE in a line (`ECHO_INLINE_SIGNATURES`), so
 * a single reference in the body (a markdown table cell, inline code, a sentence discussing
 * this protocol) fired the whole segment and swallowed everything after it. Now every
 * signature must be at LINE START, and a role-prefixed form only counts when `Assistant:` is
 * at line start too.
 */
const ECHO_SIGNATURES: readonly RegExp[] = [
  /^\[\s*Tool Result\b/i,
  /^\[\s*status\s*:\s*[a-z_]+\s*\]$/i,
  /^\[\s*(?:System|Assistant)\s*\]$/i,
  // History framing that must never reach the user. The model used to echo the trailing
  // "=== END OF HISTORY ===" footer (removed 2026-09-22); these are the safety net in case
  // it reproduces the framing anyway.
  /^===\s*END OF HISTORY\s*===$/i,
  /^===\s*CONVERSATION HISTORY\b/i,
  /^The conversation (?:above|below) is already in progress\b/i,
  /^Continue it seamlessly:/i,
  // Truncation placeholders from the transcript serializer. They used to be matched
  // ANYWHERE in a line (the now-removed ECHO_INLINE_SIGNATURES); anchored at line start
  // they still catch a real echo line, but no longer fire on a mere mention inside prose
  // or a markdown table cell.
  /^\[\s*Truncated\s*\]$/i,
  /^\[\s*\d+\s*chars?\s+omitted\s*\]$/i,
  /^assistant\s+truncated\b/i,
]
/** Transcript turn line: a single line may be body text; a run of them is an echo. */
const ECHO_TURN_RE = /^(?:User|Assistant)\s*:/

/** A bare `Assistant:` / `User:` (nothing after the colon) -- the model is starting a fake transcript line. */
const ECHO_BARE_TURN_RE = /^(?:User|Assistant)\s*:\s*$/

/** A HALF PREFIX of an echo marker (seen when the stream is cut mid-line) -- also garbage; must not reach the screen. */
const ECHO_PREFIXES = ['[tool result', '[status:', '[system]', '[assistant]']

/** Whether the line is the start fragment of some echo marker. */
function looksLikeEchoPrefix(line: string): boolean {
  const t = line.trim().toLowerCase()
  return t.length > 0 && ECHO_PREFIXES.some((p) => p.startsWith(t))
}

/**
 * Could this partial (unterminated) line still grow into an echo signature or a fence?
 *
 * Echo signatures are anchored at line start, so the moment a line diverges from every
 * possible marker prefix it is ordinary prose and can be streamed right away. This keeps
 * first-token latency low on long paragraphs that contain no newline (2026-09-19).
 */
function couldBeEchoLineStart(text: string): boolean {
  const t = text.replace(/^[ \t]+/, '')
  if (!t) return true
  // Fence opener/closer: 1+ backticks or 1+ tildes.
  if (t[0] === '`' || t[0] === '~') return true
  // Bracket markers: [Tool Result ...], [status: ...], [System], [Assistant], [Truncated],
  // [N chars omitted].
  if (t[0] === '[') {
    const low = t.toLowerCase()
    const markers = ['[tool result', '[status:', '[system]', '[assistant]', '[truncated]']
    if (markers.some((m) => low.startsWith(m))) return true
    if (t.length <= 24 && markers.some((m) => m.startsWith(low))) return true
    if (/^\[\s*\d+\s+chars?\s+omitted/i.test(t)) return true
    if (/^\[\s*\d*$/.test(t)) return true
    return false
  }
  // History-framing lines (2026-09-22): the model used to echo the trailing footer, and these
  // must be held so `classify()` can drop them. `=` covers both `=== END OF HISTORY ===` and
  // `=== CONVERSATION HISTORY ... ===`; the sentences below are the continuation instruction.
  if (t[0] === '=') return true
  const lowSentence = t.toLowerCase()
  const framing = [
    'continue it seamlessly',
    'the conversation above is already in progress',
    'the conversation below is already in progress',
  ]
  if (framing.some((f) => f.startsWith(lowSentence))) return true
  if (framing.some((f) => lowSentence.startsWith(f))) return true
  const low = t.toLowerCase()
  // Role turn lines and their building prefixes: "U", "Us", "User", "User:", "A", ...
  if ('user:'.startsWith(low) || 'assistant:'.startsWith(low)) return true
  if (/^(?:user|assistant)\b/i.test(t)) return true
  return false
}

/**
 * Line-by-line guard: once an echo signature is hit, DROP EVERYTHING FROM THAT LINE ON.
 *
 * Why this design:
 *  - an echo almost always appears at the end (the model is 'continuing the transcript'), and
 *    the real answer is before it -> truncating preserves more content than dropping everything;
 *  - not evaluated inside fenced code blocks -- a normal answer may reference these markers
 *    (e.g. when discussing this plugin);
 *  - line-buffered with the trailing unfinished half-line kept -> streaming never pushes
 *    garbage to the user and then tries to swallow it back.
 */
export class TranscriptEchoGuard {
  private pending = ''
  private inFence = false
  /** A held, not-yet-decided transcript turn line (waits for the next line to decide echo vs body). */
  private turnCandidate: string | null = null
  private fired = false
  // NOTE: there is no fixed hold-back length. Whether the current (unterminated) line can
  // still become a marker is decided by `couldBeEchoLineStart` on every push, so prose
  // streams immediately instead of waiting for a newline (2026-09-19).

  /**
   * @returns `text` = the part safe to display; `echoed` = whether an echo occurred this round (that part was dropped).
   */
  push(text: string): { text: string; echoed: boolean } {
    if (this.fired) return { text: '', echoed: true }
    this.pending += text
    let out = ''
    for (;;) {
      const nl = this.pending.indexOf('\n')
      if (nl === -1) break
      const line = this.pending.slice(0, nl + 1)
      this.pending = this.pending.slice(nl + 1)
      const verdict = this.classify(line)
      if (verdict === 'echo') {
        this.fired = true
        this.pending = ''
        this.turnCandidate = null
        return { text: out, echoed: true }
      }
      if (verdict === 'turn') {
        // Transcript turn line: a single line may be normal body text, so HOLD it first and let
        // the next line decide (otherwise the first line leaks to the screen first)
        if (this.turnCandidate !== null) {
          this.fired = true
          this.pending = ''
          this.turnCandidate = null
          return { text: out, echoed: true }
        }
        this.turnCandidate = line
        continue
      }
      // ordinary/fence line: only when this line is NOT blank does the held line turn out to be
      // just the text `User:` inside the body -> release it
      if (this.turnCandidate !== null && line.trim() !== '') {
        out += this.turnCandidate
        this.turnCandidate = null
      }
      out += line
    }
    // Stream everything that can no longer become a marker. Echo signatures are anchored at
    // line start, so once the current (unterminated) line has diverged from every possible
    // marker prefix, the whole line is prose and is released immediately. This is what makes
    // streaming feel as fast as the web UI — a long paragraph without a newline no longer
    // waits for its end (2026-09-19).
    if (this.pending) {
      // A held `User:`/`Assistant:` line is only an echo if the NEXT line is also a turn
      // line. If the next line is ordinary prose, both are just text mentioning the prefix.
      const holdable =
        this.turnCandidate !== null
          ? couldBeEchoLineStart(this.pending) || ECHO_TURN_RE.test(this.pending.trimStart())
          : this.inFence
            ? /^[ \t]*[`~]/.test(this.pending) // possible closing fence
            : couldBeEchoLineStart(this.pending)
      if (!holdable) {
        if (this.turnCandidate !== null) {
          out += this.turnCandidate
          this.turnCandidate = null
        }
        out += this.pending
        this.pending = ''
      }
    }
    return { text: out, echoed: false }
  }

  flush(): { text: string; echoed: boolean } {
    if (this.fired) return { text: '', echoed: true }
    let out = ''
    // only a lone turn line -> treat as body text, release it
    if (this.turnCandidate !== null) {
      out += this.turnCandidate
      this.turnCandidate = null
    }
    const rest = this.pending
    this.pending = ''
    // the stream was cut mid-line: a complete echo cannot be judged, but a half marker prefix is
    // garbage too, so drop it
    if (rest && (this.classify(rest) === 'echo' || looksLikeEchoPrefix(rest))) {
      this.fired = true
      return { text: out, echoed: true }
    }
    return { text: out + rest, echoed: false }
  }

  private classify(line: string): 'echo' | 'turn' | 'fence' | 'plain' {
    const t = line.trim()
    if (t.startsWith('```') || t.startsWith('~~~')) {
      this.inFence = !this.inFence
      return 'fence'
    }
    if (this.inFence) return 'plain'
    for (const re of ECHO_SIGNATURES) if (re.test(t)) return 'echo'
    // fragments left after a truncation placeholder was cut in half (e.g. a lone line `truncated]`)
    if (/^\]?\s*truncated\s*\]?\s*$/i.test(t)) return 'echo'
    // transcript playback, but with a role prefix (measured shape `Assistant: [Tool Result for call_xxx]`).
    // WARNING: this requires the ROLE PREFIX AT LINE START. The old implementation matched
    // `[Tool Result...]` / `[Truncated]` etc. ANYWHERE in a line, so a single reference in the
    // body (a markdown table cell, inline code, a sentence discussing this protocol) fired the
    // whole segment and swallowed everything after that line (2026-09-18 user-measured: a table
    // reply was cut in half).
    if (ECHO_TURN_RE.test(t) && /\[\s*(?:Tool Result\b|status\s*:|Truncated\s*\]|System\s*\]|Assistant\s*\])/i.test(t)) {
      return 'echo'
    }
    // `Assistant:` with nothing after the colon -- almost never in a real answer; letting it
    // through means waiting for a playback
    if (ECHO_BARE_TURN_RE.test(t)) return 'echo'
    if (ECHO_TURN_RE.test(t)) return 'turn'
    return 'plain'
  }
}
