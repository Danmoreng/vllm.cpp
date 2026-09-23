# SPEC-SERVE-REQUEST-LENGTH-GUARD — a refusing byte bound at the request boundary

**Issue:** [#1541](https://github.com/mudler/vllm.cpp/issues/1541), filed by the
operator at the merge of [#1539](https://github.com/mudler/vllm.cpp/pull/1539)
and listed under `## Owed` in
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md).
**Kind:** one refusal added at three request handlers, plus one derived quantity
read off the tokenizer. No kernel, no device code, no model numerics, no new
config key, no new CLI flag. Every gate in this spec runs on a CPU host.
**Row:** `SERVE-REQUEST-LENGTH-GUARD` in
[`engine-matrix.md`](../engine-matrix.md), section "Serving and the OpenAI API".
**Base:** `db648fb88`. Every local line number in this document is read there.
**Pull request shape:** ONE pull request. No answer is recorded for this row
under `## Git integration` in `.agents/developer-preferences.md`, none of
AGENTS.md's three split cases applies — no helper needs a base-reachable spec,
the scope is three call sites, and every wave writes product code — so the
repository default applies.

## Now

`DONE`. The guard bounds every chat request's RENDERED prompt, template and
tools included, by `Tokenizer::MaxPromptBytes(max_model_len)` before any encode,
on HTTP, the C ABI chat entry points and run_batch, and answers a 400. The fixed
200,000-character default this amendment removed no longer refuses prompts the
context can hold. The operator reran `## Gates` on the tree merged with main
(`ec8de4e73`): all twelve suites passed with non-zero case counts. Deleting the
rendered-prompt check fails two api-server cases and one C ABI case.

## Scope

In scope:

- `vllm::tok::Tokenizer::MaxTokenBytes()`, the longest stored token text in the
  loaded vocabulary, computed once in
  `src/vllm/tokenizer/tokenizer.cpp::FinalizeTables`.
- `ApiServer::max_prompt_bytes()` and `ApiServer::refuse_oversized_prompt`, the
  derived bound and the refusal, in
  `src/vllm/entrypoints/openai/api_server.cpp`.
- The bound's derivation moving into `ApiServer::set_tokenizer`, which is the
  one place that already receives both of its factors and is called by the
  single production wiring seam `ConfigureUtilityEndpoints`
  (`src/vllm/entrypoints/openai/server_main.cpp:1640`).
- Three call sites: `ApiServer::handle_completions`,
  `ApiServer::handle_chat_completions` and `ApiServer::handle_tokenize`.
- Red-first socket cases in
  `tests/vllm/entrypoints/openai/test_api_server.cpp`.

Out of scope, each with a stated reason:

- **Authentication.** There is none anywhere in `src/vllm/entrypoints/`, which
  is why an unauthenticated caller reaches the tokenizer at all. Adding one is a
  separate product decision with its own surface, and a bound is worth having
  whether or not it lands.
- **A raw request-BODY byte bound.** Argued and rejected under `## Design`: it
  would refuse legitimate multimodal chat requests, whose inline base64 media is
  never tokenized as text.
- **A bound on the NUMBER of messages or prompts.** vLLM has one
  (`VLLM_MAX_COMPLETION_PROMPTS`); we do not accept a prompt LIST at all
  (`CompletionRequest::prompt` is a bare `std::string`,
  `include/vllm/entrypoints/openai/protocol.h:190`), so there is nothing to
  bound. Named under `## Owed` so the gap is visible if the list form lands.
- **`/v1/embeddings`, `/v1/audio/*`, `/v1/videos*`, `/detokenize`, the C ABI.**
  Enumerated under `## Design`, with what each one is and is not exposed to.
- **`src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`.** Untouched by
  design. It is the post-encode token check and it stays exactly where it is.

## Our baseline

Read at `db648fb88`, and checked rather than assumed:

- **The only size bound in the stack is httplib's default.**
  `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` at `third_party/httplib/httplib.h:129-130` is
  `100 * 1024 * 1024`. Nothing in `src/`, `include/`, `cmake/` or
  `CMakeLists.txt` overrides it, and nothing calls `set_payload_max_length`.
- **There is no authentication anywhere in `src/vllm/entrypoints/`.** A
  case-insensitive grep for `api_key`, `api-key`, `bearer`, `authorization` and
  `authenticat` over `src/vllm/entrypoints/` and `include/vllm/entrypoints/`
  returns nothing, exit status 1.
- **`/tokenize` needs no engine and no model.** It is registered whenever a
  tokenizer is attached (`src/vllm/entrypoints/openai/api_server.cpp`,
  `register_routes`), and its handler encodes directly.
- **The generate paths pay the encode BEFORE the length check.**
  `src/vllm/v1/engine/input_processor.cpp:259-260` encodes and `:265` validates,
  so `max_model_len` bounds nothing until the expensive step is already paid.
- **The cost is no longer quadratic and is still unbounded.** `67823aee2` took
  64 KB in one pretoken from 23,620.695 ms to 7.797 ms and moved the exponent
  from about 2 to about 1 ([`bpe-quadratic-merge.md`](bpe-quadratic-merge.md)
  `## Outcome`). Linear against a 100 MB body is a smaller problem than
  quadratic, not the absence of one.

**The longest stored token text, measured over the four committed goldens** with
`json.load` on each `tokenizer.json` and `len(k.encode('utf-8'))` over
`model.vocab` and `added_tokens`:

| golden | vocab entries | longest stored token, bytes | which |
|---|---:|---:|---|
| `tests/parity/goldens/tokenizer_qwen36/tokenizer.json` | 248,044 | **256** | a 128-space run, `Ġ` x 128, `Ġ` being 2 UTF-8 bytes |
| `tests/parity/goldens/tokenizer_deepseek_v2/tokenizer.json` | 100,000 | **256** | the same shape |
| `tests/parity/goldens/tokenizer_muse_glimmer/tokenizer.json` | 200,000 | **192** | `Âł` x 48 |
| `tests/parity/goldens/tokenizer_mistral/tokenizer.json` | 32,768 | **48** | `▁` x 16, the metaspace mark being 3 UTF-8 bytes |

At a 40,960-token `max_model_len` the derived bound is therefore 10,485,760
bytes on a Qwen3.6-class checkpoint and 1,966,080 on a Mistral-class one — a
10x to 53x reduction against httplib's 100 MB, on a quantity that adapts to the
checkpoint instead of being transcribed.

## Upstream chain

**vLLM has NO equivalent for a prompt BYTE bound.** That is a search result with
the paths named, not a conclusion. Read at pin `5559679229bc961848b121ccdeaa8fa5d79bec98`:

| Layer | What it bounds | Anchor | Why it is not the equivalent |
|---|---|---|---|
| uvicorn/h11 | the request line + headers | `vllm/entrypoints/openai/cli_args.py:292-294` `h11_max_incomplete_event_size`, default 4 MB at `vllm/entrypoints/serve/utils/constants.py:9` | vLLM's docstring says "header or body", but h11 applies the cap to the UNDRAINED receive buffer only (`h11/_connection.py:485`, whose own comment reads "431 is Request header fields too large which is pretty much the only situation where we can get here"), and a body arrives as drained DATA events. It does not bound a body |
| request validation | the COUNT of prompts in a list | `vllm/entrypoints/openai/completion/protocol.py:536-553` `validate_prompt_list_length`, `VLLM_MAX_COMPLETION_PROMPTS` default 1024 at `vllm/envs.py:110,1095-1100` | a count, not bytes. It IS the register this refusal mirrors: refused inside a pydantic `model_validator`, so ahead of the router's `check_model`, with the limit named in the message |
| request boundary | the BYTES of an audio upload | `vllm/entrypoints/speech_to_text/base/utils.py:38-46` `read_upload_with_limit`, `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` default 25 at `vllm/envs.py:79,972-976` | a refusing byte bound at the request boundary, checked from `Content-Length` before materialization — but on the audio-upload surface, not on a text prompt |
| input processing | the TOKEN count, after the encode | `vllm/v1/engine/input_processor.py:387-432` `_validate_prompt_len` | the very ordering this row exists to get ahead of |

So the SHAPE is mirrored — refuse during request validation, name the limit,
never truncate — and the NUMBER is ours, which is why `## Design` derives it
instead of transcribing one.

## Port map

| Upstream | Ours today | After |
|---|---|---|
| `completion/protocol.py:536-553` (refuse in request validation, message names the limit) | nothing between the body and the encode | `ApiServer::refuse_oversized_prompt`, called from three handlers before any tokenization |
| `speech_to_text/base/utils.py:38-46` (a refusing BYTE bound at the boundary) | nothing on the text surfaces | the same shape, on the prompt text |
| `envs.py:110` / `envs.py:79` (an ARBITRARY policy number, therefore configurable) | — | a DERIVED number, therefore not configurable. See `## Design` |
| `input_processor.py:387-432` `_validate_prompt_len` | `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` | unchanged, and deliberately so |
| no counterpart | no way to ask a tokenizer its worst-case bytes per token | `vllm::tok::Tokenizer::MaxTokenBytes()` |

## Design

**The bound is `max_model_len * MaxTokenBytes()`, and it is derived rather than
chosen.** The token texts of an encode concatenate back to the input, so a
prompt of `B` bytes costs at least `B / MaxTokenBytes()` tokens. Any prompt
longer than `max_model_len * MaxTokenBytes()` therefore exceeds `max_model_len`
tokens, and `ValidatePromptLen` would refuse it after the encode. The guard
refuses it before. **It rejects nothing the server would have served**; it only
moves an already-certain refusal ahead of the work that pays for it.
**One exception, recorded 2026-09-23:** a SentencePiece tokenizer with
`fuse_unk: true` and no byte fallback encodes a run of unknown characters of
any length as ONE `<unk>` (`src/vllm/tokenizer/tokenizer.cpp`, the `fuse_unk_`
branches of the SentencePiece encode). Its token texts do not concatenate back
to the input, so such a prompt can fit in `max_model_len` tokens and still be
longer than the bound. vLLM's `max_chars_per_token` bound has the same
exception. See `#### Second review repair 2026-09-23`.

`MaxTokenBytes()` is the longest STORED token text, which is an OVER-estimate of
the decoded bytes one token can carry, and the over-estimate is the direction
soundness needs. On the byte-level family a stored token holds one mapped
codepoint per input byte, at 1-2 UTF-8 bytes each; on the SentencePiece family
it holds the literal text with the metaspace mark (3 bytes) standing in for one
space and `<0xNN>` (6 bytes) for one fallback byte. Both are at least as long as
what they decode to, so the bound is never tighter than the true one.

**It is FIXED, not configurable, and that is an argument rather than an
omission.** Below the derived value the guard would refuse prompts the server
would serve, which is a behaviour change and not a defence. Above it the guard
is inert on the generate paths, because `ValidatePromptLen` refuses anyway. It
already adapts per checkpoint, through both of its factors, which is what a
config key would otherwise be used for. vLLM makes its analogues configurable
because each of them is an arbitrary policy number — 1024 prompts, 25 MB of
audio — with nothing to derive it from; ours has a derivation, so there is
nothing to tune. This is a divergence from vLLM's register and it is recorded
here as one. No config key, no CLI flag, and therefore no `docs/USAGE.md` key
row.

**Rejected: a raw request-BODY byte bound.** It is checkable earlier still, and
it would bound the JSON parse as well as the encode. It is wrong here because
`/v1/chat/completions` accepts inline base64 media
(`src/vllm/entrypoints/openai/chat_mm.cpp`, `DecodeDataUri`), which is never
tokenized as text: a bound derived from text-token arithmetic would refuse
legitimate multimodal requests, and a bound loose enough not to would not bound
the text. The guard therefore measures the text that reaches the tokenizer.

**Rejected: truncation.** Named as forbidden by
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md) `## Defence in depth`, and
restated because it is the shortcut somebody reaches for. A shortened prompt
returns model output for text the caller did not send. The refusal is tested by
its ABSENCE of a token list, not only by its status code, so a truncating guard
cannot satisfy the gate.

**Rejected: putting it in `ValidatePromptLen`.** It needs the token count the
expensive step produces (`input_processor.cpp:259-260` then `:265`), so placing
the guard there reproduces the exact ordering that made the original defect
reachable.

### Which surfaces, and which not

| Surface | Covered | Why |
|---|---|---|
| `POST /tokenize` | **yes** | the surface this row exists for: no engine, no model, no credential. Measured on the FINAL prompt, after the chat form's template render, because that string is what the encode is handed |
| `POST /v1/completions` | **yes** | `request.prompt.size()`, before `check_model` and before `create_completion` |
| `POST /v1/chat/completions` | **yes** | the SUM of `ChatMessage::content` over the messages, which is what the chat template concatenates into the one prompt. Inline base64 media lives in `content_parts` and is correctly not counted |
| `POST /detokenize` | no | it takes token ids, not text; its cost is bounded by the id count and it never reaches the BPE merge loop |
| `POST /v1/embeddings` | no | reaches the tokenizer through a different seam (`EmbedFn`), is registered only when an embedder is attached, and `ApiServer` does not parse its inputs into a shape the bound can read. NAMED GAP, `## Owed` |
| `POST /v1/audio/transcriptions` | no | a multipart audio upload, not text. vLLM bounds this one by BYTES (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`); we do not, and that is a separate mirror. NAMED GAP, `## Owed` |
| `POST /v1/audio/speech`, `POST /v1/videos`, `POST /v1/videos/sync` | no | opt-in routes registered only with their backing attached, and their cost is dominated by generation rather than tokenization. NAMED GAP, `## Owed` |
| the C ABI (`include/vllm.h`) | no | an in-process caller is not an unauthenticated remote one, and it already chooses its own prompt. The bound is a property of the HTTP boundary |
| `/v1/messages` | not applicable | no such route exists in this tree; `grep -rn "v1/messages" src/ include/` returns nothing |

**A guard on one surface is a guard with holes, and the holes above are named
rather than implied.** The three covered surfaces are the three that hand a
caller-supplied STRING to `Tokenizer::Encode` from a registered route.

### The error shape

`MakeError(400, "BadRequestError", ...)` — the register every sibling refusal in
`src/vllm/entrypoints/openai/api_server.cpp` uses, and the status
`InputValidationError` already maps to for the post-encode token refusal
(mirroring `serve/utils/error_response.py:62-65`). The message names what
arrived, the limit, and the derivation:

```
prompt length 289 bytes exceeds the maximum allowed prompt length of 288 bytes
(max_model_len 32 x 9 bytes, the longest token in this tokenizer's vocabulary).
A prompt this long cannot fit in 32 tokens, so it is refused here rather than
tokenized first. The request is refused, not truncated.
```

## Reachability

The chain, at this row's own merge commit:

```
server_main.cpp:1640  ConfigureUtilityEndpoints(server, tokenizer, max_model_len, ...)
  -> api_server.cpp   ApiServer::set_tokenizer(&tokenizer, max_model_len)
                        -> max_prompt_bytes_ = max_model_len * tok.MaxTokenBytes()
  -> register_routes  POST /tokenize, POST /v1/completions, POST /v1/chat/completions
                        -> handle_* -> refuse_oversized_prompt
```

Every gate case drives it over a real socket through
`ConfigureUtilityEndpoints`, the one seam `server_main.cpp` uses, so the tests
enter through the production entry point rather than constructing the guard by
hand. `## Outcome` records the deletion mutation.

## Dependencies

**Code:** none. Host code only, no oracle run, no lease, no checkpoint mount.
The four goldens the derivation was measured on are already committed.

**Record:** none, and this row appends no [`issue-index.md`](../completed/issue-index.md)
row of its own. #1541's row already landed at
[`issue-index.md`](../completed/issue-index.md), appended by the closing commit of
`SPEC-BPE-QUADRATIC-MERGE`, and the index is append-only: a second row for the
same issue is what `scripts/check-agent-record.py` reports as `issue #1541
listed twice`. That row names no owning row ID and is owned through `## Owed` in
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md), which stays true.

## Work breakdown

Non-overlapping, four commits in ONE pull request.

| W | Deliverable | Reviewable on its own because |
|---|---|---|
| **W1** | The socket cases of `## Tests to port` items 1-4, against UNCHANGED code | they are red, and that red is the finding: `/tokenize` answers 200 to a prompt the server can never serve |
| **W2** | `Tokenizer::MaxTokenBytes()` | one derived quantity, with the four measured goldens behind it; nothing calls it yet |
| **W3** | The bound, the refusal, and the three call sites | W1 turns green and nothing else moves |
| **W4** | The records: this spec's `## Outcome`, the matrix row, `docs/STATUS.md`, `docs/FEATURES.md`, the gate-command baseline | the code is frozen |

## Risks/decisions

| Risk | Why it is real | Control |
|---|---|---|
| The bound refuses a servable prompt | it is new refusal behaviour on a shipped route | it is derived to be an OVER-estimate on both tokenizer families; the "at the bound" case asserts a prompt of exactly `max_prompt_bytes` is tokenized whole |
| A future edit turns the refusal into a truncation | truncation is the obvious shortcut, and a status-only test would accept it | the `/tokenize` case asserts the response carries NO `tokens` and NO `count` key, so a 200 with a shortened encode fails it. Mutation recorded in `## Outcome` |
| The guard is unreached because `set_tokenizer` was never called | `max_prompt_bytes()` is 0 in that state and the guard is silently inert | the "no tokenizer attached" case pins the inert state EXPLICITLY, and every refusal case drives `ConfigureUtilityEndpoints`, the seam `server_main.cpp` actually uses. Deletion mutation in `## Outcome` |
| `/tokenize` now refuses something vLLM accepts | vLLM's `/tokenize` has no bound at all | deliberate and recorded. Counting tokens for text the server can never serve is not a servable use, and the message says why. `## Stop conditions` makes a real complaint a `NEEDS_DECISION` rather than a quiet widening |
| A hostile `max_model_len` overflows the product | `int64_t * size_t` | `set_tokenizer` clamps to `SIZE_MAX` instead of wrapping |
| The chat sum misses text the template adds | a template emits per-message framing, so the rendered prompt exceeds the sum | the residual is bounded by the message COUNT, which the 100 MB body bound still caps. Named under `## Owed` |

## Tests to port

There is no upstream test to port: vLLM has no prompt-byte bound, so item 5 is
the closest mirror rather than a port, and this list says so instead of claiming
a provenance it does not have. All of these run over a real socket unless
stated, because a guard that only fires when a test calls the parse function is
`.agents/reachability.md`'s unpassed-parameter shape.

1. **One byte over the bound is refused, and nothing is tokenized.** `POST
   /tokenize` with `max_prompt_bytes + 1` bytes: 400, `BadRequestError`, the
   message naming both the length received and the limit, and the body carrying
   NEITHER `tokens` NOR `count`. That last pair is the truncation detector.
   RED today: the route answers 200.
2. **A prompt AT the bound is tokenized in FULL.** The same route, exactly
   `max_prompt_bytes` bytes: 200, and `count` equal to
   `Tokenizer::EncodeWithSpecialTokens(...).size()`. Passes today, and must keep
   passing: it is what stops the bound being satisfied by a shortening step
   hidden anywhere on the path.
3. **`/v1/completions` and `/v1/chat/completions` refuse on BYTES.** Both
   already answer 400 on an over-long prompt, from `ValidatePromptLen`, AFTER
   the encode. The cases therefore assert the MESSAGE: the byte limit named, and
   `maximum model length` absent. RED today on the message.
4. **The bound is decided before the model lookup.** `/v1/completions` with an
   unknown model AND an oversized prompt answers 400, not 404 — mirroring vLLM,
   where `validate_prompt_list_length` is a pydantic `model_validator` and so
   runs ahead of the router's `check_model`. RED today: 404.
5. **The derivation, pinned as a derivation.** `max_prompt_bytes()` equals
   `max_model_len * Tokenizer::MaxTokenBytes()` exactly; it is 0 with no
   tokenizer attached and 0 at `max_model_len <= 0`, which is the same "no
   context length is known" state `ValidatePromptLen` early-outs on. A literal
   in items 1-4 would still pass if the bound came from somewhere else.

## Gates

The CPU tier proves all of it; nothing here needs a GPU or a lease.

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
ninja -C build test_openai_api_server test_openai_conformance test_openai_serving \
      test_bpe test_bpe_equivalence test_detokenizer test_tokenizer_parity \
      test_tokenizer_parity_mistral test_tokenizer_parity_deepseek \
      test_tokenizer_parity_gpt4o test_tokenizer_metaspace_split
./build/tests/test_openai_api_server
./build/tests/test_bpe
./build/tests/test_bpe_equivalence
```

Run each suite as its own executable so `Status:` can be read beside
`assertions:`, and assert a NON-ZERO case count: a `-tc` filter typo reports
`0 cases ran` and `SUCCESS!`.

The tokenizer suites are in the gate because `FinalizeTables` is edited, and the
serving suites because three handlers are. `scripts/agent-preflight.sh` is the
full gate.

## Stop conditions

- Stop and return `NEEDS_DECISION` if the derived bound refuses any request the
  server would otherwise have served. That would mean the derivation is wrong,
  not that the constant needs loosening.
- Stop and return `NEEDS_DECISION` if a real caller needs `/tokenize` to count
  tokens for text longer than the bound. The answer is a config key with a
  `docs/USAGE.md` row, and that is a developer decision rather than a quiet
  widening.
- Stop if the guard needs a change to
  `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`. A design that
  needs it is the wrong design, by this row's second binding constraint.
- Do not take a GPU or an `rc` lease for anything in `## Tests to port`.

## Owed

Named gaps, none of them a defect this row leaves behind:

- **`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech` and the
  `/v1/videos*` routes carry no length bound.** Enumerated under `## Design`.
  The transcription one has a direct vLLM mirror to port
  (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `speech_to_text/base/utils.py:38-46`); the
  others do not. Owned by this row.
- **The chat guard measures the summed message text, not the rendered prompt.**
  A template's per-message framing is not counted, so a request with very many
  tiny messages is bounded only by the 100 MB body limit. Owned by this row.
  **Closed 2026-09-23** by `### Review repair 2026-09-23` in the amendment
  below: the same derived bound now also applies to the rendered chat prompt,
  before the encode.
- **No prompt-LIST form exists to bound.** If `CompletionRequest::prompt` ever
  becomes a list, `VLLM_MAX_COMPLETION_PROMPTS`
  (`completion/protocol.py:536-553`) is the mirror to port at the same time.
  Owned by this row.

## Amendment 2026-09-23: the fixed prompt-character cap

**Issue:** `ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN`, owned by this row.
**Base:** `0132c65e7`. Every local line number in this section is read there.
**Pull request shape:** one pull request, spec commit first, for the same
reason as the row: no split case applies.

### The defect

A second prompt-size refusal exists beside this row's derived bound, and this
row did not add it. `OpenAIServingChat::create_chat_completion`
(`src/vllm/entrypoints/openai/serving_chat.cpp:663-685`) refuses a RENDERED chat
prompt above `VT_SERVER_MAX_PROMPT_CHARS`, which defaults to 200,000 when the
variable is unset. `git log -S VT_SERVER_MAX_PROMPT_CHARS` names its origin:
`0c2827c18` (#154), a Gemma-4 performance pull request that added it as a "lab
guardrail" against one client sending a very large system prompt. It has three
faults:

1. **The number is not derived from the context.** A 64k-token prompt of English
   prose is about 268,000 characters, so a server started with
   `--max-model-len 262144` refuses every such prompt above about 48k tokens.
   Observed on `dgx:gpu0` on 2026-09-23: `prompt too large for this server
   (268439 chars > VT_SERVER_MAX_PROMPT_CHARS=200000)`. This violates this
   row's own guarantee, stated under `## Design`: a pre-tokenization bound
   "rejects nothing the server would have served".
2. **The status is 500.** The refusal is a `std::runtime_error`, which
   `ApiServer::handle_chat_completions` maps to `InternalServerError`
   (`api_server.cpp:377-382`). The request is at fault, so the status is 400.
3. **The message blames an unrelated client** ("Hermes is likely injecting a
   full system SOUL").

### Upstream, at the current pin `e126687a9a`

`## Upstream chain` above was read at `5559679229` and says vLLM has no prompt
byte bound. **At `e126687a9a` that is no longer true**, and this is the anchor
the fix mirrors:

| What | Anchor | Behaviour |
|---|---|---|
| the pre-tokenization text bound | `vllm/renderers/params.py:342-365` `TokenizeParams._text_len_check` | refuses `len(text) > max_input_tokens * tokenizer.max_chars_per_token` with `VLLMValidationError`. There is no fixed number and no environment variable |
| the per-token factor | `vllm/tokenizers/hf.py:131` | `max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)`, the longest vocabulary entry. It is the same quantity as our `Tokenizer::MaxTokenBytes()`, counted in characters instead of bytes |
| `max_input_tokens` | `vllm/renderers/params.py:204-210`; `chat_completion/protocol.py:608-626` | `max_model_len - (max_completion_tokens or max_tokens or 0)` |
| where it runs | `vllm/renderers/base.py:527,563` | on the rendered prompt string, immediately before the encode |
| the error text | `vllm/renderers/params.py:352-362` | `This model's maximum context length is {max_total_tokens} tokens. However, you requested {max_output_tokens} output tokens and your prompt contains {len(text)} characters (more than {max_input_chars} characters, which is the upper bound for {max_input_tokens} input tokens). Please reduce the length of the input prompt or the number of requested output tokens.` |
| the status | `vllm/entrypoints/serve/exception_handling/error_response.py:39-41` | `VLLMValidationError` is `BadRequestError`, HTTP 400 |
| the token refusal after the encode | `vllm/renderers/params.py:436-461` `_token_len_check`; `vllm/v1/engine/input_processor.py:436-476` `_validate_prompt_len` | 400, token count against `max_model_len` |

So vLLM's only pre-tokenization bound is derived from the context and the
vocabulary, and it cannot refuse a prompt that could fit, except for the
`fuse_unk` case that `## Design` records, which both bounds share. This row's
`max_model_len * MaxTokenBytes()` is the same kind of bound. It uses
`max_model_len` instead of `max_input_tokens`, so it is looser by the requested
output tokens. It counts bytes where vLLM counts characters, so on multi-byte
text it can be tighter than vLLM's, but it still refuses no prompt that fits in
`max_model_len` tokens, with the same `fuse_unk` exception (`## Design`). vLLM applies its bound to the RENDERED
prompt, template and tools included. This row's HTTP guard measures the summed
message content before the template renders, which is not the same string;
`### Review repair 2026-09-23` below closes that difference.

### How the derived bound compares with vLLM's

| Factor | vLLM at `e126687a9a` | Ours | Difference |
|---|---|---|---|
| per-token length | `max_chars_per_token`, the longest vocabulary entry in CHARACTERS, computed once at tokenizer load (`vllm/tokenizers/hf.py:131`) | `vllm::tok::Tokenizer::MaxTokenBytes()`, the longest stored token text in UTF-8 BYTES, computed once in `FinalizeTables` (`src/vllm/tokenizer/tokenizer.cpp`, `include/vllm/tokenizer/tokenizer.h`) | the prompt is measured in bytes too, and this row's `## Design` argues the stored text over-estimates the decoded bytes per token on both tokenizer families |
| token budget | `max_input_tokens = max_model_len - requested output tokens` (`params.py:204-210`) | `max_model_len` | larger by the requested output tokens |
| status | `VLLMValidationError`, a `VLLMClientError` (`vllm/exceptions.py:19-27`), mapped to 400 (`error_response.py:39-41`) | 400 `BadRequestError` | equal |
| measured string | the rendered prompt, after `render_messages(tokenize=False)` (`params.py:386-399`) | the rendered chat prompt, in `create_chat_completion` (`### Review repair 2026-09-23`); the HTTP guard also measures the summed message content earlier | equal on the rendered prompt |
| message | the `_text_len_check` text above | the rendered-prompt check uses that text, in bytes and without the output-token clause; the HTTP guard keeps this row's own byte message (`### The error shape`) | the HTTP guard's message difference is recorded in `ISSUE-LOCAL-01M37A43C0XFW3PAYV1399HFAE` with the token-refusal one, so the two text mirrors land together |

**No fixed absolute ceiling is added.** The only absolute one is httplib's
100 MB `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` on the whole body, which this change
does not touch. It can bind below the derived bound only when
`max_model_len * MaxTokenBytes()` exceeds 100 MB, for example a
1,048,576-token context with 256-byte tokens, and then only for a prompt that
averages more than about 100 bytes per token. Ordinary text averages about
four. It is a transport limit that predates this row, not a prompt policy, and
it is named here so that nobody reads it as one.

### Design

- **The default is unset, and unset means no fixed cap.** The
  pre-tokenization bound on the default configuration is the derived one this
  row landed, `max_model_len * MaxTokenBytes()`. **Correction, from the fresh
  review:** the first version of this bullet said that
  `ApiServer::refuse_oversized_prompt` alone keeps the #1541 guarantee. It does
  not. That guard sums `messages[].content` before the template renders, so
  bytes in `tools`, in assistant `tool_calls` arguments and in template framing
  reached the encode unbounded once the fixed 200,000 default was gone. The
  reviewer's 4 MiB tool description was tokenized in full and refused only
  after the encode ("length 514"). `### Review repair 2026-09-23` applies the
  same derived bound to the rendered prompt, and the #1541 guarantee holds only
  with that check in place.
- **An explicitly set positive value is kept as an operator ceiling.** vLLM has
  no such variable, so this is a divergence, and it is kept for one reason:
  the variable is documented, and removing it would silently drop a ceiling an
  operator set on purpose. Only an operator who sets it can make it bind below
  the context, and `docs/ENVIRONMENT.md` says so. `0` still means off.
- **The refusal is a 400.** It throws `vllm::v1::InputValidationError`, which
  `handle_chat_completions` already maps to `BadRequestError`
  (`error_response.py:39-41`). The message names the length received, the
  limit and the variable, and blames no client.
- **The variable is read on each request, not once into a function-local
  static.** A static fixes the first value the process saw, so no test can
  exercise both the unset and the set arms in one process, and the gate would
  be blind to one of them. vLLM also reads its environment lazily
  (`vllm/envs.py`, module `__getattr__`). One `getenv` beside a template render
  costs nothing measurable.
- **Rejected: deriving a new default for this variable.** The derived bound
  already exists one layer up. A second copy of the same derivation in
  `serving_chat.cpp` would be a duplicated fact, and the next edit to one copy
  would make them disagree.
- **Rejected: changing the token refusal's text to the renderer's text.**
  vLLM's server reaches `_token_len_check` before `_validate_prompt_len`, so
  its over-long-prompt message differs from ours. That changes
  `ValidatePromptLen`, which this row's `## Stop conditions` forbid, and it
  adds a `prompt + max_tokens` check the server does not make today. Filed as
  `ISSUE-LOCAL-01M37A43C0XFW3PAYV1399HFAE`.

### `VT_SERVER_MAX_NEW_TOKENS`

The sibling variable has a different defect by a different mechanism. It
silently CLAMPS a positive `max_tokens` to 4096 by default. It does not refuse
input. vLLM has no default output ceiling: `get_max_tokens`
(`vllm/entrypoints/serve/utils/api_utils.py:169-206`) takes the minimum of the
remaining context, the request value, an operator `override_max_tokens` that
is unset by default (`chat_completion/serving.py:172-176`) and the platform
limit. It is filed as `ISSUE-LOCAL-01M37A3S7N7GSZC37JH515QXFE` and is not
changed here.

### Tests

All cases run over a real socket through `ConfigureUtilityEndpoints`, the seam
`server_main.cpp` uses, on a fixture tokenizer whose longest token is 8,192
bytes. That makes a prompt above 200,000 bytes fit in the 32-token test
context, so the red needs no large engine.

1. **RED today, GREEN after:** a 204,800-byte chat prompt of 25 tokens, with
   `VT_SERVER_MAX_PROMPT_CHARS` unset, answers 200 with an assistant choice.
   Today it answers 500 naming `VT_SERVER_MAX_PROMPT_CHARS=200000`.
2. **A prompt over `max_model_len` in tokens is still refused:** 33 tokens in
   253,954 bytes, under the derived byte bound, answers 400 `BadRequestError`
   with the `_validate_prompt_len` text, `maximum model length of 32`.
3. **The pre-tokenization bound still refuses a pathological body:** a 4 MiB
   message answers 400 with the byte-bound message, not the token message, and
   the engine holds no request.
4. **An explicit operator ceiling refuses with 400:** with
   `VT_SERVER_MAX_PROMPT_CHARS=1000`, a 204,800-byte prompt answers 400
   `BadRequestError` naming the variable and not naming any client. RED today:
   500.

### Reachability

`POST /v1/chat/completions` -> `ApiServer::handle_chat_completions` ->
`OpenAIServingChat::create_chat_completion`. The deletion mutation restores the
fixed default in a scratch worktree and must turn case 1 red.

### Gates

The row's `## Gates`, plus `python3 tests/scripts/test_check_env_doc.py`
because `docs/ENVIRONMENT.md` changes. No GPU and no lease.

### Stop conditions

- Stop with `NEEDS_DECISION` if removing the default would leave any text
  route without a pre-tokenization bound.
- Stop if the fix needs a change to `ValidatePromptLen`.

### Evidence, recorded on the branch

CPU build, `cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=OFF`, `ninja -C build -j 4`. The focused command is
`test_openai_api_server -tc="api_server: the chat prompt cap never refuses what
the context holds"`, run with `VT_SERVER_MAX_PROMPT_CHARS` removed from the
environment.

| stage | commit | cases | assertions | `Status:` | exit |
|---|---|---:|---:|---|---:|
| RED, production code unchanged | `efe6ed9b6` | 1 | 53, **11 failed** | `FAILURE!` | **1** |
| GREEN | `1cb56af24` | 1 | 55, 0 failed | `SUCCESS!` | 0 |
| deletion mutation: `serving_chat.cpp` replaced by its `origin/main` bytes, in a scratch worktree | `1cb56af24` + mutation | 1 | 53, **11 failed** | `FAILURE!` | **1** |

The red is the defect: the 204,800-byte, 25-token prompt answered
`CHECK( 500 == 200 )` with the old `VT_SERVER_MAX_PROMPT_CHARS=200000` message;
the 33-token prompt answered 500 from the same cap instead of the token
refusal; the explicit 1000 ceiling answered 500. The 4 MiB subcase passed in
the red, because the derived byte bound already refuses it first, and it still
passes. The mutated file hashed `1701e40f...` against `866b2bb7...` for the
committed file, and the restored file hashed `866b2bb7...` again.

At `1cb56af24` the serving and tokenizer suites are green, each run as its own
executable: `test_openai_api_server` 99 cases / 1397 assertions,
`test_openai_conformance` 23 / 252, `test_openai_serving` 48 / 1365,
`test_openai_serving_chat_stream` 2 / 210, `test_openai_protocol` 37 / 269,
`test_bpe` 29 / 1009, `test_bpe_equivalence` 2 / 334, all `SUCCESS!`, exit 0.
`scripts/check-env-doc.py` reports the same three undocumented variables it
reports on `origin/main` (`VT_CUDA_ALLOC_STATS`, `VT_V4_W32_COLS`,
`VT_V4_W32_WARPS`) and no new one.

### Review repair 2026-09-23

A fresh review of `b83feab14` failed the amendment on one blocking finding and
one test gap. Both are repaired on the same branch.

**F1, blocking: the rendered prompt had no pre-tokenization bound.** The #1541
HTTP guard sums `messages[].content` before the template renders. Bytes in
`tools` (descriptions and parameter schemas), in assistant `tool_calls`
arguments and in template framing were not counted, and after the fixed 200,000
default was removed nothing else measured them. The reviewer's 4 MiB tool
description was tokenized in full and refused only after the encode ("length
514"). This case is the one `### Stop conditions` names, and the first version
of `### Design` wrongly said the #1541 guarantee held.

The repair mirrors vLLM, which runs `_text_len_check` on the rendered prompt
(`vllm/renderers/params.py:342-370`, reached through `apply_pre_tokenization`
at `:386-399`, after `render_messages(tokenize=False)`):

- `OpenAIServingChat::create_chat_completion` refuses a rendered prompt longer
  than `max_model_len * MaxTokenBytes()`. The check is at the point where the
  old fixed cap measured the prompt: after the template renders, before the
  engine encode, the beam-search encode and the multimodal seam. The first
  version of this repair gated only the engine encode; `#### Second review
  repair 2026-09-23` adds the gates for the other two. It throws
  `InputValidationError`, which is HTTP 400 `BadRequestError`. The message is
  `_text_len_check`'s, counted in bytes and without the output-token clause,
  because this bound does not subtract the requested output tokens:
  `This model's maximum context length is {max_model_len} tokens. However, your
  prompt contains {N} bytes (more than {bound} bytes, which is the upper bound
  for {max_model_len} input tokens). Please reduce the length of the input
  prompt.`
- **One copy of the derivation.** `Tokenizer::MaxPromptBytes(max_model_len)`
  holds it, with the 0-when-unknown and overflow-clamp rules that
  `ApiServer::set_tokenizer` had. `set_tokenizer` now calls it, and so does
  `InputProcessor::max_prompt_bytes()`. The chat handler reads the bound from
  its own engine's `InputProcessor` (new `input_processor()` accessors on
  `LLMEngine` and `AsyncLLM`), which holds the tokenizer and the resolved
  `max_model_len` that `ValidatePromptLen` refuses against. In
  `server_main.cpp` both come from the same `LoadedEngine`
  (`model_loader.cpp:2288`, `server_main.cpp:1984`).
- **Reach.** Every caller of `create_chat_completion` gets the check, because
  every `OpenAIServingChat` holds an engine:
  - `POST /v1/chat/completions` through `ApiServer::handle_chat_completions`;
  - the C ABI `vllm_chat` and `vllm_chat_stream` (`src/capi/vllm_c.cpp`,
    through `EnsureChatServing`, which builds the handler over
    `engine->loaded->async_engine()`). The C ABI has no HTTP guard in front of
    it, so before this repair an ABI chat prompt had no pre-tokenization bound
    at all. The new `test_capi` case proves the refusal on both entry points
    with zero encodes;
  - `RunBatch` (`run_batch.cpp:118`). It has no production caller in this tree
    (`grep -rn RunBatch src tools examples` finds only its own file), so it is
    reached structurally through the same handler and no separate case is
    added.
- **Completions and `/tokenize` have no equivalent gap.** `/tokenize` already
  measures the final prompt after the chat form's template render
  (`api_server.cpp`, `handle_tokenize`). `/v1/completions` renders nothing:
  `request.prompt` is the string the engine encodes, and the HTTP guard
  measures exactly it. The C ABI's completion entry points (`vllm_complete`,
  `vllm_complete_stream`, `vllm_request_submit`) have no pre-tokenization
  bound. That is the existing C ABI gap under the row's `## Owed`, not a
  counted-before-rendering gap, so it is not changed here.
- The HTTP content-sum guard stays. It is cheaper, it runs before the template
  renders, and it keeps its own message.

**F2, non-blocking: the operator ceiling's comparison survived off-by-one
mutations.** A boundary pair now pins it: with
`VT_SERVER_MAX_PROMPT_CHARS=204800`, a rendered prompt of exactly 204,800 bytes
is served (one encode), and 204,801 bytes are refused with the variable named
(zero encodes). The derived bound has the same pair: 262,144 rendered bytes
reach the encode and the token refusal, and 262,145 are refused before it.

**The default resolution is pinned directly.** The resolution moved into
`OperatorMaxPromptChars()` (declared in `serving_chat.h`). End to end, a fixed
default above the fixture's 262,144-byte derived bound is indistinguishable
from no default, because the derived bound refuses first. A real server's
derived bound is tens of megabytes, so such a default would still refuse
prompts the context holds. The new case asserts that unset, empty and `0`
resolve to 0, which is no ceiling, and that `1000` resolves to 1000.

**The encode spy.** `InputProcessor::num_prompt_encodes()` is a relaxed counter,
incremented where the text `process_inputs` overload hands a prompt to the
tokenizer. The cases assert it stays 0 for every pre-encode refusal and is 1
where the prompt must reach the encode, so a message-only pass cannot satisfy
them.

#### Evidence

Same CPU build and command form as above, with `VT_SERVER_MAX_PROMPT_CHARS`
removed from the environment.

| stage | commit | selection | cases | assertions | exit |
|---|---|---|---:|---:|---:|
| RED: tools and tool_calls reach the encode | `6278e226d` | `the derived bound applies to the RENDERED chat prompt` | 1, 1 failed | 45, **12 failed** | **1** |
| GREEN | `19b2bb3fc` | the three prompt-cap cases in `test_openai_api_server` | 3 | 129, 0 failed | 0 |
| GREEN | `19b2bb3fc` | `capi: vllm_chat refuses a prompt over the derived byte bound before the encode` | 1 | 10, 0 failed | 0 |

The red is the defect: the 4 MiB tool description and the 4 MiB of
`tool_calls` arguments each answered the post-encode `The decoder prompt
(length 513)` refusal with `num_prompt_encodes() == 1`, and the rendered prompt
one byte over the derived bound answered `(length 33)`, also with one encode.
The exact-bound half passed.

Mutations of `src/vllm/entrypoints/openai/serving_chat.cpp` at `19b2bb3fc`, in
a separate scratch worktree, each rebuilt and run on the three api-server cases
and the C ABI case. The committed file hashed
`0ddc717e2315453462fca8032f6815b6be015f9c85a76d30c081fec777205a8f` before every
mutation and again after every restore.

| mutation | mutated sha256 | api-server cases | C ABI case | killed by |
|---|---|---|---|---|
| delete the rendered-prompt check | `64730cfd...` | 12 of 129 failed | 4 of 10 failed | the tools, tool_calls and one-byte-over cases (encode count 1, token message); both ABI entry points |
| derived `>` to `>=` | `4706ed2d...` | 3 failed | green | the exact-bound case (byte refusal, encode count 0) |
| derived `> bound + 1` | `c3a799fd...` | 4 failed | green | the one-byte-over case (encode count 1) |
| operator `>` to `>=` | `d6859200...` | 4 failed | green | the exact-ceiling case (refused, encode count 0) |
| operator `> ceiling + 1` | `18baf44c...` | 6 failed | green | the one-byte-over-ceiling case (served) |
| default 300000 when unset | `b657fda8...` | 2 failed | green | the resolution case, unset and empty |
| measure message content only | `396b8bcb...` | 12 failed | green | the tools, tool_calls and one-byte-over cases. The ABI case stays green by construction: its content is the whole prompt |

After the last restore the tree was rebuilt and all four cases passed again.

At `19b2bb3fc`, each run as its own executable, all `SUCCESS!` and exit 0:
`test_openai_api_server` 101 cases / 1471 assertions, `test_openai_serving`
48 / 1365, `test_openai_conformance` 23 / 252,
`test_openai_serving_chat_stream` 2 / 210, `test_openai_protocol` 37 / 269,
`test_openai_run_batch` 7 / 80, `test_capi` 73 / 730, `test_chat_prompt`
5 / 15, `test_bpe` 29 / 1009, `test_bpe_equivalence` 2 / 334.
`scripts/agent-preflight.sh --staged` reports the same 15 failed gates as on
`b83feab14`, all inherited from `origin/main`, and `scripts/check-env-doc.py`
still lists only `VT_CUDA_ALLOC_STATS`, `VT_V4_W32_COLS` and
`VT_V4_W32_WARPS`.

#### Second review repair 2026-09-23

A second fresh review of `07ae2a052` confirmed that the text path is closed and
gated. It failed the branch on two blocking findings and raised two notes.

**B1, blocking: three record anchors went stale.** Lines this branch inserted
moved three symbols that `.agents/engine-matrix.md` cites. Each cell is
re-pointed and no other row changed:

| row | before | after |
|---|---|---|
| SERVE-UTILITY-ENDPOINTS | `include/vllm/entrypoints/openai/serving_chat.h:246` (`prompt_fn()`) | `:263` |
| LOAD-SENTENCEPIECE | `include/vllm/tokenizer/tokenizer.h:129` (`GetFamily`) | `:147` |
| SPEC-BPE-QUADRATIC-MERGE | `src/vllm/v1/engine/input_processor.cpp:245` (`process_inputs`), encode at 260, length check at 265 | `:249`, encode at 265, length check at 270 |

`scripts/check-agent-record.py` then reports the five stale anchors inherited
from `origin/main` (SERVE-METRICS, SERVE-UTILITY-ENDPOINTS and SERVE-ADMIN in
`api_server.cpp` and `api_server.h`) and the `roadmap_v1.md:105-106` table
errors, and nothing else. Those belong to another change.

**B2, blocking: the claim "before the beam-search encode and the multimodal
seam" had no gate.** `InputProcessor::num_prompt_encodes()` counts only the
text `process_inputs` overload. Beam search encodes with
`beam_tokenizer_->Encode(prompt)` in `create_chat_completion`, and the
multimodal seam encodes inside `MakeQwen3VLImageChatFn` (`chat_mm.cpp`). The
reviewer moved the check below the beam-search block and every suite stayed
green.

- **Beam search.** `OpenAIServingChat::num_beam_prompt_encodes()` is a relaxed
  counter, incremented on the line before `beam_tokenizer_->Encode(prompt)`.
  It is the same mechanism as the text-path spy, at the one call site the beam
  path has. The alternatives were weaker: a counter inside `Tokenizer` would
  add state to a copyable core type for one test, and asserting only the
  response message would not show that the encode was skipped. The case sends
  a `use_beam_search` request whose rendered prompt is one byte over the
  derived bound, with `VT_SERVER_MAX_PROMPT_CHARS` unset. It asserts HTTP 400
  `BadRequestError` with the byte-bound message, no `choices`, zero beam
  encodes, zero engine encodes and no unfinished request. An in-bound beam
  request is the control: it answers 200 with choices and exactly one beam
  encode, so a counter that never moves cannot pass.
- **Multimodal seam: gated.** The fixtures can drive the REAL production seam,
  `MakeQwen3VLImageChatFn`, as the existing over-limit multimodal case already
  does. The case wraps that seam in a counter. An image request one byte over
  the bound answers the byte-bound 400 with zero seam entries, zero codec calls
  and zero engine encodes. The in-bound control enters the seam once, and the
  seam's own encode is observable because it fails: the 22-entry fixture
  vocabulary cannot encode the seam's Qwen placeholder marker, so the request
  answers 500 with the tokenizer's message. What this gate does NOT prove is a
  real Qwen3-VL tokenizer's encode; it proves the order of the check and the
  seam call, which is the claim.

**N1, a known small difference on the multimodal path.** The multimodal seam
re-renders the messages itself (`chat_mm.cpp`, `MakeQwen3VLImageChatFn` step 1)
with no tools, no `chat_template_kwargs`, and content built by
`BuildMarkerInjectedContent`: text parts joined without the `"\n"` separator
the text path uses, and one placeholder marker per media part. So the string
the seam encodes is not the string the check measured. Per image part it adds
the Qwen3-VL marker `<|vision_start|><|image_pad|><|vision_end|>` (43 bytes,
`chat_mm.cpp:191`); the dropped separators and the dropped tools only make it
shorter. The review estimated about 45 bytes per media part, and 43 is the
exact upper bound for an image part. `ValidateChatMmLimits` caps Qwen3-VL at
one image, so the seam's encode can exceed the derived bound by at most 43
bytes. The token refusal after the encode still catches such a prompt. The
difference is accepted and recorded, not repaired. The missing
`chat_template_kwargs` on this seam are already owed under
`specs/chat-template-jinja-undefined.md` (#1681).

**N2, wording.** The claim "never refuses a prompt that fits in
`max_model_len` tokens" has one exception, now named under `## Design`, in the
`serving_chat.cpp` comment and in `docs/ENVIRONMENT.md`
(`VT_SERVER_MAX_PROMPT_CHARS`): a SentencePiece tokenizer with `fuse_unk: true`
and no byte fallback collapses a run of unknown characters of any length into
one `<unk>`, so such a prompt can fit in tokens and exceed the byte bound.
vLLM's `max_chars_per_token` bound has the same exception.

##### Evidence

CPU build as above, `VT_SERVER_MAX_PROMPT_CHARS` removed from the environment.
The new case is `api_server: the rendered-prompt bound runs before the beam and
multimodal encodes`, committed at `7b1553024`.

The mutations ran in one scratch worktree at `7b1553024`, each rebuilt and run
on the four prompt-cap cases of `test_openai_api_server` (172 assertions) and
the C ABI case (10 assertions). The committed `serving_chat.cpp` hashed
`bb6f29d6b824dc33965c03068cf31f28e41e6fbd7598a58ee7b9bb77cfd0015d` before every
mutation and again after every restore. The unmutated baseline was 4 of 4 cases
and 1 of 1 case, 0 failed.

| mutation | mutated sha256 | api-server | C ABI | killed by |
|---|---|---|---|---|
| (a) delete the check | `8be41ce3...` | 2 of 4 cases, 19 assertions failed | 4 of 10 failed | the tools, tool_calls and one-byte-over cases; the beam and image cases |
| (b) measure content only | `d6542d9a...` | 2 of 4, 19 failed | green (content is the whole prompt) | the same cases as (a) |
| (c) derived `>` to `>=` | `c5b2dc8c...` | 1 of 4, 3 failed | green | the exact-bound case (byte refusal, encode count 0) |
| (d) operator `>` to `>=` | `dad57298...` | 1 of 4, 4 failed | green | the exact-ceiling case |
| (e) default 300000 when unset | `271fe068...` | 1 of 4, 2 failed | green | the resolution case, unset and empty |
| (g) check moved below the beam-search block | `8d6e27ea...` | 1 of 4, 7 failed | green | the beam case: `The decoder prompt (length 33)` in place of the byte message, `num_beam_prompt_encodes() == 1`; and the image case: status 500, seam entered once |
| (h) check moved below the multimodal seam, above beam search | `71b9dfc1...` | 1 of 4, 4 failed | green | the image case: status 500 from the seam's encode, `seam_calls == 1` |

Red before, under (g), in the beam subcase:
`CHECK( msg.find("The decoder prompt") == std::string::npos )` failed with the
body `The decoder prompt (length 33) is longer than the maximum model length of
32`, and `CHECK( h.chat.num_beam_prompt_encodes() == 0 )` failed with
`1 == 0`. Green after, at `7b1553024`: the case passes, 1 case, 43 assertions,
0 failed.

At `7b1553024`, each run as its own executable, all `SUCCESS!` and exit 0:
`test_openai_api_server` 102 cases / 1508 assertions (one case more than
`19b2bb3fc`), `test_capi` 73 / 730, `test_openai_serving` 48 / 1365,
`test_openai_run_batch` 7 / 80, `test_chat_prompt` 5 / 15, `test_bpe`
29 / 1009. `test_model_registry` is excluded because it does not compile on
`origin/main` for an unrelated reason.

## Outcome

Recorded on the branch, before the merge. Base `db648fb88`, branch
`row/SERVE-REQUEST-LENGTH-GUARD`, five commits.

### What landed

`vllm::tok::Tokenizer::MaxTokenBytes()` is the longest stored token text,
computed once in `FinalizeTables`. `ApiServer::set_tokenizer` derives
`max_prompt_bytes_ = max_model_len * MaxTokenBytes()`, clamped on overflow and
0 when either factor is unknown. `ApiServer::refuse_oversized_prompt` returns a
400 `BadRequestError` naming the length received, the limit and the derivation,
called from `handle_tokenize` immediately before the encode and from
`handle_completions` / `handle_chat_completions` after the body parses and
before `check_model`.

**No config key, no CLI flag, no environment variable.** `## Design` carries the
argument. `docs/USAGE.md` gains a troubleshooting entry for the message rather
than a key row, because there is no key.

### Red, then green

| stage | command | cases | assertions | `Status:` | exit |
|---|---|---:|---:|---|---:|
| RED, at `bfe25bc51` | `test_openai_api_server -tc="api_server: an oversized prompt is REFUSED at the request boundary"` | 1 | 41, **15 failed** | `FAILURE!` | **1** |
| GREEN, at `a5dbf60e3` | the same command | 1 | 41, 0 failed | `SUCCESS!` | 0 |

The red is the finding, not a step toward one: `/tokenize` answered **200** with
a token list to a 289-byte prompt against a 288-byte bound, on a route that
needs no engine, no model and no credential. `/v1/completions` and
`/v1/chat/completions` already answered 400 — from `ValidatePromptLen`, after
the encode — so their cases assert the MESSAGE, and it is the message that was
red. `/v1/completions` with an unknown model answered 404, so the bound was not
yet ahead of the model lookup.

**The first red run reported 6 assertions, not 41**, because `.at("error")` on a
200 body threw out of the whole `TEST_CASE` and hid four of the five subcases.
The accessors are non-throwing for that reason, and the number recorded above is
the one taken after the fix.

### The full focused gate, at `a5dbf60e3`

One fresh `cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=OFF`, then `ninja -C build -j 16`: **1122 of 1122
targets, zero compiler warnings**, so no stale binary is printing this green.
Every suite run as its own executable so `Status:` could be read beside
`assertions:`, and every one reports a NON-ZERO case count.

| suite | cases | assertions | `Status:` | exit |
|---|---:|---:|---|---:|
| `test_openai_api_server` | 68 | 874 | `SUCCESS!` | 0 |
| `test_openai_conformance` | 23 | 252 | `SUCCESS!` | 0 |
| `test_openai_serving` | 42 | 556 | `SUCCESS!` | 0 |
| `test_bpe` | 24 | 971 | `SUCCESS!` | 0 |
| `test_bpe_equivalence` | 2 | 334 | `SUCCESS!` | 0 |
| `test_detokenizer` | 12 | 221 | `SUCCESS!` | 0 |
| `test_tokenizer_parity` | 4 | 1175 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_mistral` | 6 | 421 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_deepseek` | 6 | 2461 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_gpt4o` | 5 | 1000 | `SUCCESS!` | 0 |
| `test_tokenizer_metaspace_split` | 7 | 28 | `SUCCESS!` | 0 |

### The two mutations

Both applied in the worktree against a pre-taken sha256
(`1788a62b18f42df259ac4f761bff69a85fa01980c8917940d80985b753284429` for
`src/vllm/entrypoints/openai/api_server.cpp`), with the compiler return code and
the applied diff printed each time, and both restored byte-for-byte to that hash
with the suite re-run green after.

**REACHABILITY — delete all three production call sites.** `git diff --no-index`
against the pristine copy: `1 file changed, 9 deletions(-)`, the nine lines being
the three `if (auto refusal = refuse_oversized_prompt(...)) { return *refusal; }`
blocks. `compile_rc=0`, so this is a real mutation and not a build failure
wearing a pass.

| case | result |
|---|---|
| the five socket cases | **41 assertions, 15 failed, `FAILURE!`, exit 1** |
| the derivation case | 10 assertions, 0 failed, `SUCCESS!`, exit 0 |

**That split is itself the finding, and it is why the socket cases exist.** The
derivation case constructs the bound and reads it back; it stays green with the
guard reaching nothing, so on its own it measures a class rather than a
capability — exactly the shape `.agents/reachability.md` names. Only the cases
that enter through the registered route over a real socket detect the deletion.

**TRUNCATION — replace the refusal with `resize(max_prompt_bytes_)` in
`handle_tokenize` and `handle_completions`.** `git diff --no-index`:
`1 file changed, 4 insertions(+), 4 deletions(-)`. `compile_rc=0`. Result: **41
assertions, 13 failed, `FAILURE!`, exit 1**, and the two assertions that fire
FIRST on the truncating build are the truncation detectors —
`CHECK_FALSE(j.contains("tokens"))` and `CHECK_FALSE(j.contains("count"))` — so
the gate distinguishes a refusal from a shortened success rather than only
checking that the request did not hang.

### The post-merge rerun

`origin/main` advanced to `5539686c7` (`MUSIC3-DIT-ARM-REACH`, #1131) while this
row was being gated, so it was merged at `73964532d` and everything was run
again on the merged head. The incoming change touches the MiniMax-Music3 speech
model, its device header, one test and `docs/USAGE.md`; the only file both sides
edit is `docs/USAGE.md`, where the two additions land in different sections.

Rebuild from the same directory: **566 of 566 targets, zero compiler warnings**,
then `ninja: no work to do` on a confirming pass. All twelve suites re-run as
their own executables: identical case and assertion counts to the table above,
every one `SUCCESS!` and exit 0.

Every record checker re-run with `--base 5539686c7`: `check-commit-trailers`,
`check-commit-style`, `check-doc-checkpoint`, `check-now-current`,
`check-issue-index-append-only`, `check-pr-size`, `check-agent-record`,
`check-public-doc-tables`, `check-symbol-anchors`, `check-test-registration`,
`check-surface-coverage`, `check-conflict-markers`, `check-env-doc`,
`check-readme-structure`, `check-role-discipline`, `check-prompt-contract`,
`check-supported-models`, `check-quickstart-recipes`, `check-site`,
`check-model-checklist`, `check-fusion-consistency`,
`check-runner-routing-consistency`, `check-oracle-pins`, `check-snapshot-pins`,
`check-gate-commands --check` -- 25 of 25 rc 0. The range gates EXECUTED against
the real landing tree rather than skipping on a stale base.

**Two known non-findings are recorded rather than left implicit.**
`scripts/check-windows-portability.py` exits 1 on this tree AND at `db648fb88`
with the identical message, `CMakeLists.txt: MSVC /W4 /WX policy is negated on
the C/C++ compile by /w`; this row does not touch `CMakeLists.txt`.
`scripts/agent-preflight.sh` reported `test_cpu_x86_llamacpp_floor` failing at a
1-minute load average of 30.96 on a 20-core box shared with another session's
build; re-run at load 15.21 it is 10 of 10 `OK`, exit 0. That is the load
sensitivity #618 records, not a regression.

### Why each default has its value

**The bound is `max_model_len * MaxTokenBytes()` and not a round number.** It is
the largest prompt that could still fit in the model's context, so it is the
tightest bound that provably refuses nothing `ValidatePromptLen` would have
accepted. Anything smaller is a policy choice that changes behaviour; anything
larger is inert on the generate paths. At a 40,960-token context it is 10 MB on
a Qwen3.6-class checkpoint and 1.9 MB on a Mistral-class one, against httplib's
100 MB.

**`MaxTokenBytes()` reads the STORED text, not the decoded text.** The stored
form is never shorter, so the bound is never tighter than the true one. An
under-estimate would refuse a servable prompt, which is the one failure this
design cannot have.

**The refusal is 400 `BadRequestError`.** The register of every sibling refusal
in the same file, and the status `InputValidationError` already maps to for the
post-encode token refusal. 413 was considered and not taken: this is not a
payload-size limit at the transport, it is a request-validation refusal, and it
must carry the OpenAI `ErrorResponse` body an SDK reads.

**It is not configurable.** Argued in `## Design`. This diverges from vLLM's
register for its own analogues, and the divergence is recorded rather than
hidden: vLLM's numbers are arbitrary policy and must be tunable; ours is derived
and has nothing to tune.

### What was rejected

- **A raw request-BODY byte bound.** Earlier still, and it would bound the JSON
  parse too — but `/v1/chat/completions` carries inline base64 media that is
  never tokenized as text, so a text-derived bound would refuse legitimate
  multimodal requests and a bound loose enough not to would not bound the text.
- **Truncation.** Forbidden by the row's first binding constraint, and the gate
  detects it (above) rather than trusting the prohibition.
- **`ValidatePromptLen`.** Forbidden by the second, and it needs the token count
  the expensive step produces.
- **A number transcribed from vLLM.** `h11_max_incomplete_event_size`'s 4 MB was
  the tempting one. It bounds the header block, not the body, and taking it
  would have been a constant with a citation that does not support it.

### Limitations, disclosed rather than closed

1. **No end-to-end measurement of what the bound saves.** After `67823aee2` a
   64 KB prompt encodes in tens of milliseconds, so the guard protects against a
   regression and against a 100 MB body rather than against today's measured
   cost. No throughput or latency figure is claimed, and `docs/BENCHMARKS.md` is
   therefore not edited.
2. **Four routes and the C ABI are unbounded**, listed under `## Owed` with the
   one that has a direct vLLM mirror to port named.
3. **The chat arm measures the summed message text, not the rendered prompt.**
   Also under `## Owed`.

### Amendment 2026-09-23: what was decided, and why

Three fresh reviews ran before this landed. Review 1 found that removing the
fixed default left `tools`, `tool_calls` arguments and template framing unbounded
before the encode, because the #1541 guard summed only message content. A 4 MiB
tool description was fully tokenized, and 64 MB of tools costs about 63 s of
encode on one worker. The repair moved the bound onto the rendered prompt, where
vLLM's `_text_len_check` runs. Review 2 found the ordering before the beam and
multimodal encodes untested, and gates now cover both. Review 3 passed.

Rejected: keeping any fixed character default. vLLM has none, and the derived
bound already refuses a pathological body before tokenization. Also rejected:
making `VT_SERVER_MAX_NEW_TOKENS` part of this change. It clamps output tokens
rather than refusing input, so it is a different mechanism with its own issue.

The one stated exception to "a prompt that fits is never refused" is a
SentencePiece tokenizer with `fuse_unk` and no byte fallback. vLLM's bound shares
it. The multimodal seam re-renders with up to 43 more bytes per image part than
the checked string. Both are recorded under the amendment's design notes.
