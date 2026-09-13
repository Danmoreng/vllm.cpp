ID: ISSUE-LOCAL-01M2EAQ6R63BSRF1GVZ3JAR4A4
Title: The OpenAI chat path re-parses the Jinja chat template on every request, which is most of the short-prompt TTFT gap to exllamav3 on GB10
Row: SERVE-CHAT-TEMPLATE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

`MakeChatTemplatePromptFn` (`src/vllm/entrypoints/chat_template.cpp:302`) captures the template STRING, and every request calls `apply_chat_template`, which runs `minja::Parser::parse(template_str, ...)` (`chat_template.cpp:144`) before it renders.

Measured on the Qwen3.8-27B EXL3 target's own `chat_template.jinja` (8952 chars), rendering the 592-char prompt the variadic benchmark's S band sends:
- On an AMD Ryzen 9 9950X3D at -O2, parse took 45.2 ms and render took 0.10 ms per request (warm, mean of 15).
- On dgx:gpu0 (GB10), the server's `--verbose` stage log put HTTP ingress to `stage=templated` at a median of 130 ms, with some requests at 180-207 ms, over 10 warm requests at c = 1. The engine part (queued to first SSE token) was 533 ms. Evidence is in `/workspace/exl3-short-ttft/stages-*/` on the rc share.

Against exllamav3 on the same prompt and box, our TTFT p50 is 669 ms and theirs is 595 ms (`/workspace/exl3-short-ttft/20260913-205841/`). The re-parse is more than that whole 74 ms gap.

Upstream compiles a chat template once and caches it. transformers' `_compile_jinja_template` is `lru_cache`d, and vLLM's renderer resolves the template once per model (`vllm/renderers/hf.py`). The fix is to parse once, when the prompt fn is built, and render the cached `TemplateNode` per request. That requires minja's render to be safe to call concurrently on one shared root with per-call contexts, which has to be verified in `third_party/minja/minja.hpp` rather than assumed.

## Resolution

-

## Progress 2026-09-13: parse once per prompt fn (branch row/SERVE-CHAT-TEMPLATE-PARSE-CACHE, not yet reviewed)

`MakeChatTemplatePromptFn` now parses the template once when it is built and renders the shared parsed tree per request. A template that does not parse still fails per request, with the same `ChatTemplateError` text a fresh parse gives, because transformers compiles inside `apply_chat_template` and upstream reports a broken template per request. `apply_chat_template(template_str, ...)` keeps its behaviour and still parses on each call.

**Thread safety.** A minja render does not write to the tree. `TemplateNode::do_render` and `Expression::do_evaluate` are `const` (`third_party/minja/minja.hpp:865`, `:656`), and no node has a `mutable` member. Every name a render binds goes into the per-call `Context` or a `Value` it owns: `SetNode` `:1149` and `:1152`, `ForNode` `:1010-1013`, `MacroNode` `:1106`, `SetTemplateNode` `:1166`, `CallNode` `:1736`. `LiteralExpr` (`:1190`) holds only scalars (`parseConstant`, `:1891-1912`) and returns them by value. List and dict literals build a new `Value` on each evaluation (`:1204`, `:1219`). `Context::builtins()` builds a new globals object on each call (`:2818`).

**Tests** (`tests/vllm/entrypoints/test_chat_template.cpp`): one parse across 18 renders through one prompt fn, observed through the `ChatTemplateParseCountForTesting()` seam; before the fix this read 18 == 1 and failed. Byte equality against the fresh-parse path for the Qwen3.8 fixture (six shapes: reasoning default, on, off with an effort, tools, multi-turn with a tool call, its result and prior reasoning, and no generation prompt), the Qwen3.5 and Gemma4 fixtures and the two in-file templates. Per-request parse errors equal to a fresh parse's. Eight threads rendering different requests through one prompt fn. Mutation: re-parsing per call in the prompt fn read 19 == 1 and failed; restored byte for byte (sha256 checked).

**Measurement** (AMD Ryzen 9 9950X3D, CPU-only Release `-O3`, same machine and binary recipe; the real 8952-char Qwen3.8 template and the 592-char S-band prompt with `enable_thinking=true`; the prompt fn called 35 times, mean of the last 30). Before: 46.19 ms and 48.03 ms per request over two runs. After: 0.031 ms and 0.041 ms per request, plus one 50 ms parse when the prompt fn is built. The rendered bytes of six shapes are identical before and after.

**Sanitizers.** `test_chat_template` passes 41 of 41 under `-DVLLM_CPP_SANITIZE=thread` (run with `setarch -R`, because TSan otherwise aborts on this kernel's mmap layout) with no ThreadSanitizer report, and under `address,undefined` with no report. TSan is live on this test: a scratch mutation that added an unsynchronised static counter to the render path produced `WARNING: ThreadSanitizer: data race` in `RenderChatTemplate`; the file was restored byte for byte. `test_chat_prompt`, `test_capi`, `test_chat_mm`, `test_openai_api_server` and `test_openai_serving` also pass on the CPU Release build.
