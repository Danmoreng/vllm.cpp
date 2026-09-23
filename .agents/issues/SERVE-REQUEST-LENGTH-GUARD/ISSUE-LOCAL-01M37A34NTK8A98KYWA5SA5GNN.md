ID: ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN
Title: serve: the fixed 200,000-character chat prompt cap refuses prompts the context can hold, with HTTP 500
Row: SERVE-REQUEST-LENGTH-GUARD
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

OpenAIServingChat::create_chat_completion (src/vllm/entrypoints/openai/serving_chat.cpp) refuses any rendered chat prompt above VT_SERVER_MAX_PROMPT_CHARS, which defaults to 200000 characters. The number is not derived from max_model_len, so a server started with --max-model-len 262144 refuses ordinary English prompts above about 48k tokens (a 64k-token prompt is about 268k characters). The refusal is a std::runtime_error, so ApiServer::handle_chat_completions answers HTTP 500 InternalServerError, and the message blames an unrelated client (Hermes). Observed on dgx:gpu0 on 2026-09-23: 'prompt too large for this server (268439 chars > VT_SERVER_MAX_PROMPT_CHARS=200000)'. vLLM at the pin e126687a9a bounds prompt text before tokenization only by max_input_tokens * tokenizer.max_chars_per_token (vllm/renderers/params.py:342-365) and refuses with VLLMValidationError, which is HTTP 400 (vllm/entrypoints/serve/exception_handling/error_response.py:39-41). The derived pre-tokenization bound this row already landed (ApiServer::refuse_oversized_prompt, max_model_len * MaxTokenBytes) is the equivalent. The fixed cap must not refuse a prompt the configured context can hold.

## Resolution

2026-09-23: fixed on branch row/SERVE-REQUEST-LENGTH-GUARD-CHARCAP at 1cb56af24. VT_SERVER_MAX_PROMPT_CHARS now defaults to unset (no fixed cap); the pre-tokenization bound on the default path is the derived max_model_len * MaxTokenBytes() in ApiServer::refuse_oversized_prompt, the equivalent of vLLM's max_input_tokens * max_chars_per_token (vllm/renderers/params.py:342-365 at e126687a9a). An explicit value refuses with HTTP 400 BadRequestError via InputValidationError. Red at efe6ed9b6: 53 assertions, 11 failed (204,800-byte 25-token chat prompt answered 500). Green at 1cb56af24: 55 assertions, 0 failed. Deletion mutation in a scratch worktree: 11 failed; file restored byte for byte (sha256 866b2bb7d18151190a7b64a91630141b9132599e0366e8f3012aa995dc99321e both sides). Evidence in .agents/specs/serve-request-length-guard.md, 2026-09-23 amendment.

2026-09-23, review repair: the fresh review of b83feab14 found that the content-only #1541 guard is not vLLM's equivalent, because vLLM's _text_len_check measures the RENDERED prompt (vllm/renderers/params.py:342-370,386-399 at e126687a9a), so bytes in tools, tool_calls arguments and template framing reached the encode unbounded. Fixed at 19b2bb3fc: create_chat_completion refuses a rendered prompt above max_model_len * MaxTokenBytes() (one derivation, Tokenizer::MaxPromptBytes) with HTTP 400 before any encode, which also reaches vllm_chat, vllm_chat_stream and RunBatch. Red at 6278e226d: 45 assertions, 12 failed, a 4 MiB tool description reaching the encode (InputProcessor::num_prompt_encodes() == 1). Green at 19b2bb3fc: 129 assertions over the three api-server cases and 10 in the C ABI case, 0 failed. Seven mutations in a scratch worktree, all killed, file restored to sha256 0ddc717e2315453462fca8032f6815b6be015f9c85a76d30c081fec777205a8f each time. Evidence in .agents/specs/serve-request-length-guard.md, "Review repair 2026-09-23".
