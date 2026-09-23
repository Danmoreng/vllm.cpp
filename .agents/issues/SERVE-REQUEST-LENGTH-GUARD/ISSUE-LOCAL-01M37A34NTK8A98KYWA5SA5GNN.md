ID: ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN
Title: serve: the fixed 200,000-character chat prompt cap refuses prompts the context can hold, with HTTP 500
Row: SERVE-REQUEST-LENGTH-GUARD
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

OpenAIServingChat::create_chat_completion (src/vllm/entrypoints/openai/serving_chat.cpp) refuses any rendered chat prompt above VT_SERVER_MAX_PROMPT_CHARS, which defaults to 200000 characters. The number is not derived from max_model_len, so a server started with --max-model-len 262144 refuses ordinary English prompts above about 48k tokens (a 64k-token prompt is about 268k characters). The refusal is a std::runtime_error, so ApiServer::handle_chat_completions answers HTTP 500 InternalServerError, and the message blames an unrelated client (Hermes). Observed on dgx:gpu0 on 2026-09-23: 'prompt too large for this server (268439 chars > VT_SERVER_MAX_PROMPT_CHARS=200000)'. vLLM at the pin e126687a9a bounds prompt text before tokenization only by max_input_tokens * tokenizer.max_chars_per_token (vllm/renderers/params.py:342-365) and refuses with VLLMValidationError, which is HTTP 400 (vllm/entrypoints/serve/exception_handling/error_response.py:39-41). The derived pre-tokenization bound this row already landed (ApiServer::refuse_oversized_prompt, max_model_len * MaxTokenBytes) is the equivalent. The fixed cap must not refuse a prompt the configured context can hold.

## Resolution

-
