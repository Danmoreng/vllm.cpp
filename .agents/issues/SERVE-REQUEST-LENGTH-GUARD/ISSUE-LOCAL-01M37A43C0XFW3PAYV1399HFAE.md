ID: ISSUE-LOCAL-01M37A43C0XFW3PAYV1399HFAE
Title: serve: an over-long prompt is refused with the engine-layer message, not the renderer message vLLM's server returns
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

Our OpenAI server refuses a prompt longer than max_model_len tokens from InputProcessor::ValidatePromptLen (src/vllm/v1/engine/input_processor.cpp:109-127), whose text mirrors vLLM's engine-layer check (vllm/v1/engine/input_processor.py:436-476 at e126687a9a: 'The decoder prompt (length N) is longer than the maximum model length of M. ...'). Both are HTTP 400. At the pin, vLLM's OpenAI server reaches a different check first: the renderer tokenizes with max_length = max_input_tokens + 1 and TokenizeParams._token_len_check raises VLLMValidationError 'This model's maximum context length is {max_model_len} tokens. However, you requested {max_tokens or 0} output tokens and your prompt contains [at least ]{n} input tokens, for a total of [at least ]{n + max_tokens} tokens. Please reduce the length of the input prompt or the number of requested output tokens.' (vllm/renderers/params.py:436-461), where max_input_tokens = max_model_len - requested output tokens (params.py:204-210, chat_completion/protocol.py:608-626). So vLLM also refuses prompt + max_tokens > max_model_len at the server, which our server does not check before admission. Status matches; text and the output-token term do not. Found while fixing ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN; not fixed there because it changes the token refusal ValidatePromptLen owns. The pre-tokenization byte refusal (ApiServer::refuse_oversized_prompt) also answers 400 with this row's own text instead of vLLM's _text_len_check text (vllm/renderers/params.py:342-365: 'This model's maximum context length is ... your prompt contains N characters (more than M characters, which is the upper bound for K input tokens) ...'); mirror both texts together.

## Resolution

-
