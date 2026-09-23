ID: ISSUE-LOCAL-01M37A3S7N7GSZC37JH515QXFE
Title: serve: VT_SERVER_MAX_NEW_TOKENS defaults to a 4096-token output ceiling that vLLM does not have
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

OpenAIServingChat::create_chat_completion (src/vllm/entrypoints/openai/serving_chat.cpp) clamps a chat request's positive max_tokens / max_completion_tokens to VT_SERVER_MAX_NEW_TOKENS, which defaults to 4096 when unset, and does so silently: the client asks for 8192 and receives finish_reason length at 4096. /v1/completions has no such clamp. vLLM at the pin e126687a9a has no default output ceiling: get_max_tokens (vllm/entrypoints/serve/utils/api_utils.py:169-206) returns min(max_model_len - input_length, the request max_tokens or the generation-config default, override_max_tokens, the platform limit), where override_max_tokens is set only by the operator through --override-generation-config max_new_tokens or a generation config (vllm/entrypoints/openai/chat_completion/serving.py:172-176, :312-320). The mechanism differs from the prompt-character cap (a silent clamp on output, not a refusal of input), so it is filed separately from ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN. Related: ISSUE-GH-544 asks for the ceiling to also apply when max_tokens is unset, which is how vLLM's override_max_tokens behaves; the defect here is the non-zero DEFAULT, not the operator ceiling.

## Resolution

-
