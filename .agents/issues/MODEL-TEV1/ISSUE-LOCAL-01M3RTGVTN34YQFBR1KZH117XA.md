ID: ISSUE-LOCAL-01M3RTGVTN34YQFBR1KZH117XA
Title: Tokenizer eos_token is not a stop token when config.json lacks it and no generation_config.json ships
Row: MODEL-TEV1
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

Upstream resolves the primary eos from the tokenizer (renderers/base.py:310-317 get_eos_token_id returns tokenizer.eos_token_id, i.e. tokenizer_config.json eos_token) and adds generation-config ids (sampling_params.py:629-653; try_get_generation_config falls back to GenerationConfig.from_model_config, config.py:1074-1090 @ 5559679229). InputProcessor (src/vllm/v1/engine/input_processor.cpp:39-83) takes config.json eos_token_id first, then tok::Tokenizer::EosId(), which reads only the tokenizer.json post_processor; tokenizer_config.json eos_token never reaches the stop set. togethercomputer/Tev1-0.8B-experimental @6bb2dff1 and Tev1-4B-experimental @0b7becf0 ship no generation_config.json and tokenizer_config eos_token <|im_end|> (248046), so /v1/chat/completions at max_tokens=8 returns 'A<|im_end|>\n<|endoftext|>...' with finish_reason=length instead of 'A' with stop. Measured on this branch, CPU, 2026-09-30. Workaround: send stop_token_ids [248046]. The fix changes the stop set of every model, so it needs its own spec and review.

## Resolution

-
