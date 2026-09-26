ID: ISSUE-LOCAL-01M375BKV9KF9NP6W1SK1Y2GSW
Title: qwen3_5_gguf_weights defaults rotary_dim to 0 when GGUF lacks rope.dimension_count
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp line 1015 defaults c.rotary_dim to 0 when the GGUF file omits rope.dimension_count. llama.cpp's converter omits this key when it equals head_dim (full rotary). A rotary_dim of 0 crashes RoPE. The same bug was already fixed in qwen3_gguf_weights.cpp (commit 363bfbe92), but qwen3_5_gguf_weights.cpp was not updated.

## Resolution

Fixed in the same change that filed this issue. The loader now defaults
`rotary_dim` to `c.head_dim` when `rope.dimension_count` is absent, mirroring
the qwen3 fix (commit `363bfbe92`). A test case verifies the default. The fix
is on `row/BACKEND-GATE-ROCM-SGLANG-qwen35-rotary-dim`.
