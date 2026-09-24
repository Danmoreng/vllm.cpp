ID: ISSUE-LOCAL-01M37A490GB73VNYFE7NBEA3EV
Title: qwen3 GGUF: port LoadMerged to keep homogeneous merged weights quantized
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

The qwen3 GGUF loader (qwen3_gguf_weights.cpp:471) forces ALL merged weights through NoKeepQuant (bf16 expansion), even homogeneous gate_up_proj (Q4_K+Q4_K). This causes 3/6 token mismatches vs llama.cpp in the Q4_K_M gate. The LoadMerged pattern in muse_glimmer_gguf_weights.cpp:255-345 already handles this correctly: homogeneous shards stay on keep-quant (block-level byte concat, native quant GEMM), only heterogeneous shards expand to bf16. Port this pattern to the qwen3 loader.

## Resolution

Falsified by the tree. The `LoadMerged` port from `muse_glimmer` was
implemented on `row/BACKEND-GATE-ROCM-SGLANG-q4-keepquant`, tested, and
abandoned. `LoadMerged` keeps homogeneous shards on the keep-quant path,
but vLLM (the PRIMARY oracle) always expands merged weights to bf16 for
Q4_K_M, so `LoadMerged` makes vllm.cpp diverge from the primary oracle.
The result was still 3/6 matches — the mismatch pattern shifted but the
count did not improve. See commit `d1e26c9cb` and the Evidence section of
`.agents/specs/qwen3-gguf-arch-support.md` for the full analysis.

The 3/6 mismatches are not a bug in vllm.cpp's loader. They are a property
of Q4_K_M's heterogeneous quantization (Q4_K + Q6_K), which forces bf16
expansion for merged weights in both vLLM and vllm.cpp. The gate
denominator is vLLM eager-mode, not llama.cpp.
