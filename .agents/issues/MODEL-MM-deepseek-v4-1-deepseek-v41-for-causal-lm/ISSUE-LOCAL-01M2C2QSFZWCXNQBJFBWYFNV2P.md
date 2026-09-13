ID: ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P
Title: DeepSeek-V4.1 W3b: MXFP8 32x32 UE8M0 host reference
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

DeepSeek-V4.1-Flash routes a LinearBase whose weight_block_size == [32, 32] and is_scale_e8m0 to ModelOptLinearMethod(kMxfp8Static / kMxfp8Dynamic) (vllm/models/deepseek_v4_1/quant_config.py:186-201 @ e77daef89e). This tree has no MXFP8 reader at all: src/vllm/model_executor/layers/quantization/modelopt_mixed_precision.h:1087-1091 refuses the algo by name with 'this build has no MXFP8 loader'. W3b owes the portable host reference for the family, per the spec's wave table: the checkpoint-to-runtime scale row expansion (repeat_interleave(32, dim=0) BEFORE TP slicing), the runtime per-32-column dequant, the dynamic activation quantizer, the emulation linear arm, and the weight_scale / weight_scale_inv checkpoint name split.

## Resolution

2026-09-13, PARTIAL — the host reference lands; the issue stays OPEN until a
production entry point reaches it.

Landed `include/vllm/model_executor/models/deepseek_v4_1_mxfp8.h`,
`src/vllm/model_executor/models/deepseek_v4_1_mxfp8.cpp` and
`tests/vllm/models/test_deepseek_v4_1_mxfp8.cpp`. The five pieces are
`ExpandMxfp8CheckpointScale`, `DequantMxfp8ToF32` / `DequantMxfp8ToBf16`,
`QuantizeMxfp8E4m3`, `Mxfp8LinearEmulation` and `Mxfp8LinearScaleParamName`.

The unit gate is 13 cases / 7827 assertions green. It ports
`tests/quantization/test_fp8.py:67-238` @ vLLM `e77daef89e`, including that
test's bit-exact (`rtol=0, atol=0`) equivalence between the checkpoint-layout
32x32 dequant and the expanded per-32-column runtime dequant, over both
`tp_rank` values and the six checkpoint shapes it names.

Five mutations are killed, each with the binary proved changed: the
`repeat_interleave` stride, the runtime scale row index, the upper clamp bound,
the lower clamp, the `block_cols != 32` refusal, the bf16 narrowing in the
linear arm, the `expert_dtype` half of the name split, and the bias term. Two
survive and are recorded rather than papered over: the MULTIPLY-versus-divide
form and the `amax` TINY floor are both indistinguishable on a host without
flush-to-zero, and W5's device arm owes those measurements.

WHAT IS STILL OPEN: nothing calls any of it. `deepseek_v41` is not registered
(W1), so the wiring is owed by W8 (loader) and W4 (host forward assembly) on
this row. `.agents/specs/deepseek-v4-1-flash.md` `## Owed` carries the same
record.
