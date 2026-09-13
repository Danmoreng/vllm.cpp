ID: ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA
Title: DeepSeek-V4.1 W3a: no Engram host reference exists -- the n-gram hash, the prime bucket layout, the fp8/ue8m0 table lookup and the hyper-connection gate are all unimplemented, and the ue8m0 decode the lookup needs is the BITCAST form our tree does not have
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

vLLM implements Engram at `vllm/models/deepseek_v4_1/common/engram.py` (1033 lines at `e77daef89e`): an n-gram hash over a compressed token vocabulary, two ~384M-row fp8 tables with ue8m0 per-32 scales, and a sigmoid gate that injects the looked-up rows into the [T, hc_mult, hidden] hyper-connection manifold between mhc_post and mhc_pre. This tree has no counterpart: `grep -rn Engram src/ include/` is empty. W3a of .agents/specs/deepseek-v4-1-flash.md owns the portable host reference. Two upstream constants cannot be derived here without new machinery -- the compressed token map needs NFKC/NFD/StripAccents/Lowercase normalizers our tokenizer has none of, and the hash multipliers come from NumPy PCG64 -- and the published GGUF artifact carries both as metadata, so this reference consumes them as data and validates them. The lookup also needs a ue8m0 decode by BITCAST (engram.py:613-614), which yields +0.0 at byte 0 where our arithmetic `E8M0ToF32` yields 2^-127.

## Resolution

2026-09-13, PARTIAL and the issue stays OPEN. W3a landed the Engram host reference at include/vllm/model_executor/models/deepseek_v4_1_engram.h and src/vllm/model_executor/models/deepseek_v4_1_engram.cpp: the prime bucket layout (EngramLayoutFromConfig, engram.py:178-210), the three-tier n-gram hash with its sticky blocked flag (NgramHashState, engram.py:213-236, :248-380, :439-546), the head-sharded fp8/ue8m0 table lookup (MakeEngramEmbeddingShard, LoadEngramHeadShard, EngramLookup; engram.py:549-561, :564-619, :630-748) and the injection gate (EngramPostWkv, engram.py:765-854). The BITCAST ue8m0 decode landed as a SECOND entry point, vllm::E8M0BitsToF32 in mxfp4_dequant.{h,cpp}; E8M0ToF32 and its six callers are untouched. Gate: tests/vllm/models/test_deepseek_v4_1_engram.cpp, 17 cases and 106,646 assertions green, porting tests/kernels/test_engram.py at vLLM e77daef89e including its two scalar oracles at rtol=0/atol=0. Fourteen mutations of the claimed guarantees were each detected; the three that initially SURVIVED -- a padded head on a rank that also owns rows, the pad id being the COMPRESSED id of the pad token, and the byte-0 ue8m0 decode -- were each ungated by upstream's own fixtures and each got a case upstream cannot express. STILL OPEN: nothing reaches this code from a production entry point (W1 registry, W4 forward, W8 loader), the compressed token map CONSTRUCTION and the NumPy PCG64 multiplier DERIVATION are read from the artifact rather than derived, and the private-use sentinel is documented but not gated. All four are recorded in the row spec.
