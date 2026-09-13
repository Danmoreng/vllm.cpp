ID: ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142
Title: QsaGatherAttentionKernel is 34.5% of decode GPU time at 2.545 ms a launch, and at decode it runs 24 blocks on 48 SMs
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

MEASURED on `dgx:gpu0` (GB10 sm_121a) 2026-09-13, `nsys` over a 60 s window inside a 3000-token decode at `ee0644eab` (W6+W7 landed, step 77.8 ms, 12.85 tok/s): `QsaGatherAttentionKernel` is **34.5% of all GPU kernel time** -- 16.079 s across 6,319 instances, mean **2.545 ms**, median 2.555, min 1.268, max 4.987. That is ~8.2 instances and **~20.4 ms per decode step**, and it is the largest kernel by a factor of **ten** over the next one's mean (cuBLAS `gemvx` at 242 us).

THIS IS A NEW #1 AND IT WAS INVISIBLE BEFORE. In the pre-W6 trace the same kernel was 2.7% at 231 us, buried under 2.87 s/step of allocator churn. W6 removed the allocator and W7 removed `HcGroupedNormKernel` (which was 40.7%), and this is what the profile shows underneath. The next row was going to be the dense GEMV path on the strength of the OLD ranking; scoping from a stale profile would have aimed the work at the wrong kernel. Re-rank before scoping, and re-rank again after this lands.

THE STRUCTURAL OBSERVATION, which is the same shape W7 had. `cuda_qwen4_exp_qsa.cu:609-612` launches `grid = pairs < 4096 ? pairs : 4096` with `width = BlockWidthFor(DH)`. At decode `T = 1`, `pairs = T * HQ`, and the released Qwen3.8-Flash-Next has `num_attention_heads = 24` (`qwen4_exp.h:216`) with `head_dim = 128` (`:45`). So the decode launch is **24 blocks of 128 threads on a 48-SM GB10** -- half the SMs idle before the kernel does any work. W7's predecessor was four threads on 48 SMs; this is 24 blocks, less extreme but the same family, and the W7 result (435.7 us -> 16.0 us per launch once the work was spread) is the reason to look here first.

WHAT IS NOT YET KNOWN, and must not be assumed. Whether the 2.545 ms is dominated by (a) the grid being narrower than the machine, (b) the per-block walk over selected KV blocks, (c) the gather's memory pattern, or (d) something in the paged path that the fixture never exercises. The instances-per-step figure (~8.2) also implies only about 8 of the 48 layers are `kQwenSparseAttention` on this checkpoint; confirm that against `layer_types` before sizing any fix, because a per-layer cost is a different target from a per-head one.

OWED BEFORE A FIX IS SCOPED: an `nsys`/Nsight-Compute attribution of where the 2.545 ms goes inside the kernel, on `dgx:gpu0`, against the CURRENT 77.8 ms step. Do not scope from this issue's paragraph alone -- that is the mistake this row already made once, when `~1,400 kernel launches` was carried as a cause for months without anyone counting them.

## Resolution

-
