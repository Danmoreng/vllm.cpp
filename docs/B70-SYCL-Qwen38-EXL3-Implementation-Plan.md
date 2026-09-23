# Implementation Plan: Native B70 EXL3 Path

**Target revision:** vllm.cpp `9e63db5dd33b35e7cc57d0f0e80fe6c7d5ababa6`<br>
**Target model:** `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`, revision `19441ac874c4018295da848e250f23511361cda4`<br>
**Hardware:** single Intel Arc Pro B70; SYCL/Level Zero; compressed weights kept resident on-device.<br>
**Status:** implementation work plan; no assumption that the required GPU kernels already exist. Source references `[Lxx]` refer to the source index in the technical report.

## 1. Working Rules and Target States

The first milestone is a **correct text-only forward pass followed by decode**, batch size 1, short context, BF16 residual/KV path, greedy sampling, with MTP and vision disabled. It must support every EXL3 bit width actually used by the checkpoint. A production Python configuration that selects FP16 must not be treated as proof that an equivalent end-to-end C++ path already exists.

The second milestone is **fast decode and prefill** using separate EXL3 strategies, chunked GDN, and native attention. The third milestone adds **FP8 KV cache, long contexts, batch size 4, and serving parity**. Graph capture and MTP come only after memory and state correctness have been demonstrated. Vision is a separate expansion path.

Every pull request must include a regression test, a shape/dtype trace, and a list of newly reachable operators. An operator registration is not considered complete until the real model path reaches it correctly. For device USM, keep `DeviceMemoryIsHostAddressable=false`. Any unexpected CPU reference fallback is a release-blocking failure. When native XPU execution is explicitly selected, missing hardware must produce a clear failure rather than silently falling back to CPU. [L09–L14]

### Dependencies

```text
PR00 → PR01 → PR02
               ├→ PR03 EXL3 ─────────────┐
               ├→ PR04 Conv/GDN ─────────┼→ PR06 Text MVP
               └→ PR05 Attention ────────┘       │
                         ┌───────────────────────┼─────────────────────┐
                         ↓                       ↓                     ↓
                   PR07 Decode             PR08 Prefill          PR09 Chunk-GDN
                         └───────────────────────┼─────────────────────┘
                                             PR10 FP8/Attention
                                                   │
                                             PR11 Serving
                                               ┌───┴───┐
                                               ↓       ↓
                                          PR12 Graph PR13 MTP
                                                       │
                                                PR14 Vision optional
```

PR12 and PR13 do not have to be strictly serial: MTP can first be validated in eager mode. The final MTP graph path requires both. GPTQ is not part of this critical path.

## 2. Pull Requests and Acceptance Criteria

### PR00 — Reproducible Baseline, Header Inventory, and Op Trace

**Priority:** P0. **Complexity:** medium. **Dependencies:** none.

Recreate the full target checkout and donor dependencies from the pinned revisions. Run the supplied audit. In addition, read the real Safetensors headers from the model shards and generate a loader plan with byte counts split into text, MTP, vision, embedding, head, and remaining weights. Resolve the 516 tensors that are not yet described in detail; verify markers and actual MTP dtypes. Treat index/header inconsistencies as errors, not as silent defaults. [L01–L03, L25]

**Files:** existing Safetensors/EXL3 loaders; a new read-only inventory utility; `src/vt/op_provider.cpp` for a machine-readable native/fallback trace; a separate microbenchmark configuration containing the 401 matrix descriptors.

**Acceptance:** all model and source pins are recorded; the 3/4/5/6-bit distribution is confirmed or any header deviation is explained; no automatic vision staging in the text MVP; inventory generation does not require runtime execution. The wheel/source discrepancy is either resolved or explicitly documented as remaining work.

### PR01 — XPU Build, Device Context, Memory, and Queues

**Priority:** P0. **Complexity:** high. **Dependencies:** PR00.

Introduce `VLLM_CPP_XPU` as an isolated CMake option. Add `src/vt/xpu/xpu_backend.cpp`, `xpu_common.h`, and `src/vllm/platforms/xpu.cpp`. Implement one device context per GPU, device USM, VT queue mapping, pinned host allocation, and event ownership. Use a single in-order queue as the initial reference path; keep additional streams disabled until correctness is established. `MemoryInfo` must accurately distinguish total, budgeted, and allocated bytes. [L10–L11]

**Acceptance:** Alloc/Copy/Memset/Free tests in both transfer directions; two independent queues; explicit cross-queue events; memory is released only after its final use; a clear error is produced when no GPU is present. Log device ID, driver, runtime, compiler, subgroup support, and supported matrix combinations. Run one small matrix kernel on the B70. CPU-only builds remain unchanged and green. Do not claim graph or compressed-state capabilities yet.

### PR02 — Core Operators, Dtypes, and Greedy Sampling

**Priority:** P0. **Complexity:** high. **Dependencies:** PR01.

Add `xpu_elementwise.cpp`, `xpu_norm.cpp`, `xpu_sampling.cpp`, and register them through `xpu_ops.cpp`. Required functionality includes F16/BF16/F32 casts, strided copy, row gather/scatter, embedding, residual/add, standard and zero-centered RMSNorm, SiLU/MoeSiluMul, sigmoid gating, and matmul variants for unquantized residual/BA weights. Add greedy argmax with a small host readback. [L07, L09, L14]

**Acceptance:** cover dtype/stride combinations observed in the model trace; tail lengths; alias-safe in-place variants where required by the operator contract; RMSNorm semantics for `weight` versus `1+weight`; NaN/Inf handling and tie behavior matching the existing operator contract. Device buffers must never be dereferenced by the host. Do not make full-logit GPU-to-host downloads the default path.

### PR03 — EXL3 Correctness: mul1, Had128, and Baseline GEMM

**Priority:** P0. **Complexity:** high. **Dependencies:** PR02.

Add `xpu_exl3.cpp`, initially without XMX-specific tuning. Port mul1, the cyclic codeword reader, and logical tile permutation from the CPU reference path. Support 3/4/5/6 bit as mandatory; a generic 1–8-bit path is preferable. Implement FP16 input scaling, Had128, FP32 accumulation, output Hadamard, and F16/F32 output according to the existing contract. Do not enable an unmeasured reconstruct dispatch yet. [L04–L06]

**Acceptance:** validate all 65,536 mul1 values on GPU against CPU; bitstream tests for every required bit width and wrap boundary; tile permutation tests; uploads from intentionally unaligned host-mmap addresses; small synthetic matrices and real K/N shapes. Test Hadamard separately. Compare F16 output and F32→BF16 output independently. Do not create a fully dequantized BF16 model copy. Any failure must identify the exact matrix.

### PR04 — Conv1d and GDN: Complete Native Reference Path

**Priority:** P0. **Complexity:** very high. **Dependencies:** PR02; integration also depends on PR03.

Add `xpu_gdn.cpp`. Implement CausalConv1dFwd and Update, GdnPostConv, GdnDecode, a simple GdnPrefill, RmsNormGated, and state gather/scatter. Keep the SSM state in FP32. The first prefill implementation may be serial but must be correct; it is explicitly not the final performance implementation. Conform to VT operator contracts and do not directly apply raw Intel GDN functions to VT inputs that have already been normalized/transformed. [L14–L17]

**Acceptance:** support 10,240 convolution channels; kernel width 4; state stores raw history; Hv/Hk=3; variable sequence lengths; negative/null slots; permuted requests. Compare full prefill against split prefill plus decode. Test lengths 1/2/3/4/63/64/65. Validate both outputs and every final state tensor. No double L2 normalization and no duplicate beta/decay transformation.

### PR05 — Full Attention and Native BF16 KV Cache

**Priority:** P0. **Complexity:** high. **Dependencies:** PR02; integration also depends on PR03.

Add `xpu_attention.cpp` implementing Q/gate split, Q/K normalization, partial RoPE, KV write, and generic paged attention. Respect the real VT layouts and query dtypes, including the possibility of a generic FP32 query path. Use FP32 online softmax, 24 Q heads / 4 KV heads, head dimension 256, scale 1/16, and 64 rotary channels. Only add a fused QK-norm/RoPE/gate operator after proving equivalence against the unfused path. [L02, L09, L14]

**Acceptance:** appended queries with existing context; correct causal alignment; multiple block sizes and non-trivial strides; batch 1/4; correct GQA mapping; block boundaries; M=1 and M>1. Do not hard-code the Python donor layout with block size 1,664. Precision changes must be documented explicitly instead of being forced through an inaccurate FA2 capability flag.

### PR06 — End-to-End Text MVP and Residency

**Priority:** P0, Milestone B. **Complexity:** high. **Dependencies:** PR03–PR05.

Connect the existing Qwen3.5 Dense/EXL3 model path to the XPU platform. Validate `needs_weight_staging`, residency, weight lifetime, and kernel reachability together. Make text-only behavior explicit and preserve the existing gather before the LM head. Add any small helper operators discovered by the first real execution trace. [L06–L12]

**Acceptance:** use the same pinned EXL3 checkpoint and tokenizer; run short prefill followed by at least 64 decode steps, greedy sampling, MTP disabled; compare intermediate states against an appropriate reference; zero CPU reference fallbacks in repeated forward passes. One-time host initialization must be reported separately. VRAM and host RSS stabilize after warmup. The LM head must actually execute through the 6-bit EXL3 path. Only after this gate should performance claims be made.

### PR07 — Optimize EXL3 Decode and Small-M Regimes

**Priority:** P1. **Complexity:** very high. **Dependencies:** PR06.

Benchmark a vectorized M=1 GEMV against very-small-M XMX kernels. Implement coalesced trellis reads, register/SLM tiling, FP32 reduction, and deterministic split-K if needed. Support M=2/4/5/8/16/20 with weight reuse. Tune Head6 and MLP3 separately. Build a strategy cache keyed by device, compiler/kernel version, bit width, K/N, M, and dtype. [L03–L06]

**Acceptance:** all numerical gates from PR03 remain intact; microbenchmark each real projection family and measure end-to-end TPOT. Choose the winning strategy per regime from repeated measurements, not from peak XMX throughput alone. Avoid an unbounded number of specialized binaries. Define regression limits in advance and report both kernel-level and model-level timings.

### PR08 — EXL3 XMX Prefill and Bounded Reconstruction

**Priority:** P1. **Complexity:** very high. **Dependencies:** PR06; PR07 strategy infrastructure is helpful.

Implement decode-plus-XMX FP16 matmul with FP32 accumulation, M reuse, and double buffering. Optionally use a bounded FP16 panel plus native GEMM as an intermediate path. Budget scratch per serialized execution context; initially investigate 8–64 MiB. Make the global M>144 decision backend-specific, or avoid registering reconstruction until it is tuned for B70. [L04–L06]

**Acceptance:** cover M=128 through 6,656, all relevant K/N shapes and bit widths, including Head6. Improve actual prompt TTFT, not only isolated GEMM time. Scratch memory must not scale with layer count or graph-slot count. Never materialize a fully dequantized model. Document trade-offs affecting decode, chunk size, and KV budget.

### PR09 — Chunked GDN on Xe2/XMX

**Priority:** P1. **Complexity:** very high. **Dependencies:** PR04 and PR06.

Port the Xe2 chunk-64 pipeline from the Intel donor implementation: Prepare, A, triangular system, W/U, Output/State. Replace Torch tensors and runtime allocations with VT views, queues, and preplanned workspaces. Keep the existing native reference path for comparison and for valid cases that are not optimized. [L15–L17, L25]

**Acceptance:** validate initial state, variable lengths, chunk boundaries, and prefill→decode continuation; measure long-sequence state drift; do not silently reduce FP32 state precision. Workspace allocation must be bounded and persistent. Measure prefill and full-model TTFT on short and long prompts; document chunk-size and subgroup choices. Do not assume donor-wheel parity.

### PR10 — FP8 KV, Split-KV Decode, and Prefill Fast Paths

**Priority:** P1, essential for production-scale context lengths. **Complexity:** very high. **Dependencies:** PR05–PR06.

Implement the complete E4M3 KV path from write/scale through attention read. Use the Q128/Xe2 donor code only as a starting point; all cache strides and sequence bounds must be explicit. Add stable split-KV reduction for long decode contexts and reuse for GQA. Also accelerate small M=2–5 without excluding batch 4 or tail groups. [L14, L18–L19]

**Acceptance:** independently test FP8 encode/decode and scaling; validate against the actual VT KV layouts; compare against a BF16 reference and run long-context quality checks. Validate 32k first, then 128k/200,704 only after proving the memory budget. Eligibility violations must fall back to a correct generic XPU kernel, never to CPU and never to an incompatible layout. Report cache bytes, quantization error, TPOT, and TTFT.

### PR11 — Batch, Prefix/State Lifecycle, and Sampling

**Priority:** P1. **Complexity:** high. **Dependencies:** PR06; PR10 for long-context serving.

Support up to four requests with admission, completion, reordering, mixed prefill/decode, and reserved KV budgets. Validate prefix and GDN/Conv snapshot lifecycles against the manual production patches. Add all sampling behavior required by the target serving path, including seed/penalty semantics and small token readback. [L01, L20–L24, L28]

**Acceptance:** request reordering during asynchronous accepted-token transfers; no duplicate gather; previously published prefix states remain unchanged; cancel/resume and empty padded rows work correctly. Validate top-k/top-p behavior statistically. Admission control must prevent four maximum-length contexts from overcommitting memory. Throttled or rejected requests must fail cleanly rather than ending in GPU OOM.

### PR12 — Graph Capture and Launch/Cast Fusion

**Priority:** P2. **Complexity:** high. **Dependencies:** stable optimized eager path; PR11.

Implement backend graph APIs plus runner integration. Add warmup, stable metadata buffers, defined shape buckets, and bounded graph/scratch resources. After measurement, fuse input cast/Hadamard, output Hadamard/scale/cast, and residual/norm. Keep projections with different `suh` transformations separate. [L06–L10; W3 in the report]

**Acceptance:** eager and graph outputs/states are equivalent; support changing lengths and state indices; two execution slots run without scratch races; graph capture warmup does not repeat unnecessarily. Host-side or direct Level Zero work must not be assumed to be captured automatically. Document graph memory usage and the steady-state memory plateau. Do not claim speedups without end-to-end measurements.

### PR13 — EXL3 MTP: First One, Then Four Draft Tokens

**Priority:** P2. **Complexity:** very high. **Dependencies:** PR06 and PR11; performance paths PR07/PR10; graph operation also depends on PR12.

Reuse the existing EXL3 MTP loader/forward path. Implement native SpecConv, GdnSpecDecode, accepted-token handling, state snapshots, and rejection sampling. The 39 MTP headers identified in PR00 determine any additional kernel/dtype requirements. Do not automatically reuse GPTQ draft INT4 modules. [L14, L20–L24]

**Acceptance:** accepted draft-token counts 0–4, API counts 1–5, rejection at every position, EOS handling, shortened final groups, request switching, and prefix rewind. Validate not only tokens but also Conv/SSM/KV state and cache publication. Benchmark M=5 and aggregated M=20. Measure benefit as accepted output tokens per second including draft cost; use the safe non-speculative path when acceptance is poor.

### PR14 — Vision/MRoPE as a Separate Expansion

**Priority:** P3, optional. **Complexity:** high. **Dependencies:** stable text path; real vision headers from PR00.

Inventory the existing Qwen vision integration and add missing native vision operators. Investigate image preprocessing, vision projections, patch/merge layout, multimodal positions, and memory budget. The model index containing vision tensors does not prove that a working multimodal runtime path already exists. [L02–L03]

**Acceptance:** image/text prompts with verified MRoPE positioning, correct image boundaries, vision-enabled peak-memory measurements, and no regression in the text path. Only then include the production image option in the parity scope. Do not make a GPTQ conversion a hidden prerequisite.

## 3. Test Matrix for Codex and CI

| Test group | Exact focus | Gate |
|---|---|---|
| Format | 65,536 mul1 values; bits 1–8; cyclic windows; tile permutation | bit-exact |
| Had/Matmul | F16 pre-scaling; F32 accumulation; output casts; real K/N; M tails | documented reference and tolerances |
| Conv/GDN | raw history; 63/64/65; initial state; slot masks | output and final state |
| Attention | Q24/KV4/D256; RoPE64; query/KV strides; scale; causal offset | reference plus long-context validation |
| Memory | weight staging; queue-free; panel reuse; graph slots | no races, no unbounded growth |
| Serving | batch reordering; prefix rollback; cancel; EOS | no request/state mixing |
| MTP | rejection positions; tail context; 0–4 accepted drafts | correct state and useful throughput |

Do not hard-code one universal numerical tolerance for the first acceptance pass. Decoder values can be tested exactly; FP32 GEMM reductions and long recurrent states require normalized error metrics, absolute-error checks near zero, and comparisons over sequence progression. Accuracy acceptance and performance optimization must be documented separately.

## 4. Measurement Protocol and Required Deliverables for Each Performance PR

Every result set must include at minimum: model/kernel commit hash, compiler, driver, GPU ID, power configuration, dtypes, context length, batch size, MTP status, graph status, shape/bit width, warmup count, and repetition count. Record kernel time, TTFT, TPOT distribution, output tokens/s, native/fallback op counts, VRAM high-water mark, and persistent plus temporary workspace usage.

For MTP, additionally record draft time, verification time, and acceptance rate. For FP8, record scaling strategy and quality comparison. For XMX tuning, record evidence of matrix-instruction use, SLM usage, register pressure, and spills. Compare runs using identical prompts and sampling parameters. Label GPTQ production measurements separately as a different quantized model path.

## 5. What Should Explicitly Not Be Implemented First

Do not create a new Qwen architecture branch solely because of the 3.8 naming. Do not keep a fully dequantized EXL3→BF16 model in VRAM. Do not build an INT4-only kernel. Do not literally port the B70 attention wrapper with its special block size. Do not switch GDN state to BF16 without validation. Do not combine MTP, graph, or vision work into a large patch before the first correct text path. Do not derive performance numbers from hardware TOPS or code comments.

**Initial Codex task:** implement PR00 and PR01 first, with the deliberately narrow objective of establishing a reproducible B70 toolchain, an accurate backend memory contract, and a verified matrix/queue probe. Then implement core ops plus EXL3, GDN, and attention as separate, reference-tested paths. The first shared merge milestone is PR06, not an isolated GEMM benchmark.
