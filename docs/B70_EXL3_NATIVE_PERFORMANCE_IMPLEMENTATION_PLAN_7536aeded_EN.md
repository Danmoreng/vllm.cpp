# B70 EXL3 native performance: implementation plan after 7536aeded

**Review date:** 2026-10-04  
**Native baseline:** `7536aededc049b2157f12eadb0f4cc6bbf29b7b7`  
**Repository / branch:** `Danmoreng/vllm.cpp`, `b70-gptq-int4`  
**Purpose:** reduce real prefill and MTP serving latency without undoing implemented functionality or concealing incomplete numerical/lifecycle qualification.

## 0. Decision and first assignment

Continue from the current native implementation. Do not restart EXL3 loading, compact-head integration, native MTP, IndexCopy, graph ownership, prefix handling or the recent norm rewrite. Those features now exist. The next phase is a sequence of small, measured performance changes, with an explicit qualification ledger rather than another open-ended S1 investigation.

Two independent first opportunities have concrete source support:

1. **Decode:** qualify the existing native Xe2 packed verification design for the actual **page-1600, Q4, C1** operands. Current native admission still requires page 1664, whereas the matching original extension's source admits page 1600. Prove the actual original route and arithmetic before enabling it.
2. **Prefill:** replace the native W8A8 **128-output-column reconstruction/GEMM loop** with larger, bounded, group-correct panels. The current profile records 91,392 target reconstruction calls and 91,392 oneDNN stream calls over only three prefill chunks. This is not the original's larger per-group organization.

**First concrete implementation deliverable:** a wider-panel W8A8 path for one real grouped projection, first at M256 and then at the actual M1600 prefill shape, with the existing 128-column path as an internal A/B control. Start with panel widths 1024 and 2048 only. Implement the planning/allocation contract as well as the kernel loop. Record intermediate and final numerical comparisons, operator latency, panel count and peak scratch. See P2.

Before that patch, perform only the bounded P0 readiness check and the small P1 route/operand capture. If the original GPU fixture needed by P1 is unavailable, record that dependency and implement P2 immediately; do not wait through repeated full-model runs.

**Next decode implementation deliverable:** P1's page-1600 Q4 verifier, followed by Q2/Q3/Q5 boundaries and a distinct C4 subtask. Merely deleting the page guard is not the deliverable.

These changes do not require a new quantization scheme, new weights, different MTP depth, more VRAM, a dependency upgrade or a new broad benchmark campaign.

## 1. Immutable scope and explicit non-goals

Keep the checkpoint `turboderp/Qwen3.8-27B-exl3` at revision `113cf7ab958054860e43fb7f3063b1af19171095`, the existing 4.00-bpw body and full 248,320-token **6-bpw EXL3 target head**. Keep the exact 65,536-token draft subset, its global-token mapping and complete 128-token Hadamard blocks, and MTP3. Preserve FP16 activation/materialization boundaries, FP32 recurrent state and FP8 KV storage. Large-M arithmetic is **rotated EXL3 W8A8**, not GPTQ W4A8. [E01, E05–E07]

Use one 32-GB B70 and the 32-GB host objective, fixed 180 W, existing compiler 2026.1 / oneDNN 3.13, and the supplied image/source pins. Production stays stopped while the authorized development tests run; original and native use the GPU sequentially. Use one implementation session on the current branch, small commits and focused tests. Do not add background polling loops, new dependencies or model downloads.

Keep the 262,144 total-context objective, page-1600 launch, fixed workload tokenizer and scheduler settings for the first performance comparisons. The configured maximum and cache-capacity arithmetic are not evidence of a successful 262,144-token generation.

Do not retune power, weights, draft vocabulary, MTP depth, sampler semantics, max batched tokens or acceptance thresholds to manufacture a speedup. Do not expand the target head to FP16 or cache the whole expanded model. Do not use original captured arrays in product inference. A generated mathematical function table proposed in P3 is not a model/reference tensor substitution.

A generic native implementation can remain an internal correctness/control route. The intended product is still one native EXL3 engine, not a Python fallback or two public quality/performance profiles.

## 2. Independent audit: what is supported

### 2.1 What this review actually checked

The archive's 9,262 manifest entries match their byte counts and SHA-256 hashes; it identifies 8,864 tracked native files. All 13 captured active-runtime source guards match. The vendored `exl3_esimd.h` is byte-identical to the supplied pinned donor header. Forty-two supplied host tests passed in this review: projection comparator, runtime-layout, grouped-capture and inventory tests. [E01]

The review recalculated the raw C1 serving metrics and compared all 1,024 output IDs and non-timing cycle metadata between the preceding native candidate and the latest graph/eager runs. It checked the controlled 65-row comparison summary. **The external large tensors were not independently re-compared, and the 451,699,956-byte raw GPU profile is not included.** The packaged cost summary was inspected for scope/count consistency; it was not regenerated from that external profile. No B70 execution, native compilation or driver validation occurred during this review.

### 2.2 Current performance, correctly scoped

| Arm | TTFT (s) | Emitted decode tokens/s | End-to-end (s) | Accepted / proposed |
|---|---:|---:|---:|---:|
| Native before latest norm | 4.848126 | 28.774967 | 40.399888 | 642 / 1152 |
| Native latest graph | 4.841378 | 34.731260 | 34.296143 | 642 / 1152 |
| Native latest eager | 4.858865 | 32.384278 | 36.448299 | 642 / 1152 |
| Earlier pinned original Python | 1.948672 | 63.334398 | 18.101045 | 664 / 1083 |

These are single unprofiled trials, warm O32 followed by a cold request prefix, P4096/O1024, MTP3. Startup/loading is outside the interval. A timestamp represents an emitted MTP chunk, not a separately timed arrival for each token. [E02]

The norm improvement is especially credible within this bounded case: **20.70% higher native decode throughput with identical native tokens and semantic cycle metadata**, not a more favorable acceptance trajectory. Native latest versus original is descriptively 54.84% of emitted decode rate, 2.484× TTFT and 1.895× end-to-end time. Original/native outputs diverge at zero-based output 53, and the original was not rerun after the norm change. These ratios are not same-continuation or final qualification claims.

The native run has 384 decode cycles, averaging **76.699 ms per cycle** and **2.66406 emitted tokens per cycle**. At that unchanged yield, 63.334 tokens/s would require about **42.063 ms per cycle**. This is a derived engineering budget, **not a measured original cycle duration**. The original raw record does not supply matched target-cycle timings.

A rough acceptance-only counterfactual, holding native cycle cost fixed and applying `1 + 3 * acceptance`, reaches about **36.91 tokens/s**, not 63.33. Different aggregation and continuations limit the calculation, but acceptance alone is plainly not an adequate explanation for the observed gap. Do not solve the speed problem by cherry-picking easier outputs.

### 2.3 Functional work is substantial; qualification remains bounded

Native autonomous target generation, large-M W8A8, variable-length GDN, compact draft, MTP1/MTP3, C2/C4 state/lifecycle plumbing, recurrent-prefix integration and graphs have been implemented and locally exercised. The latest norm work preserves its documented arithmetic tree and materially improves serving. This is no longer a missing-full-model problem. [E01, E03, E04, E16]

Keep these unresolved gates visible:

* The full P128+D64 target record has **all 65 complete 248,320-element rows exact to a separately named controlled deterministic-BA original**. The default frozen original still fails four TV/KL gates: D27 TV 0.0305754 / KL 0.00212581 and D29 TV 0.212752 / KL 0.163144; top-10 minimum is 9. Do not rename the controlled protocol into the default oracle.
* The isolated compact-draft/head proof is strong and bounded: hidden, compact logits, independent head replay and token mapping passed. It predates the last shared norm/graph changes and is not every production MTP iteration.
* The practical 8/8 suite and 393 native assertions predate the latest two shared changes. It is sequential with capacity for four, not a concurrent C4 quality screen. Latest snapshot-wide practical coverage is still pending.
* Latest C1 graph/eager equivalence is demonstrated. All context, EOS/cancel, row reuse, sampled-RNG, prefix-state and C4 graph transitions are not thereby proven.
* The 262K setting remains a capacity objective rather than a newly measured full-length success.

Performance work on identical-input operators can proceed with this ledger open. Product qualification cannot be declared complete by the assertion fraction or by transferring old practical results to new source.

### 2.4 A small environmental comparison gap to close, not a new campaign

The archived native commands use `--memory 12g --cpus 2`; the original serving command uses `--memory 32g --cpus 2`. Native reports peak process RSS around 13.58 GB. Process high-water RSS and cgroup accounting are different quantities: **this does not prove swapping, throttling or an OOM**, but it requires a small resource receipt before attributing all long-run differences to kernels. [E18]

Record cgroup memory.current/peak/events, swap policy/usage, major-fault deltas and CPU throttling where available. Verify actual worker identity and warmed state. For final comparisons use comparable safe effective limits within the 32-GB host budget, or explicitly retain/document the difference. Do not casually set both processes to consume all host RAM, and do not run them concurrently. The norm before/after improvement remains valid under its unchanged native resource setup.

## 3. Ranked bottlenecks and what the numbers do not mean

### 3.1 Decode: separate eager diagnostic attribution

| Area | Mean device-stage cost per target decode cycle | Interpretation |
|---|---:|---|
| SmallM DPAS | 28.351 ms | Largest measured component, but already uses the pinned donor family; not automatically the largest implementation gap. |
| Generic Split-K attention partial | 15.306 ms | Current page-1600 verification misses the opt-in Xe2 path. Strong route-level opportunity after identical-operand proof. |
| Speculative GDN | 10.607 ms | Scalar K-dimension row accumulation and private state array; geometry/coalescing opportunity with strict FP32-state checks. |
| Gemma5120 norm | 3.756 ms | Already reduced from 20.527 ms; preserve the successful rewrite. |
| Gated GDN norm | 1.835 ms | Secondary candidate after primary changes. |
| SiLU/multiply | 1.082 ms | Modest decode share; much more important in prefill. |

The eager target-forward span averages 74.434 ms; draft span 6.585 ms. Do not add overlapping parent/child ranges or paste these profiled eager values onto the unprofiled graph cycle. The primary evidence does not provide matching original per-operator times. [E03]

### 3.2 Prefill: three chunks, not one mean request

| Area | Mean reported cost per prefill chunk | Count over three chunks |
|---|---:|---:|
| oneDNN W8A8 stream spans | 676.257 ms | 91,392 |
| SiLU/multiply | 200.667 ms | 192 |
| W8A8 row-finite validation | 180.663 ms | 768 |
| W8A8 weight reconstruction | 120.179 ms | 91,392 |
| Gated GDN producer norm | 67.306 ms | 144 |
| Output Hadamard | 39.393 ms | 768 |
| oneDNN SDPA | 17.559 ms | 192 |

These are the actual three chunks of the request; average target span is 1,600.837 ms **per chunk**. oneDNN stream spans are not isolated pure GEMM execution and can include scheduling gaps. Some donor launches have less detailed attribution. Use a short event timeline after each major improvement; do not assume every remaining millisecond is already assigned. [E03]

The source explains the striking launch count. Per chunk:

```text
48 GDN layers * (16384 + 5120 + 34816 + 5120) / 128 = 23040 panels
16 attention layers * (14336 + 5120 + 34816 + 5120) / 128 = 7424 panels
Total = 30464 panels/chunk; three chunks = 91392
```

Each panel launches reconstruction **and** calls oneDNN. The original reconstructs and multiplies larger contiguous source-group spans instead. This is a concrete organizational difference, not speculation about Python overhead. [E05, E06]

## 4. Execution order and bounded work packages

Suggested order: **P0 → small P1 fixture → P2 → complete P1 → P3 → P4/P5 as indicated by the refreshed profile → bounded P6 → P7**. P1 and P2 are independent; the same assistant implements them serially. P3 can proceed if external P1 fixtures are unavailable. Do not require a new full performance baseline campaign before a dependency-ready patch.

Every package needs one conceptual change per commit, a short receipt, its smallest correctness check and an unprofiled improvement check. Keep rejected experiments as receipts, not enabled code. Two failed performance variants without new causal information are a stop signal for that package, not a reason to repeat full serving runs indefinitely.

### P0 — Freeze a useful measurement boundary and repair only blocking evidence gaps

**Files/areas:** `docs/b70-exl3/RECOVERY_STATUS.json`, `EXECUTION_MAP.md`, `tools/bench/b70_exl3_serving_timing.py`, `tools/exl3_reference/serving_timing.py`, `tests/vllm/models/test_xpu_exl3_mtp.cpp`, existing R11 receipts.

**Deliverable:** one receipt identifying current binary/source, original image, model/subset hashes, resource limits, physical/padded shapes, page policy and selected routes. Reuse the latest native raw baseline. Add cgroup observations from §2.4. Record current logical and padded target rows; C4 MTP3 is normally up to 16 logical verifier rows, not 20 from the old MTP4 configuration.

**Smallest checks:** latest compact draft and the small graph/eager reset/EOS/cancel sentinel once, because the latest shared changes postdate their broader proof. Do not regenerate all external tensors merely to get another expected D64 failure. Existing original fixture paths may remain external and hash-identified; unavailable data is a skip/blocker, never a pass.

**Separate numerical track:** propose, rather than silently adopt, a reference-contract correction if deterministic-BA original behavior is intended. Require an independently justified arithmetic definition, same-input state checks and two held-out prefixes not used to tune rounding. Preserve all 65 default frozen rows and failures. A genuinely new state, indexing or safety defect blocks affected optimization; the already understood reference question does not block independent panel implementation.

**Stop:** end readiness work once the receipt and critical sentinel outcomes are known. Do not change drivers, power, dependencies or weight formats to make this receipt cleaner.

### P1 — Native packed verification for page1600, first C1/Q4 and then C4

**Files/functions:**
- `src/vt/xpu/xpu_attention_verify_xe2.cpp`: `PagedAttentionXe2VerifyKernel` admission, split choice, packing, donor argument construction and output unpack/copy.
- `src/vt/xpu/xpu_attention.cpp`: verify dispatch and opt-in-to-qualified admission.
- `src/vt/xpu/xpu_attention_split.cpp`: retained generic control and partition policy.
- `tests/vt/test_xpu_attention_fast.cpp`, `tests/vt/test_xpu_attention.cpp`; model test and graph owner integration.
- `src/vllm/model_executor/models/qwen3_5.cpp`, `src/vt/xpu/xpu_backend.cpp` only as needed for policy identity and lifetime.

**Hypothesis:** the current 15.306-ms generic partial cost can be substantially reduced by using the original packed/XMX verification family on the actual layout. The native fast wrapper rejects page1600 and is opt-in; the matching original source admits positive page sizes divisible by64 and supports multiple requests. [E09–E11]

**Evidence provenance:** the newest archive's imported `shared_kv_verify.py` implementation and native extension wrapper are not complete there. This review recovered them from the user's already supplied 2026-10-02 production archive and checked their manifest hashes. The dispatcher hash and original image identity match the current captured reference. This strengthens the admission comparison; it is **not a fresh executed-route measurement**. Reconfirm the live guard and binary hash in the single original operand-capture run.

**First change:** capture one actually executed original C1 page1600 Q4 verification call, including Q, raw cache bytes, strides, exact active lengths, block table, query offsets, scaling and output. Inspect the original wrapper's packing and compiled arithmetic. Adapt the native wrapper with a precise page1600/layout admission rather than unconditional guard deletion.

The packed original maps multiple logical verification rows into the head dimension. Its physical query metadata is one row per request; the causal cutoff for logical row r is `L - Q + r`. Validate this explicitly. Native and original must agree about whether query offsets are logical or packed; do not reuse C1's convenient `{0,1}` for a C4 batch.

**Invariants:**
- Original FP8 decoding, scales, head mapping (24 Q /4 KV, D256), row-specific causality and output ordering.
- Preserve FP16 probability/output boundaries and F32 reductions as implemented by the qualified original kernel. A generic F32 attention match alone does not identify the original arithmetic.
- No reads from poisoned inactive keys/pages or another request. No stores into another request's output or metadata.
- Separate contiguous/interleaved/padded cache layouts using actual strides. Initially retain restrictions on non-unit scales unless an identical-input original test qualifies them; unsupported forms use the existing native route.
- Complete pack/kernel/unpack cost and scratch ownership, including aliases and asynchronous output copies.
- Graph keys/retirement include route, physical layout, active-page and partition policy. Keep the recently fixed 960/4096 and page-boundary retirement behavior. A faster kernel cannot reuse an arithmetically incompatible captured graph.

**Smallest checks:** C1 Q4 at active L4096; then Q2/Q3/Q5 on the same layout. Boundary operator cases around 1599/1600/1601 and 4799/4800/4801, and one L32768 case. Compare actual original outputs first, then eager/graph native with all own-state and output fields. Keep noncausal/unsupported-scale/stride refusals and poisoned padding tests.

**C4 subtask after C1 passes:** uniform Q4 for four requests, unequal active lengths, distinct poisoned caches, permutation and output-copy tests; then one ragged Q2/Q4 batch via a qualified per-request metadata contract or documented generic remainder. Do not serialize four C1 calls and assume that is a fast batch implementation. The original C4 direct-output-copy behavior is already documented and must be retained or independently matched.

**Performance:** operator medians include packing/copy; then C1 P4096/O1024 and one C4 sentinel. Compare exact-input kernel latency and actual emitted-cycle throughput separately. If switching from generic F32 to the original FP16/XMX arithmetic changes native tokens, do not advertise an identical-native-output speedup. Require original same-input proof and full-model qualification, and separate the acceptance effect.

**Stop:** reject the candidate on masking/state/lifetime failure. If two implementation variants fail to improve the complete operator beyond noise, retain the generic route, record the unresolved cause and move to P2/P3. Do not alter global page layout to satisfy an obsolete guard.

### P2 — Larger bounded W8A8 panels, preserving the quantization arithmetic

**Files/functions:**
- `src/vt/exl3_grouped.cpp`, `include/vt/exl3_grouped.h`: `PlanExl3W8A8`, workspace/shape checks and bounded panel geometry.
- `src/vt/xpu/xpu_exl3_w8a8.cpp`: `Reconstruct`, `Exl3GroupedW8A8Kernel`, per-group iteration.
- `include/vllm/model_executor/models/dense_attn_block.h`: both grouped and single/head `DBuf` panel allocation paths.
- `src/vt/xpu/xpu_gptq4*` / the definition of `Exl3W8A8Matmul`: support the actual strided destination and panel width without changing scale application.
- `tests/vt/test_xpu_exl3_w8a8.cpp`, existing real grouped fixtures.

**Hypothesis:** 128-column decomposition produces too many reconstruct submissions and oneDNN calls; wider group-correct panels reduce these costs and improve GEMM geometry. This is supported by the 91,392-call counts and donor organization, but the gain is unmeasured. [E05–E07]

**Implementation:** construct contiguous runs of equal `shard_of_nb`, and split each run into panels of up to1024, then2048 columns. Never cross a different input-scale/Hadamard source group. Respect noncontiguous source-group maps by making distinct runs; do not silently assume a monotonic map. Tail widths stay multiples of128. Pass width/offset into the donor reconstruction and oneDNN descriptors; preserve full output leading dimension N.

Use one completion-owned reusable panel/scratch pool rather than allocating a larger persistent panel for every model layer. At K17408, an int8 panel of2048 columns is **34 MiB**, versus2.125 MiB for128 columns. This is a scratch example, not additional permission to retain all expanded weights. Budget target, draft, graph captures, KV and recurrent state together. Address both grouped body projections and single head/projection wrappers, but do not turn the already small-row head into a large-M route.

**Arithmetic invariants:** identical trellis decoding and int8 rounding; unchanged product-to-FP16 and Hadamard-to-FP16 activation boundaries; identical zero-row scale behavior and padding; unchanged S32 accumulation contract and overflow bounds; identical order of weight/row scales; materialized FP16 intermediate Y **before** output Hadamard and output scaling. Keep K unchanged; only change output-column decomposition. A different oneDNN implementation may alter the F16 epilogue, so compare actual intermediate Y rather than assuming equal integer sums prove equal final results.

**Failure invariants:** existing invalid-input/overlap/range checks retain their meaning. No public output or data workspace is partially committed on validation failure. Existing reserved status scratch behavior remains as documented. Retain a safe completion fence until explicit panel leases make it unnecessary; do not delete the current synchronizations as part of the first width patch.

**Smallest checks:** one real grouped QKVZ or gate/up fixture at M256 with two distinct scale groups, then M1600 and final-chunk M896. Verify padded rows, 128-wide tails and a non-monotonic group map. Check 4-bpw body and one 6-bpw large-M fixture if that route is reachable. Compare reconstructed panels, Xq/scales, pre-Hadamard Y and final outputs. Preserve M128/M129 route boundaries and alias/refusal tests.

**Performance:** one operator plus one complete unprofiled P4096 prefill. Count actual submissions: report reduction from30464 panel calls per chunk, not just a favorable single GEMM. Capture total operator elapsed, oneDNN device time, reconstruction, allocator/host gaps and peak scratch. Cache/compile cold-start cost is reported separately from warmed serving.

**Stop:** after1024 and2048 candidates, choose the smallest bounded panel that produces a clear total-operator/model improvement. If output differences appear, localize the first intermediate mismatch; do not relax the original gate. Keep the old panel route for diagnosis and move to P3 if no compliant candidate wins.

### P3 — Faster exact FP16 SiLU, not a reversal of the numerical repair

**Files/functions:** `src/vt/xpu/xpu_elementwise.cpp::SiluAndMulKernel`, `tests/vt/test_xpu_ops.cpp`, existing same-input SwiGLU/GDN21 receipts. Any optional device table belongs to the existing device/context owner, not a global untracked allocation.

**Hypothesis:** the measured ~200.667 ms per prefill chunk includes a generic dynamic-view kernel executing exp and correctly rounded division for every gate element. A typed contiguous path, and possibly a small exact finite-domain function table, may remove substantial repeated work. Neither speedup has been measured. [E08]

**First variant:** specialize contiguous FP16 input/output with direct typed indexing and bounded row/channel workgroups. Preserve the existing alias-safe `WithOutput` semantics and fallback for other types/strides. Evaluate `F16(fdiv_rn(g, 1 + exp(-g)))` before multiplying by up in the existing precision and materializing FP16 output. Do not fuse this into oneDNN in a way that eliminates the first FP16 boundary.

**Second, optional variant:** FP16 has65,536 possible bit patterns. The scalar mapping from a finite FP16 gate to its materialized FP16 SiLU can be represented in a **128-KiB uint16 table**, generated by native code using the existing expression on the same device/compiler/math configuration. Runtime performs the exact lookup, widens as before, multiplies by up and narrows as before. This is a review proposal, not existing product code and not an approximation or a table of captured model activations.

Build before capture; own the table until all graphs/kernels retire; key it by device/context and arithmetic implementation. Keep the original expression for exceptional values if needed to preserve NaN/Inf behavior. Do not assume changing launch geometry leaves transcendental results identical: prove the table against the original kernel exhaustively.

**Smallest checks:** all65,536 gate bit patterns, finite exactness, signed zero, subnormals and nonfinite behavior; several up values including zero/extremes plus real MLP rows. Include the recorded gate `-2.724609375`, up `-0.346923828125` midpoint and the actual MLP/GDN21 fixture. Validate poisoned inactive rows, aliases and unsupported strides. Compare the complete operator, not lookup alone.

**Memory/performance:** 128 KiB persistent per relevant device/context is small but must be accounted for. Random lookups can be slower than computation or interfere with caches. Measure real M1600/I17408 and M896; retain the old small-row route if decode does not benefit. Then verify full native C1 tokens/cycles unchanged.

**Stop:** reject any finite bit mismatch or lost FP16 boundary. If typed and table variants both fail to improve total latency, stop; do not replace `fdiv_rn` with approximate reciprocal, native_exp or unrestricted fast-math. The previous numerical bug was real.

### P4 — Reduce W8A8 guard/transform duplication without removing protection

**Files/functions:** `xpu_exl3_w8a8.cpp::ValidateRows`, `ValidateOutputScale`, `Exl3GroupedW8A8Kernel`; loader-owned grouped metadata; W8A8 tests and buffer ownership.

**Hypothesis:** row validation repeats FP16 product and Hadamard work that the subsequent activation-quantization stage performs again. Its180.663-ms/chunk cost is not explained by the tiny output-scale check (~0.361 ms). The already implemented merged status readback must not be re-presented as a new optimization. [E05]

First profile the validator itself on M1600 using real scales/outliers. Improve typed geometry/reuse or carry validated transform results into quantization using a bounded **private preparation scratch**. Preserve both FP16-overflow checks, including the65520 midpoint boundary, NaN/Inf handling, zero-row behavior and error priority for invalid output scales.

Do not simply run a potentially undefined float-to-int8 conversion on bad input. A fused prepare path must detect invalid values and use defined safe placeholder conversions, then prevent GEMM/public output/state commit. The public API's no-partial-result tests remain. If the old public workspace must remain unchanged on failure, private scratch is needed; changing the contract is not an optimization.

Immutable maps/scales can be validated at load under a versioned owner, avoiding repeated host map reads in the **model-owned** route. This requires device, shape, generation and lifetime validity, not a pointer-only cache. Keep the untrusted public operator checks. Do not claim cached output-scale validation removes the dominant row-transform cost.

**Smallest checks:** M129/M1600, two scale groups, zero rows, finite near-overflow rows, product overflow, post-Hadamard overflow, NaN/Inf in input/suh/svh, invalid map, poisoned scratch and aliases. Test error priority and output/state non-modification before measuring.

**Stop:** one validator-geometry variant and one bounded reuse variant at most. Accept only a reduction in complete operator/request latency at the same arithmetic/safety contract. If additional scratch exceeds budget or synchronization merely moves elsewhere, retain the current safe path.

### P5 — Speculative GDN row/state execution

**Files/functions:** `src/vt/xpu/xpu_gdn.cpp::GdnSpecDecodeKernel`; `tests/vt/test_xpu_gdn.cpp`; MTP state/lifecycle tests. Avoid unrelated prefill changes in this patch.

**Hypothesis:** current workgroups still assign one logical work item a private `float s[128]` row and perform two ascending K loops for each verification token. The64-thread C1 workgroup setting changes scheduling but does not itself replace that scalar row computation. Better register placement, typed loads and sharing across independent value rows may reduce10.607 ms/cycle. Register spilling is a question for generated-code/profile evidence, not a proven diagnosis. [E12]

First inspect actual register/scratch usage for Q4/C1 and Q4/C4. Try a specialized Dk=Dv=128 FP16-input/F32-state implementation that vectorizes **independent value rows** or uses explicit SIMD storage while retaining the existing ascending K accumulation order. Hoist q/k/decay/beta where exact and beneficial. Preserve the initial snapshot selection from previous accepted counts and every provisional snapshot, including aliasing with the starting slot.

Do not transplant the ordinary SG16 decode reduction and call it numerically identical: a parallel K reduction changes the arithmetic order. A different order requires a distinct original same-input proof and explicit numerical review, not only matching generated text.

**Smallest checks:** C1 Q1/Q4 with accepted-count selectors1..4; full FP32 output snapshots, not merely final FP16 output. Then C4 distinct/permuted requests, unequal query counts, negative inactive slots, poisoned unused states, cancel/reuse and prefix-state continuation. Retain cross-request alias rejection and no writes to inactive slots.

**Performance:** state update including snapshots and materialized outputs. Compare C1 and C4 independently; the current C1-only WG64 default does not establish a good C4 geometry. One MTP target cycle and graph/eager transition follow the operator pass.

**Stop:** at most two justified geometry/register variants after the initial inspection. Reject state drift, lost snapshots or greater model latency. Do not repeat an old width-only WG sweep without new code-level evidence, and do not reduce recurrent-state precision.

### P6 — SmallM DPAS: measure the donor gap before inventing another kernel

**Files/functions:** `src/vt/xpu/xpu_exl3_smallm.cpp`, `src/vt/exl3_grouped.cpp`, `third_party/exl3xpu/exl3_esimd.h`, `tests/vt/test_xpu_exl3_smallm.cpp`, focused reference projection tools. [E13]

**Hypothesis:** a remaining native/donor discrepancy may be in shapes, split-K, grouped launch selection, intermediate buffers, compiled code or graph behavior. The native donor header is already byte-identical; this is not a missing donor-port task.

Compare actual M4/C1 and M16/C4 target verification, M1/M4 draft and selected-token head rows; include 4-bpw QKVZ, QKV, gate/up, down/out and 6-bpw full/compact heads. Record actual padding, MB/NT, group maps, partition count, register mode, compiler flags and mutable donor tuning settings. The current defaults use GEMV for M<=2, MB8/16 with1408 target threads for typical verify shapes, matching the inspected donor defaults; a runtime override still needs identification.

Use a short sequence of the **real different layer weights** or a weighted representative set, not an indefinitely hot single matrix. Compare direct GPU execution and complete operator cost, including Hadamard/reduction. Run device-core equivalence on identical packed bytes and scales.

If native is near the same-input donor time within a predeclared practical comparison band (suggested10%, with observed noise reported), document that this28-ms component is largely shared cost and move on. A large absolute cost alone is not evidence it explains the engine gap.

If a real discrepancy exists, first repair launch/compile mismatch. Only then try at most two geometry variants that keep arithmetic partitions fixed. A split-K change can change F32 reductions and is not merely scheduling. Never enable the optional fully fused donor path solely because of its name: its captured source defaults it off and records worse behavior under graphs.

**Stop:** no broad tile search, global fast-math switch, widened head or all-weight expansion. Keep only operator and serving wins with the same numerical contract.

### P7 — Remaining overhead, bounded final comparison and honest qualification

Use the refreshed short event timeline to decide whether host gaps, remaining gated norms, state movement or draft work are now material. Graph versus eager currently improves native C1 from32.384 to34.731 tokens/s; graph functionality is present. Do not spend the first optimization patch implementing it again.

Preserve the recent owner retirement when attention partition policy changes. Any new panel pool, table, verifier scratch or graph cache must have completion-owned lifetime, stable capture addresses, correct invalidation and bounded retention. No host polling loops. Optimize remaining waits only with an identified dependency and complete ownership test; the presence of a wait is not proof it is unnecessary.

Refresh the unchanged practical screen once on the final candidate and add only the changed-path C4/EOS/cancel/prefix/graph-boundary sentinels. Revisit the full default/controlled target and isolated compact draft proofs at this checkpoint. Keep default failures visible unless an independently reviewed reference-contract change closes them.

Execute the bounded matrix in §6. Use three unprofiled repetitions only for the final primary C1/C4 P4096/O1024 MTP3 cases; profile separately. Use one warmed representative execution for the remaining sentinels, repeating only a noisy/regressing result to resolve it. Alternate original/native order where feasible, never simultaneous. No full262K or new power campaign is required before these dependency-ready kernel improvements.

**Finish condition:** supported modes pass their declared arithmetic/state/lifecycle gates, the current capability screen is refreshed, and primary emitted-token/TTFT/E2E comparisons are reported with fixed identities and memory. Claim parity only for workloads that actually meet it. The desired target is no slower than original in the primary like-for-like cases; do not mark it achieved because one kernel or one favorable continuation wins. If a mode remains slower, leave its measured gap and next dominant stage explicit instead of renaming completion.

## 5. Receipt and acceptance protocol for every change

Keep this record small enough to complete with each commit:

```json
{
  "task": "P2",
  "hypothesis": "Larger group-correct output panels reduce W8A8 launch cost",
  "baseline_source": "7536aededc049b2157f12eadb0f4cc6bbf29b7b7",
  "candidate_source": "RECORD_ACTUAL_COMMIT",
  "binaries_and_libraries": "paths, byte sizes, SHA256, loaded implementation",
  "inputs": "checkpoint/subset/fixture hashes, exact shapes, strides, scales and states",
  "arithmetic_change": false,
  "tests": {"exit_code": null, "assertions": null, "skips_are_not_passes": true},
  "numerical": "intermediate/final/state evidence with original gates unchanged",
  "memory": "GPU active/reserved/native/graph + host/cgroup, peak and retired values",
  "performance": "unprofiled operator/model times; separate profiler receipt",
  "tokens": "emitted, accepted, proposed, cycles; first difference when applicable",
  "decision": "retain | reject | blocked",
  "stop_reason_or_next_dependency": "concrete reason"
}
```

No forged zero exits for diagnostic summaries, no captured-value substitutions and no assertion fraction described as model accuracy. A missing GPU fixture can block a particular proof without blocking an independent implementation package.

Suggested practical performance triage, not a mathematical guarantee: investigate only candidates with a clear focused gain beyond noise; promote them after a visible complete-operator/model gain. Aim for at least several percent model benefit for invasive changes. Small robustness fixes do not need an artificial speed threshold. Once a candidate worsens arithmetic/safety or loses on the complete path, reject it regardless of its central-kernel TFLOPS.

For unchanged arithmetic/geometry-equivalent patches, expect identical full native outputs and non-timing cycle fields on the frozen C1 case. For a deliberate switch to the original's different attention arithmetic, require original same-input evidence and an explicitly scoped model comparison; do not demand identical old generic-native text while also claiming a new reference-matching route.

## 6. Bounded benchmark matrix and metrics

Use O for total emitted tokens, including the first. Use Q for per-request verification rows and M for the actual possibly padded matrix rows. MTP3 normally gives up toQ4. Do not reuse MTP4 row counts from earlier GPTQ plans.

| ID | Requests / input | Output / MTP | Purpose | Final repetitions |
|---|---|---|---|---:|
| C1-4K-M3 | C1,4096 cold | O1024,MTP3 | Main decode/TTFT target, raw cycles | 3 per engine |
| C4-4K-M3 | C4,4096 each, distinct prefixes | O1024 each,MTP3 | True batch verification and overlap | 3 per engine |
| C1-4K-M0 | C1,4096 cold | O256,no MTP | Target-only kernel/route isolation | 1 per engine |
| C4-4K-M0 | C4,4096 each | O256 each,no MTP | Batch target-only isolation | 1 per engine |
| C1-32K-M3 | C1,32768 cold | O256,MTP3 | Long-KV attention, wider prefill | 1 per engine |
| C4-32K-M3 | C4,32768 each, distinct prefixes | O256 each,MTP3 | Memory/admission and long batch | 1 per engine, within safe admission |
| MIXED-4K-32K | Begin C1 4K decode; admit32K after16 emitted tokens | O256 each,MTP3 | Mixed routing, pauses and resource ownership | 1 per engine |
| PREFIX-32K | Frozen32K request then exact repeated prefix | O64,MTP3 | Prefix correctness/reuse, not raw prefill-TFLOPS | 1 native + original fixture check |
| LIFECYCLE | Small unequal C4 including earlyEOS/cancel/reuse and page-policy transition | bounded64–256,MTP3 | State, graph and buffer lifetime | focused native test |

C4-32K is an admission test as well as a throughput test. Never force four requests to remain resident if the actual32-GB budget cannot support them with states/workspaces. Record queuing and active-request intervals; do not label a nonoverlapping result fully overlapped C4. Keep262K as the unchanged end capability objective, not a new factorial sweep here.

Report:

1. pure target-forward time and complete MTP-cycle time under identified inputs;
2. accepted/proposed/emitted tokens and cycle counts, not only acceptance percentage;
3. TTFT and actual emitted post-first decode rate;
4. chunk-observed TPOT and longest streaming pause;
5. common fully overlapped aggregate decode for C4 only when that interval exists;
6. per-request and batch end-to-end time;
7. prompt tokens actually computed versus prefix hits, no confusion with effective cached throughput;
8. active/padded forms, routes, graph capture/replay/retirement and resource peaks.

Keep warmup separate, reset request-prefix cache for cold cases, hash the actual prompts and token IDs, and exclude loading/startup only consistently. Same seed is not same continuation. A controlled replay must fix proposed tokens, accepted lengths and starting states as well as input shapes; do not merely force one repeated token and call it an equivalent MTP workload.

## 7. What stays deferred

No draft-vocabulary/depth re-search, no target/draft bit-rate changes, no GPTQ restoration, no public dual-profile product, no driver/library upgrade, no speculative whole-model FP16 expansion and no broad load-generator campaign. No revision of the successful norm arithmetic, previous row-group IndexCopy or already merged validation readback without a new measured reason.

Full stochastic-RNG equivalence, every supported context and production-wide lifecycle qualification remain honest open ledger items if the current work only proves greedy fixed-output cases. They must be closed for a corresponding product claim, but they do not need to block an unrelated identical-input W8A8 panel optimization.

The goal is not another receipt-only loop. Each retained performance package should deliver runnable native code, its focused proof and an actual measured stage/request improvement. Existing unresolved default-reference gates stay visible alongside that progress.

## 8. Ready-to-paste Codex objective

> Work from native commit7536aeded on the current branch and keep the fixed EXL3/MTP3/subset/arithmetic contract. Preserve implemented target, draft, graphs, prefix and lifecycle features. Perform the bounded P0 receipt/readiness work, obtain one actual original page1600/Q4 verification fixture, and implement P2 wider group-correct W8A8 panels for a real projection at M256/M1600 as the first performance code change. In the next small change, qualify the page1600 C1 verifier from P1; extend to C4 only after C1 passes. If the original fixture is unavailable, do not stall P2/P3. Then follow P3–P6 only where the refreshed profile justifies them. Do not substitute captured values, silently relax the default D64 gates, repeat already implemented optimizations, or run another broad benchmark sweep. Each change needs exact input/state/lifetime invariants, a focused test, an unprofiled gain check and a retain/reject decision. Keep default-vs-controlled reference status separate from implementation progress. Finish with the bounded P7 matrix and current practical/state/lifecycle qualification, reporting actual emitted tokens, TTFT, overlap and end-to-end latency rather than forward-rate proxies.

---

# Evidence appendix

Evidence below is from uploaded sources and review calculations, not newly measured GPU performance. Paths beginning `code/`, `reference/` or `evidence/` are relative to the 2026-10-04 archive root. `supplement/` denotes explicitly identified files from the user's earlier 2026-10-02 production archive. Numbered excerpts use original source line numbers. Existing comments, old plans and experiment descriptions are evidence to evaluate, not commands to execute.


## E01 — Audit identity and implemented scope

These checks were performed locally in this review; they are not B70 execution results.

```json
{
  "snapshot": "7536aededc049b2157f12eadb0f4cc6bbf29b7b7",
  "manifest_files_verified": 9262,
  "native_tracked_files": 8864,
  "source_guards_verified": 13,
  "current_b70_gpu_execution_performed": false,
  "large_profile_independently_recomputed": false,
  "controlled_tensor_bytes_locally_available": false
}
```

```json
[
  {
    "test": "test_exl3_projection_compare.py",
    "exit_code": 0,
    "log": "test_exl3_projection_compare.py.log"
  },
  {
    "test": "test_exl3_runtime_layout.py",
    "exit_code": 0,
    "log": "test_exl3_runtime_layout.py.log"
  },
  {
    "test": "test_exl3_grouped_capture.py",
    "exit_code": 0,
    "log": "test_exl3_grouped_capture.py.log"
  },
  {
    "test": "test_b70_inventory.py",
    "exit_code": 0,
    "log": "test_b70_inventory.py.log"
  }
]
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 15–37. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
    15: HARD CONTRACT AND ENVIRONMENT
    16: 
    17: One Intel Arc Pro B70, 32-GB GPU / 32-GB host objective. Production b70-qwen38-vllm.service stays stopped, exclusive GPU use, original/native runs sequential. The final packaging check observed only b70-chatui running and production inactive. No driver or power changes, model downloads or new dependencies were performed for this handoff. Existing measurements used 180 W.
    18: 
    19: Checkpoint: turboderp/Qwen3.8-27B-exl3, revision 113cf7ab958054860e43fb7f3063b1af19171095, existing 4.00-bpw EXL3 model. Target uses the full 248,320-token 6-bpw head. Exact draft subset has 65,536 global tokens in complete 128-token Hadamard blocks, MTP depth3. FP16 activation/parameter boundaries, FP32 GDN recurrent state and FP8 KV storage are retained. No hidden Python/GPTQ fallback; captured original arrays are diagnostic operands, never product inference substitutions.
    20: 
    21: Large-M projection route is rotated W8A8, including reconstructed/rotated integer weight panels and activation rounding; the model's packed EXL3 bpw and head bpw are distinct from W8A8 operator arithmetic. Do not describe this as a generic W4A8 kernel or replace it without a clearly stated arithmetic contract and focused identical-input proof. SmallM uses the pinned EXL3 ESIMD/DPAS route.
    22: 
    23: Measured launch: max_model_len262144, max_num_batched_tokens1600, max_num_seqs4, 180 KV blocks, physical page1600, FP8 KV, recurrent prefix reuse enabled, exact compact draft/MTP3, greedy sampler ignoring EOS for fixed output budgets. C1 means one active request, not simultaneous C4. The 262144 setting/capacity arithmetic and bounded long-context tests do not constitute an actual 262144-token successful generation claim.
    24: 
    25: Pinned donor: exl3xpu c59d9442aba8610188837e37724600f1517d7335. Pinned original runtime image sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918. Native builder sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca; oneAPI compiler2026.1, oneDNN3.13. Exact runtime/source/library/fixture/subset identities and commands are in the receipts and producer manifests. The packaged earlier reference manifest contains historical native identities; it does not replace the new snapshot manifest.
    26: 
    27: IMPLEMENTATION STATUS
    28: 
    29: R01 native autonomous C1 feedback/output/EOS/reset: delivered and locally tested.
    30: R02 same-input state attribution and required norm geometry: bounded deliveries and local tests complete; integrated original target/state qualification still open.
    31: R03 grouped large-M W8A8 and R04 variable-length GDN/workspace: delivered and locally tested within recorded families/shapes.
    32: R05 practical eager 4K target: delivered and locally tested.
    33: R06 compact draft/head: implemented, current bounded original numerical replay passes; broader qualification open.
    34: R07 MTP1/MTP3, R08 C2/C4 lifecycle, R09 recurrent prefix/long-context, R10 graph ownership: implemented and locally exercised; complete flags remain false because their integrated qualification is incomplete.
    35: R11 practical screen and measured optimization: substantial evidence and improvements exist; final qualification and performance coverage remain incomplete.
    36: 
    37: The status file records R01–R05 complete within their local delivery scopes, R06–R10 implemented/operator-tested but not complete, and R11 incomplete. Target-qualified/serving-qualified remain false. Recorded milestones A (native short output) and B (4K plus chunked32K native output) are complete; C/D/E are not recorded complete. This is no reason to restart functional R06 or abandon the engine; it is a reason to preserve the distinction between features, bounded numerical evidence and complete supported-mode qualification.
```

## E02 — Raw serving metrics independently recalculated

Calculated from the supplied raw request/cycle records using the supplied timing contract and independent token/time consistency checks. The original has no directly matched target-cycle timer in these records.

```json
{
  "perf": {
    "baseline": {
      "ttft_s": 4.848126379,
      "decode_tps": 28.774967067769566,
      "e2e_s": 40.399887516,
      "accepted": 642,
      "proposed": 1152,
      "decode_cycle_ms": 92.57650351041667,
      "mean_emitted_per_decode_cycle": 2.6640625
    },
    "graph": {
      "ttft_s": 4.841377878,
      "decode_tps": 34.73126032235613,
      "e2e_s": 34.296143261,
      "accepted": 642,
      "proposed": 1152,
      "decode_cycle_ms": 76.6990317395833,
      "mean_emitted_per_decode_cycle": 2.6640625
    },
    "eager": {
      "ttft_s": 4.8588650630000005,
      "decode_tps": 32.38427814480701,
      "e2e_s": 36.448299476,
      "accepted": 642,
      "proposed": 1152,
      "decode_cycle_ms": 82.25763803385429,
      "mean_emitted_per_decode_cycle": 2.6640625
    },
    "python": {
      "ttft_s": 1.948671716003446,
      "decode_tps": 63.33439782856978,
      "e2e_s": 18.101044948998606,
      "accepted": 664,
      "proposed": 1083,
      "decode_cycle_ms": null,
      "mean_emitted_per_decode_cycle": null
    }
  },
  "derived": {
    "norm_speedup": 0.20699565843319845,
    "native_python_decode_ratio": 0.5483791038223002,
    "ttft_ratio": 2.4844502222925677,
    "e2e_ratio": 1.8947051597094313,
    "python_first_diff_index": 53,
    "same_native_tokens": true,
    "same_native_semantic_cycles": true,
    "cycles": 387,
    "equivalent_cycle_budget_ms_at_python_tps_current_native_yield": 42.06343774217203,
    "acceptance_only_hypothetical_native_tps": 36.90804745017191
  }
}
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 66–79. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
    66: LATEST PERFORMANCE: C1 P4096/O1024, MTP3
    67: 
    68: Each row is one UNPROFILED trial with O32 real-output warmup followed by prefix reset/cold request prefix; startup/loading is excluded. Decode is actual emitted serving tokens/s over the reported decode interval, not forward count or raw target token throughput. Timestamps are one per emitted MTP chunk; TPOT is therefore chunk-observed. TTFT includes more than pure prefill kernel time.
    69: 
    70:                                    TTFT(s)  emitted decode(tok/s) TPOT(ms) E2E(s)   accepted/proposed
    71: Native attention-policy-v1          4.84813  28.77497              34.75243 40.39989 642/1152
    72: Native latest norm-workgroup-v1     4.84138  34.73126              28.79251 34.29614 642/1152
    73: Pinned Python post-numerics arm     1.94867  63.33440              15.78921 18.10104 664/1083
    74: 
    75: Latest native versus preceding native has exactly the same1024 IDs/cycle metadata:20.7% higher decode throughput and15.1% lower E2E elapsed time, with effectively unchanged TTFT. This is a useful bounded improvement, not three-repeat final qualification.
    76: 
    77: The Python arm is a separately captured earlier pinned same-workload/configuration original trial and was NOT rerun after norm optimization. Original/native continuations first differ at zero-based output index53 (native2838, original561), and MTP acceptance differs. Latest native is about54.8% of that Python emitted decode rate and1.895x its E2E elapsed time, descriptively. These are own-continuation numbers, not proven same-continuation arithmetic or final speed parity. Improve the speed while retaining this caveat; do not optimize to a favorable acceptance shift and call it a kernel speedup.
    78: 
    79: Evidence: current graph report/receipt plus evidence/r11-producer-serving-v1/c1-report-post-numerics-v1.json and its raw JSON; receipt-c1-post-numerics-v2.json preserves both old arms and the earlier graph/eager mismatch.
```

## E03 — Separate eager profile and numerical scope

This is an extraction from the packaged cost summary, not an independent regeneration from the omitted 451.7-MB raw GPU trace. Parent and stream spans are not interchangeable with exclusive kernel time.

```json
{
  "target_prefill": {
    "cycles": 3,
    "span_mean_ms": 1600.8366669999998,
    "top_stages": [
      {
        "stage": "onednn_exl3_w8a8_stream",
        "count": 91392,
        "total_ms": 2028.7711369996184,
        "mean_ms_per_cycle": 676.2570456665395,
        "stream_span_count": 91392
      },
      {
        "stage": "silu_and_mul",
        "count": 192,
        "total_ms": 602.0014589999997,
        "mean_ms_per_cycle": 200.6671529999999,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_w8a8_validate_rows",
        "count": 768,
        "total_ms": 541.989362,
        "mean_ms_per_cycle": 180.66312066666669,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_w8a8_weight_reconstruct",
        "count": 91392,
        "total_ms": 360.5365410000479,
        "mean_ms_per_cycle": 120.17884700001598,
        "stream_span_count": 0
      },
      {
        "stage": "gdn_gated_norm_fp16_producer",
        "count": 144,
        "total_ms": 201.9185410000001,
        "mean_ms_per_cycle": 67.30618033333336,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_w8a8_output_hadamard",
        "count": 768,
        "total_ms": 118.17863600000005,
        "mean_ms_per_cycle": 39.39287866666668,
        "stream_span_count": 0
      },
      {
        "stage": "rms_norm_gemma5120_fp16",
        "count": 387,
        "total_ms": 97.98136099999999,
        "mean_ms_per_cycle": 32.66045366666666,
        "stream_span_count": 0
      },
      {
        "stage": "conv1d_prefill",
        "count": 144,
        "total_ms": 89.899381,
        "mean_ms_per_cycle": 29.966460333333334,
        "stream_span_count": 0
      },
      {
        "stage": "attn_qk_norm_rope_gate_subgroup",
        "count": 48,
        "total_ms": 67.41093699999998,
        "mean_ms_per_cycle": 22.470312333333325,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_w8a8_input_quantize",
        "count": 768,
        "total_ms": 64.38699600000028,
        "mean_ms_per_cycle": 21.462332000000092,
        "stream_span_count": 0
      }
    ]
  },
  "target_decode": {
    "cycles": 384,
    "span_mean_ms": 74.4344691015625,
    "top_stages": [
      {
        "stage": "exl3_smallm_dpas",
        "count": 98688,
        "total_ms": 10886.948396999722,
        "mean_ms_per_cycle": 28.351428117186774,
        "stream_span_count": 0
      },
      {
        "stage": "attention_split_partial",
        "count": 6144,
        "total_ms": 5877.562059999997,
        "mean_ms_per_cycle": 15.306151197916657,
        "stream_span_count": 0
      },
      {
        "stage": "gdn_spec_decode_wg",
        "count": 18432,
        "total_ms": 4072.9328589999313,
        "mean_ms_per_cycle": 10.606595986978988,
        "stream_span_count": 0
      },
      {
        "stage": "rms_norm_gemma5120_fp16",
        "count": 49536,
        "total_ms": 1442.2448029999791,
        "mean_ms_per_cycle": 3.755845841145779,
        "stream_span_count": 0
      },
      {
        "stage": "gdn_gated_norm_fp16_producer",
        "count": 18432,
        "total_ms": 704.750325999992,
        "mean_ms_per_cycle": 1.8352873072916458,
        "stream_span_count": 0
      },
      {
        "stage": "silu_and_mul",
        "count": 24576,
        "total_ms": 415.48136200003205,
        "mean_ms_per_cycle": 1.0819827135417501,
        "stream_span_count": 0
      },
      {
        "stage": "attn_qk_norm_rope_gate_subgroup",
        "count": 6144,
        "total_ms": 357.48212200000165,
        "mean_ms_per_cycle": 0.930943026041671,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_smallm_output_hadamard",
        "count": 98688,
        "total_ms": 332.4908029999615,
        "mean_ms_per_cycle": 0.865861466145733,
        "stream_span_count": 0
      },
      {
        "stage": "attention_split_reduce_cooperative",
        "count": 6144,
        "total_ms": 272.52125500000307,
        "mean_ms_per_cycle": 0.7096907682291747,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_smallm_input_hadamard",
        "count": 98688,
        "total_ms": 253.3572700001007,
        "mean_ms_per_cycle": 0.6597845572919289,
        "stream_span_count": 0
      }
    ]
  },
  "draft_decode": {
    "cycles": 384,
    "span_mean_ms": 6.585420466145833,
    "top_stages": [
      {
        "stage": "exl3_smallm_gemv",
        "count": 4992,
        "total_ms": 862.936329000002,
        "mean_ms_per_cycle": 2.2472300234375053,
        "stream_span_count": 0
      },
      {
        "stage": "attention_split_partial",
        "count": 1152,
        "total_ms": 568.3367480000002,
        "mean_ms_per_cycle": 1.4800436145833338,
        "stream_span_count": 0
      },
      {
        "stage": "mapped_greedy_argmax",
        "count": 1152,
        "total_ms": 250.66063799999952,
        "mean_ms_per_cycle": 0.6527620781249988,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_smallm_dpas",
        "count": 1920,
        "total_ms": 184.45091200000024,
        "mean_ms_per_cycle": 0.4803409166666673,
        "stream_span_count": 0
      },
      {
        "stage": "rms_norm_gemma5120_fp16",
        "count": 5760,
        "total_ms": 84.32181999999949,
        "mean_ms_per_cycle": 0.21958807291666535,
        "stream_span_count": 0
      },
      {
        "stage": "attn_qk_norm_rope_gate_subgroup",
        "count": 1152,
        "total_ms": 68.27498700000035,
        "mean_ms_per_cycle": 0.17779944531250091,
        "stream_span_count": 0
      },
      {
        "stage": "attention_split_reduce_cooperative",
        "count": 1152,
        "total_ms": 38.49385499999972,
        "mean_ms_per_cycle": 0.10024441406249927,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_smallm_output_hadamard",
        "count": 6912,
        "total_ms": 26.201682000000154,
        "mean_ms_per_cycle": 0.0682335468750004,
        "stream_span_count": 0
      },
      {
        "stage": "exl3_smallm_input_hadamard",
        "count": 6912,
        "total_ms": 22.85845199999974,
        "mean_ms_per_cycle": 0.059527218749999326,
        "stream_span_count": 0
      },
      {
        "stage": "staged_d2h_copy",
        "count": 11520,
        "total_ms": 15.216622000001038,
        "mean_ms_per_cycle": 0.03962661979166937,
        "stream_span_count": 0
      }
    ]
  }
}
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 39–64. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
    39: NUMERICAL AND PRACTICAL EVIDENCE
    40: 
    41: 1. Full target P128+64 native-owned replay, measured before the last graph-policy/norm changes:
    42:    - All65 full248320 logit rows byte-exact to a separately named controlled deterministic-BA original protocol. No shortened passing subset.
    43:    - Unchanged original frozen default reference gate still returns804/808, exit1: D27 TV/KL and D29 TV/KL fail. D27 TV.0305754/KL.00212581; D29 TV.212752/KL.163144; minimum top10 overlap9.
    44:    - Frozen TV<=.02, KL<=.002 and top10>=9 budgets are unchanged. Controlled-original equality does not silently promote that original protocol as the default reference.
    45:    - Scope C1/eager/no-MTP/P128+64 forced original prefix, not all contexts/states. Any reference-contract correction needs independently justified arithmetic, held-out prefix evidence and an explicit proposal.
    46:    Evidence: evidence/r02-D29-head-v1/receipt-controlled-D64-v103.json, controlled-D64-comparison-v102.json, controlled-D64-v101.log.
    47: 
    48: 2. Earlier causal repair: one FP16 SwiGLU midpoint at P128 block20 changed320 MLP halves and GDN21 history. Explicitly rounded F32 division before FP16 materialization closes the same-input MLP42/42 and actual GDN21 history230/230; observed active Conv/SSM become byte-exact. Subsequent controlled D29 all30 logits/all192 outer boundaries are exact. These are bounded positive results, not global strict-state clearance.
    49:    Evidence: receipt-swiglu-repair-v95.json and receipt-D29-swiglu-v98.json.
    50: 
    51: 3. Existing original isolated P128 compact draft/head replay passes131644/131644, exit0 on the post-numerics source: all655360 hidden values,65536 compact logits and independent65536 identical-input head outputs byte-exact after F32 widening; token map/mapped argmax exact. Earlier34-hidden/77-logit failures remain in historical receipts. This does not qualify every MTP3 iteration, graph or context, and was not rerun after the two latest shared changes.
    52:    Evidence: evidence/r06-compact-head-v1/receipt-current-numerical-v3.json and native-current-v3.log.
    53: 
    54: 4. Post-numerics practical suite passes native393/393 and semantic8/8: four code functions with20 execution cases, two60872-token retrieval tasks, two structured/tool tasks; all8 prompts/raw output sequences equal the historical passing v10, all finish stop. Requests were sequential with C4 capacity, not concurrent C4. Device peak30461556091B and graph bytes after release0.
    55:    IMPORTANT: this full suite predates the graph-policy and norm optimizations; current full practical-screen rerun remains pending. Do not label latest snapshot8/8 solely by transfer.
    56:    Evidence: receipt-capability-post-numerics-v1.json, capability-native-post-numerics-v1.json and capability-check-post-numerics-v1.json.
    57: 
    58: 5. Latest norm snapshot has its own focused unchanged original fixture proof:
    59:    - Focused build of both affected test binaries exit0.
    60:    - Physical rows3/4/12/16:821/821, one case, exit0; residual/no-residual, poisoned inactive row, alias, guards and strided refusal exact.
    61:    - Original P128/D1 Gemma boundaries:64/64, one case, exit0, all four outputs/residual boundaries exact.
    62:    - Current C1 graph/eager/separate eager profile:1975/1975 each, exit0. All1024 output IDs/all387 non-timing cycle fields match each other and the prior attention-policy candidate. Graph2 captures382 replays; accepted642/proposed1152. Graph versus eager comparison excludes mode-specific capture/replay fields; before/after graph capture/replay counts agree.
    63:    This proves the recorded bounded scopes, not a fresh full D64/state/draft/capability release gate.
    64:    Evidence: receipt-norm-workgroup-v1.json, original fixture logs, raw current graph/eager JSON and comparison-norm-workgroup-v1.json.
```

## E04 — Successful norm geometry to preserve

The latest short-row kernel reorganizes physical workgroups while retaining the virtual reduction tree and precision boundaries. The new plan does not redo the rejected width-only specialization.

**Source:** `code/src/vt/xpu/xpu_norm.cpp`, lines 1–97. SHA-256: `ed2581794e8f34b76782b1ed31625a09f9ab514048c6102ce554648f6d88b407`.

```text
     1: #include "xpu_common.h"
     2: #include "xpu_kernels.h"
     3: #include "xpu_qk_norm.h"
     4: namespace vt::xpu {
     5: namespace {
     6: template <bool Residual>
     7: float Gemma5120Value(View src, View res, int64_t row, int col) {
     8: #pragma clang fp contract(off)
     9:   const float value = Load(src, row * src.stride[0] + col);
    10:   if constexpr (Residual)
    11:     return value + Load(res, row * res.stride[0] + col);
    12:   return value;
    13: }
    14: 
    15: float Gemma5120OutputValue(float value, float inverse, float weight) {
    16: #pragma clang fp contract(off)
    17:   // Keep each producer F32 boundary with native arithmetic. The XPU target
    18:   // also uses -fno-fast-math/-ffp-contract=off; no fused multiply/add or
    19:   // reassociation is permitted. Avoid external directed-rounding library calls
    20:   // in this finite FP16-operand path.
    21:   const float effective_weight = 1.0f + weight;
    22:   const float normalized = value * inverse;
    23:   return normalized * effective_weight;
    24: }
    25: 
    26: template <bool Residual>
    27: void Gemma5120ShortRowsKernel(Queue& q, View dst, View src, View w, View res,
    28:                               int64_t rows, int width, float eps) {
    29:   // One work-group per row distributes the original virtual-lane partials.
    30:   // SLM exchanges preserve the same descending chunk tree, then the same
    31:   // ascending SG16 tree for each half of the producer's logical SG32.
    32:   constexpr int lanes = 16;
    33:   const int workgroup = width / 2, chunks = width / 32;
    34:   const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    35:     sycl::local_accessor<float, 1> sums(sycl::range<1>(width + 1), h);
    36:     h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * workgroup),
    37:                                     sycl::range<1>(workgroup)),
    38:         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    39: #pragma clang fp contract(off)
    40:       const int64_t row = item.get_group_linear_id();
    41:       const int local = item.get_local_linear_id();
    42:       const int chunk = local / lanes, lane = local % lanes;
    43:       for (int half = 0; half < 2; ++half) {
    44:         float regs[4] = {};
    45:         for (int col = 4 * (32 * chunk + lane + half * lanes); col < 5120;
    46:              col += 4 * width) {
    47:           for (int j = 0; j < 4; ++j) {
    48:             const float value = Gemma5120Value<Residual>(src, res, row, col + j);
    49:             regs[j] += value * value;
    50:           }
    51:         }
    52:         sums[half * workgroup + local] = ((regs[0] + regs[1]) + regs[2]) + regs[3];
    53:       }
    54:       item.barrier(sycl::access::fence_space::local_space);
    55:       for (int offset = chunks / 2; offset > 0; offset /= 2) {
    56:         if (chunk < offset) {
    57:           sums[local] += sums[local + offset * lanes];
    58:           sums[workgroup + local] += sums[workgroup + local + offset * lanes];
    59:         }
    60:         item.barrier(sycl::access::fence_space::local_space);
    61:       }
    62:       if (chunk == 0) {
    63:         auto group = item.get_sub_group();
    64:         float a = sums[lane], b = sums[workgroup + lane];
    65:         for (int offset = 1; offset < lanes; offset *= 2) {
    66:           a += sycl::shift_group_left(group, a, offset);
    67:           b += sycl::shift_group_left(group, b, offset);
    68:         }
    69:         const float mean = sycl::group_broadcast(group, a + b, 0) * (1.0f / 5120.0f);
    70:         if (lane == 0) sums[width] = sycl::rsqrt(mean + eps);
    71:       }
    72:       item.barrier(sycl::access::fence_space::local_space);
    73:       const float inverse = sums[width];
    74:       for (int col = local; col < 5120; col += workgroup) {
    75:         const float value = Gemma5120Value<Residual>(src, res, row, col);
    76:         if constexpr (Residual) Store(res, row * res.stride[0] + col, value);
    77:         Store(dst, row * dst.stride[0] + col,
    78:               Gemma5120OutputValue(value, inverse, Load(w, col)));
    79:       }
    80:     });
    81:   });
    82:   RecordProfileEvent(q, "rms_norm_gemma5120_fp16", event);
    83: }
    84: 
    85: template <bool Residual>
    86: void Gemma5120Kernel(Queue& q, View dst, View src, View w, View res,
    87:                      int64_t rows, float eps) {
    88:   // Pinned Torch ReduceConfig: max WG1024, logical SG32, contiguous vec4.
    89:   // Output count determines group_height; group_x_reduce first halves the
    90:   // virtual lanes down to32, then uses ascending subgroup offsets.
    91:   int height = 1;
    92:   while (height < 32 && height * 2 <= rows) height *= 2;
    93:   const int width = 1024 / height;
    94:   if (rows > 0 && rows <= 16) {
    95:     Gemma5120ShortRowsKernel<Residual>(q, dst, src, w, res, rows, width, eps);
    96:     return;
    97:   }
```

## E05 — Current W8A8 panel loop and guard cost

The 128-column panel is imposed in reconstruction, the GEMM loop and the public planner. Validation repeats a transformed-row computation. The already merged two-word status readback is visible and must not be proposed as a new change.

**Source:** `code/src/vt/xpu/xpu_exl3_w8a8.cpp`, lines 29–65. SHA-256: `ac21dd9ab8ad303c64cc44d4e6e82ddcec687193c121398ac39c6009e2d5492f`.

```text
    29: // The donor's float->INT8 conversion has no defined NaN/Inf contract. Reject
    30: // those inputs, including overflow at either FP16 Hadamard rounding boundary,
    31: // before writing any output. Bit tests remain valid in this fast-math TU.
    32: struct ValidateRows {
    33:   ::exl3::HadInQ8Kernel<sycl::half> input;
    34:   uint32_t* valid;
    35:   void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    36:     using namespace sycl::ext::intel::esimd;
    37:     const int kb_n = input.Kdim / 128;
    38:     const size_t id = it.get_global_id(0);
    39:     if (id >= size_t(input.S) * input.M * kb_n) return;
    40:     const int kb = int(id % kb_n), m = int(id / kb_n % input.M);
    41:     const int g = int(id / kb_n / input.M);
    42:     const auto xi = block_load<uint16_t, 128>(
    43:         reinterpret_cast<const uint16_t*>(input.x) + size_t(m) * input.Kdim + kb * 128);
    44:     const auto su = block_load<uint16_t, 128>(
    45:         reinterpret_cast<const uint16_t*>(input.suh) + size_t(g) * input.Kdim + kb * 128);
    46:     bool finite = !((xi & 0x7c00u) == 0x7c00u).any() &&
    47:                   !((su & 0x7c00u) == 0x7c00u).any();
    48:     // Check the product rounding separately: a subsequent max reduction is
    49:     // not a valid detector for a NaN that some lanes may ignore.
    50:     auto product_f32 =
    51:         convert<float>(block_load<sycl::half, 128>(input.x + size_t(m) * input.Kdim + kb * 128)) *
    52:         convert<float>(block_load<sycl::half, 128>(input.suh + size_t(g) * input.Kdim + kb * 128));
    53:     // 65520 is the round-to-nearest FP16 overflow boundary. In this fast-math
    54:     // TU a half round-trip can be optimized away, so validate the F32 bits
    55:     // before narrowing as well as checking the resulting half bits.
    56:     finite &= !((product_f32.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    57:     simd<sycl::half, 128> product = convert<sycl::half>(product_f32);
    58:     finite &= !((product.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
    59:     // Mirror blk's transform, checking the second boundary before its half
    60:     // conversion for the same reason. The producer itself remains unchanged.
    61:     auto transformed = convert<float>(product);
    62:     ::exl3::fwht128(transformed);
    63:     transformed *= ::exl3::kRsqrt128;
    64:     finite &= !((transformed.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    65:     transformed = convert<float>(convert<sycl::half>(transformed));
```

**Source:** `code/src/vt/xpu/xpu_exl3_w8a8.cpp`, lines 87–150. SHA-256: `ac21dd9ab8ad303c64cc44d4e6e82ddcec687193c121398ac39c6009e2d5492f`.

```text
    87: 
    88: template<int Bits>
    89: void Reconstruct(Queue& q, const Tensor& tr, Tensor& panel, int k, int n, int nb) {
    90:   ::exl3::ReconstructKernel<Bits, 2, 8, int8_t> kernel{
    91:       static_cast<const uint32_t*>(tr.data), static_cast<int8_t*>(panel.data),
    92:       n / 16, nb * 8, 128, 1, k / 16, 127.0f / 3.453125f};
    93:   Launch(q, k / 16, 8, kernel, "exl3_w8a8_weight_reconstruct");
    94: }
    95: }  // namespace
    96: 
    97: void Exl3GroupedW8A8Kernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    98:     const Tensor& suh, const Tensor& svh, const Tensor& shard,
    99:     Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
   100: #ifndef VLLM_CPP_XPU_GPTQ4
   101:   VT_CHECK(false, "EXL3 W8A8 requires the pinned oneDNN 3.13 build");
   102: #else
   103:   TraceXpuOp(OpId::kExl3GroupedW8A8, q,
   104:              {&out, &in, &tr, &suh, &svh, &shard, &workspace, &panel});
   105:   const int m = int(in.shape[0]), k = int(in.shape[1]), n = int(out.shape[1]);
   106:   const int groups = int(suh.shape[0]);
   107:   const auto plan = PlanExl3W8A8(m, k, n, groups, args.bits);
   108:   for (const Tensor* t : std::initializer_list<const Tensor*>{
   109:            &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel})
   110:     VT_CHECK(reinterpret_cast<uintptr_t>(t->data) % 16 == 0,
   111:              "EXL3 W8A8 storage requires 16-byte alignment");
   112:   for (const Tensor* dst : {&out, &workspace, &panel})
   113:     for (const Tensor* src : {&in, &tr, &suh, &svh, &shard})
   114:       VT_CHECK(!Overlap(*dst, *src), "EXL3 W8A8 writable storage may not overlap operands");
   115:   VT_CHECK(!Overlap(out, workspace) && !Overlap(out, panel) && !Overlap(workspace, panel),
   116:            "EXL3 W8A8 output and scratch may not overlap");
   117: 
   118:   // Eager initial implementation: mapping is small, and the completion wait
   119:   // also ensures that the panel's previous use has retired before reuse.
   120:   std::vector<int32_t> mapping(n / 128);
   121:   auto& backend = GetBackend(q.device);
   122:   backend.Copy(q, mapping.data(), shard.data, mapping.size() * sizeof(int32_t));
   123:   backend.Synchronize(q);
   124:   for (const int group : mapping)
   125:     VT_CHECK(group >= 0 && group < groups, "EXL3 W8A8 shard_of_nb group out of range");
   126:   const auto* sv_bits = static_cast<const uint16_t*>(svh.data);
   127: 
   128:   auto* bytes = static_cast<uint8_t*>(workspace.data);
   129:   auto* xq = reinterpret_cast<int8_t*>(bytes + plan.activation_offset);
   130:   auto* sx = reinterpret_cast<float*>(bytes + plan.row_scale_offset);
   131:   auto* y = reinterpret_cast<sycl::half*>(bytes + plan.intermediate_offset);
   132:   auto* sw = reinterpret_cast<float*>(bytes + plan.weight_scale_offset);
   133:   // Both GPU checks use spare words in the existing 64-byte scale region.
   134:   // Read their results together before output/panel/activation writes. This
   135:   // avoids a separate metadata allocation, readback and retirement drain.
   136:   auto* valid = reinterpret_cast<uint32_t*>(sw + 1);
   137:   NativeQueue(q).single_task([=] { valid[0] = 1; valid[1] = 1; });
   138:   Launch(q, n / 128, 8, ValidateOutputScale{sv_bits, n, valid + 1},
   139:          "exl3_w8a8_validate_svh");
   140:   ::exl3::HadInQ8Kernel<sycl::half> input{
   141:       static_cast<const sycl::half*>(in.data), static_cast<const sycl::half*>(suh.data),
   142:       xq, sx, m, k, groups, k, plan.padded_rows};
   143:   Launch(q, int64_t(groups) * m * (k / 128), 8, ValidateRows{input, valid},
   144:          "exl3_w8a8_validate_rows");
   145:   std::array<uint32_t, 2> finite{};
   146:   backend.Copy(q, finite.data(), valid, sizeof(finite));
   147:   backend.Synchronize(q);
   148:   // Preserve the previous error priority if both checks fail.
   149:   VT_CHECK(finite[1], "EXL3 W8A8 requires finite svh");
   150:   VT_CHECK(finite[0], "EXL3 W8A8 requires finite inputs and finite FP16 transformed rows");
```

**Source:** `code/src/vt/xpu/xpu_exl3_w8a8.cpp`, lines 153–191. SHA-256: `ac21dd9ab8ad303c64cc44d4e6e82ddcec687193c121398ac39c6009e2d5492f`.

```text
   153:   // Poison/stale padding is never consumed by GEMM. The active producer writes
   154:   // every byte of each real row. For zero rows its exact fallback is scale1.
   155:   NativeQueue(q).memset(xq, 0, size_t(groups) * plan.padded_rows * k);
   156:   NativeQueue(q).parallel_for(sycl::range<1>(size_t(groups) * plan.padded_rows),
   157:                             [=](sycl::id<1> i) { sx[i[0]] = 1.0f; });
   158:   NativeQueue(q).single_task([=] { *sw = 3.453125f / 127.0f; });
   159:   if (k / 128 <= 48) {
   160:     ::exl3::HadInQ8WgKernel<sycl::half, 8, 6> had{
   161:         input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
   162:     Launch(q, int64_t(groups) * m * 8, 8, had, "exl3_w8a8_input_quantize");
   163:   } else if (k / 128 <= 144) {
   164:     ::exl3::HadInQ8WgKernel<sycl::half, 16, 9> had{
   165:         input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
   166:     Launch(q, int64_t(groups) * m * 16, 16, had, "exl3_w8a8_input_quantize");
   167:   } else {
   168:     Launch(q, int64_t(groups) * m, 8, input, "exl3_w8a8_input_quantize");
   169:   }
   170: 
   171:   const auto weight_scale = Tensor::Contiguous(sw, DType::kF32, q.device, {1});
   172:   for (int nb = 0; nb < n / 128; ++nb) {
   173:     if (args.bits == 4) Reconstruct<4>(q, tr, panel, k, n, nb);
   174:     else Reconstruct<6>(q, tr, panel, k, n, nb);
   175:     const int group = mapping[nb];
   176:     const auto a = Tensor::Contiguous(xq + size_t(group) * plan.padded_rows * k,
   177:         DType::kI8, q.device, {plan.padded_rows, k});
   178:     const auto scales = Tensor::Contiguous(sx + size_t(group) * plan.padded_rows,
   179:         DType::kF32, q.device, {plan.padded_rows});
   180:     auto dst = Tensor::Contiguous(y + nb * 128, DType::kF16, q.device,
   181:                                   {plan.padded_rows, 128});
   182:     dst.stride[0] = n;
   183:     Exl3W8A8Matmul(q, dst, a, panel, scales, weight_scale);
   184:   }
   185:   // oneDNN rounds to F16 before the output Hadamard, as the production donor
   186:   // does. The fallback I32 HadOutQ8 recipe is a different arithmetic route.
   187:   ::exl3::HadOutKernel<sycl::half, sycl::half> tail{
   188:       y, static_cast<const sycl::half*>(svh.data), static_cast<sycl::half*>(out.data),
   189:       m, n, 1, n};
   190:   Launch(q, int64_t(m) * (n / 128), 8, tail, "exl3_w8a8_output_hadamard");
   191:   const char* setting = std::getenv("VT_XPU_EXL3_TRACE");
```

## E06 — Original larger-group W8A8 organization

The source reconstructs larger contiguous group spans before GEMM. Changing native output-panel width must still preserve the exact source-group input Hadamard/scales and the F16 intermediate before the output Hadamard.

**Source:** `reference/exl3-qualified-source/csrc/exl3_ops.sycl`, lines 718–777. SHA-256: `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`.

```text
   718:     if (g_int8 && cb == 2 && (K == 4 || K == 6) && M > 16) {
   719:         auto& q = queue_of(x2);
   720: #ifdef EXL3_DNNL
   721:         // GEMM rows padded to 256: bounds the number of oneDNN primitives (one per (Ms, K, N)) on real traffic
   722:         const int64_t Ms = (M + 255) / 256 * 256;
   723: #else
   724:         const int64_t Ms = M;
   725: #endif
   726:         auto xq = at::empty({S, Ms, k}, x.options().dtype(at::kChar));
   727:         auto sx = at::empty({S, Ms}, x.options().dtype(at::kFloat));
   728:         int n_in = S * M, local = 8;
   729:         auto run_in = [&](auto tag) {
   730:             using TIn = decltype(tag);
   731:             int kb_n = (int)(k / 128);
   732:             auto wg = [&](auto kern, int tpr) {
   733:                 q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)n_in * tpr, tpr), kern); });
   734:             };
   735:             if (kb_n <= 8 * 6) {
   736:                 wg(HadInQ8WgKernel<TIn, 8, 6>{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
   737:                                               reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
   738:                                               (int)x2.stride(0), (int)Ms}, 8);
   739:                 return;
   740:             }
   741:             if (kb_n <= 16 * 9) {
   742:                 wg(HadInQ8WgKernel<TIn, 16, 9>{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
   743:                                                reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
   744:                                                (int)x2.stride(0), (int)Ms}, 16);
   745:                 return;
   746:             }
   747:             HadInQ8Kernel<TIn> kin{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
   748:                                    reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
   749:                                    (int)x2.stride(0), (int)Ms};
   750:             q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n_in, local), local), kin); });
   751:         };
   752:         if (x2.scalar_type() == at::kHalf) run_in(fp16{});
   753:         else if (x2.scalar_type() == at::kBFloat16) run_in(sycl::ext::oneapi::bfloat16{});
   754:         else TORCH_CHECK(false, "x dtype");
   755:         const float sw = kQ8WMax / 127.0f;
   756: #ifdef EXL3_DNNL
   757:         {
   758:             static std::vector<at::Tensor> swd(16);
   759:             auto& swt = swd.at(x.device().index());
   760:             if (!swt.defined()) swt = at::full({1}, sw, x.options().dtype(at::kFloat));
   761:             auto y = at::empty({Ms, n}, x.options().dtype(at::kHalf));
   762:             for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
   763:                 int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1], ng = g1 - g0;
   764:                 auto w = at::empty({k, ng}, x.options().dtype(at::kChar));
   765:                 if (K == 4) launch_reconstruct_q8<4>(trellis, w, g0, q); else launch_reconstruct_q8<6>(trellis, w, g0, q);
   766:                 q8dnnl::gemm(q, x, reinterpret_cast<const int8_t*>(xq.data_ptr()) + g * Ms * k,
   767:                              reinterpret_cast<const int8_t*>(w.data_ptr()), swt.data_ptr<float>(),
   768:                              sx.data_ptr<float>() + g * Ms, reinterpret_cast<fp16*>(y.data_ptr()) + g0, Ms, k, ng, n);
   769:             }
   770:             exl3_had_out_h(y.narrow(0, 0, M), svh, out);
   771:             return out.view(shape);
   772:         }
   773: #endif
   774:         for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
   775:             int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1], ng = g1 - g0;
   776:             auto w = at::empty({k, ng}, x.options().dtype(at::kChar));
   777:             if (K == 4) launch_reconstruct_q8<4>(trellis, w, g0, q); else launch_reconstruct_q8<6>(trellis, w, g0, q);
```

## E07 — Panel allocation and public shape contract

Both the plan and model allocation sites currently require K×128. A real larger-panel implementation must change these contracts together, with bounded completion-owned reuse.

**Source:** `code/src/vt/exl3_grouped.cpp`, lines 6–70. SHA-256: `946d73662e370a478fab866e9c139de68cac09c1419f9d61177fe23d04bc1ff9`.

```text
     6: Exl3W8A8Plan PlanExl3W8A8(int64_t m, int64_t k, int64_t n,
     7:                          int64_t groups, int bits) {
     8:   VT_CHECK(m > 128 && m <= 4096, "EXL3 W8A8 requires physical M in [129,4096]");
     9:   VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
    10:                k <= std::numeric_limits<int>::max() &&
    11:                n <= std::numeric_limits<int>::max() && groups > 0 &&
    12:                groups <= std::numeric_limits<int>::max() / m / 16,
    13:            "EXL3 W8A8 invalid geometry or I32 input launch overflow");
    14:   VT_CHECK(bits == 4 || bits == 6, "EXL3 W8A8 supports 4/6bpw only");
    15:   VT_CHECK(k <= std::numeric_limits<int32_t>::max() / (127 * 127),
    16:            "EXL3 W8A8 K exceeds bounded signed INT32 accumulation");
    17:   VT_CHECK(m * (n / 128) <= std::numeric_limits<int>::max(),
    18:            "EXL3 W8A8 output launch exceeds I32 indexing");
    19:   const int ms = int((m + 255) / 256 * 256);
    20:   auto multiply = [](size_t a, size_t b) {
    21:     VT_CHECK(b == 0 || a <= (std::numeric_limits<size_t>::max() - 63) / b,
    22:              "EXL3 W8A8 workspace size overflow");
    23:     return a * b;
    24:   };
    25:   size_t cursor = 0;
    26:   auto region = [&](size_t bytes) {
    27:     const size_t start = cursor;
    28:     VT_CHECK(bytes <= std::numeric_limits<size_t>::max() - cursor - 63,
    29:              "EXL3 W8A8 workspace size overflow");
    30:     cursor += (bytes + 63) / 64 * 64;
    31:     return start;
    32:   };
    33:   const size_t rows = multiply(size_t(groups), size_t(ms));
    34:   const size_t act = region(multiply(rows, size_t(k)));
    35:   const size_t sx = region(multiply(rows, sizeof(float)));
    36:   const size_t y = region(multiply(multiply(size_t(ms), size_t(n)), size_t{2}));
    37:   const size_t sw = region(sizeof(float));
    38:   return {ms, act, sx, y, sw, cursor, multiply(size_t(k), size_t{128})};
    39: }
    40: 
    41: void Exl3GroupedW8A8(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    42:     const Tensor& suh, const Tensor& svh, const Tensor& shard,
    43:     Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
    44:   VT_CHECK(in.rank == 2 && out.rank == 2 && suh.rank == 2,
    45:            "EXL3 W8A8 requires rank-2 input/output/suh");
    46:   const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
    47:   const auto plan = PlanExl3W8A8(m, k, n, suh.shape[0], args.bits);
    48:   VT_CHECK(args.codebook == 2 && out.shape[0] == m &&
    49:                in.dtype == DType::kF16 && out.dtype == DType::kF16,
    50:            "EXL3 W8A8 requires mul1 and matching F16 input/output");
    51:   VT_CHECK(tr.dtype == DType::kI8 && tr.rank == 3 && tr.shape[0] == k / 16 &&
    52:                tr.shape[1] == n / 16 && tr.shape[2] == 32 * args.bits,
    53:            "EXL3 W8A8 packed trellis layout mismatch");
    54:   VT_CHECK(suh.dtype == DType::kF16 && suh.shape[1] == k &&
    55:                svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
    56:            "EXL3 W8A8 requires F16 suh[S,K] and svh[N]");
    57:   VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
    58:            "EXL3 W8A8 requires I32 shard_of_nb[N/128]");
    59:   VT_CHECK(workspace.dtype == DType::kI8 && workspace.rank == 1 &&
    60:                workspace.shape[0] >= 0 && size_t(workspace.shape[0]) >= plan.workspace_bytes,
    61:            "EXL3 W8A8 byte workspace too small or wrong layout");
    62:   VT_CHECK(panel.dtype == DType::kI8 && panel.rank == 2 &&
    63:                panel.shape[0] == k && panel.shape[1] == 128,
    64:            "EXL3 W8A8 requires bounded I8[K,128] weight panel");
    65:   for (const Tensor* t : std::initializer_list<const Tensor*>{
    66:            &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel}) {
    67:     VT_CHECK(t->IsContiguous(), "EXL3 W8A8 requires contiguous tensors");
    68:     VT_CHECK(t->device == q.device, "EXL3 W8A8 device mismatch");
    69:   }
    70:   reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedW8A8, q.device.type))(
```

**Source:** `code/include/vllm/model_executor/models/dense_attn_block.h`, lines 348–388. SHA-256: `a4717b99b870803129042b6bb355017d86323595903d329bff2c0048c7710f85`.

```text
   348: inline DBuf Exl3GroupedMatmulD(Dev d, const vt::Tensor& x,
   349:                                const Exl3GroupedWeight& w) {
   350:   const int64_t M = x.shape[0], K = w.suh.shape[1], N = w.svh.shape[0];
   351:   VT_CHECK(d.q.device.type == vt::DeviceType::kXPU &&
   352:                d.activation_dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 &&
   353:                x.rank == 2 && x.shape[1] == K && w.codebook == 2,
   354:            "exl3 grouped model: requires scoped XPU FP16 and matching K/mul1");
   355:   const int bits = static_cast<int>(w.trellis.shape[2] / 32);
   356:   // Validate arithmetic/extent before uploading weights. Large prefills use
   357:   // the independent signed INT8 producer route, without dense reconstruction.
   358:   if (M > 128) (void)vt::PlanExl3W8A8(M, K, N, w.suh.shape[0], bits);
   359:   else (void)vt::PlanExl3SmallM(M, K, N, bits);
   360:   const bool upload = !w.trellis.d_dev || !w.suh.d_dev || !w.svh.d_dev || !w.source_map.d_dev;
   361:   auto trellis = ResidentWeight(d, w.trellis);
   362:   auto suh = ResidentWeight(d, w.suh);
   363:   auto svh = ResidentWeight(d, w.svh);
   364:   auto map = ResidentWeight(d, w.source_map);
   365:   if (upload) {
   366:     d.b.Synchronize(d.q);
   367:     w.trellis.ReleaseHost();
   368:     w.suh.ReleaseHost();
   369:     w.svh.ReleaseHost();
   370:     w.source_map.ReleaseHost();
   371:   }
   372:   DBuf out(d, vt::DType::kF16, {M, N});
   373:   if (M > 128) {
   374:     const auto plan = vt::PlanExl3W8A8(M, K, N, w.suh.shape[0], bits);
   375:     DBuf workspace(d, vt::DType::kI8, {static_cast<int64_t>(plan.workspace_bytes)});
   376:     DBuf panel(d, vt::DType::kI8, {K, 128});
   377:     vt::Exl3GroupedW8A8(d.q, out.t(), x, trellis, suh, svh, map,
   378:                        workspace.t(), panel.t(), {bits, w.codebook, w.name.c_str()});
   379:     // DBuf retirement completes these eager consumers before scratch reuse.
   380:     return out;
   381:   }
   382:   const auto plan = vt::PlanExl3SmallM(M, K, N, bits);
   383:   DBuf had(d, vt::DType::kF16, {w.suh.shape[0], K / 16, plan.padded_rows, 16});
   384:   DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
   385:   vt::Exl3GroupedLinear(d.q, out.t(), x, trellis, suh, svh, map,
   386:                        had.t(), parts.t(), {bits, w.codebook, w.name.c_str()});
   387:   return out;
   388: }
```

**Source:** `code/include/vllm/model_executor/models/dense_attn_block.h`, lines 410–435. SHA-256: `a4717b99b870803129042b6bb355017d86323595903d329bff2c0048c7710f85`.

```text
   410:     if (M > 128) (void)vt::PlanExl3W8A8(M, K, N, 1, w.Bits());
   411:     else (void)vt::PlanExl3SmallM(M, K, N, w.Bits());
   412:     auto trellis = ResidentWeight(d, w.trellis);
   413:     auto suh = Reshape(ResidentWeight(d, w.suh), {1, K});
   414:     auto svh = ResidentWeight(d, w.svh);
   415:     if (w.single_source_map.rank == 0) {
   416:       w.single_source_map.dtype = vt::DType::kI32;
   417:       w.single_source_map.rank = 1;
   418:       w.single_source_map.shape[0] = N / 128;
   419:     }
   420:     Tensor shard = ResidentWeight(d, w.single_source_map, {}, [&](Tensor& t) {
   421:       d.b.Memset(d.q, t.data, 0, t.Bytes());
   422:     });
   423:     DBuf out(d, vt::DType::kF16, {M, N});
   424:     if (M > 128) {
   425:       const auto plan = vt::PlanExl3W8A8(M, K, N, 1, w.Bits());
   426:       DBuf workspace(d, vt::DType::kI8, {static_cast<int64_t>(plan.workspace_bytes)});
   427:       DBuf panel(d, vt::DType::kI8, {K, 128});
   428:       vt::Exl3GroupedW8A8(d.q, out.t(), x, trellis, suh, svh, shard,
   429:                          workspace.t(), panel.t(), {w.Bits(), w.codebook, w.name.c_str()});
   430:       return out;
   431:     }
   432:     const auto plan = vt::PlanExl3SmallM(M, K, N, w.Bits());
   433:     DBuf had(d, vt::DType::kF16, {1, K / 16, plan.padded_rows, 16});
   434:     DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
   435:     vt::Exl3GroupedLinear(d.q, out.t(), x, trellis, suh, svh, shard,
```

## E08 — Exact FP16 SiLU boundary and the proposed finite-domain optimization

The exact lookup proposal is reviewer reasoning derived from the FP16 input/output boundary: 65,536 bit patterns ×2 bytes =128 KiB. It is not present in the source and is not a measured gain. The table must reproduce this actual operator, not an approximate analytic sigmoid.

**Source:** `code/src/vt/xpu/xpu_elementwise.cpp`, lines 65–110. SHA-256: `b9c75e6b4ddc9dbb757f731ade3383ecd0ef8254a80452be91d76c9006e2f1f3`.

```text
    65:     const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
    66:       const auto i = item[0]; const float g = Load(gv, gv.offset(i));
    67:       const float act = Round(gv.dtype, g / (1.0f + sycl::exp(-g)));
    68:       Store(dst, dst.offset(i), act * Load(uv, uv.offset(i)));
    69:     });
    70:     RecordProfileEvent(q, "moe_silu_mul", event);
    71:   });
    72: }
    73: void SiluAndMulKernel(Queue& q, Tensor& out, const Tensor& in) {
    74:   TraceXpuOp(OpId::kSiluAndMul, q, {&out, &in});
    75:   FloatTensor(out); FloatTensor(in);
    76:   WithOutput(q, out, {&in}, [&](Tensor& target) {
    77:     const View dst(target), src(in);
    78:     const auto width = out.shape[1];
    79:     const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
    80:       const auto i = item[0]; const auto off = (i / width) * src.stride[0] + i % width;
    81:       const float gate = Load(src, off);
    82:       // Preserve the eager FP16 SiLU boundary before multiplying by up.
    83:       // An approximate division can cross a half midpoint on real MLP rows.
    84:       const float denominator = 1.0f + sycl::exp(-gate);
    85:       const float silu = src.dtype == DType::kF16 && dst.dtype == DType::kF16
    86:           ? sycl::ext::intel::math::fdiv_rn(gate, denominator) : gate / denominator;
    87:       Store(dst, dst.offset(i), Round(src.dtype, silu) * Load(src, off + width));
    88:     });
    89:     RecordProfileEvent(q, "silu_and_mul", event);
    90:   });
    91: }
    92: void SigmoidGateKernel(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
    93:   TraceXpuOp(OpId::kSigmoidGateBf16, q, {&out, &attn, &gate});
    94:   FloatTensor(out); FloatTensor(attn); FloatTensor(gate);
    95:   WithOutput(q, out, {&attn, &gate}, [&](Tensor& target) {
    96:     const View dst(target), av(attn), gv(gate);
    97:     const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
    98:       const auto i = item[0];
    99:       if (dst.dtype == DType::kF16 && av.dtype == DType::kF16) {
   100:         // Eager FP16 attention multiplies by a materialized FP16 sigmoid.
   101:         // Keep the gate input's F32 values, then narrow only this result.
   102:         const float value = Load(gv, gv.offset(i));
   103:         const float sigmoid = Round(DType::kF16,
   104:             sycl::ext::intel::math::fdiv_rn(1.0f, 1.0f + sycl::exp(-value)));
   105:         Store(dst, dst.offset(i), Load(av, av.offset(i)) * sigmoid);
   106:         return;
   107:       }
   108:       Store(dst, dst.offset(i), Load(av, av.offset(i)) * (1.0f / (1.0f + sycl::exp(-Load(gv, gv.offset(i))))));
   109:     });
   110:     RecordProfileEvent(q, "sigmoid_gate", event);
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 41–49. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
    41: 1. Full target P128+64 native-owned replay, measured before the last graph-policy/norm changes:
    42:    - All65 full248320 logit rows byte-exact to a separately named controlled deterministic-BA original protocol. No shortened passing subset.
    43:    - Unchanged original frozen default reference gate still returns804/808, exit1: D27 TV/KL and D29 TV/KL fail. D27 TV.0305754/KL.00212581; D29 TV.212752/KL.163144; minimum top10 overlap9.
    44:    - Frozen TV<=.02, KL<=.002 and top10>=9 budgets are unchanged. Controlled-original equality does not silently promote that original protocol as the default reference.
    45:    - Scope C1/eager/no-MTP/P128+64 forced original prefix, not all contexts/states. Any reference-contract correction needs independently justified arithmetic, held-out prefix evidence and an explicit proposal.
    46:    Evidence: evidence/r02-D29-head-v1/receipt-controlled-D64-v103.json, controlled-D64-comparison-v102.json, controlled-D64-v101.log.
    47: 
    48: 2. Earlier causal repair: one FP16 SwiGLU midpoint at P128 block20 changed320 MLP halves and GDN21 history. Explicitly rounded F32 division before FP16 materialization closes the same-input MLP42/42 and actual GDN21 history230/230; observed active Conv/SSM become byte-exact. Subsequent controlled D29 all30 logits/all192 outer boundaries are exact. These are bounded positive results, not global strict-state clearance.
    49:    Evidence: receipt-swiglu-repair-v95.json and receipt-D29-swiglu-v98.json.
```

## E09 — Native verification rejects page1600 and is not default-on

This is the direct native admission/dispatch evidence. Actual original execution and numerical/layout equivalence are still required before changing admission.

**Source:** `code/src/vt/xpu/xpu_attention_verify_xe2.cpp`, lines 10–83. SHA-256: `8095e1813fe859852437663cb5189733e76e8499dd86e2af28cbe4451bd24ae8`.

```text
    10: namespace vt::xpu {
    11: namespace {
    12: using namespace cute;
    13: using VerifyQ8 = PagedDecodeConfig<Shape<_8, _64, _64>,
    14:     Shape<_8, _32, _64>, Shape<_8, _256>, Layout<Shape<_1, _4, _1>>,
    15:     void, 1, true, false, false, half_t, float_e4m3_t, float_e4m3_t, half_t>;
    16: 
    17: constexpr size_t Align(size_t bytes) { return (bytes + 63) & ~size_t{63}; }
    18: }  // namespace
    19: 
    20: bool PagedAttentionXe2VerifyKernel(Queue& q, Tensor& out, const Tensor& query,
    21:     const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    22:     const Tensor& seq_lens, const Tensor& query_start_loc,
    23:     const PagedAttentionArgs& args) {
    24:   const auto& device = NativeQueue(q).get_device();
    25:   const int64_t tokens = query.shape[0];
    26:   const int64_t page = key_cache.shape[1];
    27:   if (tokens < 2 || tokens > 5 || query.rank != 3 || out.rank != 3 ||
    28:       query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
    29:       query.shape[1] != 24 || query.shape[2] != 256 ||
    30:       out.shape[0] != tokens || out.shape[1] != 24 || out.shape[2] != 256 ||
    31:       query.stride[0] != 24 * 256 || query.stride[1] != 256 ||
    32:       query.stride[2] != 1 || out.stride[0] != 24 * 256 ||
    33:       out.stride[1] != 256 || out.stride[2] != 1 ||
    34:       key_cache.rank != 4 || value_cache.rank != 4 ||
    35:       key_cache.dtype != DType::kI8 || value_cache.dtype != DType::kI8 ||
    36:       args.kv_cache_dtype != Fp8KVCacheDataType::kFp8E4M3 ||
    37:       key_cache.shape[0] != value_cache.shape[0] ||
    38:       page != 1664 || value_cache.shape[1] != page ||
    39:       key_cache.shape[2] != 4 || value_cache.shape[2] != 4 ||
    40:       key_cache.shape[3] != 256 || value_cache.shape[3] != 256 ||
    41:       key_cache.stride[1] != 4 * 256 ||
    42:       value_cache.stride[1] != key_cache.stride[1] ||
    43:       key_cache.stride[2] != key_cache.stride[1] / 4 ||
    44:       value_cache.stride[2] != key_cache.stride[2] ||
    45:       key_cache.stride[3] != 1 || value_cache.stride[3] != 1 ||
    46:       key_cache.stride[0] != value_cache.stride[0] ||
    47:       key_cache.stride[0] % key_cache.stride[1] != 0 ||
    48:       block_table.rank != 2 || block_table.dtype != DType::kI32 ||
    49:       block_table.shape[0] != 1 || block_table.stride[1] != 1 ||
    50:       seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 ||
    51:       seq_lens.Numel() != 1 || query_start_loc.rank != 1 ||
    52:       query_start_loc.dtype != DType::kI32 || query_start_loc.Numel() != 2 ||
    53:       args.max_seq_len < tokens ||
    54:       block_table.shape[1] < (args.max_seq_len + page - 1) / page ||
    55:       !args.causal || args.window_size || args.logits_soft_cap != 0 ||
    56:       args.k_scale != 1.0f || args.v_scale != 1.0f ||
    57:       args.scale != 1.0f / 16 ||
    58:       (reinterpret_cast<uintptr_t>(query.data) & 15) ||
    59:       (reinterpret_cast<uintptr_t>(out.data) & 15) ||
    60:       (reinterpret_cast<uintptr_t>(key_cache.data) & 15) ||
    61:       (reinterpret_cast<uintptr_t>(value_cache.data) & 15) ||
    62:       !device.has(sycl::aspect::ext_intel_device_id) ||
    63:       device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
    64:       std::string_view(__VERSION__) !=
    65:           "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
    66:       device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10" ||
    67:       key_cache.shape[0] > std::numeric_limits<int>::max() /
    68:           (key_cache.stride[0] / key_cache.stride[1]))
    69:     return false;
    70: 
    71:   const int splits = tokens == 2 ? 32 : tokens == 3 ? 8 : 16;
    72:   const size_t packed_bytes = size_t(tokens) * 24 * 256 * sizeof(uint16_t);
    73:   const size_t temp_bytes = packed_bytes * splits;
    74:   const size_t stats_bytes = size_t(tokens) * 24 * splits * sizeof(float);
    75:   const size_t total = 2 * Align(packed_bytes) + Align(temp_bytes) +
    76:       2 * Align(stats_bytes) + Align(2 * sizeof(float));
    77:   // Split-K may reuse this queue later and needs the full persistent workspace.
    78:   VT_CHECK(total <= 16 * 1024 * 1024, "verification workspace exceeds Split-K allocation");
    79:   return WithAttentionWorkspace(q, 16 * 1024 * 1024, [&](void* storage) {
    80:     auto* ptr = static_cast<unsigned char*>(storage);
    81:     auto* packed_q = reinterpret_cast<uint16_t*>(ptr); ptr += Align(packed_bytes);
    82:     auto* packed_out = reinterpret_cast<uint16_t*>(ptr); ptr += Align(packed_bytes);
    83:     auto* temp = reinterpret_cast<uint16_t*>(ptr); ptr += Align(temp_bytes);
```

**Source:** `code/src/vt/xpu/xpu_attention_verify_xe2.cpp`, lines 95–155. SHA-256: `8095e1813fe859852437663cb5189733e76e8499dd86e2af28cbe4451bd24ae8`.

```text
    95:       const int row = (head / 6) % int(tokens);
    96:       const int group_head = head % 6;
    97:       packed_q[index] = src[(row * 24 + kv_head * 6 + group_head) * 256 + dim];
    98:     });
    99:     RecordProfileEvent(q, "attention_verify_pack", pack);
   100: 
   101:     paged_decode_args_t donor{};
   102:     donor.query = packed_q;
   103:     donor.key = key_cache.data;
   104:     donor.value = value_cache.data;
   105:     donor.out = packed_out;
   106:     donor.tem_out = temp;
   107:     donor.exp_sums = sums;
   108:     donor.max_logits = maxima;
   109:     donor.block_table = block_table.data;
   110:     donor.cu_seqlens_q = query_start_loc.data;
   111:     donor.cu_seqlens_k = seq_lens.data;
   112:     donor.max_queries = 1;
   113:     donor.max_keys = args.max_seq_len;
   114:     donor.total_seqlen_q = 1;
   115:     donor.total_seqlen_k = key_cache.shape[0] *
   116:         (key_cache.stride[0] / key_cache.stride[1]);
   117:     donor.k_scale = scales;
   118:     donor.v_scale = scales + 1;
   119:     donor.sm_scale = args.scale;
   120:     donor.batch_size = 1;
   121:     donor.num_heads_q = tokens * 24;
   122:     donor.num_heads_k = 4;
   123:     donor.head_size = 256;
   124:     donor.v_head_size = 256;
   125:     donor.max_blocks_per_seq = block_table.shape[1];
   126:     donor.block_size = page;
   127:     donor.is_varlen = true;
   128:     donor.is_paged = true;
   129:     donor.is_causal = true;
   130:     donor.num_kv_splits = splits;
   131:     donor.q_stride_seq = tokens * 24 * 256;
   132:     donor.q_stride_heads = 256;
   133:     donor.k_stride_page = key_cache.stride[0];
   134:     donor.k_stride_seq = key_cache.stride[1];
   135:     donor.k_stride_heads = key_cache.stride[2];
   136:     donor.v_stride_page = value_cache.stride[0];
   137:     donor.v_stride_seq = value_cache.stride[1];
   138:     donor.v_stride_heads = value_cache.stride[2];
   139:     donor.page_stride_elements = key_cache.stride[0] / key_cache.stride[1];
   140:     VerifyQ8::kernel_dispatch(NativeQueue(q), donor);
   141: 
   142:     auto* dst = static_cast<uint16_t*>(out.data);
   143:     const auto unpack = NativeQueue(q).parallel_for(sycl::range<1>(count),
   144:         [=](sycl::id<1> item) {
   145:       const int index = item[0], dim = index % 256;
   146:       const int head = index / 256;
   147:       const int row = head / 24, kv_head = (head / 6) % 4;
   148:       const int group_head = head % 6;
   149:       dst[index] = packed_out[((kv_head * int(tokens) + row) * 6 +
   150:                                 group_head) * 256 + dim];
   151:     });
   152:     RecordProfileEvent(q, "attention_verify_unpack", unpack);
   153:   });
   154: }
   155: }  // namespace vt::xpu
```

**Source:** `code/src/vt/xpu/xpu_attention.cpp`, lines 488–512. SHA-256: `d1711439eddab0a5bae82f1aa51c92f4b87ffa871ff0bc9ca46f70a7d9845f56`.

```text
   488:     if (mode == "exl3_onednn" || (automatic && tokens > 128))
   489:       onednn = PagedAttentionExl3OneDnnKernel(q, target, query, key_cache, value_cache,
   490:           block_table, seq_lens, query_start_loc, args);
   491: #endif
   492:     bool verify = false;
   493: #ifdef VLLM_CPP_XPU_XE2_VERIFY
   494:     const char* verify_setting = std::getenv("VT_XPU_XE2_VERIFY");
   495:     // Original short C1 decode uses FP16 probabilities and XMX P*V, with
   496:     // one split below16 KV tiles. The same native donor is already used for
   497:     // optional packed verification; its own admission guards both routes.
   498:     if ((mode == "verify" || automatic) && tokens == 1)
   499:       verify = PagedAttentionXe2DecodeKernel(
   500:           q, target, query, key_cache, value_cache,
   501:           block_table, seq_lens, query_start_loc, args);
   502:     if (!verify && (mode == "verify" || (automatic && verify_setting &&
   503:                               std::string_view(verify_setting) == "1")))
   504:       verify = PagedAttentionXe2VerifyKernel(
   505:           q, target, query, key_cache, value_cache,
   506:           block_table, seq_lens, query_start_loc, args);
   507: #endif
   508:     const bool try_split = !onednn && !verify &&
   509:         (automatic || mode == "split" || mode == "prefill" || mode == "verify" || mode == "exl3_onednn");
   510:     const bool split = try_split && PagedAttentionSplitKernel(
   511:         q, target, query, key_cache, value_cache,
   512:         block_table, seq_lens, query_start_loc, args);
```

## E10 — Matching original page1600/C4-capable verifier source, recovered from prior upload

The two implementations below were recovered from the already uploaded `B70_EXL3_Current_Production_and_Power_Review_2026-10-02.zip`, not guessed from the old GPTQ/M04 implementation. Their hashes match that archive's manifest. The current and earlier captured production dispatcher both hash to `e3284ba808faeb53670a7546c686c7cf77156d78cdb54884ecd602aecb8beda5`; the captured original image is the same `8d0e1d...f97918`. This establishes matching captured identity, not a fresh GPU dispatch trace. The bounded P1 original capture is still required. The wrapper supports batch1–16 but the production dispatcher limits the qualified domain; do not widen it merely because the inner wrapper permits more.

**Source:** `supplement/exl3-qualified-source/exl3xpu/shared_kv_verify.py`, lines 21–46. SHA-256: `efb06f59250cbf4efb16f3eb6eb119bbd4c24c9ff086459b4afb85cc21adacb2`.

```text
    21: def eligible(d):
    22:     if set(d) - KNOWN or any(d.get(n) is None for n in
    23:                             ('q', 'k', 'v', 'cu_seqlens_q', 'seqused_k',
    24:                              'block_table', 'k_descale', 'v_descale')):
    25:         return False
    26:     q, k, v = d['q'], d['k'], d['v']
    27:     cq, used, bt = d['cu_seqlens_q'], d['seqused_k'], d['block_table']
    28:     batch, rows = cq.numel() - 1, d.get('max_seqlen_q')
    29:     if not isinstance(rows, int) or not 2 <= rows <= 5 or not 1 <= batch <= 16:
    30:         return False
    31:     if (q.device.type != 'xpu' or q.dtype != torch.float16 or
    32:             tuple(q.shape) != (batch * rows, 24, 256) or not q.is_contiguous()):
    33:         return False
    34:     if (k.dtype != torch.float8_e4m3fn or v.dtype != k.dtype or k.ndim != 4 or
    35:             k.shape[1] <= 0 or k.shape[1] % 64 or tuple(k.shape[2:]) != (4, 256) or
    36:             v.shape != k.shape or v.stride() != k.stride()):
    37:         return False
    38:     strides = k.stride()
    39:     if (strides[3] != 1 or strides[2] < 256 or strides[1] < 4 * strides[2] or
    40:             strides[0] < k.shape[1] * strides[1]):
    41:         return False
    42:     if (cq.ndim != 1 or cq.dtype != torch.int32 or not cq.is_contiguous() or
    43:             used.ndim != 1 or used.numel() != batch or used.dtype != torch.int32 or
    44:             not used.is_contiguous() or bt.ndim != 2 or bt.shape[0] != batch or
    45:             bt.dtype != torch.int32 or not bt.is_contiguous()):
    46:         return False
```

**Source:** `supplement/exl3-qualified-source/exl3xpu/shared_kv_verify.py`, lines 87–123. SHA-256: `efb06f59250cbf4efb16f3eb6eb119bbd4c24c9ff086459b4afb85cc21adacb2`.

```text
    87: def packed_queries(q, batch, rows):
    88:     return q.view(batch, rows, 4, 6, 256).permute(0, 2, 1, 3, 4).reshape(batch, rows * 24, 256).contiguous()
    89: 
    90: 
    91: def unpacked_output(y, batch, rows):
    92:     return y.view(batch, 4, rows, 6, 256).permute(0, 2, 1, 3, 4).reshape(batch * rows, 24, 256).contiguous()
    93: 
    94: 
    95: def run(d, splits=None, tile=8):
    96:     batch, rows = d['cu_seqlens_q'].numel() - 1, d['max_seqlen_q']
    97:     q = packed_queries(d['q'], batch, rows)
    98:     y = torch.empty_like(q)
    99:     # Initial historical policy, to be measured independently for C1/C4.
   100:     splits = splits if splits is not None else {2: 32, 3: 8, 4: 16, 5: 16}[rows]
   101:     temp = torch.empty((batch, rows * 24 * splits, 256), dtype=q.dtype, device=q.device)
   102:     sums = torch.empty((batch, rows * 24, splits), dtype=torch.float32, device=q.device)
   103:     maxima = torch.empty_like(sums)
   104:     cu = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
   105:     ks = d['k_descale'].as_strided((1,), (1,))
   106:     vs = d['v_descale'].as_strided((1,), (1,))
   107:     torch.ops.b70_exl3_attention.shared_kv_verify_out(
   108:         q, d['k'], d['v'], d['block_table'], cu, d['seqused_k'], ks, vs,
   109:         y, temp, sums, maxima, d['max_seqlen_k'], splits, tile)
   110:     out = d.get('out')
   111:     if out is not None:
   112:         # Matched q4 repeats support this copy at C4; C1 was slightly
   113:         # slower with the direct strided copy. Keep its qualified layout path.
   114:         if batch == 4:
   115:             out.view(batch, rows, 4, 6, 256).copy_(
   116:                 y.view(batch, 4, rows, 6, 256).permute(0, 2, 1, 3, 4))
   117:         else:
   118:             out.copy_(unpacked_output(y, batch, rows))
   119:         return out
   120:     return unpacked_output(y, batch, rows)
   121: 
   122: 
   123: def dispatch(original, **d):
```

**Source:** `supplement/engine-code/benchmarks/experiments/exl3-shared-kv-verify/csrc/shared_kv_verification.cpp`, lines 25–78. SHA-256: `8e28d338b95ceee1d57c7828324d0f27ab81a6714099b233e47727ebedf331dd`.

```text
    25:               "q and out must be FP16");
    26:   TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn &&
    27:                   v.scalar_type() == k.scalar_type(),
    28:               "k and v must be E4M3FN");
    29:   TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= 16 && q.size(2) == 256 &&
    30:                   q.size(1) >= 48 && q.size(1) <= 120 && q.size(1) % 24 == 0,
    31:               "packed q must have shape [B,48|72|96|120,256], B in [1,16]");
    32:   TORCH_CHECK(k.dim() == 4 && k.size(1) > 0 && k.size(1) % 64 == 0 && k.size(2) == 4 &&
    33:                   k.size(3) == 256 && v.sizes() == k.sizes() &&
    34:                   v.strides() == k.strides(),
    35:               "k/v must share a positive multiple-of-64 page shape and strides");
    36:   TORCH_CHECK(q.is_contiguous() && out.is_contiguous() &&
    37:                   out.sizes() == q.sizes(),
    38:               "q/out must be contiguous and have equal shapes");
    39:   TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == q.size(0) &&
    40:                   block_table.scalar_type() == at::kInt &&
    41:                   block_table.is_contiguous(),
    42:               "block_table must have one contiguous int32 row per sequence");
    43:   TORCH_CHECK(cu_seqlens_q.numel() == q.size(0) + 1 &&
    44:                   cu_seqlens_q.scalar_type() == at::kInt &&
    45:                   seqused_k.numel() == q.size(0) &&
    46:                   seqused_k.scalar_type() == at::kInt,
    47:               "invalid sequence metadata");
    48:   TORCH_CHECK(k_scale.scalar_type() == at::kFloat &&
    49:                   v_scale.scalar_type() == at::kFloat &&
    50:                   k_scale.numel() == 1 && v_scale.numel() == 1,
    51:               "k/v scales must be scalar FP32 tensors");
    52:   TORCH_CHECK(num_splits >= 1 && num_splits <= 32,
    53:               "num_splits must be in [1,32]");
    54:   TORCH_CHECK(q_tile == 8 || q_tile == 16, "q_tile must be 8 or 16");
    55:   TORCH_CHECK(temp_out.numel() == q.numel() * num_splits &&
    56:                   temp_out.scalar_type() == at::kHalf,
    57:               "invalid split output scratch");
    58:   TORCH_CHECK(exp_sums.numel() == q.size(0) * q.size(1) * num_splits &&
    59:                   max_logits.numel() == exp_sums.numel() &&
    60:                   exp_sums.scalar_type() == at::kFloat &&
    61:                   max_logits.scalar_type() == at::kFloat,
    62:               "invalid reduction scratch");
    63:   TORCH_CHECK(max_seqlen_k >= q.size(1) / 24 &&
    64:                   max_seqlen_k <= block_table.size(1) * k.size(1),
    65:               "maximum KV length must cover query rows and fit the block table");
    66:   TORCH_CHECK(k.stride(3) == 1 && k.stride(2) >= 256 &&
    67:                   k.stride(1) >= 4 * k.stride(2) &&
    68:                   k.stride(0) >= k.size(1) * k.stride(1),
    69:               "unsupported KV layout");
    70:   for (const auto* t : {&k, &v, &out, &block_table, &cu_seqlens_q,
    71:                          &seqused_k, &k_scale, &v_scale, &temp_out,
    72:                          &exp_sums, &max_logits}) {
    73:     TORCH_CHECK(t->device() == q.device(), "all tensors must share the XPU device");
    74:   }
    75:   TORCH_CHECK(cu_seqlens_q.is_contiguous() && seqused_k.is_contiguous() &&
    76:                   k_scale.is_contiguous() && v_scale.is_contiguous() &&
    77:                   temp_out.is_contiguous() && exp_sums.is_contiguous() &&
    78:                   max_logits.is_contiguous(), "metadata/scratch must be contiguous");
```

**Source:** `supplement/engine-code/benchmarks/experiments/exl3-shared-kv-verify/csrc/shared_kv_verification.cpp`, lines 91–131. SHA-256: `8e28d338b95ceee1d57c7828324d0f27ab81a6714099b233e47727ebedf331dd`.

```text
    91:   paged_decode_args_t args{};
    92:   args.query = q.data_ptr();
    93:   args.key = k.data_ptr();
    94:   args.value = v.data_ptr();
    95:   args.out = out.data_ptr();
    96:   args.tem_out = num_splits == 1 ? out.data_ptr() : temp_out.data_ptr();
    97:   args.exp_sums = exp_sums.data_ptr();
    98:   args.max_logits = max_logits.data_ptr();
    99:   args.block_table = block_table.data_ptr();
   100:   args.cu_seqlens_q = cu_seqlens_q.data_ptr();
   101:   args.cu_seqlens_k = seqused_k.data_ptr();
   102:   args.max_queries = 1;
   103:   args.max_keys = max_seqlen_k;
   104:   args.total_seqlen_q = q.size(0);
   105:   args.total_seqlen_k = get_paged_kv_cache_effective_total_seqlen(k);
   106:   args.k_scale = k_scale.data_ptr();
   107:   args.v_scale = v_scale.data_ptr();
   108:   args.sm_scale = 0.0625f;
   109:   args.batch_size = q.size(0);
   110:   args.num_heads_q = q.size(1);
   111:   args.num_heads_k = 4;
   112:   args.head_size = 256;
   113:   args.v_head_size = 256;
   114:   args.max_blocks_per_seq = block_table.size(1);
   115:   args.block_size = k.size(1);
   116:   args.is_varlen = true;
   117:   args.is_paged = true;
   118:   args.is_causal = true;
   119:   args.num_kv_splits = num_splits;
   120:   args.q_stride_seq = q.stride(0);
   121:   args.q_stride_heads = q.stride(1);
   122:   args.k_stride_page = k.stride(0);
   123:   args.k_stride_seq = k.stride(1);
   124:   args.k_stride_heads = k.stride(2);
   125:   args.v_stride_page = v.stride(0);
   126:   args.v_stride_seq = v.stride(1);
   127:   args.v_stride_heads = v.stride(2);
   128:   args.page_stride_elements = get_paged_kv_cache_page_stride_elements(k);
   129:   auto& queue = c10::xpu::getCurrentXPUStream().queue();
   130:   if (q_tile == 8) {
   131:     PagedDecodeConfig<typename VerifyPolicyQ8::ShapeQK,
```

## E11 — Generic Split-K work and partition-sensitive graph arithmetic

The current generic path shares work across pairs of GQA heads but still schedules separate query tokens. A new packed verifier changes more than one admission integer; it needs the right causal rows and graph policy.

**Source:** `code/src/vt/xpu/xpu_attention_split.cpp`, lines 115–222. SHA-256: `907b10da7b4e2f6ffcfdc0692b842acbbaf61492988bb2d08f73a15f7f647274`.

```text
   115:         maximum[r] = -std::numeric_limits<float>::infinity();
   116:         #pragma unroll
   117:         for (int c = 0; c < Components; ++c)
   118:           if (pair * Reuse + r < ratio && lane + c * SG < dim)
   119:             qv[r][c] = Load(qs, (token * heads + head + r) * dim + lane + c * SG);
   120:       }
   121:       for (int64_t key = begin; key < end; ++key) {
   122:         const int64_t block = table[request * bt.stride[0] + (key / page) * bt.stride[1]];
   123:         const int64_t kb = block * kc.stride[0] + (key % page) * kc.stride[1] + kh * kc.stride[2];
   124:         const int64_t vb = block * vc.stride[0] + (key % page) * vc.stride[1] + kh * vc.stride[2];
   125:         float kval[Components], vval[Components];
   126:         #pragma unroll
   127:         for (int c = 0; c < Components; ++c) {
   128:           kval[c] = lane + c * SG < dim ? LoadKV(kc, kb + lane + c * SG, kscale) : 0;
   129:           vval[c] = lane + c * SG < dim ? LoadKV(vc, vb + lane + c * SG, vscale) : 0;
   130:         }
   131:         // Adjacent GQA heads share every K/V load; tail heads are masked.
   132:         #pragma unroll
   133:         for (int r = 0; r < Reuse; ++r) {
   134:           float dot = 0;
   135:           #pragma unroll
   136:           for (int c = 0; c < Components; ++c) dot += qv[r][c] * kval[c];
   137:           float score = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
   138:           if (cap > 0) score = cap * sycl::tanh(score / cap);
   139:           const float next = sycl::max(maximum[r], score);
   140:           const float old = sycl::exp(maximum[r] - next), p = sycl::exp(score - next);
   141:           denominator[r] = denominator[r] * old + p;
   142:           #pragma unroll
   143:           for (int c = 0; c < Components; ++c) acc[r][c] = acc[r][c] * old + p * vval[c];
   144:           maximum[r] = next;
   145:         }
   146:       }
   147:       #pragma unroll
   148:       for (int r = 0; r < Reuse; ++r) if (pair * Reuse + r < ratio) {
   149:         auto* result = partial + ((token * heads + head + r) * parts + part) * stride;
   150:         if (lane == 0) { result[dim] = maximum[r]; result[dim + 1] = denominator[r]; }
   151:         #pragma unroll
   152:         for (int c = 0; c < Components; ++c) if (lane + c * SG < dim) result[lane + c * SG] = acc[r][c];
   153:       }
   154:     });
   155:     RecordProfileEvent(q, "attention_split_partial", partial_event);
   156:     const char* reduce_setting = std::getenv("VT_XPU_ATTN_SPLIT_REDUCE");
   157:     const std::string_view reduce_mode = reduce_setting ? reduce_setting : "auto";
   158:     VT_CHECK(reduce_mode == "auto" || reduce_mode == "scalar" ||
   159:                  reduce_mode == "cooperative",
   160:              "VT_XPU_ATTN_SPLIT_REDUCE must be auto, scalar or cooperative");
   161:     if (b70_fp8 && args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
   162:         heads == 24 && dim == 256 && reduce_mode != "scalar") {
   163:       constexpr int ReduceLanes = 64;
   164:       const int64_t component_groups = (dim + ReduceLanes - 1) / ReduceLanes;
   165:       const int64_t groups = tokens * heads * component_groups;
   166:       const auto reduce_event = NativeQueue(q).submit([&](sycl::handler& h) {
   167:         sycl::local_accessor<float, 1> weights(sycl::range<1>(256), h);
   168:         h.parallel_for(sycl::nd_range<1>(groups * ReduceLanes, ReduceLanes),
   169:             [=](sycl::nd_item<1> item) {
   170:           const auto group = item.get_group(0);
   171:           const int64_t row = group / component_groups;
   172:           const int64_t component = group % component_groups * ReduceLanes +
   173:                                     item.get_local_id(0);
   174:           const int lane = item.get_local_id(0);
   175:           const auto* src = partial + row * parts * stride;
   176:           float local_max = -std::numeric_limits<float>::infinity();
   177:           for (int p = lane; p < parts; p += ReduceLanes)
   178:             if (src[p * stride + dim + 1] > 0)
   179:               local_max = sycl::max(local_max, src[p * stride + dim]);
   180:           const auto workgroup = item.get_group();
   181:           const float maximum = sycl::reduce_over_group(
   182:               workgroup, local_max, sycl::maximum<float>{});
   183:           float local_sum = 0;
   184:           for (int p = lane; p < parts; p += ReduceLanes) {
   185:             const float denominator = src[p * stride + dim + 1];
   186:             const float factor = denominator > 0 ?
   187:                 sycl::exp(src[p * stride + dim] - maximum) : 0.0f;
   188:             weights[p] = factor;
   189:             local_sum += factor * denominator;
   190:           }
   191:           const float sum = sycl::reduce_over_group(
   192:               workgroup, local_sum, sycl::plus<float>{});
   193:           item.barrier(sycl::access::fence_space::local_space);
   194:           if (component < dim) {
   195:             float value = 0;
   196:             for (int p = 0; p < parts; ++p)
   197:               value += weights[p] * src[p * stride + component];
   198:             Store(dst, row * dim + component, value / sum);
   199:           }
   200:         });
   201:       });
   202:       RecordProfileEvent(q, "attention_split_reduce_cooperative", reduce_event);
   203:       return;
   204:     }
   205:     const auto reduce_event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * heads * dim), [=](sycl::id<1> item) {
   206:       const int64_t row = item[0] / dim, d = item[0] % dim;
   207:       const auto* src = partial + row * parts * stride;
   208:       float maximum = -std::numeric_limits<float>::infinity();
   209:       for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0)
   210:         maximum = sycl::max(maximum, src[p * stride + dim]);
   211:       float sum = 0, value = 0;
   212:       for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0) {
   213:         const float factor = sycl::exp(src[p * stride + dim] - maximum);
   214:         sum += factor * src[p * stride + dim + 1];
   215:         value += factor * src[p * stride + d];
   216:       }
   217:       Store(dst, item[0], value / sum);
   218:     });
   219:     RecordProfileEvent(q, "attention_split_reduce", reduce_event);
   220:   });
   221: }
   222: }  // namespace vt::xpu
```

## E12 — Speculative GDN private row and per-token snapshots

The explicit WG64 selection is C1-specific and the inner row algorithm remains scalar over K. No register spill is claimed without generated-code/profile evidence.

**Source:** `code/src/vt/xpu/xpu_gdn.cpp`, lines 534–639. SHA-256: `10a03e7ec3cb294f48324ac389d635fad44f42872d3bcaa749b68c5e8b0ad96b`.

```text
   534: void GdnSpecDecodeKernel(Queue& q, Tensor& out, const Tensor& qi,
   535:                          const Tensor& ki, const Tensor& vi, const Tensor& g,
   536:                          const Tensor& beta, Tensor& state, const Tensor& qsl,
   537:                          const Tensor& indices, const Tensor& accepted,
   538:                          const GdnArgs& args) {
   539:   TraceXpuOp(OpId::kGdnSpecDecode, q,
   540:              {&out, &qi, &ki, &vi, &g, &beta, &state, &qsl, &indices, &accepted});
   541:   const int64_t requests = indices.shape[0], cols = indices.shape[1];
   542:   const int64_t tokens = qi.shape[0], slots = state.shape[0];
   543:   const int64_t hk = qi.shape[1], hv = state.shape[1];
   544:   const int64_t dv = state.shape[2], dk = state.shape[3];
   545:   VT_CHECK(state.dtype == DType::kF32, "XPU spec GDN requires F32 state");
   546:   VT_CHECK(dk > 0 && dk <= 128, "XPU spec GDN supports Dk <= 128");
   547:   for (const auto* operand :
   548:        std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta,
   549:                                             &qsl, &indices, &accepted})
   550:     VT_CHECK(!Overlap(state, *operand), "XPU spec GDN state must have separate storage");
   551:   const auto* offsets = static_cast<const int32_t*>(qsl.data);
   552:   const auto* ids = static_cast<const int32_t*>(indices.data);
   553:   const auto* nat = static_cast<const int32_t*>(accepted.data);
   554:   CheckDeviceMetadata(q, [=] {
   555:     if (offsets[0] != 0 || offsets[requests] != tokens) return false;
   556:     for (int64_t r = 0; r < requests; ++r) {
   557:       if (offsets[r] < 0 || offsets[r + 1] < offsets[r] ||
   558:           offsets[r + 1] > tokens || offsets[r + 1] - offsets[r] > cols ||
   559:           nat[r] < 1 || nat[r] > cols) return false;
   560:       for (int64_t c = 0; c < cols; ++c) {
   561:         const int32_t slot = ids[r * cols + c];
   562:         if (slot >= slots) return false;
   563:         if (slot < 0) continue;
   564:         for (int64_t prior = 0; prior < r; ++prior)
   565:           for (int64_t pc = 0; pc < cols; ++pc)
   566:             if (ids[prior * cols + pc] == slot) return false;
   567:       }
   568:     }
   569:     return true;
   570:   }, "XPU spec GDN invalid offsets, accepted count or state slot",
   571:       {&qsl, &indices, &accepted});
   572:   WithOutput(q, out, {&qi, &ki, &vi, &g, &beta, &qsl, &indices, &accepted},
   573:              [&](Tensor& target) {
   574:     const View dst(target), qs(qi), ks(ki), vs(vi), gs(g), bs(beta);
   575:     auto* cache = static_cast<float*>(state.data);
   576:     const float scale = args.scale;
   577:     const int64_t items = requests * hv * dv;
   578:     if (!items) return;
   579:     const auto work = [=](int64_t index) {
   580:       const int64_t request = index / (hv * dv);
   581:       const int64_t head = (index / dv) % hv, value = index % dv;
   582:       const int64_t first = offsets[request], last = offsets[request + 1];
   583:       const int32_t initial = ids[request * cols + nat[request] - 1];
   584:       const int64_t out_channel = head * dv + value;
   585:       if (initial < 0) {
   586:         for (int64_t token = first; token < last; ++token)
   587:           Store(dst, token * hv * dv + out_channel, 0.0f);
   588:         return;
   589:       }
   590:       if (first == last) return;
   591:       float s[128];
   592:       const int64_t initial_base = ((static_cast<int64_t>(initial) * hv + head) * dv + value) * dk;
   593:       for (int64_t j = 0; j < dk; ++j) s[j] = cache[initial_base + j];
   594:       const int64_t key_head = head / (hv / hk);
   595:       for (int64_t token = first; token < last; ++token) {
   596:         const int64_t key_base = (token * hk + key_head) * dk;
   597:         const float decay = sycl::exp(Load(gs, token * hv + head));
   598:         float prediction = 0.0f;
   599:         for (int64_t j = 0; j < dk; ++j) {
   600:           s[j] *= decay;
   601:           prediction += s[j] * Load(ks, key_base + j);
   602:         }
   603:         const float delta =
   604:             (Load(vs, token * hv * dv + out_channel) - prediction) *
   605:             Load(bs, token * hv + head);
   606:         float output = 0.0f;
   607:         for (int64_t j = 0; j < dk; ++j) {
   608:           s[j] += delta * Load(ks, key_base + j);
   609:           output += s[j] * (Load(qs, key_base + j) * scale);
   610:         }
   611:         Store(dst, token * hv * dv + out_channel, output);
   612:         const int32_t snapshot = ids[request * cols + token - first];
   613:         if (snapshot >= 0) {
   614:           const int64_t base = ((static_cast<int64_t>(snapshot) * hv + head) * dv + value) * dk;
   615:           for (int64_t j = 0; j < dk; ++j) cache[base + j] = s[j];
   616:         }
   617:       }
   618:     };
   619:     const char* setting = std::getenv("VT_XPU_GDN_SPEC_WG");
   620:     // The 27B MTP verification shape benefits from explicit B70 workgroups.
   621:     // Keep other shapes on the existing range launch until they are measured.
   622:     const int wg = setting ? std::atoi(setting) :
   623:         requests == 1 && hk == 16 && hv == 48 && dk == 128 && dv == 128 &&
   624:                 tokens <= 5 ? 64 : 0;
   625:     VT_CHECK(wg == 0 || wg == 32 || wg == 64 || wg == 128 || wg == 256,
   626:              "VT_XPU_GDN_SPEC_WG must be 0, 32, 64, 128 or 256");
   627:     sycl::event event;
   628:     if (wg == 0) {
   629:       event = NativeQueue(q).parallel_for(sycl::range<1>(items),
   630:           [=](sycl::id<1> item) { work(item[0]); });
   631:     } else {
   632:       const int64_t rounded = ((items + wg - 1) / wg) * wg;
   633:       event = NativeQueue(q).parallel_for(
   634:           sycl::nd_range<1>(rounded, wg), [=](sycl::nd_item<1> item) {
   635:             const int64_t index = static_cast<int64_t>(item.get_global_id(0));
   636:             if (index < items) work(index);
   637:           });
   638:     }
   639:     RecordProfileEvent(q, wg ? "gdn_spec_decode_wg" : "gdn_spec_decode", event);
```

## E13 — SmallM already uses the donor and its default launch family

The copied ESIMD header is byte-identical to the supplied original. The optional original fully fused route is disabled by default and explicitly described as slower under graphs; blindly enabling it is not an evidence-backed optimization.

```json
{
  "donor_header_exact": true
}
```

**Source:** `code/src/vt/exl3_grouped.cpp`, lines 74–98. SHA-256: `946d73662e370a478fab866e9c139de68cac09c1419f9d61177fe23d04bc1ff9`.

```text
    74: Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits) {
    75:   VT_CHECK(m > 0 && m <= 128, "EXL3 SmallM requires physical M in [1,128]");
    76:   VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
    77:                k <= std::numeric_limits<int>::max() &&
    78:                n <= std::numeric_limits<int>::max(),
    79:            "EXL3 SmallM requires positive I32 K/N multiples of 128");
    80:   VT_CHECK(bits == 4 || bits == 6, "EXL3 SmallM supports 4/6bpw only");
    81:   // c59d944 exl3_ops.sycl defaults: vector M<=2, DPAS MB24/40/48 enabled,
    82:   // max MB64, NT2 at MB>=24, and thread targets 1024/1408/2048.
    83:   const bool vector = m <= 2;
    84:   const int mb = vector ? int(m) : m <= 8 ? 8 : m <= 16 ? 16 :
    85:                  m <= 24 ? 24 : m <= 32 ? 32 : m <= 40 ? 40 : m <= 48 ? 48 : 64;
    86:   const int blocks = vector ? 1 : int((m + mb - 1) / mb);
    87:   const int nt = vector ? (bits == 4 && m == 1 ? 8 : 4) : mb <= 16 ? 4 : 2;
    88:   const int64_t units = (n / 16 / nt) * blocks;
    89:   const int target = vector ? 1024 : mb <= 16 ? 1408 : mb >= 40 ? 2048 : 1024;
    90:   const int tk = int(k / 16);
    91:   const int requested = int(std::max(int64_t{1}, std::min(int64_t(tk),
    92:                                          (target + units - 1) / units)));
    93:   const int rps = (tk + requested - 1) / requested;
    94:   return {vector, mb, blocks * mb, nt, (tk + rps - 1) / rps, rps};
    95: }
    96: 
    97: void Exl3GroupedLinear(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    98:     const Tensor& suh, const Tensor& svh, const Tensor& shard,
```

**Source:** `code/src/vt/xpu/xpu_exl3_smallm.cpp`, lines 19–88. SHA-256: `97310c02517bc398fef49ee6c71464cb0410098e8c39e3c4850f3adad0de4222`.

```text
    19: namespace {
    20: template<class Kernel> struct LargeGrf {
    21:   Kernel kernel;
    22:   void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL { kernel(item); }
    23:   auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    24:     return sycl::ext::oneapi::experimental::properties{
    25:         sycl::ext::intel::experimental::grf_size<256>};
    26:   }
    27: };
    28: template<class Kernel>
    29: void Launch(Queue& q, int64_t count, Kernel kernel, const char* stage) {
    30:   constexpr int local = 8;
    31:   const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    32:     h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + local - 1) / local * local),
    33:                                     sycl::range<1>(local)), kernel);
    34:   });
    35:   RecordProfileEvent(q, stage, event);
    36: }
    37: 
    38: template<int Bits, int MB, int NT>
    39: void Dpas(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
    40:           Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
    41:   const int strips = n / 16 / NT, blocks = (m + MB - 1) / MB;
    42:   ::exl3::DpasKernel<Bits, 2, MB, NT> kernel{
    43:       static_cast<const sycl::half*>(in.data), static_cast<const uint32_t*>(trellis.data),
    44:       static_cast<const int*>(shard.data), static_cast<float*>(partials.data),
    45:       m, k, n, n / 16, plan.tile_rows_per_split, strips, blocks, blocks * MB, {}, 0};
    46:   const int64_t count = int64_t(strips) * blocks * plan.splits;
    47:   if constexpr (MB >= 40) {
    48:     // Same default 256-GRF domain as the pinned MB40/48/64 launchers.
    49:     const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    50:       h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + 7) / 8 * 8),
    51:                                       sycl::range<1>(8)), LargeGrf{kernel});
    52:     });
    53:     RecordProfileEvent(q, "exl3_smallm_dpas", event);
    54:   } else {
    55:     Launch(q, count, kernel, "exl3_smallm_dpas");
    56:   }
    57: }
    58: 
    59: template<int Bits, int MR, int NT>
    60: void Gemv(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
    61:           Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
    62:   const int strips = n / 16 / NT;
    63:   ::exl3::GemvKernel<Bits, 2, MR, NT> kernel{
    64:       static_cast<const sycl::half*>(in.data), static_cast<const uint32_t*>(trellis.data),
    65:       static_cast<const int*>(shard.data), static_cast<float*>(partials.data),
    66:       m, k, n, n / 16, plan.tile_rows_per_split, strips, {}, 0};
    67:   Launch(q, int64_t(strips) * plan.splits, kernel, "exl3_smallm_gemv");
    68: }
    69: 
    70: template<int Bits>
    71: void Gemm(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
    72:           Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
    73:   if (plan.vector) {
    74:     if (m == 1) Gemv<Bits, 1, Bits == 4 ? 8 : 4>(q, in, trellis, shard, partials, m, k, n, plan);
    75:     else Gemv<Bits, 2, 4>(q, in, trellis, shard, partials, m, k, n, plan);
    76:     return;
    77:   }
    78:   switch (plan.row_block) {
    79:     case 8: Dpas<Bits, 8, 4>(q, in, trellis, shard, partials, m, k, n, plan); break;
    80:     case 16: Dpas<Bits, 16, 4>(q, in, trellis, shard, partials, m, k, n, plan); break;
    81:     case 24: Dpas<Bits, 24, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    82:     case 32: Dpas<Bits, 32, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    83:     case 40: Dpas<Bits, 40, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    84:     case 48: Dpas<Bits, 48, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    85:     case 64: Dpas<Bits, 64, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    86:     default: VT_CHECK(false, "EXL3 SmallM invalid DPAS row block");
    87:   }
    88: }
```

**Source:** `reference/exl3-qualified-source/csrc/exl3_ops.sycl`, lines 175–185. SHA-256: `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`.

```text
   175: static int dpas_nt(int MB) { return MB == 4 ? EXL3_DPAS4_NT : MB == 8 ? EXL3_DPAS8_NT : MB == 24 ? EXL3_DPAS24_NT : (MB == 40 || MB == 48) ? 2 : (MB <= 16 ? 4 : (MB == 32 ? EXL3_DPAS32_NT : EXL3_DPAS64_NT)); }
   176: static int g_vec_max_m = 2;   // vector kernel for M<=2, DPAS (flat cost up to M=8) above
   177: 
   178: // Split-K sizing: threads to aim for when choosing the number of K splits. Measured on B70 (all linears,
   179: // graph-captured): 1024 beats 4096 by 10-20% for M=1..32 (fewer fp32 partials for had_out to reduce, and the
   180: // 256-GRF DPAS kernels only hold ~1024 resident threads); MB=64 prefers 2048 (77 vs 88 ms).
   181: static int g_target_threads = 1024;          // vector GEMV (M<=2) and DPAS MB=32
   182: static int g_target_threads_mb16 = 1408;     // DPAS MB=8/16: plateau 1280-1536 (M=4 35.1->32.2 ms, M=16 38.7->36.0), cliff at 1664
   183: static int g_target_threads_mb64 = 2048;
   184: 
   185: // Column-strip width (tiles per thread) per case, from bench sweeps on B70:
```

**Source:** `reference/exl3-qualified-source/csrc/exl3_ops.sycl`, lines 250–268. SHA-256: `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`.

```text
   250:         DpasKernel<K, CB, 8, EXL3_DPAS8_NT, true> k{nullptr, trp, shp, pp, M, Kdim, N, tiles_n, rps, n_strips, 1, 8, fa, P};
   251:         launch_fused(k, n, q);
   252:     }
   253:     return true;
   254: }
   255: 
   256: static bool g_fused = false;   // measured slower under XPU graphs (serialized split-K tail); opt in via exl3_set_fused
   257: void exl3_set_fused(int64_t on) { g_fused = on != 0; }
   258: 
   259: void exl3_gemm_small(at::Tensor x, at::Tensor trellis, at::Tensor suh, at::Tensor svh, at::Tensor shard_of_nb,
   260:                      at::Tensor out, int64_t K, int64_t cb) {
   261:     TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "x must be 2D row-major");
   262:     TORCH_CHECK(trellis.scalar_type() == at::kShort && trellis.is_contiguous());
   263:     auto& q = queue_of(x);
   264:     int M = x.size(0), Kdim = x.size(1), N = svh.size(0), S = suh.size(0);
   265:     if (g_fused && M <= 8 && x.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf && out.stride(1) == 1 &&
   266:         N / 128 <= (1 << 16)) {
   267:         bool done = false;
   268:         if (K == 4 && cb == 2) done = fused_small<4, 2>(x, trellis, suh, svh, shard_of_nb, out, M, Kdim, N, q);
```

## E14 — Recent graph retirement fix to retain

Owner retirement includes changes that affect captured dispatch arithmetic even when a workspace partition cap is already saturated. This is a correctness invariant, not wasted host work that can be removed without replacement.

**Source:** `code/src/vllm/model_executor/models/qwen3_5.cpp`, lines 13148–13194. SHA-256: `dcda5e883cb3d8377d3a011dc36d162c8741d710873e2de962c708dc15231bb5`.

```text
 13148:     pgm = gdn_meta;
 13149:   } else {
 13150:     // The padding helper copies GDN indices into S entries. Without consumers,
 13151:     // the caller's unused metadata has no validated bound and must stay inert.
 13152:     BuildPaddedDecode(S, token_ids, positions, attn_meta,
 13153:                       has_gdn ? gdn_meta : GDNAttentionMetadata{}, ptok, ppos,
 13154:                       pam, pgm);
 13155:   }
 13156: 
 13157:   // A block-table column-count change reallocates the persistent block_table (the
 13158:   // staged/baked H2D dest shape moves) → invalidate this slot's graph + device inputs.
 13159:   const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
 13160:   // Device lengths cannot refresh host dispatch/partitioning baked into a
 13161:   // graph. Retire at the short-decode, long Split-K and active-page boundaries.
 13162:   // Page changes conservatively retire even when a workspace cap has already
 13163:   // saturated the partition count; no device metadata download is needed.
 13164:   std::array<int64_t, 3> attention_policy{{0, 0, 0}};
 13165:   if (d.q.device.type == vt::DeviceType::kXPU) {
 13166:     const char* active_pages = std::getenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP");
 13167:     const bool active_page_cap = !active_pages || std::string_view(active_pages) == "1";
 13168:     attention_policy = {{
 13169:         S == 1 && Q == 1 && vt::PagedAttnXpuShortDecodeBound(pam.max_seq_len),
 13170:         vt::PagedAttnXpuLongSplitBound(pam.max_seq_len),
 13171:         active_page_cap && !attn_kv.empty() ?
 13172:             vt::PagedAttnXpuActivePages(pam.max_seq_len, attn_kv[0].block_size) : 0}};
 13173:   }
 13174:   const bool attention_policy_changed = s.xpu_attention_policy[0] != -1 &&
 13175:       s.xpu_attention_policy != attention_policy;
 13176:   s.Refresh(ptok, ppos, pam, pgm);
 13177:   s.fa_cols = cols;
 13178:   s.xpu_attention_policy = attention_policy;
 13179:   bool seq_continuation = true;  // no seq_lens -> no boundary to detect
 13180:   // TT-27B-STEP-DECOMPOSE: the per-step TT refresh block is the "warmup
 13181:   // passes" phase — every Warm* call the captured arm makes per step is
 13182:   // inside this bracket, and `warm_calls` counts them (cos|sin + one paged-KV
 13183:   // shadow per full-attn layer + cur_pos + RAC idx + PA meta).
 13184:   const StepPhaseClk::time_point sph_tw0 =
 13185:       sph.on ? StepPhaseClk::now() : sph.t0;
 13186:   // TT captured arm (the WarmRopeCosSin populate-outside / content-HIT-inside
 13187:   // pattern; qwen3.cpp:827-900 ported to the dense driver): refresh every
 13188:   // persistent device input the captured region reads, OUTSIDE capture, on
 13189:   // every step (warm, capture, replay). No-op off the TT host-free decode lane.
 13190:   if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
 13191:     // Fused-preamble cos|sin table: refreshed so the captured
 13192:     // kAttnQkNormRopeGate serves a table matching the in-region
 13193:     // RopeCosSinCacheKernel refill of these same positions.
 13194:     vt::tenstorrent::WarmAttnCosSin(
```

## E15 — Existing W8A8 refusal/reuse tests and model scratch lifetime

Extend the current tests; do not weaken them to make a larger panel or fused validation path pass. Record private preparation scratch separately from public result storage.

**Source:** `code/tests/vt/test_xpu_exl3_w8a8.cpp`, lines 41–62. SHA-256: `ecd5f4a5d91f3479dd6b995dd24850fca601bada39841403bb3e8775cd9d132d`.

```text
    41: TEST_CASE("EXL3 W8A8 plan: explicit boundary bounded panel and aligned regions") {
    42:   CHECK(vt::PlanExl3SmallM(128, 5120, 16384, 4).padded_rows == 128);
    43:   CHECK_THROWS(vt::PlanExl3SmallM(129, 5120, 16384, 4));
    44:   CHECK_THROWS(vt::PlanExl3W8A8(128, 5120, 16384, 2, 4));
    45:   for (int m : {129, 256, 4096}) {
    46:     const auto p = vt::PlanExl3W8A8(m, 5120, 16384, 2, 4);
    47:     CHECK(p.padded_rows == (m + 255) / 256 * 256);
    48:     CHECK(p.weight_panel_bytes == 5120 * 128);
    49:     CHECK(p.row_scale_offset == size_t(2) * p.padded_rows * 5120);
    50:     CHECK(p.intermediate_offset >= p.row_scale_offset + size_t(2) * p.padded_rows * 4);
    51:     CHECK(p.weight_scale_offset >= p.intermediate_offset + size_t(p.padded_rows) * 16384 * 2);
    52:     CHECK(p.workspace_bytes == p.weight_scale_offset + 64);
    53:     for (const size_t offset : {p.activation_offset, p.row_scale_offset, p.intermediate_offset,
    54:                                 p.weight_scale_offset, p.workspace_bytes}) CHECK(offset % 64 == 0);
    55:   }
    56:   CHECK_THROWS(vt::PlanExl3W8A8(4097, 5120, 16384, 2, 4));
    57:   CHECK_THROWS(vt::PlanExl3W8A8(129, 5130, 16384, 2, 4));
    58:   CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 0, 4));
    59:   CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 2, 5));
    60:   CHECK_THROWS(vt::PlanExl3W8A8(129, 133248, 128, 1, 4));
    61:   CHECK_THROWS(vt::PlanExl3W8A8(4096, 2147483520LL, 2147483520LL, 32767, 6));
    62: }
```

**Source:** `code/tests/vt/test_xpu_exl3_w8a8.cpp`, lines 125–180. SHA-256: `ecd5f4a5d91f3479dd6b995dd24850fca601bada39841403bb3e8775cd9d132d`.

```text
   125:     CHECK(quantized_exact); CHECK(scales_exact); CHECK(padding_zero);
   126:     const auto ybytes = size_t(m) * n * 2;
   127:     Accuracy(std::vector<unsigned char>(raw.begin() + p.intermediate_offset,
   128:         raw.begin() + p.intermediate_offset + ybytes), f.Get("y_m" + std::to_string(m)), "F16_intermediate");
   129:     if (m == 129) {
   130:       if (groups > 1) {
   131:         // H128 is block-local. Permuting complete packed output blocks, SV
   132:         // and their source IDs must give exactly the same permutation of the
   133:         // original W8A8 output, including a noncontiguous source-group map.
   134:         auto packed_bytes = trellis.download(), sv_bytes = svh.download();
   135:         auto map_bytes = mapping.download();
   136:         const size_t block_bytes = size_t(8) * 32 * bits;
   137:         for (int tile = 0; tile < k / 16; ++tile) {
   138:           const size_t first = size_t(tile) * (n / 16) * 32 * bits;
   139:           const size_t last = first + size_t(n / 128 - 1) * block_bytes;
   140:           std::swap_ranges(packed_bytes.begin() + first, packed_bytes.begin() + first + block_bytes,
   141:                            packed_bytes.begin() + last);
   142:         }
   143:         std::swap_ranges(sv_bytes.begin(), sv_bytes.begin() + 256, sv_bytes.end() - 256);
   144:         for (int i = 0; i < 4; ++i) std::swap(map_bytes[i], map_bytes[map_bytes.size() - 4 + i]);
   145:         REQUIRE(vt::LoadUnaligned<int32_t>(map_bytes.data()) !=
   146:                 vt::LoadUnaligned<int32_t>(map_bytes.data() + map_bytes.size() - 4));
   147:         trellis.upload(packed_bytes.data()); svh.upload(sv_bytes.data()); mapping.upload(map_bytes.data());
   148:         const auto& expected = f.Get("output_m129");
   149:         std::vector<unsigned char> permuted(expected.data, expected.data + expected.nbytes);
   150:         for (int row = 0; row < m; ++row) {
   151:           const size_t start = size_t(row) * n * 2;
   152:           std::swap_ranges(permuted.begin() + start, permuted.begin() + start + 256,
   153:                            permuted.begin() + start + size_t(n - 128) * 2);
   154:         }
   155:         vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
   156:             svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
   157:         xpu_test::SameBytes(out.download(), permuted);
   158:         trellis.upload(tr.data); svh.upload(sv.data); mapping.upload(map.data);
   159:       }
   160:       const auto before = out.download();
   161:       auto bad = scratch.tensor; bad.shape[0] = int64_t(p.workspace_bytes) - 1;
   162:       CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
   163:           svh.tensor, mapping.tensor, bad, panel.tensor, args));
   164:       auto alias = panel.tensor; alias.data = out.tensor.data;
   165:       CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
   166:           svh.tensor, mapping.tensor, scratch.tensor, alias, args));
   167:       auto map_raw = mapping.download();
   168:       const int32_t invalid = groups; std::memcpy(map_raw.data(), &invalid, sizeof(invalid));
   169:       mapping.upload(map_raw.data());
   170:       CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
   171:           svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
   172:           doctest::Contains("group out of range"), std::runtime_error);
   173:       xpu_test::SameBytes(out.download(), before);
   174:       mapping.upload(map.data);
   175:       const auto original_input = in.download();
   176:       const auto original_svh = svh.download();
   177:       const auto original_suh = suh.download();
   178:       auto reject = [&](const char* message, const std::string& boundary = "nonfinite operand") {
   179:         CAPTURE(boundary);
   180:         CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor,
```

## E16 — Previously requested draft selection has already been implemented

The current code selects hidden rows before the compact head and uses the model's native selection method. The previous plan's full-logits CPU argmax task is not a fresh primary optimization now.

**Source:** `code/src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, lines 67–94. SHA-256: `85608acea736cf73d09d9fafd00ebef99ad5d46b4534db47d253b3b09589e8ab`.

```text
    67:   // ── The one paged draft forward (I5c) + shared lm_head. ──────────────────────
    68:   vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
    69:       spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue);
    70:   PrefillOutcome out;
    71:   // ── Greedy draft pick over each request's last (sampled) row
    72:   // (spec_decode/speculator.py:276-280). ──────────────────────────────────────
    73:   out.sampled_rows.assign(
    74:       spi.last_token_indices.begin(),
    75:       spi.last_token_indices.begin() + static_cast<size_t>(num_reqs));
    76:   // Keep all forward rows for draft KV/feedback, but apply the head only to the
    77:   // sampled rows, as the pinned producer does before sample_draft.
    78:   const auto selected = draft.GatherHiddenRows(hidden.tensor, out.sampled_rows, queue);
    79:   const auto logits = draft.ComputeLogits(selected.tensor, queue);
    80:   VT_CHECK(logits.on_device() && logits.rows == num_reqs,
    81:            "MtpProposePrefill: unexpected selected draft logits shape");
    82:   out.draft_tokens = draft.SelectDraftTokens(logits, queue);
    83: 
    84:   // :346 — the positions of those same rows. The decode half advances from them.
    85:   // The k=1 caller drops them, and dropping a host vector costs nothing.
    86:   out.positions.resize(static_cast<size_t>(num_reqs));
    87:   for (int64_t r = 0; r < num_reqs; ++r) {
    88:     out.positions[static_cast<size_t>(r)] =
    89:         positions32[static_cast<size_t>(out.sampled_rows[
    90:             static_cast<size_t>(r)])];
    91:   }
    92:   out.hidden = std::move(hidden);
    93:   return out;
    94: }
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 107–113. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
   107: PRIOR RESULTS: HISTORICAL CONTEXT, NOT CURRENT CLAIMS
   108: 
   109: - The previously discussed11.23% slower result was only C4 P4096/O1024 MTP3 E2E medians across three trials: native v10 84.821714s versus original76.257753s, with different continuations. It was not decode-only, prefill-only, no-MTP or a current norm-candidate result.
   110: - Historical warm32K native v10: TTFT39.98335s versus original18.57784s, MTP emitted decode12.63421 versus49.44291tok/s, E2E60.16680 versus23.73536s; one trial, first output difference56. These motivate long-context work but require current remeasurement.
   111: - Historical mixed short-decode/long-prefill native70.79778s versus original27.68211s E2E; original common decode overlap unavailable, so no matched concurrent decode rate claim.
   112: - Earlier row-group IndexCopy changed total device scatter137.36s ->.205s and mixed E2E224.04s ->84.47s on identical native outputs in a bounded trial. W8A8 GPU validation merge later changed mixed70.79778s ->66.31599s,6.33% single-trial elapsed reduction. These improvements are already in current source, not proposals to repeat.
   113: - Current latest-candidate no-MTP,32K,mixed,C4 and small final three-repeat coverage remains pending. No broad benchmark was launched for this review handoff.
```

## E17 — Unchanged qualification ledger and raw timing interpretation

The report explicitly preserves default-original failures and marks practical evidence that predates the current snapshot. The new performance work must maintain this distinction rather than closing flags based on old results.

**Source:** `PRO_REVIEW_REPORT.txt`, lines 27–64. SHA-256: `5811a6e748e99a744f84799be8f2a4c30c88ca8b85be95f9169b996e4a43f3d4`.

```text
    27: IMPLEMENTATION STATUS
    28: 
    29: R01 native autonomous C1 feedback/output/EOS/reset: delivered and locally tested.
    30: R02 same-input state attribution and required norm geometry: bounded deliveries and local tests complete; integrated original target/state qualification still open.
    31: R03 grouped large-M W8A8 and R04 variable-length GDN/workspace: delivered and locally tested within recorded families/shapes.
    32: R05 practical eager 4K target: delivered and locally tested.
    33: R06 compact draft/head: implemented, current bounded original numerical replay passes; broader qualification open.
    34: R07 MTP1/MTP3, R08 C2/C4 lifecycle, R09 recurrent prefix/long-context, R10 graph ownership: implemented and locally exercised; complete flags remain false because their integrated qualification is incomplete.
    35: R11 practical screen and measured optimization: substantial evidence and improvements exist; final qualification and performance coverage remain incomplete.
    36: 
    37: The status file records R01–R05 complete within their local delivery scopes, R06–R10 implemented/operator-tested but not complete, and R11 incomplete. Target-qualified/serving-qualified remain false. Recorded milestones A (native short output) and B (4K plus chunked32K native output) are complete; C/D/E are not recorded complete. This is no reason to restart functional R06 or abandon the engine; it is a reason to preserve the distinction between features, bounded numerical evidence and complete supported-mode qualification.
    38: 
    39: NUMERICAL AND PRACTICAL EVIDENCE
    40: 
    41: 1. Full target P128+64 native-owned replay, measured before the last graph-policy/norm changes:
    42:    - All65 full248320 logit rows byte-exact to a separately named controlled deterministic-BA original protocol. No shortened passing subset.
    43:    - Unchanged original frozen default reference gate still returns804/808, exit1: D27 TV/KL and D29 TV/KL fail. D27 TV.0305754/KL.00212581; D29 TV.212752/KL.163144; minimum top10 overlap9.
    44:    - Frozen TV<=.02, KL<=.002 and top10>=9 budgets are unchanged. Controlled-original equality does not silently promote that original protocol as the default reference.
    45:    - Scope C1/eager/no-MTP/P128+64 forced original prefix, not all contexts/states. Any reference-contract correction needs independently justified arithmetic, held-out prefix evidence and an explicit proposal.
    46:    Evidence: evidence/r02-D29-head-v1/receipt-controlled-D64-v103.json, controlled-D64-comparison-v102.json, controlled-D64-v101.log.
    47: 
    48: 2. Earlier causal repair: one FP16 SwiGLU midpoint at P128 block20 changed320 MLP halves and GDN21 history. Explicitly rounded F32 division before FP16 materialization closes the same-input MLP42/42 and actual GDN21 history230/230; observed active Conv/SSM become byte-exact. Subsequent controlled D29 all30 logits/all192 outer boundaries are exact. These are bounded positive results, not global strict-state clearance.
    49:    Evidence: receipt-swiglu-repair-v95.json and receipt-D29-swiglu-v98.json.
    50: 
    51: 3. Existing original isolated P128 compact draft/head replay passes131644/131644, exit0 on the post-numerics source: all655360 hidden values,65536 compact logits and independent65536 identical-input head outputs byte-exact after F32 widening; token map/mapped argmax exact. Earlier34-hidden/77-logit failures remain in historical receipts. This does not qualify every MTP3 iteration, graph or context, and was not rerun after the two latest shared changes.
    52:    Evidence: evidence/r06-compact-head-v1/receipt-current-numerical-v3.json and native-current-v3.log.
    53: 
    54: 4. Post-numerics practical suite passes native393/393 and semantic8/8: four code functions with20 execution cases, two60872-token retrieval tasks, two structured/tool tasks; all8 prompts/raw output sequences equal the historical passing v10, all finish stop. Requests were sequential with C4 capacity, not concurrent C4. Device peak30461556091B and graph bytes after release0.
    55:    IMPORTANT: this full suite predates the graph-policy and norm optimizations; current full practical-screen rerun remains pending. Do not label latest snapshot8/8 solely by transfer.
    56:    Evidence: receipt-capability-post-numerics-v1.json, capability-native-post-numerics-v1.json and capability-check-post-numerics-v1.json.
    57: 
    58: 5. Latest norm snapshot has its own focused unchanged original fixture proof:
    59:    - Focused build of both affected test binaries exit0.
    60:    - Physical rows3/4/12/16:821/821, one case, exit0; residual/no-residual, poisoned inactive row, alias, guards and strided refusal exact.
    61:    - Original P128/D1 Gemma boundaries:64/64, one case, exit0, all four outputs/residual boundaries exact.
    62:    - Current C1 graph/eager/separate eager profile:1975/1975 each, exit0. All1024 output IDs/all387 non-timing cycle fields match each other and the prior attention-policy candidate. Graph2 captures382 replays; accepted642/proposed1152. Graph versus eager comparison excludes mode-specific capture/replay fields; before/after graph capture/replay counts agree.
    63:    This proves the recorded bounded scopes, not a fresh full D64/state/draft/capability release gate.
    64:    Evidence: receipt-norm-workgroup-v1.json, original fixture logs, raw current graph/eager JSON and comparison-norm-workgroup-v1.json.
```

```json
{
  "controlled_summary_rows": 65,
  "controlled_tensor_bytes_locally_available": false
}
```

## E18 — Container memory ceilings differ in the recorded original/native commands

Selected command facts below are literal extractions from the supplied receipts. This is a comparison-contract check, not evidence that the native worker was actually swapping. Peak process RSS does not alone determine cgroup charged memory.

```json
[
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-norm-workgroup-v1.json",
    "receipt_sha256": "a3143f9182136cd5de70430f1893867a8f848bb0279c4628a9ef7afd05ec60e6",
    "command_key": "graph",
    "memory_argument": "12g",
    "cpu_argument": "2",
    "async_sched_0": true,
    "async_runner_0": true
  },
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-norm-workgroup-v1.json",
    "receipt_sha256": "a3143f9182136cd5de70430f1893867a8f848bb0279c4628a9ef7afd05ec60e6",
    "command_key": "eager",
    "memory_argument": "12g",
    "cpu_argument": "2",
    "async_sched_0": true,
    "async_runner_0": true
  },
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-norm-workgroup-v1.json",
    "receipt_sha256": "a3143f9182136cd5de70430f1893867a8f848bb0279c4628a9ef7afd05ec60e6",
    "command_key": "eager_profile",
    "memory_argument": "12g",
    "cpu_argument": "2",
    "async_sched_0": true,
    "async_runner_0": true
  },
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-c1-post-numerics-v2.json",
    "receipt_sha256": "4edf469a61100c3a7bb5fe5f9283befa11407b77592851071639917a6f45ae0f",
    "command_key": "native",
    "memory_argument": "12g",
    "cpu_argument": "2",
    "async_sched_0": true,
    "async_runner_0": true
  },
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-c1-post-numerics-v2.json",
    "receipt_sha256": "4edf469a61100c3a7bb5fe5f9283befa11407b77592851071639917a6f45ae0f",
    "command_key": "eager_profile",
    "memory_argument": "12g",
    "cpu_argument": "2",
    "async_sched_0": true,
    "async_runner_0": true
  },
  {
    "receipt": "evidence/r11-serving-timing-v1/receipt-c1-post-numerics-v2.json",
    "receipt_sha256": "4edf469a61100c3a7bb5fe5f9283befa11407b77592851071639917a6f45ae0f",
    "command_key": "producer",
    "memory_argument": "32g",
    "cpu_argument": "2",
    "async_sched_0": false,
    "async_runner_0": false
  }
]
```

**Source:** `code/tests/vllm/models/test_xpu_exl3_mtp.cpp`, lines 1871–1883. SHA-256: `3382dc5be65b19cefef988071fa32486df9286eeb6b3abdc8018c3f8ec5b2ba8`.

```text
  1871:     CHECK(trace.at("first_scheduled_position") == 0);
  1872:   }
  1873:   if (profile || host_profile) drain_profiles();
  1874:   result["backend_peak_device_bytes"] = vt::xpu::GetMemoryInfo().peak_allocated_bytes;
  1875:   CHECK(result["backend_peak_device_bytes"].get<uint64_t>() <= (uint64_t(32) << 30));
  1876:   struct rusage usage{};
  1877:   REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
  1878:   result["host_peak_rss_bytes"] = uint64_t(usage.ru_maxrss) * 1024;
  1879:   CHECK(result["host_peak_rss_bytes"].get<uint64_t>() <= (uint64_t(32) << 30));
  1880:   CHECK(vt::GetReferenceTierHits() == 0);
  1881:   loaded.reset();
  1882:   result["after_release_graph_bytes"] = vt::xpu::GetMemoryInfo().graph_device_bytes;
  1883:   CHECK(result["after_release_graph_bytes"] == 0);
```

## Evidence handling and reproducibility note

The companion audit contains `recompute.py`, `audit.json`, the host-test outputs and the selected supplemental verifier sources. Re-extract the current archive under an appropriate local source root and pass it to the audit script as documented in the package README. No font/model/driver/library binaries are included. Source snippets in this document retain the original licenses and attribution context; the complete native code and its license files remain in the user's archive.

The performance hypotheses and proposed thresholds in P0–P7 are engineering recommendations from this review. They are not measurements or claims that parity has already been reached.
