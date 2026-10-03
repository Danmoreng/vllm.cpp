# B70 native C++: EXL3 migration and parity plan

**Decision date:** 2026-10-02.  
**Native starting point:** `c1342199e09bd4c2c018c4694330152d283bf7ad` (`b70-gptq-int4`).  
**New production oracle:** qualified EXL3 source `c59d9442aba8610188837e37724600f1517d7335`, vLLM 0.30.0, production EXL3 v2.  
**Purpose:** one actionable, self-contained handoff for the next Codex session. This is a migration plan, not a claim that native EXL3 already matches production.

## 1. Decision and immediate next task

**Move the native project to the current EXL3 checkpoint now. Do not require a complete GPTQ performance/release campaign first. Build on the current C++ branch, not on an unreviewed wholesale merge of the old EXL3 branch.**

The supplied C++ checkout already contains EXL3 model loaders, target/MTP integration and registered XPU operators. The missing work is therefore **production-compatible execution, qualification and optimization**, not inventing EXL3 support from zero. Its existing EXL3 paths still differ from the new Python producer in precision, projection grouping and kernel dispatch. [E06–E11]

Keep GPTQ as a frozen **diagnostic/regression reference**, not a second production objective. A shared state, ownership, mask or sampler defect must be fixed before the affected infrastructure is reused. A discrepancy isolated to the GPTQ projection implementation must remain documented, but need not block a separately qualified EXL3 engine. Never mark the old GPTQ gate as passed merely because EXL3 passes.

### First concrete task for Codex

Create a small **EXL3 execution-contract and first-projection replay** change, with no full benchmark campaign:

1. Read the local repository instructions. Record actual HEAD and working-tree status without resetting anything. Create a dedicated migration branch from the current native base after preserving any local changes. Compare the old EXL3 branch read-only (`merge-base`, scoped diff and commit inspection); retain only demonstrably missing changes. The old branch is not included in these archives, so its content is not yet reviewed.
2. Produce `docs/b70-exl3/REFERENCE_MANIFEST.json` and `EXECUTION_MAP.md`. Record the two references in §3, the checkpoint inventory, real FP16 boundaries, selected kernel families, grouping, head policy and actual KV/state layouts. Resolve the installed Python draft-sampling mode rather than inferring it from the request temperature.
3. Add a fixture/replay for one real 4-bpw MLP projection and 128-aligned first/last blocks of the 6-bpw head, initially at **M=1 and M=4**. Record exact input/output dtypes and hashes. Run the existing native EXL3 implementation against that fixture and report the first differing stage: packed decode, input Hadamard, GEMM or output Hadamard/scaling.
4. In the same diagnostic tooling, preserve the old first-GDN-to-QKV GPTQ replay. Use it only to identify whether the existing failure is in shared state/normalization or GPTQ-specific math. Do not launch a GPTQ tuning sweep.

**First-task completion:** an executable, correctly failing-or-passing replay and a precise implementation map. A failed existing FP16 route is a useful result; silently widening to BF16 or using a different checkpoint is not. If the required local weights/captures are absent, report the exact missing assets and finish the host inventory checks; do not fabricate parity numbers.

## 2. Independent review: what is established

### 2.1 Work performed for this review

The two archives were unpacked and their selected active code inspected. This review independently checked:

- **1,867 EXL3 payload entries:** manifest SHA-256 and byte size, with zero mismatches.
- **13 declared active runtime-source dependencies:** included bytes match the expected guards. The union of the two active guard declarations was also inspected, rather than assuming a partial directory was complete.
- **147 native C++ source files:** match the supplied verification of the tracked `c1342199e` checkout, with zero mismatches. The archive is still a curated subset, not the complete repository.
- The production-policy hash and the original four-cap raw measurement files, worker identities, counts and published prefill/decode/wall medians.
- Locally rerun **10 host inventory tests**, the **host C4 output-copy test** with all its batch/row/output subcases, and the compiled **host bounded-cache policy test**. All passed. These are CPU/host checks, not new B70 numerical or performance tests.

No native model build, GPU inference, current live-container inspection or new BF16 inference was performed here. The eight live runtime artifact hashes are supported by the included read-only live receipt; omitted binaries cannot be independently rehashed from this ZIP. Large Q/K/V/logit captures and some task fixtures are also omitted. Supplied GPU pass receipts are therefore reviewed evidence, not independently reproduced GPU results. [E01–E05, E20]

### 2.2 Current EXL3 release: no newly demonstrated blocker in the reviewed scope

The latest release decision, promotion receipt and report say EXL3 v2 is deployed. Earlier pending/promotion states in nested historical documents must not override those later receipts. The final child image has the qualified parent's filesystem layers and adds release metadata/explicit cache environment. [E01–E03]

The source and receipts support these bounded conclusions:

| Area | Supported finding | Limitation / native-port requirement |
|---|---|---|
| Source identity | All 13 active guarded files are captured and match their expected hashes. | This is closure of the declared guards, not an audit of all Torch/driver dependencies. |
| Checkpoint loading | Producer receipts cover 409/409 quantized modules, including 8 MTP modules; host inventory rejects missing, duplicate and incompatible entries. | Native loader must reproduce the inventory and verify actual tensor shapes/bytes. |
| Partition cache | Capacity 64 per thread/queue; evict completed LRU entry; wait on one victim only when a full-cache miss finds all entries busy; last-use event chains cover prior uses. | Cache lifetime, graph references and VT allocations need an explicit native ownership contract. The bound is not global across unlimited queues. |
| Actual cache pressure | 147 observations: peak 64 entries, 7,937 hits, 87 misses, 23 evictions, zero pressure waits; RSS about 4.311–4.326 GiB. | The all-entries-busy GPU branch was not exercised by that worker. Host policy tests cover its logic, not GPU allocator lifetime. |
| C4 output copy | C4 with supplied output uses the direct permuted-view copy; C1–C3 and no-output calls retain the prior route. Host bit-pattern test and supplied GPU comparisons pass. | The approximately 0.8% microcopy observation is not a whole-server throughput gain. |
| Numerical behavior | Matched-state long-prefill and true C4 verification-graph probes pass rtol 0.01 / atol 0.003; v1/v2 short-reference aggregates match. | Finite probes do not establish universal determinism or broad task-quality superiority. |
| Capacity/serving | 261,120 input + 1,024 output; moderate C4, prefix operations, abort/recovery and restart receipts pass. | C16 is pressure load: four preemptions were recorded; no promise of 16 simultaneous maximum contexts. |

**No release-critical new defect is demonstrated by the reviewed source and receipts.** There are nevertheless specific ownership assumptions that must not be copied blindly into a different C++ allocator or multi-queue runtime. The producer's global INT8 primitive/scratch machinery is not covered by the per-queue SDPA capacity limit. Make its native counterpart explicitly context/queue-owned and safely reusable. [E04, E11, E19]

The user's observation that EXL3 needs fewer correction cycles is a valid reason to choose it as the product target. It is not a universal quality claim established by the power campaign. The short panel reports PPL 3.646722 and BF16-relative KL 0.032481; the long suffix is a compact 128-position/8-distribution window. Historical Flappy and QueueKit results remain dated task evidence; QueueKit is 7/8 checks and 1/2 tasks, with the same failure in its native control. Do not relabel these as newly rerun or perfect results. [E03]

### 2.3 Four-cap audit and the baseline to retain

Each cap contains **20 scenarios, 70 measured waves, 124 requests and 126,976 output tokens**. Request markers and prompt/completion counts agree across caps. The original worker ID and image in each raw result match its separate worker receipt. All included outputs finish at the requested 1,024-token limit; there are no preemptions or excess recomputed prefill tokens in these matrices.

The following values were independently recomputed from original per-wave wall time and card-energy counters:

| Cap | Sum measured wave time | Card energy | Matrix output throughput | Tokens/joule |
|---:|---:|---:|---:|---:|
| 150 W | 50.05 min | 125.16 Wh | 42.28 tokens/s | 0.2818 |
| **180 W** | **41.68 min** | **125.10 Wh** | **50.78 tokens/s** | **0.2819** |
| 230 W | 35.77 min | 137.16 Wh | 59.17 tokens/s | 0.2571 |
| 275 W | 33.89 min | 155.34 Wh | 62.44 tokens/s | 0.2271 |

Use `sum(output_tokens) / sum(wave_wall_seconds)` and `sum(output_tokens) / sum(card_joules)`, not the average of scenario rates. These figures include prefill, decode and request overhead. They exclude warmup, worker startup and the separate isolated 64K-prefix probe; they are **card energy, not whole-system energy**, and **matrix throughput, not decode speed**. Concurrent per-request native decode seconds overlap and must not be added to create elapsed server time. [E02]

Keep **180 W** for the initial native comparison. Here, 150 W took 20.09% longer with essentially identical energy (about 0.051% higher); higher caps traded additional energy for time. The non-random sequence was 180 → 230 → 275 → 150 W, with no fresh repeated 180 W thermal control and no comparable 180 W thermal series. This supports a stable project baseline, not a universal efficiency optimum. Do not reopen power tuning during the port.

Useful current 180 W serving anchors, not native parity proofs:

| Workload | Native prefill rate | Native request-weighted decode rate | Other metric |
|---|---:|---:|---|
| 4K C1 | 2,241.38 tokens/s | 57.14 tokens/s | TTFT 1.832 s |
| 32K C1 | 1,977.65 tokens/s | 54.50 tokens/s | TTFT 16.595 s |
| 128K C1 | 1,262.55 tokens/s | 39.93 tokens/s | TTFT 103.984 s |
| 4K C4 | not comparable to C1 prefill | 41.70 tokens/s, median request-weighted rate across waves | **180.07 tokens/s aggregate fully overlapped decode** |

These are sampled MTP3 measurements. Different continuations and acceptance can change their work per emitted token. A fresh small matched replay is required before attributing a difference to native kernels.

## 3. Freeze the correct references

### Reference A — historical GPTQ diagnostic oracle

- Native HEAD: `c1342199e09bd4c2c018c4694330152d283bf7ad`.
- GPTQ model revision: `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.
- Frozen Python source: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, archived patches and historical image.
- Preserve each fixture's actual page geometry, dtype, MTP depth, token inputs and timing contract. The current MTP measurements used physical/effective geometry recorded as page 1664; older non-speculative documents also contain 1600. Do not silently normalize them.
- The target-TV failures of 0.02445/0.02329 versus 0.02 remain open. With identical Python FP16 K/V inputs, the native FP8 writer is bit-identical; the earlier K/V difference already exists upstream. [E16]

### Reference B — current EXL3 production and future native EXL3 oracle

| Field | Frozen value |
|---|---|
| Serving source | `ed47c5b84614f5654440355b12fa61b6eea8ee5e` |
| EXL3 source | `c59d9442aba8610188837e37724600f1517d7335` |
| Production image | `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918` |
| Qualified runtime parent | `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737` |
| Policy SHA-256 | `1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839` |
| Checkpoint | `turboderp/Qwen3.8-27B-exl3` @ `113cf7ab958054860e43fb7f3063b1af19171095` |
| Representation | EXL3 mul1; 4-bpw body, **6-bpw full target head** |
| Arithmetic | FP16 model execution; native SmallM at M ≤ 128; rotated-basis W8A8 at larger M |
| Context/KV | 262,144 total tokens; FP8 KV; actual physical layout must be captured, not inferred from old recipes |
| Speculation | MTP3; full 248,320 target vocabulary; **65,536-entry draft subset** |
| Attention | Guarded eager oneDNN at Q ≥ 64 and exact K ≥ 4,096, up to 262,144; query bucket 256; M04 uniform verification Q2–Q5 |
| Batching/graphs | 4,096 scheduled token budget; clean qualified C4; FULL_DECODE_ONLY graphs |
| Budget | One 32 GB B70, 32 GB host RAM, 180 W; GPU fraction 0.965 is the producer policy, not a promise that a different allocator has identical overhead |

**Do not carry the former GPTQ 196,608 oneDNN ceiling, 512-row W4A8 threshold, MTP4 or full draft vocabulary into the EXL3 contract.** Those were decisions for another representation/runtime. Conversely, do not change the current EXL3 thresholds before establishing parity.

The current producer has conditional draft sampling: probabilistic sampling when a draft-logit buffer is configured, otherwise argmax. Its launch JSON specifies MTP3 but not that resolved option. Record the resolved configuration from the pinned worker/config implementation. Do not infer draft sampling from target temperature 1.0 or assume the native historical greedy proposer has identical semantics. [E14]

**Never use raw GPTQ-versus-EXL3 logits to diagnose an engine bug.** Numerical engine comparisons must share checkpoint bytes, token prefix, embedding/normalization precision, projection grouping, cache/state initialization and visibility, quantization scales, attention geometry and output rounding. References A and B are separate test assets, not two user-facing native profiles.

## 4. What can be reused, and what must change

| Area | Existing native implementation | Required EXL3 action |
|---|---|---|
| Checkpoint loader | EXL3 trellis/scales in full attention, GDN, MLP, head and MTP already exist. | Validate all 409 modules against the authoritative safetensors index, including MTP entries absent from some quantization metadata; preserve 4/6 bits and mul1. |
| Activation precision | Default is BF16; `DenseDev` applies the explicit FP16 precision only to GPTQ; EXL3 Q/K/V and down projection have explicit BF16 outputs. | Introduce/extend a real precision contract independent of `gptq4_checkpoint`; use FP16 throughout the EXL3 model boundaries required by the producer. Do not impersonate a GPTQ checkpoint or enable the explicitly refused global F32 switch. |
| Existing XPU EXL3 | Packed/reference/fused/panel paths are registered. Old automatic dispatch requires F32 output (BF16 is remapped); **FP16 output falls through to packed**. | Correct dispatch and port qualified SmallM ESIMD/DPAS kernels. Merely fixing output dtype may make the old path slower until dispatch is fixed. |
| Projection topology | Full-attention EXL3 Q/K/V remain separate; GDN and MLP use older EXL3 seams. | Group-aware QKV, QKVZ and gate/up operators, retaining each source group's input scaling/Hadamard transform. Do not concatenate weights and apply one shared `suh`. |
| Large-M linear | Old native path reconstructs FP16 panels with matrix work; it is not producer W8A8. | Port native input Hadamard/INT8 quantization, reconstructed INT8 weights, oneDNN integer GEMM/scales and output Hadamard. Keep packed EXL3 as storage. |
| State/attention | Native GDN, FP8 writer, paged attention and optional Xe2 verifier already exist. | Qualify with EXL3 FP16 activations and current exact cache layouts. Reuse interfaces, not unproved old whole-model correctness. |
| Draft/head | EXL3 draft loading exists; historical proposer computes all head rows then downloads FP32 logits for CPU selection. | Preserve current 65,536 draft subset, gather required hidden rows **before** the head, then native selection and explicit buffer ownership. |
| Graph/prefix | Target `hidden_tap` bypasses graph selection; recurrent prefix snapshots plus speculation are refused. | Implement explicit graph outputs and speculative prefix/state commit before claiming full native production parity. |

The qualified producer's `csrc/exl3_ops.sycl` already implements the heavy EXL3 operations in C++/SYCL. Replace its ATen/Torch registration, allocation and tensor wrappers with native VT interfaces; **do not link a Python/Torch runtime just to call the donor `.so`**. Retain upstream attribution/licenses. Match the pinned native math first; newer unreviewed upstream kernels are a separate experiment. [E06–E15]

### Representation details that are easy to get wrong

The producer stores trellis as int16 `[K/16, N/16, 16*bits]`. The existing byte view `[K/16, N/16, 32*bits]` can represent the same storage, but it needs a bit-preserving view, not a numeric int16-to-byte conversion. Validate tile/bit order and alignment.

A fused projection needs source-group boundaries and `shard_of_nb` mapping for 128-output blocks. The GDN Q/K/V source and Z source can have different input transforms. Gate and up can also differ. The authoritative inventory and grouping report, not assumed model names alone, define what may be merged.

The draft subset consists of **whole 128-token Hadamard blocks**. Validate unique block IDs, range and order; retain the exact global token-ID map and out-of-subset behavior. In the producer, absent vocabulary entries receive negative infinity before sampling. For a later compact native sampler, preserve global tie-breaking, proposal support and probability semantics. A permutation of subset storage must not change ties. [E10, E12]

## 5. Implementation sequence and gates

### S0 — Freeze, map and isolate common risks

**Files:** `docs/b70-exl3/`, native EXL3 test fixtures; inspect `qwen3_5_dense_weights.cpp`, `qwen3_5_mtp.cpp`, shared EXL3 headers and XPU registration. Some shared headers are referenced but omitted from the curated archive: inspect them in the real repository before editing.

**Deliver:** §1's first task, two immutable oracle manifests, actual route/precision map, source-branch diff and single-projection fixtures. Record selected driver/compiler/library bindings and configuration-derived sampling defaults.

For the old TV failure, use one frozen prefix and the first three GDN blocks into the first full-attention block. Compare identical-input stage replays: normalization → projection → Q/K norm/RoPE → FP8 write; V helps isolate errors before RoPE. If a shared cache/state/normalization defect is found, fix it now with a regression test. If only GPTQ projection differs, record the remaining failure and proceed independently with EXL3. No broad GPTQ campaign.

**Stop:** any missing/mismatched checkpoint, incomplete inventory, unsafe ownership or unknown layout. Do not patch around failure by weakening the old TV gate.

### S1 — Native EXL3 FP16 vertical slice, then eager target

**Files:** `src/vllm/model_executor/models/{qwen3_5_dense_weights,qwen3_5,qwen3_5_mtp}.cpp`, the shared `dense_exl3`/linear seams, `src/vt/xpu/{xpu_exl3,xpu_exl3_strategy,xpu_ops}.*` and tests.

1. Validate loader metadata and packed bytes without full dequantization. Count all 409 modules and eight MTP modules even though speculation is initially disabled.
2. Make EXL3 FP16 activation/output boundaries explicit, including embedding, residual/norm, QKV, GDN gates, MLP, final norm and head. Maintain FP32 recurrent accumulation/state where the captured producer requires it. Resolve types from the oracle rather than copying BF16-era comments.
3. Port SmallM mul1 4/6-bpw kernels through a typed grouped-linear VT API. The CPU-order-preserving native path remains a small diagnostic oracle, not the production performance target.
4. Match one block, then a complete eager target at P128/D1 and P128/D64, no MTP and no graphs. Keep the target head at 6 bpw and full vocabulary. Gather final prompt rows before head evaluation.

**Gate:** exact loader/mapping checks; discrete cache/position metadata exact; real projection output relative norm < 2e-3 against the appropriate independent/reference arithmetic; finite outputs; first differing model stage localized. Attention uses the existing producer rtol 0.01 / atol 0.003 on identical inputs. A passing local projection does not waive model/state checks.

**Stop:** silent fallback to BF16, scalar packed production execution without a route witness, incorrect group transforms, duplicated full dense head or unbounded logit temporary.

### S2 — Production prefill and long-context building blocks

**Files:** new native implementations behind the existing EXL3 linear/attention API; `xpu_exl3_prefill.cpp`, backend allocator/queue interfaces, attention dispatch and cache metadata code. Donors: `exl3_ops.sycl`, `exl3_esimd.h`, `attention_dispatch.py`, `attention_metadata.py`, `target_runtime.py`.

- Port large-M **rotated-basis W8A8**, not GPTQ W4A8. Reproduce Hadamard normalization, FP16 intermediate rounding, per-row activation scales, mul1 reconstruction bound `3.453125`, INT8 arithmetic, output scales and FP16 output Hadamard.
- Preserve the SmallM boundary M=128/129 and row-padding contract. A padded graph's physical row count is part of dispatch; test poisoned padding and large-first → small-call sequences.
- Port FP8 gather and exact-K oneDNN prefill using the current guarded metadata decision once per step. Preserve bottom-right causal alignment (`j <= L-Q+r`) and the actual logical/physical page strides. Do not pad K to improve cache reuse.
- Use bounded, completion-aware compiled partitions; explicit dependency events and scratch lifetime; no unconditional per-layer synchronization. Do not port inactive old `EXL3_KV_BLOCK_EXACT` or legacy sparse-retirement monkeypatches instead of current V2 behavior.
- Large W8A8/oneDNN prefill stays eager initially. FULL_DECODE_ONLY is the producer's profile; capturing large prefill is not a migration prerequisite.

**Gate:** M129/256/512 and actual 4096-row budget; one representative family each for GDN-QKVZ, attention-QKV, gate/up, down and head blocks. Compare W8A8 against an independently quantized reference of the **same** arithmetic, not require it to equal unquantized FP16. Continue target/state tests at P4096/D64 and a chunked continuation to 32K. Confirm cache writes occur once, even when attention routes are split.

**Stop:** memory exceeds the declared budget, tail/continuation mask fails, cache eviction releases live resources or oneDNN splits into an unintended unfused/memory-heavy graph. Report the actual unsupported shape and retain a qualified native implementation for it; never silently change checkpoint/precision.

### S3 — MTP1 bridge, then the production MTP3 contract

**Files:** `qwen3_5_mtp.cpp`, MTP model entry points in `qwen3_5.cpp`, `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, input preparation, sampler/rejection and state commit code.

- Use existing native speculative Conv/GDN and separate draft KV infrastructure after S1/S2 qualification. First MTP1 gives a small acceptance/rollback test; then fix the production setting at **MTP3**.
- Load the eight EXL3 MTP modules exactly; share intended read-only target resources. Create the exact 65,536-entry, 128-block-aligned draft head and token map. Do not reconstruct a full FP16 target/draft head or turn on GPTQ draft overrides.
- Move selected hidden-row gathering before draft-head computation. Keep any other hidden rows needed for draft KV preparation. Capture whether logits-hidden and feedback-hidden are identical or distinct in the resolved producer; preserve its final-normalization boundary.
- Replace full-logit CPU downloads with native token selection. A first bounded version may return only selected IDs with an explicit event; then keep successive draft steps device-resident. Removing a wait without retaining the old hidden/metadata allocation is forbidden.
- Match resolved draft sampling, target temperature/top-k/top-p, prefix-dependent processors, EOS/stop handling, RNG identity per request/token/draft position and rejection behavior. Probabilistic proposals require their actual proposal distribution; a greedy proposal requires the corresponding deterministic-proposal contract. Do not assume these are interchangeable.
- Qualify native verification Q2–Q4 (MTP1–3) first. Q5 remains a useful compatibility boundary for the existing generic kernel, not a request to enable MTP4.

**Gate:** forced accepted counts from 0 through k, first/middle/last rejection, full acceptance/bonus, output limit, EOS and cancellation. Commit the correct KV length, Conv history, GDN snapshot and hidden state per request. Teacher-forced target distributions must pass at the same prefix; sampled-text identity is not the only correctness criterion. Add deterministic RNG/reorder replay tests for the sampler contract.

**Stop:** rejected tokens remain visible, a compact draft ID leaks as a target ID, wrong state survives cancellation, output limit is exceeded or target parity fails. No speed claim based only on accepted-token ratio or one altered sampled continuation.

### S4 — Graph ownership and real C2/C4 MTP

**Files:** target graph driver (`qwen3_5_dense.cpp`), platform graph dispatch, MTP proposer, speculative Conv/GDN kernels, `xpu_attention_verify_xe2.cpp`, state/slot lifecycle.

- Replace the early `hidden_tap` eager-only return with an explicit persistent graph output contract. Stable pointers, recorded input copies, output handoff and retired graph-resource references must be owned through completion.
- Verify single-request eager versus graph before batch graphs. Check request-history dependence, not just repeated replay of one unchanged request. Retain the old C8 near-tie case as a regression fixture; do not first enlarge the graph batch cap.
- Generalize the real verification batching using current producer M04 layout/causality. Do not implement C4 as four assumed-fast serial C1 calls without measuring it. The old C1 GDN workgroup-64 and verifier wins are not evidence of C4 speed.
- Port C4-only direct output copy, preserving C1–C3/no-output behavior. Preserve the producer split table initially; tune only after attribution.
- Resolve exact CPU prefill metadata once per step and keep active decode lengths correct on-device. Never use an asynchronous CPU upper bound as an exact length.

**Gate:** C1/C2/C4, Q2–Q4; uniform and ragged acceptance, request permutation, compaction, arrival, EOS, cancellation/reuse and graph→eager→graph transitions. A replay with changed IDs, positions and state slots must match eager at the same restored input state. No repeated compilation, per-step host vocabulary transfer or growing retained buffers.

**Performance check:** only after correctness, use the small C1/C4 4K/1024 pair, then one 32K-decode + 128K-prefill case. Measure target, draft, sampler/state commit and host gaps separately in diagnostics; score uninstrumented runs.

### S5 — Recurrent prefix, 262K capacity and operational finish

**Files:** runner prefix guard/implementation, KV/GDN snapshot pools, sparse block retirement, request scheduler/admission, server token/stream/parser endpoints.

The current native runner explicitly rejects recurrent prefix snapshots with speculation. Implement the combination rather than leaving prefix enabled in configuration but ineffective/unsafe. Prefix keys must include relevant model/quantization/precision/position identity. Sparse retirement must scan valid processed regions past holes without freeing in-flight or shared states. Use the current V2 behavior, not a legacy patch selected for another allocator.

**Gate:** one 32K exact resend/extension, one isolated 64K cold/warm test, one 128K continuation and one exact **261,120 input + 1,024 output** boundary. Test long-request admission with a short concurrent request, completion/cancellation and reclaimed slots. Run only the long cases whose predecessors pass; no four-cap or 124-request matrix here.

C4 refers to moderate concurrent requests within capacity. Queue or reject clearly when the one-card budget cannot admit another large context; do not claim four or sixteen simultaneous maximum contexts. Image/video parity is a separate workstream: the initial native text release is not a complete replacement for all multimodal features of the Python server.

### S6 — Finish and optimize only measured bottlenecks

After S1–S5, compare the representative cases in §6 against current EXL3 Python at the same card cap and timing boundary. Rank **measured** costs. Prefer grouped SmallM/verification reuse, removing proven synchronization and safe scratch reuse before numerical changes. Keep the checkpoint, MTP3, draft subset and precision contract frozen.

A text capability milestone may be recorded before speed parity, explicitly naming remaining gaps. A release may be called **performance parity** only when matched primary workload medians are at least as fast within the predefined measurement band, with no unresolved quality or memory failure. Use a >5% slowdown as a repeat/investigation trigger and reverse run order once; it is not permission to declare a repeatable 5% loss equal. Beyond-Python performance requires actual matched measurements.

## 6. Bounded validation and performance matrix

Do not take a Cartesian product of every row, context, batch and power cap. Reuse previously passing receipts when implementation and contract hashes are unchanged.

| Layer of testing | Initial scope | Expansion only when relevant |
|---|---|---|
| Loader/packing | All 409 module metadata entries; actual 4/6-bpw tensor headers; complete MTP/head map | Bit-unpack spot checks at first/last tiles and every family; negative missing/duplicate/codebook cases |
| SmallM math | M=1,4,12,16,128 on representative real families | 2,3,5,8 and physical graph rows 24,32,40,48,56,64 only for routed cases |
| LargeM boundary | M=129,256,512 plus actual 4096-row prefill | Tail rows 127/128/129 and 255/256/257; actual last chunk, not all lengths on every family |
| Attention/state | Q1/Q2/Q3/Q4, exact K4096 and one 32K continuation | K4095/4096/4097 and actual page-boundary ±1; long exact-K sample; Q5 compatibility separately |
| Eager target | P128/D1, P128/D64, P4096/D64 | State/logit checkpoints after 1,64,256,1024 steps in one selected long decode |
| MTP lifecycle | k1 then k3, C1; every accepted-prefix length | C2/C4 ragged, EOS, cancellation, compaction, graph transitions |
| Scored serving | C1 P4096/O1024, C4 P4096/O1024; 3 repetitions per arm | C1 P32768/O1024; one controlled mixed 32K/128K case |
| Long capacity | Prefix32K/64K, P128K continuation, exact maximum case | A second long held-out task only if a prior discrepancy needs it |

Use a small number of frozen prompts representing natural text, code and structured/tool output. For quality, add a few executable multi-file coding tasks and long-context retrieval/review checks not used to tune kernels. The user's completion-with-fewer-repairs objective belongs in a separate task outcome/time-to-success report; it is not measured by native tokens/s alone.

### Parity and measurement rules

1. **Exact invariants:** checkpoint bytes, packed layout, token IDs/positions, subset map, request identity, causal visibility, accepted counts, logical lengths and ownership. FP8 writer bytes must agree when given identical source values and scales.
2. **Operator arithmetic:** producer's existing real-linear relative-norm threshold <2e-3; attention rtol .01/atol .003 on matched inputs. SmallM and W8A8 use their respective independent references. Record max error and finite status, not only one aggregate metric. Exact data rearrangements use bit-pattern equality, including direct-copy tests.
3. **Whole-target comparison:** begin with the frozen GPTQ diagnostic envelope (TV .02, KL .002, top-10 overlap ≥9) as a conservative **EXL3 investigation trigger**, not evidence already validated for EXL3. Calibrate an EXL3 same-checkpoint/route baseline before accepting a different envelope. Near ties may explain a token mismatch, never an ownership/mask failure. Do not weaken a gate after observing a failure just to green the port.
4. **Quantization quality versus engine correctness:** current EXL3-vs-BF16 differences are part of the chosen checkpoint/arithmetic. Reimplementing that recipe should not introduce an unexamined new quality tradeoff. Use held-out application tests to supplement, not replace, matched-state checks.
5. **Three timing scopes:** synchronized operator replay; complete worker step including the target head and required sampling/state work; end-to-end emitted tokens/TTFT/request wall. Never compare one scope's number with another.
6. **MTP accounting:** count proposed, accepted and emitted tokens, target cycles and draft steps separately. MTP3 normally verifies up to four rows per request, not five. O counts total emitted tokens; D counts extra target forwards in non-speculative tests. First output arises from prefill. Do not infer work from an output token repeated 1,024 times.
7. **Performance attribution:** for device-core comparisons restore identical full states, tokens, proposals, accepted lengths and physical batch forms. For serving, keep prompts and sampling controls fixed but report different continuations and acceptance; the same seed need not produce the same text across arithmetic implementations.
8. **Operational isolation:** do not run Python and native models concurrently on the B70 to “pair” them. Use existing captures first. A scheduled exclusive test may temporarily use the GPU with explicit authorization; preserve and restore the production identity. Compilation/host tests must also respect 32 GB host RAM.

## 7. Memory and asynchronous ownership contract

### No dense checkpoint expansion

Memory-map/stream checkpoint shards and upload packed modules. Do not build a full dequantized 27B host or device model. Avoid simultaneously retaining duplicate grouped/un-grouped packed copies after ownership-safe loader completion.

Useful planning calculations, not measured peaks:

- A `[4096,248320]` FP32 logit tensor alone is **3.789 GiB**. Gather requested hidden rows before both target and draft heads.
- Raw 6-bpw target-head weights are about **0.888 GiB**, versus about 2.368 GiB in FP16, excluding scales/packing. A 65,536-row 6-bpw draft subset is **240 MiB** raw packed data. Keep the qualified subset rather than expanding the draft for speculation.
- Given the archived 16 full-attention layers, 4 KV heads and D256, raw target FP8 K/V at 262,144 tokens are **8 GiB**. A draft full-attention layer with the same KV geometry adds up to 0.5 GiB at that length. Physical pages, padding, scales and actual active lengths change allocation requirements.
- The archived GDN matrix geometry is 3 MiB per layer per state slot, about 144 MiB over 48 layers, before Conv history, speculation or prefix snapshots. Calculate the actual number/lifetime of state slots; do not allocate a dense historical state at every token.
- Unpacking FP16 K/V for one KV head at 262,144 tokens uses **256 MiB** before query/output/SDPA scratch. Do not keep per-layer FP16 copies of the entire cache.

**Important donor detail:** the `slice_n=16384` parameter bounds the FP16 reconstruction fallback. In the active oneDNN W8A8 branch, the source actually allocates reconstructed INT8 weights for an entire source group (`[K, ng]`), not that slice loop. Inspect/measure this real allocation. In particular, accidentally running a large-M full-vocabulary head could reconstruct roughly 1.184 GiB of INT8 weights in addition to large logits. Row selection avoids that normal serving case. Do not claim all donor reconstruction is already capped at 16K columns. [E11]

### Ownership requirements

Every asynchronous output, metadata buffer, scale, reconstructed-weight panel, state snapshot, compiled partition and graph binding must have an owner through its last GPU consumer. Express dependencies explicitly or prove the in-order queue contract. A wait removed from the old proposer must be replaced with valid retained ownership, not just deleted.

Cache keys include device/context/queue identity and lifetime, exact shape/dtype/strides/causal options and relevant precision configuration. Do not use a recycled queue address as a lasting identity. Retiring a partition must account for graph-held references as well as its last ordinary event. Drain safely on shutdown/queue destruction and asynchronous error. One worker/queue is enough initially; concurrent requests do not require unsafely sharing global scratch across multiple queues.

Keep application caches bounded and observe native allocator, oneDNN internal cache and graph residency separately. The Python 64-entry SDPA bound does not prove all native caches are bounded. Add host tests for completion-aware eviction, all-busy pressure, factory failure, owner destruction and reordered completion; one tiny GPU eviction/teardown test belongs to the native integration, not another full serving campaign.

## 8. What not to do, and when to stop

**Keep deferred:** complete GPTQ performance parity; a broad old-branch merge; new MTP-depth/vocabulary sweeps; full-vocabulary draft expansion; GPTQ W4A8 or its 196,608 ceiling transplanted into EXL3; power-cap sweeps; TP>1; C16 optimization; large-prefill graph capture; new quantization algorithms; vision/video porting before native text parity.

Do not re-run the 124-request serving matrix at each milestone. Do not turn historical plans into instructions to restart old experiments. Do not expand the quality scope endlessly: use the bounded tests above and add a case only for a concrete failure or missing production capability.

The native EXL3 text path is ready to promote when:

- The exact EXL3 checkpoint, 6-bpw target head and 65,536-entry MTP3 draft are loaded and traced; Python/Torch/Triton are not inference dependencies.
- Matched-state eager/graph/operator and MTP commit/RNG tests pass; no known shared correctness defect is hidden by a different quantization.
- Moderate C4, prefix reuse, cancellation/recovery and the 262K total-token boundary fit GPU/host budgets without unbounded resource retention.
- The fixed representative serving measurements are reproducible and accurately labeled. Any remaining speed gap is explicit; no historical GPTQ percentage is relabeled as EXL3 parity.
- One native EXL3 production policy is documented. Internal shape-specific safe kernel selection is allowed; silently loading GPTQ or silently changing precision is not. The existing Python service remains the production service until the explicit deployment decision.

**Bottom line:** reuse the current C++ model/state/serving infrastructure, replace or extend the EXL3 execution seams to match the now-qualified producer, and optimize the same-checkpoint EXL3 path. Do the few shared correctness checks that protect that reuse; do not spend another campaign finishing a weight format the user no longer runs in production.

## 9. Compact machine-readable milestone contract

The following is a planning contract, not an executable command list. Paths under `docs/b70-exl3` are proposed output paths in the native repository.

```json
{
  "schema": 1,
  "decision": "migrate_native_exl3_now_reuse_current_cpp",
  "date": "2026-10-02",
  "cpp_base": "c1342199e09bd4c2c018c4694330152d283bf7ad",
  "exl3_source": "c59d9442aba8610188837e37724600f1517d7335",
  "checkpoint_revision": "113cf7ab958054860e43fb7f3063b1af19171095",
  "constraints": {
    "gpu_count": 1,
    "gpu_memory_gb": 32,
    "host_memory_gb": 32,
    "power_w": 180,
    "total_context_tokens": 262144,
    "mtp_depth": 3,
    "target_vocab": 248320,
    "draft_vocab": 65536,
    "clean_concurrency": 4,
    "python_inference_dependency": false,
    "preserve_existing_production_worker": true
  },
  "milestones": [
    {
      "id": "S0",
      "deliverable": "immutable references, branch/precision map, real projection replay",
      "stop_on": [
        "wrong identity",
        "missing packed data",
        "shared ownership or state bug"
      ]
    },
    {
      "id": "S1",
      "deliverable": "FP16 EXL3 eager target and qualified native SmallM",
      "gate": [
        "4/6-bpw packing",
        "relative norm < 0.002 on appropriate real linear oracle",
        "P128 D64 target/state replay"
      ]
    },
    {
      "id": "S2",
      "deliverable": "rotated W8A8 prefill and exact-K native SDPA",
      "gate": [
        "M128/129 boundary",
        "padding poison",
        "P4096 D64",
        "32K continuation",
        "bounded memory/events"
      ]
    },
    {
      "id": "S3",
      "deliverable": "MTP1 bridge then exact production MTP3 with draft subset",
      "gate": [
        "accepted counts 0..k",
        "correct global token mapping",
        "state commit",
        "EOS/cancel",
        "sampling contract"
      ]
    },
    {
      "id": "S4",
      "deliverable": "owned graphs and C2/C4 MTP",
      "gate": [
        "mutated inputs",
        "ragged acceptance",
        "request reorder/reuse",
        "same-state eager comparison"
      ]
    },
    {
      "id": "S5",
      "deliverable": "prefix, long context and text serving qualification",
      "gate": [
        "32K/64K prefix",
        "128K continuation",
        "261120 input + 1024 output",
        "capacity-safe admission"
      ]
    },
    {
      "id": "S6",
      "deliverable": "bounded matched performance and explicit promotion",
      "gate": [
        "C1/C4 emitted-token metrics",
        "no unqualified quality change",
        "no unbounded memory growth"
      ]
    }
  ],
  "scored_primary_cases": [
    {
      "concurrency": 1,
      "prompt_tokens": 4096,
      "output_tokens": 1024,
      "repetitions": 3
    },
    {
      "concurrency": 4,
      "prompt_tokens_each": 4096,
      "output_tokens_each": 1024,
      "repetitions": 3
    },
    {
      "concurrency": 1,
      "prompt_tokens": 32768,
      "output_tokens": 1024,
      "repetitions": 3
    }
  ],
  "diagnostic_not_full_sweep": true,
  "reference_rules": [
    "GPTQ logits compare only to frozen GPTQ reference",
    "EXL3 logits compare only to frozen EXL3 reference",
    "sampled serving is not fixed-work kernel timing",
    "new tolerance requires prior documented calibration, not post-failure relaxation"
  ]
}
```

## 10. Evidence appendix and source boundaries

Archive-relative roots below:

- **EXL3/** = `B70_EXL3_Current_Production_and_Power_Review_2026-10-02.zip`.
- **CPP/** = `B70_GPTQ_FP8_MTP_Review_c1342199e_2026-09-28(1).zip` (same supplied Monday checkpoint).
- **AUDIT/** = independently generated checks in this review, not a production run.

Excerpts retain original file line numbers. Hashes identify the exact inspected files. Local code comments describe their original scope and are not automatically current production guarantees. The appendices are sufficient to locate each major finding; the original archives/full local checkout are still required for implementation.

### E01 — Current identity, source closure and policy

**Source:** `EXL3/handoff/START_HERE.md`  
**SHA-256:** `7b573585efb951a39ab7563641fb7397949ded50beba0aaa9c4ea585d6fdc731`

Lines 1–19:

```text
00001 | # Current EXL3 production and native C++ review handoff — 2026-10-02
00002 | 
00003 | This bundle is an evidence snapshot for an independent review, not a new qualification run. Read `handoff/REVIEW_REQUEST_EN.md` for the requested next-step assessment. The neighboring native C++ archive is supplied separately, unchanged.
00004 | 
00005 | ## Exact current production identity
00006 | 
00007 | - Owned serving repository: `ed47c5b84614f5654440355b12fa61b6eea8ee5e`, matching local `origin/main`; tracked checkout clean at capture.
00008 | - Qualified EXL3 source: `c59d9442aba8610188837e37724600f1517d7335`.
00009 | - Production tag: `local/b70-qwen38-vllm:production-exl3-v2`.
00010 | - Immutable image: `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`.
00011 | - Qualified runtime parent: `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737`; the child has identical rootfs, with release metadata/policy and explicit cache-capacity environment settings.
00012 | - Policy SHA-256: `1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839`.
00013 | - Active user service `b70-qwen38-vllm.service`, container `b70-qwen38-vllm`, default card cap **180 W**, verified through read-only inspection.
00014 | - All eight policy-pinned native/source/profile artifact hashes match the running container. All **13 active guarded runtime-source dependencies** are included and hash-verified. See `qualified-runtime/SOURCE_GUARD_COVERAGE.json` and `handoff/LIVE_ARTIFACT_HASH_CHECK.json`.
00015 | 
00016 | vLLM 0.30.0, Torch 2.13 XPU, oneAPI 2026.1.1 and oneDNN 3.13. EXL3 4 bpw body / 6 bpw head, FP16 execution, FP8 KV, MTP3, 65,536-entry draft vocabulary and full 248,320-entry target vocabulary. GPU memory fraction 0.965; 262,144 total-token limit; 4,096 batched tokens; FULL_DECODE_ONLY graphs. Admission is 16 requests; moderate C4 is qualified, C16 is pressure load with accepted preemptions, not 16 maximum contexts. Multimodal limits: 32 images and four videos per prompt, subject to shared capacity and configured pixel limits.
00017 | 
00018 | ## Current power measurements
00019 | 
```

### E02 — Power definitions, exclusions and campaign limits

**Source:** `EXL3/handoff/START_HERE.md`  
**SHA-256:** `7b573585efb951a39ab7563641fb7397949ded50beba0aaa9c4ea585d6fdc731`

Lines 21–38:

```text
00021 | |---:|---:|---:|---:|---:|---:|---:|
00022 | | 150 | 150.05 | 50.05 | 42.28 | 0.2818 | 1014.47 | 125.16 |
00023 | | 180 | 180.10 | 41.68 | 50.78 | 0.2819 | 1014.99 | 125.10 |
00024 | | 230 | 230.09 | 35.77 | 59.17 | 0.2571 | 925.73 | 137.16 |
00025 | | 275 | 274.99 | 33.89 | 62.44 | 0.2271 | 817.42 | 155.34 |
00026 | 
00027 | Each cap used the same immutable image and serving configuration, frozen prompts/seeds, **20 scenarios, 70 measured waves, 124 requests and 126,976 output tokens**, plus a separate isolated 64K prefix-cache probe. No preemptions or recomputed extra prefills occurred in the serving matrices. The already-qualified fresh 180 W baseline was reused.
00028 | 
00029 | The table uses summed card energy and wall time for the 70 measured waves, including prefill, decode and request overhead. It excludes warmup, worker startup and the supplementary prefix probe. This is **card energy, not whole-system electricity**, and **matrix throughput, not decode-only speed**. Per-scenario decode/prefill rates remain in the published comparison and raw JSONs. Do not average scenario rates to reproduce the table.
00030 | 
00031 | At 150 W the matrix took 20.09% longer than at 180 W while measured energy was 0.051% higher: effectively tied efficiency, with better latency at 180 W. 230/275 W improve throughput but reduce tokens per joule in this workload. One campaign per cap with three/five per-scenario repeats and no randomized cap order does not establish a universal optimum. MTP acceptance and generated continuations can differ despite frozen prompt/seed/configuration.
00032 | 
00033 | Primary published comparison: `engine-code/benchmarks/results/exl3-power-20261002-with-150w/comparison.json`. All four original source-review result JSONs and isolated-prefix JSONs are retained **byte-for-byte** under `handoff/power-raw/`, with worker identities and original-path/hash provenance. Temperature observations exist for 150/230/275 W; the earlier 180 W run has no equivalent thermal series.
00034 | 
00035 | ## Release evidence and bounded improvements
00036 | 
00037 | Start with `engine-code/docs/EXL3_RELEASE_V2_REPORT.md` and `engine-code/benchmarks/results/exl3-review-release-v2/`. Release decision, promotion, final consistency, restore/rollback preflight, source coverage, serving results and diagnostic receipts are included.
00038 | 
```

### E03 — Final production qualification and finite evidence

**Source:** `EXL3/engine-code/docs/EXL3_RELEASE_V2_REPORT.md`  
**SHA-256:** `d3c6702b5c21b25bf90ff6c5bfb0127def4f03aaa4c259f4f51af4187490c148`

Lines 61–95:

```text
00061 | 
00062 | ## Operating and numerical gates
00063 | 
00064 | Host unit discovery: **17 passed, 3 skipped** because they require a pinned
00065 | B70 runtime image. [The test log](../benchmarks/results/exl3-review-release-v2/unit-tests.log)
00066 | records each case. The image-specific numerical and operational gates below
00067 | were executed separately on the actual B70.
00068 | 
00069 | - Fresh load: 409/409 modules, including 8/8 MTP; all 13 active source guards
00070 |   and their complete runtime-source capture pass.
00071 | - API: generated/streaming text, automatic tools, reasoning, image/video
00072 |   count limits and capped images pass. Permanent-service checks include real
00073 |   default-reasoning requests, exact image/policy identity and a service restart.
00074 | - Exact context boundary: 261,120 input + 1,024 output, no preemption.
00075 | - Moderate C4, long-image context, prefix extension, scheduler-confirmed
00076 |   abort/recovery, independent maximum-area images and worker restart pass.
00077 | - C16 pressure: all 16 independent 8K + 256-output requests complete;
00078 |   4 preemptions are permitted and recorded. This does not promise
00079 |   sixteen simultaneous maximum contexts or preemption-free C16.
00080 | - Fresh short-reference smoke: 16,368 scored
00081 |   positions, PPL 3.646722, mean BF16-relative KL
00082 |   0.032481; all 32 aggregate arrays bitidentical to
00083 |   v1. Original reference data is reused. This short panel alone does not
00084 |   exercise long oneDNN prefill or prove MTP graph correctness.
00085 | - Fresh existing 32K-prefix suffix smoke: 128 scored
00086 |   positions / 8 full distributions, PPL
00087 |   1.194194 versus BF16 1.184646,
00088 |   mean KL 0.000957. It is a compact engineered window, not a new
00089 |   broad model-quality campaign.
00090 | - Separate fresh candidate matched-state probes exercise actual long prefill,
00091 |   mixed serving and true C4 verification graphs. Every local comparison passes
00092 |   the original rtol 0.01 / atol 0.003. Histories/states are restored within
00093 |   each finite probe; no historical bit-exact-text or universal determinism claim.
00094 | 
00095 | ## Actual serving-worker cache observation
```

Lines 99–122:

```text
00099 | inference thread/current queue after its native attention enqueue. This is
00100 | not a separate-process cache query. Each observed cache stays at/below 64,
00101 | with hits, misses and eviction; pressure waits, compilation microseconds,
00102 | RSS and XPU allocated/reserved memory are preserved in the assessment.
00103 | These instrumented times are excluded from the throughput comparison.
00104 | 
00105 | | Worker context | Peak entries | Hits | Misses | Evictions | Pressure waits | Cumulative compile | RSS range |
00106 | |---|---:|---:|---:|---:|---:|---:|---:|
00107 | | 1 | 64 | 7937 | 87 | 23 | 0 | 56.288 ms | 4.311–4.326 GiB |
00108 | 
00109 | Zero recorded pressure waits does not exercise the GPU all-entries-busy branch;
00110 | the separate host policy test covers that branch. The
00111 | [147 raw worker observations](../benchmarks/results/exl3-review-release-v2/cache-observations.jsonl) preserve
00112 | allocated/reserved XPU memory and the individual cache samples.
00113 | 
00114 | The cap applies to each application thread/queue context. It does not globally
00115 | bound arbitrary thread/queue counts, oneDNN internal caches or every allocator,
00116 | and finite measurements do not guarantee permanent freedom from OOM.
00117 | Client p95 SSE burst gaps are delivery observations, not GPU token-step times.
00118 | 
00119 | ## Historical coding results and rollback
00120 | 
00121 | The v1 Flappy/QueueKit runs remain [separately dated historical evidence](EXL3_CODING_BENCHMARKS.md).
00122 | QueueKit's 7/8 checks and 1/2 tasks, including the same native-control failure,
```

### E04 — Completion-aware bounded cache

**Source:** `EXL3/exl3-qualified-source/csrc/bounded_partition_cache.h`  
**SHA-256:** `1b5f09229a508f71470f5f4bc4eb74cdc3c5a359129ebeec00dd5f8d648291e7`

Lines 25–53:

```text
00025 |     template <class Factory, class Ready, class Wait>
00026 |     Value& get(const Key& key, Factory factory, Ready ready, Wait wait) {
00027 |         auto found=entries_.find(key);
00028 |         if (found!=entries_.end()) {
00029 |             ++stats.hits; found->second.used=++tick_; return found->second.value;
00030 |         }
00031 |         ++stats.misses;
00032 |         if (entries_.size()==capacity_) {
00033 |             auto victim=entries_.end();
00034 |             auto oldest=entries_.end();
00035 |             for (auto it=entries_.begin();it!=entries_.end();++it) {
00036 |                 if (oldest==entries_.end() || it->second.used<oldest->second.used) oldest=it;
00037 |                 if (ready(it->second.value) && (victim==entries_.end() || it->second.used<victim->second.used)) victim=it;
00038 |             }
00039 |             if (victim==entries_.end()) {
00040 |                 victim=oldest; ++stats.pressure_waits;
00041 |                 // Only a full-cache MISS waits, and only for this victim's
00042 |                 // completion chain. No hit/normal call synchronizes the queue.
00043 |                 wait(victim->second.value);
00044 |             }
00045 |             entries_.erase(victim); ++stats.evictions;
00046 |         }
00047 |         auto start=std::chrono::steady_clock::now();
00048 |         Value value=factory();
00049 |         stats.compile_microseconds+=std::chrono::duration_cast<std::chrono::microseconds>(
00050 |             std::chrono::steady_clock::now()-start).count();
00051 |         auto inserted=entries_.emplace(key,Node{std::move(value),++tick_}).first;
00052 |         if (entries_.size()>stats.peak_entries) stats.peak_entries=entries_.size();
00053 |         return inserted->second.value;
```

### E04 — Native SDPA last-use chain

**Source:** `EXL3/exl3-qualified-source/csrc/exl3_ops.sycl`  
**SHA-256:** `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`

Lines 563–575:

```text
00563 |         vi = 6;
00564 |     }
00565 |     ins.push_back(tensor(e.ins[vi], c.eng, v.data_ptr()));
00566 |     std::vector<tensor> outs{tensor(e.outs[0], c.eng, out.data_ptr())};
00567 |     if (with_lse) outs.push_back(tensor(e.outs[1], c.eng, lse.data_ptr()));
00568 |     // Chain uses of a partition even on an out-of-order queue: completion of
00569 |     // last_use then covers every prior use, making eviction lifetime-safe.
00570 |     std::vector<sycl::event> deps;
00571 |     if (e.last_use) deps.push_back(*e.last_use);
00572 |     e.last_use=dnnl::graph::sycl_interop::execute(e.cp, c.strm, ins, outs, deps);
00573 | }
00574 | #endif
00575 | 
```

### E05 — C4 direct output copy and retained alternatives

**Source:** `EXL3/exl3-qualified-source/exl3xpu/shared_kv_verify.py`  
**SHA-256:** `efb06f59250cbf4efb16f3eb6eb119bbd4c24c9ff086459b4afb85cc21adacb2`

Lines 95–121:

```text
00095 | def run(d, splits=None, tile=8):
00096 |     batch, rows = d['cu_seqlens_q'].numel() - 1, d['max_seqlen_q']
00097 |     q = packed_queries(d['q'], batch, rows)
00098 |     y = torch.empty_like(q)
00099 |     # Initial historical policy, to be measured independently for C1/C4.
00100 |     splits = splits if splits is not None else {2: 32, 3: 8, 4: 16, 5: 16}[rows]
00101 |     temp = torch.empty((batch, rows * 24 * splits, 256), dtype=q.dtype, device=q.device)
00102 |     sums = torch.empty((batch, rows * 24, splits), dtype=torch.float32, device=q.device)
00103 |     maxima = torch.empty_like(sums)
00104 |     cu = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
00105 |     ks = d['k_descale'].as_strided((1,), (1,))
00106 |     vs = d['v_descale'].as_strided((1,), (1,))
00107 |     torch.ops.b70_exl3_attention.shared_kv_verify_out(
00108 |         q, d['k'], d['v'], d['block_table'], cu, d['seqused_k'], ks, vs,
00109 |         y, temp, sums, maxima, d['max_seqlen_k'], splits, tile)
00110 |     out = d.get('out')
00111 |     if out is not None:
00112 |         # Matched q4 repeats support this copy at C4; C1 was slightly
00113 |         # slower with the direct strided copy. Keep its qualified layout path.
00114 |         if batch == 4:
00115 |             out.view(batch, rows, 4, 6, 256).copy_(
00116 |                 y.view(batch, 4, rows, 6, 256).permute(0, 2, 1, 3, 4))
00117 |         else:
00118 |             out.copy_(unpacked_output(y, batch, rows))
00119 |         return out
00120 |     return unpacked_output(y, batch, rows)
00121 | 
```

### E06 — Existing native EXL3 target loaders

**Source:** `CPP/vllm-cpp/src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`  
**SHA-256:** `6865e8b852ef17fbd2ff36b90e833701420ec9f120937e9a4119f7bdd3338bd0`

Lines 663–682:

```text
00663 |   // the way it is actually stored.
00664 |   //
00665 |   // THIS REPLACES A REFUSAL. Until this rung existed the three projections were
00666 |   // refused by name, which was the honest behaviour then: the alternative was
00667 |   // falling through to `get(la + "in_proj_qkv.weight")` and dying on "tensor
00668 |   // not found", a sentence about a checkpoint that is complete.
00669 |   //
00670 |   // THREE projections and no merged qkvz owner, because that is what the
00671 |   // artifact ships. See `GdnLayerWeights::in_proj_qkv_exl3` for why the trellis
00672 |   // pair cannot be N-concatenated the way the bf16 pair is.
00673 |   if (dense_loaders::IsExl3Projection(has, la + "in_proj_qkv")) {
00674 |     g.in_proj_qkv_exl3 = dense_loaders::LoadExl3(get, has, la + "in_proj_qkv");
00675 |     g.in_proj_z_exl3 = dense_loaders::LoadExl3(get, has, la + "in_proj_z");
00676 |     g.out_proj_exl3 = dense_loaders::LoadExl3(get, has, la + "out_proj");
00677 |     // GEOMETRY, CHECKED rather than trusted. A trellis is self-consistent at
00678 |     // more than one reading of its own shape, so a transposed or mismatched
00679 |     // projection is invisible until the numbers are wrong. `conv1d.weight`'s
00680 |     // leading dim is the checkpoint's own statement of `conv_dim`, and
00681 |     // `out_proj`'s K is its statement of `value_dim`; both are read from the
00682 |     // artifact rather than from the config, so a config that disagrees with the
```

Lines 756–774:

```text
00756 |   // checkpoint that quantizes only some layers loads each one the way it is
00757 |   // actually stored (`Linear.is_exl3_storage`, `modules/linear.py:385-389`).
00758 |   // An EXL3 projection has NO `.weight` at all, so every probe below --
00759 |   // `IsNvfp4Projection`, the `get(name + ".weight").dtype` read, the block
00760 |   // cross-check -- would either mis-route it or die asking for a tensor the
00761 |   // checkpoint correctly does not ship. Mirrors `LoadLlamaLayer`
00762 |   // (`llama_weights.cpp:73-85`), which is the working exemplar.
00763 |   if (dense_loaders::IsExl3Projection(has, sa + "q_proj")) {
00764 |     a.q_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "q_proj");
00765 |     a.k_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "k_proj");
00766 |     a.v_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "v_proj");
00767 |     a.o_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "o_proj");
00768 |     // The per-head q/k RMSNorm weights are NOT quantized; they ship beside the
00769 |     // trellis at the checkpoint's unquantized dtype.
00770 |     a.q_norm = LoadModelVectorForScheme(get, /*exl3=*/true, sa + "q_norm.weight");
00771 |     a.k_norm = LoadModelVectorForScheme(get, /*exl3=*/true, sa + "k_norm.weight");
00772 |     return a;
00773 |   }
00774 |   // Three forms, not two. `modelopt_mixed` checkpoints quantize this tower to
```

Lines 821–835:

```text
00821 |   // MODEL-QWEN35-EXL3 (#2495 item 3). FIRST and exclusive, for the same reason
00822 |   // the attention arm is: an EXL3 projection ships no `.weight`, so every probe
00823 |   // below reads a tensor that is not there. gate and up stay SEPARATE and
00824 |   // `layers::Exl3MlpGateUpMethod` consumes the pair on the shared
00825 |   // `MlpGateUpMethodBase` seam.
00826 |   if (dense_loaders::IsExl3Projection(has, mlp + "gate_proj")) {
00827 |     m.gate_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "gate_proj");
00828 |     m.up_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "up_proj");
00829 |     m.down_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "down_proj");
00830 |     return m;
00831 |   }
00832 |   if (IsNvfp4Projection(has, mlp + "gate_proj")) {
00833 |     m.gate_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "gate_proj");
00834 |     m.up_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "up_proj");
00835 |     m.down_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "down_proj");
```

Lines 1568–1587:

```text
01568 |           : LoadBf16Direct(get, backbone + "embed_tokens.weight");
01569 |   w.final_norm =
01570 |       LoadModelVectorForScheme(get, exl3_checkpoint, backbone + "norm.weight");
01571 |   // The 27B owns an explicit head; smaller Qwen3.5 checkpoints tie logits to
01572 |   // the embedding table and omit lm_head.weight.
01573 |   if (dense_loaders::IsExl3Projection(has, "lm_head")) {
01574 |     // MODEL-QWEN35-EXL3 (#2495 item 5), mirroring `llama_weights.cpp:152-161`.
01575 |     // A REAL quantized head is preferred over a tied embedding table EVEN when
01576 |     // the config declares `tie_word_embeddings: true`: the publisher quantized
01577 |     // a separate head at its own width, so the head is what it intends to be
01578 |     // used. That width comes from the tensor and never from a config scalar --
01579 |     // `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw ships a SIX-bit head
01580 |     // under a config that says 3.0, and no shape check can catch a reader that
01581 |     // trusts the scalar.
01582 |     w.lm_head_exl3 = dense_loaders::LoadExl3(get, has, "lm_head");
01583 |   } else if (DenseCheckpointHasLmHead(has, "lm_head")) {
01584 |     LoadDenseLmHead(get, has, "lm_head", w.lm_head, w.lm_head_fp4);
01585 |   } else {
01586 |     w.tied_lm_head = true;
01587 |     w.embed_tokens.nk = true;
```

### E06 — Native EXL3 MTP loader already exists

**Source:** `CPP/vllm-cpp/src/vllm/model_executor/models/qwen3_5_mtp.cpp`  
**SHA-256:** `304b2dc2f3f85b58f7f7e4ca2054100b8b7f814da9c4dd3bc6bbe9dfaa1d9250`

Lines 99–116:

```text
00099 |   // carry. The four are loaded TOGETHER, which is what makes
00100 |   // `FullAttnLayerWeights::IsExl3`'s single `q_proj_exl3` key honest.
00101 |   if (dense_loaders::IsExl3Projection(has, attn + "q_proj")) {
00102 |     out.q_proj_exl3 = dense_loaders::LoadExl3(get, has, attn + "q_proj");
00103 |     out.k_proj_exl3 = dense_loaders::LoadExl3(get, has, attn + "k_proj");
00104 |     out.v_proj_exl3 = dense_loaders::LoadExl3(get, has, attn + "v_proj");
00105 |     out.o_proj_exl3 = dense_loaders::LoadExl3(get, has, attn + "o_proj");
00106 |     out.q_norm = LoadBf16Direct(get, attn + "q_norm.weight");
00107 |     out.k_norm = LoadBf16Direct(get, attn + "k_norm.weight");
00108 |     return out;
00109 |   }
00110 |   out.q_proj = LoadBf16RawNK(get, attn + "q_proj.weight");
00111 |   out.k_proj = LoadBf16RawNK(get, attn + "k_proj.weight");
00112 |   out.v_proj = LoadBf16RawNK(get, attn + "v_proj.weight");
00113 |   out.o_proj = LoadBf16RawNK(get, attn + "o_proj.weight");
00114 |   out.q_norm = LoadBf16Direct(get, attn + "q_norm.weight");
00115 |   out.k_norm = LoadBf16Direct(get, attn + "k_norm.weight");
00116 |   return out;
```

Lines 365–382:

```text
00365 |   // MODEL-QWEN35-EXL3-HEAD (#2495 item 5). The fc-cat projection is quantized in
00366 |   // `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (`mtp.fc.trellis I16 [640, 320, 64]`,
00367 |   // K=2H=10240, N=H=5120, bits 4, `.mul1`), so the read asks the presence
00368 |   // question first, exactly as the dense tower's resolver does, and never reads
00369 |   // `quantization_config.mtp_bits`: the width is the trellis geometry's.
00370 |   if (dense_loaders::IsExl3Projection(has, "mtp.fc")) {
00371 |     out.fc_exl3 = dense_loaders::LoadExl3(get, has, "mtp.fc");
00372 |   } else {
00373 |     out.fc = LoadBf16RawNK(get, "mtp.fc.weight");
00374 |   }
00375 |   out.pre_fc_norm_embedding =
00376 |       LoadBf16Direct(get, "mtp.pre_fc_norm_embedding.weight");
00377 |   out.pre_fc_norm_hidden =
00378 |       LoadBf16Direct(get, "mtp.pre_fc_norm_hidden.weight");
00379 |   out.final_norm = LoadBf16Direct(get, "mtp.norm.weight");
00380 | 
00381 |   for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
00382 |     const std::string base =
```

### E06 — Native XPU EXL3 registration already exists

**Source:** `CPP/vllm-cpp/src/vt/xpu/xpu_ops.cpp`  
**SHA-256:** `46fb8594e67e940b905deb0d918a1a2a9f1a1d8621cc9597fdd1b1e240c5850e`

Lines 31–37:

```text
00031 |     XPU_OP(kApplyLogitBias, ApplyLogitBiasFn, ApplyLogitBiasKernel);
00032 |     XPU_OP(kApplyTokenMask, ApplyTokenMaskFn, ApplyTokenMaskKernel);
00033 |     XPU_OP(kApplyAllowedTokenIds, ApplyAllowedTokenIdsFn, ApplyAllowedTokenIdsKernel);
00034 |     XPU_OP(kExl3HadR128, Exl3HadR128Fn, Exl3HadR128Kernel);
00035 |     XPU_OP(kExl3Gemm, Exl3GemmFn, Exl3GemmKernel);
00036 |     XPU_OP(kCausalConv1dFwd, CausalConv1dFwdFn, CausalConv1dFwdKernel);
00037 |     XPU_OP(kCausalConv1dUpdate, CausalConv1dUpdateFn, CausalConv1dUpdateKernel);
```

### E07 — FP16 is not yet a format-independent native contract

**Source:** `CPP/vllm-cpp/src/vllm/model_executor/models/qwen3_5.cpp`  
**SHA-256:** `25db0faaa5bcfc1b16df4c13c3ee2e42ae4a32ef32a94e2d6ed849805538768c`

Lines 235–239:

```text
00235 | vt::DType detail::ActDType(vt::DeviceType dev_type) {
00236 |   static const bool f32 = detail::ActF32FlagIsOn(std::getenv("VT_ACT_F32"));
00237 |   if (!f32) return vt::DType::kBF16;
00238 |   // `VT_ACT_F32=1` is REFUSED, and refusing is the point. The f32 conversion is
00239 |   // INCOMPLETE, so honouring the flag does not produce an f32 engine: it
```

Lines 2742–2751:

```text
02742 |   // emit. A token gate cannot see a dtype that is too wide.
02743 |   if (w.IsExl3()) {
02744 |     out.q_owner.emplace(dense_exl3::Linear(d, h, w.q_proj, w.q_proj_exl3, DType::kBF16));
02745 |     out.k_owner.emplace(dense_exl3::Linear(d, h, w.k_proj, w.k_proj_exl3, DType::kBF16));
02746 |     out.v_owner.emplace(dense_exl3::Linear(d, h, w.v_proj, w.v_proj_exl3, DType::kBF16));
02747 |     out.qgate = out.q_owner->t();
02748 |     out.key = out.k_owner->t();
02749 |     out.value = out.v_owner->t();
02750 |     return out;
02751 |   }
```

Lines 8037–8050:

```text
08037 |   // SHARED `layers::MlpGateUpMethodBase` seam AGENTS.md names, exactly as
08038 |   // `qwen3.cpp:136-145` does for the Llama/Qwen3 dense MLP -- the model calls
08039 |   // one method and never asks which scheme it bound.
08040 |   if (w.IsExl3()) {
08041 |     DBuf act = dense_exl3::GateUp(d, dh, w.gate_up_proj, w.gate_proj_exl3,
08042 |                                   w.up_proj_exl3, I);
08043 |     (void)T;
08044 |     return dense_exl3::Linear(d, act.t(), w.down_proj, w.down_proj_exl3,
08045 |                               DType::kBF16);
08046 |   }
08047 |   // fp4-resident W4A4 path (real 27B, notes §5 step-6a) when populated; else the
08048 |   // bf16 path (synthetic CPU tests). Exactly one representation is filled.
08049 |   const bool fp4 = !w.gate_proj_fp4.Empty();
08050 | #ifdef VT_CUTLASS_NVFP4
```

Lines 10105–10114:

```text
10105 | static Dev DenseDev(Queue& queue, const Qwen3_5DenseWeights& weights) {
10106 |   Dev d{vt::GetBackend(queue.device.type), queue};
10107 |   if (weights.gptq4_checkpoint) {
10108 |     VT_CHECK(queue.device.type == vt::DeviceType::kXPU &&
10109 |                  weights.precision.activation == DType::kF16,
10110 |              "gptq4: scoped FP16 execution requires XPU");
10111 |     d.activation_dtype = weights.precision.activation;
10112 |   }
10113 |   return d;
10114 | }
```

### E08 — Old EXL3 auto-dispatch excludes FP16 output

**Source:** `CPP/vllm-cpp/src/vt/xpu/xpu_exl3_strategy.h`  
**SHA-256:** `59830e7b6457ea55c20b56501f84a69516d7937194cba8546d663e367e61afea`

Lines 17–29:

```text
00017 | 
00018 | inline Strategy MeasuredStrategy(const StrategyDomain& domain, const Shape& shape) {
00019 |   if (domain.device_id != 57891 || domain.driver != "1.17.39758+10" || domain.runtime != "1.17" ||
00020 |       domain.compiler != "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
00021 |       domain.kernel != kKernelVersion) return Strategy::kPacked;
00022 |   const auto [bits, k, n, m, dtype] = shape;
00023 |   if (dtype != DType::kF32 || m < 1) return Strategy::kPacked;
00024 |   constexpr uint32_t large = (1u << 16) | (1u << 20);
00025 |   // Two independent five-sample medians per strategy after >=200 ms warmup.
00026 |   // Admit fusion only for >=10% mean improvement, with neither fused median
00027 |   // >5% slower than either packed median. Unmeasured regimes keep packed.
00028 |   struct Row { int bits; int64_t k, n; uint32_t rows; };
00029 |   constexpr Row winners[] = {
```

Lines 43–51:

```text
00043 |     // All eleven real families: CPU panel oracles at M=128/512/2048/6656,
00044 |     // synthetic panel tails, and same-checkpoint answer/probability gates.
00045 |     // The hardware matrix reduction is bounded-error, unlike packed/fused.
00046 |     if (m >= 512 && m <= 6656 && n != 248320) return Strategy::kPrefillAllRows;
00047 |     if (m >= 128 && m <= 6656) return Strategy::kPrefill;
00048 |     if (m <= 20 && (row.rows & (1u << m))) return Strategy::kFused;
00049 |   }
00050 |   return Strategy::kPacked;
00051 | }
```

### E08 — Dispatch call remaps BF16 but not FP16

**Source:** `CPP/vllm-cpp/src/vt/xpu/xpu_exl3.cpp`  
**SHA-256:** `edde80c73596b552e5f2c7e6ae1dc22ff7d0ff6696f6401adda12c694ef359a1`

Lines 86–100:

```text
00086 |             v[j] = ((lane & step) ? FlipSign(v[j]) : v[j]) + shared[base + (lane ^ step) * 4 + j];
00087 |           item.barrier(sycl::access::fence_space::local_space);
00088 |           if (lane < 32) for (int j = 0; j < 4; ++j) shared[base + lane * 4 + j] = v[j];
00089 |           item.barrier(sycl::access::fence_space::local_space);
00090 |         }
00091 |         if (lane < 32 && row + r < m) for (int j = 0; j < 4; ++j) {
00092 |           const int64_t column = col_base + lane * 4 + j;
00093 |           const float value = Round(out.dtype == DType::kBF16 ? DType::kF32 : out.dtype,
00094 |                                     v[j] * kInvSqrt128) * static_cast<float>(scales[column]);
00095 |           Store(out, (row + r) * n + column, value);
00096 |         }
00097 |       }
00098 |     });
00099 |   });
00100 |   RecordProfileEvent(q, "exl3_fused_gemm_hadamard", event);
```

### E09 — Old native prefill is FP16 panel math, not donor W8A8

**Source:** `CPP/vllm-cpp/src/vt/xpu/xpu_exl3_prefill.cpp`  
**SHA-256:** `b6318bc298b39f3c6f33c898587212a4ef19e820679f09fea74d3a5dceb0f192`

Lines 1–12:

```text
00001 | #include "xpu_common.h"
00002 | #include "xpu_exl3.h"
00003 | #include "xpu_kernels.h"
00004 | #include <sycl/ext/oneapi/matrix/matrix.hpp>
00005 | #include "xpu_exl3_register_gemm.h"
00006 | #include <cstdlib>
00007 | #include <string_view>
00008 | 
00009 | namespace vt::xpu {
00010 | namespace {
00011 | constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
00012 | constexpr int kBM = 32, kBN = 64, kBK = 32;
```

Lines 21–31:

```text
00021 |   return matrix && (mode == "register" || (mode == "auto" && all_rows && rows >= 512));
00022 | }
00023 | 
00024 | template<int Bits>
00025 | void DecodePanel(Queue& q, sycl::half* panel, const uint32_t* packed,
00026 |                  int64_t k, int64_t n, int64_t base_k, int64_t column, int64_t width) {
00027 |   const auto event = NativeQueue(q).parallel_for(sycl::range<1>(k * width), [=](sycl::id<1> i) {
00028 |     const int64_t r = base_k + i[0] / width, c = column + i[0] % width;
00029 |     const auto* tile = packed + ((r / 16) * (n / 16) + c / 16) * (8 * Bits);
00030 |     panel[i[0]] = sycl::half(exl3::Decode(exl3::PackedCodeword<Bits>(tile, r % 16, c % 16), 2));
00031 |   });
```

Lines 111–131:

```text
00111 |                        const Tensor& trellis, const Tensor& svh, int bits, bool matrix, bool all_rows) {
00112 |   const int64_t m = in_had.shape[0], k = in_had.shape[1], n = out.shape[1];
00113 |   const auto device = NativeQueue(q).get_device();
00114 |   if (!device.has(sycl::aspect::ext_intel_device_id) ||
00115 |       device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
00116 |       !device.has(sycl::aspect::ext_intel_matrix) || m < 1 || m > 6656) return false;
00117 |   // PERF-02 selects all-row text prefill at M>=512 on this B70. The register
00118 |   // kernel passed the fixed model gates and wins for every measured text family.
00119 |   const bool register_panel = UseRegisterPanel(matrix, all_rows, m);
00120 |   // The all-row schedule keeps every M row in the result panel, so each
00121 |   // compressed weight panel is decoded once across all rows.
00122 |   constexpr int64_t k_per_panel = 1024;
00123 |   const int64_t columns = all_rows ? 1024 : 4096;
00124 |   const int64_t rows_per_panel = all_rows ? m : 256;
00125 |   const int64_t padded_rows = ((rows_per_panel + kBM - 1) / kBM) * kBM;
00126 |   if (all_rows && !matrix) return false;
00127 |   if (columns * (2 * k_per_panel + 4 * padded_rows) > static_cast<int64_t>(kWorkspaceBytes))
00128 |     return false;
00129 |   return WithExl3Workspace(q, kWorkspaceBytes, [&](void* workspace) {
00130 |     auto* panel = static_cast<sycl::half*>(workspace);
00131 |     auto* result = reinterpret_cast<float*>(panel + k_per_panel * columns);
```

### E10 — Producer grouped packed storage

**Source:** `EXL3/exl3-qualified-source/exl3xpu/vllm_plugin.py`  
**SHA-256:** `4c89fe610ce9d5aeabec0740eed1ab9a63136b7d885d54e273ed60ba1f52c0d6`

Lines 145–173:

```text
00145 |                        output_size, params_dtype, **extra_weight_attrs):
00146 |         k = input_size_per_partition
00147 |         n = sum(output_partition_sizes)
00148 |         if get_tensor_model_parallel_world_size() != 1:
00149 |             raise NotImplementedError("exl3: tensor parallel sharding is not qualified (TP must be 1)")
00150 |         if k != input_size or n != (output_size if not isinstance(layer, ParallelLMHead) else n):
00151 |             raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
00152 |         if k % 128 != 0 or not output_partition_sizes or any(s<=0 or s%128 for s in output_partition_sizes):
00153 |             raise ValueError(f"exl3: dims must be positive multiples of 128 (k={k}, n={output_partition_sizes})")
00154 |         K = self.K
00155 |         layer.exl3_sizes = list(output_partition_sizes)
00156 |         layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
00157 |         layer.exl3_groups = set()
00158 |         layer.exl3_loaded_shards = {name:set() for name in ('trellis','suh','svh')}
00159 |         layer.exl3_codebook_loaded = False
00160 |         layer.exl3_K = K
00161 |         layer.exl3_cb = self.cb
00162 |         layer_ref = weakref.ref(layer)
00163 | 
00164 |         def reg(name, tensor, loader):
00165 |             p = Parameter(tensor, requires_grad=False)
00166 |             p.weight_loader = loader
00167 |             layer.register_parameter(name, p)
00168 | 
00169 |         reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), self._load_trellis(layer))
00170 |         reg("suh", torch.empty((len(output_partition_sizes), k), dtype=torch.float16), self._load_suh(layer))
00171 |         reg("svh", torch.empty((n,), dtype=torch.float16), self._load_svh(layer))
00172 |         if self.config.codebook != "3inst":
00173 |             def load_codebook(p,w,*a,**kw):
```

Lines 265–286:

```text
00265 |         for gi in range(len(groups)):
00266 |             shard_of_nb[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
00267 |         layer.suh = Parameter(suh, requires_grad=False)
00268 |         layer.exl3_shard_of_nb = shard_of_nb.to(dev)
00269 |         layer.exl3_bounds = bounds
00270 |         key=_norm_key(self.prefix)
00271 |         base,_,leaf=key.rpartition('.')
00272 |         members=[f'{base}.{p}' if base else p for p in FUSED.get(leaf,[leaf])]
00273 |         # GDN's combined q/k/v source shares one Hadamard transform; z has its own.
00274 |         expected_groups=[(0,1,2),(3,)] if leaf=='in_proj_qkvz' else [(i,) for i in range(len(members))]
00275 |         if self.prefix and groups!=expected_groups:
00276 |             raise ValueError(f"exl3: unexpected fused source groups for {self.prefix}: {groups} != {expected_groups}")
00277 |         for member in members:
00278 |             if self.config.storage and self.config.storage.get(member)!=self.K:
00279 |                 raise ValueError(f"exl3: loaded unexpected module/bit width {member}/{self.K}")
00280 |         layer.exl3_loader_report={'prefix':self.prefix,'members':members,'bits':self.K,
00281 |                                   'shapes':{name:list(getattr(layer,name).shape) for name in ('trellis','suh','svh')},
00282 |                                   'groups':[list(g) for g in groups],'bounds':bounds,
00283 |                                   'shard_of_nb_counts':[(bounds[i+1]-bounds[i])//128 for i in range(len(groups))],
00284 |                                   'loaded_shards':{name:sorted(v) for name,v in layer.exl3_loaded_shards.items()},
00285 |                                   'codebook':self.config.codebook,'status':'PASS'}
00286 |         if isinstance(layer, ParallelLMHead) and os.environ.get("EXL3_DRAFT_VOCAB") and _spec_is_mtp():
```

Lines 299–309:

```text
00299 |     def apply(self, layer, x, bias=None):
00300 |         from . import ops
00301 |         if layer.exl3_cpp:
00302 |             y = torch.ops.exl3xpu_C.linear(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
00303 |                                            layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
00304 |                                            ops.SMALL_M_MAX, ops.RECON_SLICE_N)
00305 |         else:
00306 |             y = ops._exl3_linear_py(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
00307 |                                     layer.exl3_bounds, layer.exl3_K, layer.exl3_cb)
00308 |         if bias is not None:
00309 |             y = y + bias
```

### E11 — Actual SmallM and W8A8 routing

The W8A8 allocation is per complete source group. The separate FP16 fallback uses `slice_n`; do not confuse the two memory contracts.

**Source:** `EXL3/exl3-qualified-source/csrc/exl3_ops.sycl`  
**SHA-256:** `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`

Lines 701–737:

```text
00701 | at::Tensor exl3_linear(at::Tensor x, at::Tensor trellis, at::Tensor suh, at::Tensor svh, at::Tensor shard_of_nb,
00702 |                        std::vector<int64_t> group_bounds, int64_t K, int64_t cb, int64_t small_m_max,
00703 |                        int64_t slice_n) {
00704 |     auto shape = x.sizes().vec();
00705 |     int64_t k = shape.back();
00706 |     int64_t n = svh.size(0);
00707 |     auto x2 = x.reshape({-1, k});
00708 |     if (!x2.is_contiguous()) x2 = x2.contiguous();
00709 |     int64_t M = x2.size(0);
00710 |     shape.back() = n;
00711 |     auto out = at::empty({M, n}, x.options());
00712 |     if (M == 0) return out.view(shape);
00713 |     if (M <= small_m_max && exl3_supported(K, cb)) {
00714 |         exl3_gemm_small(x2, trellis, suh, svh, shard_of_nb, out, K, cb);
00715 |         return out.view(shape);
00716 |     }
00717 |     int64_t S = suh.size(0);
00718 |     if (g_int8 && cb == 2 && (K == 4 || K == 6) && M > 16) {
00719 |         auto& q = queue_of(x2);
00720 | #ifdef EXL3_DNNL
00721 |         // GEMM rows padded to 256: bounds the number of oneDNN primitives (one per (Ms, K, N)) on real traffic
00722 |         const int64_t Ms = (M + 255) / 256 * 256;
00723 | #else
00724 |         const int64_t Ms = M;
00725 | #endif
00726 |         auto xq = at::empty({S, Ms, k}, x.options().dtype(at::kChar));
00727 |         auto sx = at::empty({S, Ms}, x.options().dtype(at::kFloat));
00728 |         int n_in = S * M, local = 8;
00729 |         auto run_in = [&](auto tag) {
00730 |             using TIn = decltype(tag);
00731 |             int kb_n = (int)(k / 128);
00732 |             auto wg = [&](auto kern, int tpr) {
00733 |                 q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)n_in * tpr, tpr), kern); });
00734 |             };
00735 |             if (kb_n <= 8 * 6) {
00736 |                 wg(HadInQ8WgKernel<TIn, 8, 6>{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
00737 |                                               reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
```

Lines 756–778:

```text
00756 | #ifdef EXL3_DNNL
00757 |         {
00758 |             static std::vector<at::Tensor> swd(16);
00759 |             auto& swt = swd.at(x.device().index());
00760 |             if (!swt.defined()) swt = at::full({1}, sw, x.options().dtype(at::kFloat));
00761 |             auto y = at::empty({Ms, n}, x.options().dtype(at::kHalf));
00762 |             for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
00763 |                 int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1], ng = g1 - g0;
00764 |                 auto w = at::empty({k, ng}, x.options().dtype(at::kChar));
00765 |                 if (K == 4) launch_reconstruct_q8<4>(trellis, w, g0, q); else launch_reconstruct_q8<6>(trellis, w, g0, q);
00766 |                 q8dnnl::gemm(q, x, reinterpret_cast<const int8_t*>(xq.data_ptr()) + g * Ms * k,
00767 |                              reinterpret_cast<const int8_t*>(w.data_ptr()), swt.data_ptr<float>(),
00768 |                              sx.data_ptr<float>() + g * Ms, reinterpret_cast<fp16*>(y.data_ptr()) + g0, Ms, k, ng, n);
00769 |             }
00770 |             exl3_had_out_h(y.narrow(0, 0, M), svh, out);
00771 |             return out.view(shape);
00772 |         }
00773 | #endif
00774 |         for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
00775 |             int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1], ng = g1 - g0;
00776 |             auto w = at::empty({k, ng}, x.options().dtype(at::kChar));
00777 |             if (K == 4) launch_reconstruct_q8<4>(trellis, w, g0, q); else launch_reconstruct_q8<6>(trellis, w, g0, q);
00778 |             auto y = at::_int_mm(xq[g], w);                                   // [M, ng] int32
```

Lines 791–805:

```text
00791 |     }
00792 |     auto xh = at::empty({S, M, k}, x.options().dtype(at::kHalf));
00793 |     exl3_had_in_rm(x2, suh, xh);
00794 |     auto y = at::empty({M, n}, x.options().dtype(at::kHalf));
00795 |     for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
00796 |         int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1];
00797 |         for (int64_t n0 = g0; n0 < g1; n0 += slice_n) {
00798 |             int64_t n1 = std::min(n0 + slice_n, g1);
00799 |             auto w = weight_scratch(x.device(), k * (n1 - n0)).narrow(0, 0, k * (n1 - n0)).view({k, n1 - n0});
00800 |             exl3_reconstruct(trellis, w, n0, K, cb);
00801 |             auto yslice = y.narrow(1, n0, n1 - n0);
00802 |             at::matmul_out(yslice, xh[g], w);
00803 |         }
00804 |     }
00805 |     exl3_had_out_h(y, svh, out);
```

### E12 — Exact block-aligned draft subset and full-vocabulary mapping

**Source:** `EXL3/exl3-qualified-source/exl3xpu/vllm_plugin.py`  
**SHA-256:** `4c89fe610ce9d5aeabec0740eed1ab9a63136b7d885d54e273ed60ba1f52c0d6`

Lines 334–376:

```text
00334 | def _build_draft_head(layer, path):
00335 |     with open(path) as f:
00336 |         spec = json.load(f)
00337 |     n_total_blocks = layer.svh.shape[0] // 128
00338 |     ids=spec['blocks']
00339 |     if not ids or any(type(i) is not int or i<0 or i>=n_total_blocks for i in ids) or len(set(ids))!=len(ids):
00340 |         raise ValueError(f'exl3: draft vocabulary contains empty/invalid/duplicate blocks: {path}')
00341 |     blocks = torch.tensor(ids, dtype=torch.long)
00342 |     dev = layer.trellis.device
00343 |     tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
00344 |     trellis = layer.trellis.data.index_select(1, tiles).contiguous()
00345 |     svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
00346 |     idx = (blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
00347 |     layer.exl3_draft = dict(trellis=trellis, svh=svh, idx=idx,
00348 |                             shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
00349 |                             bounds=[0, len(blocks) * 128])
00350 |     logger.info("exl3: MTP draft head uses %d of %d vocab blocks (%.1f%% of lm_head)",
00351 |                 len(blocks), n_total_blocks, 100.0 * len(blocks) / n_total_blocks)
00352 | 
00353 | 
00354 | def _patch_mtp_draft_logits():
00355 |     try:
00356 |         from vllm.model_executor.models import qwen3_5_mtp
00357 |     except Exception:
00358 |         return
00359 |     cls = qwen3_5_mtp.Qwen3_5MTP
00360 |     if getattr(cls, "_exl3_patched", False):
00361 |         return
00362 |     orig = cls.compute_logits
00363 | 
00364 |     def compute_logits(self, hidden_states, spec_step_idx: int = 0):
00365 |         lm = self.lm_head
00366 |         d = getattr(lm, "exl3_draft", None)
00367 |         if d is None:
00368 |             return orig(self, hidden_states, spec_step_idx)
00369 |         from . import ops
00370 |         sub = torch.ops.exl3xpu_C.linear(hidden_states, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
00371 |                                          lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
00372 |         logits = hidden_states.new_full((hidden_states.shape[0], lm.svh.shape[0]), float("-inf"))
00373 |         logits.index_copy_(1, d["idx"], sub)
00374 |         return logits[:, : self.config.vocab_size]
00375 | 
00376 |     cls.compute_logits = compute_logits
```

### E13 — Native draft: full head before row selection and host logits

**Source:** `CPP/vllm-cpp/src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`  
**SHA-256:** `37d5290657b4edadb9b146913d164004831bfcd0b6d09d149f3f8ac2900808ff`

Lines 27–54:

```text
00027 |                                        const std::vector<int64_t>& rows,
00028 |                                        vt::Queue& queue) {
00029 |   const int64_t vocab = logits.vocab;
00030 |   const int64_t num_rows = logits.rows;
00031 |   std::vector<float> host(static_cast<size_t>(num_rows) *
00032 |                           static_cast<size_t>(vocab));
00033 |   vt::Backend& backend = vt::GetBackend(queue.device.type);
00034 |   backend.Copy(queue, host.data(), logits.device_tensor.data,
00035 |                host.size() * sizeof(float));
00036 |   backend.Synchronize(queue);
00037 | 
00038 |   std::vector<int32_t> drafted(rows.size(), 0);
00039 |   for (size_t r = 0; r < rows.size(); ++r) {
00040 |     const int64_t row = rows[r];
00041 |     VT_CHECK(row >= 0 && row < num_rows,
00042 |              "MtpPropose: draft logits row out of range");
00043 |     const float* logit_row =
00044 |         host.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab);
00045 |     int32_t best_idx = 0;
00046 |     float best_val = logit_row[0];
00047 |     for (int64_t v = 1; v < vocab; ++v) {
00048 |       if (logit_row[static_cast<size_t>(v)] > best_val) {
00049 |         best_val = logit_row[static_cast<size_t>(v)];
00050 |         best_idx = static_cast<int32_t>(v);
00051 |       }
00052 |     }
00053 |     drafted[r] = best_idx;
00054 |   }
```

Lines 106–122:

```text
00106 |   // ── The one paged draft forward (I5c) + shared lm_head. ──────────────────────
00107 |   vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
00108 |       spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue);
00109 |   vllm::ForwardLogits logits = draft.ComputeLogits(hidden.tensor, queue);
00110 |   VT_CHECK(logits.on_device() && logits.rows == T,
00111 |            "MtpProposePrefill: unexpected draft logits shape");
00112 | 
00113 |   PrefillOutcome out;
00114 |   // ── Greedy draft pick over each request's last (sampled) row
00115 |   // (spec_decode/speculator.py:276-280). ──────────────────────────────────────
00116 |   out.sampled_rows.assign(
00117 |       spi.last_token_indices.begin(),
00118 |       spi.last_token_indices.begin() + static_cast<size_t>(num_reqs));
00119 |   out.draft_tokens = GreedySampleDraft(logits, out.sampled_rows, queue);
00120 | 
00121 |   // :346 — the positions of those same rows. The decode half advances from them.
00122 |   // The k=1 caller drops them, and dropping a host vector costs nothing.
```

### E14 — Producer selects hidden rows before the head

**Source:** `EXL3/qualified-runtime/installed-vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`  
**SHA-256:** `5b97f54e20ddf65b98e939832d050745f67ef43cfa145786da300f1a8322e3c6`

Lines 450–458:

```text
00450 |                 ret_hidden_states = self.model(**model_inputs)
00451 |         # Some MTP models declare a single-tensor contract but return
00452 |         # (logits_hidden, feedback_hidden) for final-norm correctness.
00453 |         if isinstance(ret_hidden_states, tuple):
00454 |             last_hidden_states, hidden_states = ret_hidden_states
00455 |         else:
00456 |             last_hidden_states = ret_hidden_states
00457 |             hidden_states = ret_hidden_states
00458 |         return last_hidden_states, hidden_states
```

Lines 491–507:

```text
00491 |         sample_hidden_states = last_hidden_states[last_token_indices]
00492 |         self.draft_tokens[:num_reqs, 0] = self.sample_draft(
00493 |             sample_hidden_states,
00494 |             sample_src_positions,
00495 |             idx_mapping,
00496 |             self.temperature,
00497 |             self.seeds,
00498 |             self.current_draft_step,
00499 |             self.draft_logits,
00500 |         )
00501 |         if last_hidden_states is hidden_states:
00502 |             self.hidden_states[:num_reqs] = sample_hidden_states
00503 |         else:
00504 |             self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
00505 |         self.input_buffers.positions[:num_reqs] = positions
00506 |         self.sample_src_positions[:num_reqs] = sample_src_positions
00507 | 
```

### E14 — Producer draft-sampling branch must be resolved explicitly

**Source:** `EXL3/qualified-runtime/installed-vllm/v1/worker/gpu/spec_decode/speculator.py`  
**SHA-256:** `3a1ab15bf8b20918852eb835fe9755046e99dcbac6ba6ca3d414949fc8bc695a`

Lines 400–437:

```text
00400 |     def sample_draft(
00401 |         self,
00402 |         hidden_states: torch.Tensor,
00403 |         sample_src_positions: torch.Tensor,
00404 |         idx_mapping: torch.Tensor,
00405 |         temperature: torch.Tensor,
00406 |         seeds: torch.Tensor,
00407 |         draft_step: torch.Tensor,
00408 |         draft_logits: torch.Tensor | None,
00409 |     ) -> torch.Tensor:
00410 |         if draft_logits is not None:
00411 |             logits = self.model.compute_logits(hidden_states)
00412 |             sampled = gumbel_sample(
00413 |                 logits,
00414 |                 idx_mapping,
00415 |                 temperature,
00416 |                 seeds,
00417 |                 sample_src_positions,
00418 |                 apply_temperature=True,
00419 |                 is_drafting=True,
00420 |                 logits_cache=draft_logits,
00421 |                 logits_cache_col=draft_step,
00422 |                 use_fp64=self.use_fp64_gumbel,
00423 |             )
00424 |             if self.draft_watermarker is not None:
00425 |                 sampled = self.draft_watermarker.sample(
00426 |                     logits,
00427 |                     sampled,
00428 |                     idx_mapping,
00429 |                     temperature,
00430 |                 )
00431 |         elif self.use_local_argmax_reduction:
00432 |             return self.model.get_top_tokens(hidden_states)
00433 |         else:
00434 |             logits = self.model.compute_logits(hidden_states)
00435 |             sampled = logits.argmax(dim=-1)
00436 |         self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
00437 |         return sampled
```

Lines 485–495:

```text
00485 |         # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
00486 |         # NOTE(woosuk): For draft sampling, we only consider the temperature
00487 |         # and ignore the other sampling parameters such as top_k and top_p,
00488 |         # for simplicity and performance.
00489 |         # While this may slightly degrade the acceptance rate, it does not
00490 |         # affect the output distribution after rejection sampling.
00491 |         self.temperature.copy_(temperature)
00492 |         self.seeds.copy_(seeds)
00493 |         self.idx_mapping[:num_reqs].copy_(idx_mapping)
00494 |         # idx_mapping for CG padded requests points to -1, which is ignored
00495 |         # during sampling to prevent writing stale values to draft logits.
```

### E15 — Native hidden-tap graph bypass

**Source:** `CPP/vllm-cpp/src/vllm/model_executor/models/qwen3_5_dense.cpp`  
**SHA-256:** `3ddd47bce6ff38d85db17f7acf70d3806ff1792ad833a7d09354f6cb9b867530`

Lines 202–224:

```text
00202 |       input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));
00203 | 
00204 |   // SPEC-MTP I5d-pre hidden-state tap. When the spec verify forward requests the
00205 |   // drafter's [T,H] post-final-norm hidden (I5d), route to the EXISTING
00206 |   // ForwardDeviceTap: byte-identical logits to ForwardDevice, plus the hidden
00207 |   // moved into *input.hidden_tap. Null (every spec-off run) falls through to the
00208 |   // unchanged path below, so the forward is byte-identical when spec is off.
00209 |   if (input.hidden_tap != nullptr) {
00210 |     if (weights.gptq4_checkpoint && Gptq4RouteTraceEnabled())
00211 |       std::fprintf(stderr,
00212 |                    "{\"event\":\"gptq4_route\",\"selected\":\"eager\","
00213 |                    "\"reason\":\"hidden_tap\",\"tokens\":%zu,"
00214 |                    "\"requests\":%d}\n",
00215 |                    input.token_ids.size(), input.num_reqs);
00216 |     return Qwen3_5DenseModel::ForwardDeviceTap(
00217 |         input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
00218 |         input.attn_kv, input.gdn_state, weights, input.config, input.queue,
00219 |         input.hidden_tap, input.logits_indices);
00220 |   }
00221 | 
00222 |   // SPEC-DFLASH D1 (DF-AUX-TAPS): non-null routes to ForwardDeviceMultiTap
00223 |   // (byte-identical logits + the [T,H×taps] aux capture); null is byte-identical to
00224 |   // the path below. Mutually exclusive with hidden_tap.
```

### E15 — Native recurrent-prefix/speculation restriction

**Source:** `CPP/vllm-cpp/src/vllm/v1/worker/gpu/runner.cpp`  
**SHA-256:** `869ea15a94ccc495ea204bc6b248cfa29b4e3c56056355bf7e68fc1d47312d4d`

Lines 752–760:

```text
00752 |   // a test path that skipped the ctor arg) by falling back to num_blocks.
00753 |   //
00754 |   // SPEC-MTP I5d: under speculation each sequence needs num_spec+1 CONSECUTIVE
00755 |   // GDN state slots (the k+1 draft-timestep snapshots the recurrent rollback
00756 |   // selects among, spec §3). Size the compact pool max_num_reqs*(num_spec+1) and
00757 |   // let remap_gdn_state_slots hand each sequence a base of num_spec+1 slots.
00758 |   const int spec_cols = spec_on() ? num_spec() + 1 : 1;
00759 |   const int64_t base_slots = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
00760 |   prefix_snapshot_base_ = base_slots * spec_cols;
```

### E16 — Historical GPTQ results and unresolved same-prefix gate

**Source:** `CPP/review/checkpoint_review_20260928.md`  
**SHA-256:** `598e277ffab22ad3df8d1490dac07d1cc9e1701ef52a54197be690c039eba6e7`

Lines 7–15:

```text
00007 | | Matched request scope | Native C++ | Python reference | C++ / Python | Provenance |
00008 | | --- | ---: | ---: | ---: | --- |
00009 | | P4096/O1024, sampled eager MTP4, FP8 KV, 1664 page, C1, decode emitted tokens/s | 49.1052 | 71.8318 | 68.4% | C++ five-run median after GDN workgroup change; Python earlier matched-workload run, not simultaneous |
00010 | | Same C++ build, optional Xe2 shared-KV verifier, P4096/O1024 | 62.7176 | 71.8318 | 87.3% | One unprofiled candidate run; Python earlier run, different sampled continuation |
00011 | | P4096 prefill tokens/s, FP8 KV, 1664 page, non-MTP comparison | 1004.96 | 1586.66 | 63.3% | Earlier page-geometry comparison, different from current MTP4 serving scope |
00012 | 
00013 | The MTP decode rate is `1023 / (O1024 wall - separate O1 TTFT)`, an end-to-end client estimate. The accepted GDN launch change raised C++ MTP4 decode from 41.8496 to 49.1052 tokens/s (+17.3%) in alternating route runs with the same token hash and 770/1012 accepted drafts. No new end-to-end prefill result has been measured after that change. Do not mix these emitted-token rates with non-speculative forwards/s or per-kernel timings.
00014 | 
00015 | In a fresh same-build C++ A/B, the existing default made 49.3827 decode tokens/s (O1 4.24323 s, O1024 24.959 s, 1012 proposed/770 accepted, output hash `6972477010856893408`); the optional Xe2 verifier made 62.7176 (O1 4.25066 s, O1024 20.5619 s, 1024 proposed/767 accepted, output hash `7809634488016794068`). Both passed 50/50 harness assertions with the same 4096-token prompt hash `4760835697920107937`. This is a one-run **+27.0% request-level candidate**, not a qualified speedup: the sampled continuations differ, and same-prefix Python target-logit quality plus Q2–Q5 route tests are still needed. The default remains Split-K.
```

Lines 21–28:

```text
00021 | | W0 references/contracts | Native capability matrix, call-path and benchmark evidence exist. Exact source-to-installed-wheel correspondence remains unproved. |
00022 | | W1 graph baseline | C8 case-5 first difference is a near-tie at row 2; graph stays opt-in and capped. Request/state/slot ownership still needs diagnosis. |
00023 | | W2 decode attribution | Exact top-20 native sampler qualified with a modest measured win. Non-speculative same-scope Python parity remains open. |
00024 | | W3 eager MTP1 | Native draft, conv/GDN, separate cache, speculative state and model smokes exist. EOS/cancellation and complete lifecycle qualification remain open. |
00025 | | W4 sampled MTP | Native T1/p.95/k20 MTP4 works at C1. Same-prefix Python/C++ raw target TV is 0.02445/0.02329 on verification rows 0/4, exceeding the frozen 0.02 gate. Prefix-dependent processors and broader distribution/RNG gates remain open. |
00026 | | W5 MTP4 speed | Packed INT4 draft weights, workgroup-64 GDN gain and a Torch-free optional Xe2 Q2–Q5 shared-KV attention port exist. New operator replay is near bitwise, but model-level qualification and production promotion remain open. |
00027 | | W6 C2/C4 | Functional multi-request coverage exists, but no qualified native sampled MTP4 C2/C4 performance parity. |
00028 | | W7 prefix/long context | Recurrent prefix snapshots with speculation and production long-context gates remain open. |
```

Lines 33–43:

```text
00033 | - Profiling isolated speculative GDN as a large per-cycle cost. The scoped workgroup-64 launch cut that GPU sum from 222.91 to 102.52 ms in the short profile and produced the measured +17.3% full-request decode gain.
00034 | - Two simple shared-KV C++ probes regressed the complete request, so they were not promoted. The Python M04 real-Q5 replay with identical Q/FP8 K/V measured about 0.14–0.16 ms host / 0.165 ms device span. Existing C++ Split-K measured 1.999 ms host, with 1.905 ms partial plus 0.067 ms reduction. C++ Split-K had 0.03114% relative RMS output error against Python.
00035 | - A narrow Torch-free port of the pinned Xe2 M04 shared-KV policy now runs through the VT C++ API behind `VT_XPU_XE2_VERIFY=1`. The same-input real Q5 replay passed 34/34 assertions, differed in 5 of 30,720 FP16 outputs, had relative RMS `5.21419e-06` and max absolute error `0.000244141`. Its 20-call synchronized host median was 0.213176 ms. The P4096/O32 smoke kept the default's 32 emitted token IDs and 23/29 draft acceptance. The full one-run A/B is recorded above. The default path remains Split-K until the broader quality and Q2–Q5 gates pass.
00036 | - The real first-layer 4K pre-quantization K/V source already differed between Python and C++ by 0.0965%/0.1091% relative RMS. With identical Python FP16 source, the native FP8 writer matched Python cache bytes exactly. The attention crossed replay attributed more of the Q5 output gap to cache content than to the attention arithmetic. A writer profile bounded prefill cache-writing opportunity at <0.84% of observed TTFT. No cache writer change was promoted.
00037 | - A native-GDN prefill opt-in reduced one O1 TTFT (4.0730 to 3.5625 s), but the O1024 sampled request slowed (49.2570 to 43.7578 tokens/s) with a different trajectory. It stays opt-in.
00038 | 
00039 | ## Highest-value next decisions for the revised plan
00040 | 
00041 | 1. Resolve the source difference before the first full-attention K/V cache write: compare the same real GDN-to-QKV activation and GPTQ projection/QK norm/RoPE outputs in the two engines. This also addresses the sampled row-0/4 TV failures.
00042 | 2. Qualify the new native M04-like path at Q2–Q5, then run paired P4096/O1024 C1 route A/B and report selected-route witnesses, token hashes, accepted/proposed counts, TTFT, decode rate and device/host cost. Keep it opt-in if it regresses or changes target quality.
00043 | 3. Complete W1 graph ownership diagnosis, W4 prefix-dependent processors/RNG coverage, then C2/C4 and recurrent prefix/long-context work. Retain separate plans for vision and production multimodal quality.
```

### E17 — Independent real-linear oracle and frozen operator tolerance

**Source:** `EXL3/exl3-qualified-source/tests/test_linear_rows_xpu.py`  
**SHA-256:** `2e4b37daef939595cc665ef11b0e4d5cd3224036dc26b5c2034232a72b04e839`

Lines 1–27:

```text
00001 | """Real checkpoint fused/head row, padding, large-first and graph regression gate.
00002 | 
00003 | Small-M compares with independent FP32 Hadamard/matmul. Large INT8 compares
00004 | with independently quantized torch._int_mm and the already gated transforms.
00005 | The existing 2e-3 relative-norm tolerance is fixed before this matrix.
00006 | """
00007 | import argparse
00008 | import gc
00009 | import hashlib
00010 | import json
00011 | import os
00012 | from pathlib import Path
00013 | 
00014 | import torch
00015 | from safetensors import safe_open
00016 | from exl3xpu import ops, ref
00017 | 
00018 | ROWS = [1,2,3,4,5,8,12,16,20,24,32,40,48,64,128,129,256,512]
00019 | TOLERANCE = 2e-3
00020 | 
00021 | 
00022 | def check(actual,expected):
00023 |     a,b=actual.float(),expected.float()
00024 |     relative=float((a-b).norm()/b.norm().clamp_min(1e-20))
00025 |     finite=bool(a.isfinite().all())
00026 |     assert finite and relative<TOLERANCE,(relative,finite)
00027 |     return dict(relative_norm=relative,maximum_absolute=float((a-b).abs().max()),finite=finite)
```

Lines 74–96:

```text
00074 |             fp32_gold=torch.empty((512,n),dtype=torch.float32,device='xpu')
00075 |             int8_pre=torch.empty((512,n),dtype=torch.float16,device='xpu')
00076 |             xh=torch.empty((len(keys),512,kdim),dtype=torch.float16,device='xpu')
00077 |             E.exl3_had_in_rm(x,suh,xh)
00078 |             for i,(left,right) in enumerate(zip(bounds,bounds[1:])):
00079 |                 inner=ref.had_rows(x.float()*suh[i].float())
00080 |                 q8=3.453125/127
00081 |                 sx=xh[i].float().abs().amax(1,keepdim=True)/127
00082 |                 xq=torch.round(xh[i].float()/sx).to(torch.int8)
00083 |                 # Bound reference storage even for the full248320-row head.
00084 |                 # Slice boundaries are128-aligned; output Hadamard blocks do
00085 |                 # not cross them. This preserves every real head column.
00086 |                 for start in range(left,right,16384):
00087 |                     stop=min(start+16384,right)
00088 |                     w=torch.empty((kdim,stop-start),dtype=torch.float16,device='xpu')
00089 |                     E.exl3_reconstruct(tr,w,start,bits,2)
00090 |                     fp32_gold[:,start:stop]=ref.had_rows(inner@w.float())*svh[start:stop].float()
00091 |                     wq=torch.round(w.float()/q8).to(torch.int8)
00092 |                     int8_pre[:,start:stop]=(torch._int_mm(xq,wq).float()*sx*q8).half()
00093 |                     del w,wq
00094 |                 del inner,xq,sx
00095 |             int8_gold=torch.empty_like(int8_pre);E.exl3_had_out_h(int8_pre,svh,int8_gold)
00096 |             row=dict(name=name,keys=keys,bits=bits,k=kdim,n=n,head_slice_start=offset,cases=[],padding=[],compiled=[])
```

### E18 — Native generic EXL3 packed scratch lifetime

**Source:** `CPP/vllm-cpp/src/vt/xpu/xpu_exl3.cpp`  
**SHA-256:** `edde80c73596b552e5f2c7e6ae1dc22ff7d0ff6696f6401adda12c694ef359a1`

Lines 312–346:

```text
00312 |   TraceExl3Dispatch(q, in, out, bits, cb, args.debug_name,
00313 |                     strategy, specialized ? "packed" : "reference", reason);
00314 |   const size_t raw_bytes = m * n * sizeof(float);
00315 |   auto launch = [&](void* storage) {
00316 |     auto raw = Tensor::Contiguous(storage, DType::kF32, q.device, {m, n});
00317 |     auto* result = static_cast<float*>(raw.data);
00318 |     if (specialized) {
00319 |       const auto* words = reinterpret_cast<const uint32_t*>(packed);
00320 |       switch (bits) {
00321 |         case 3: PackedGemm<3>(q, result, ah, words, m, k, n); break;
00322 |         case 4: PackedGemm<4>(q, result, ah, words, m, k, n); break;
00323 |         case 5: PackedGemm<5>(q, result, ah, words, m, k, n); break;
00324 |         case 6: PackedGemm<6>(q, result, ah, words, m, k, n); break;
00325 |       }
00326 |     } else {
00327 |       const auto event = NativeQueue(q).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> item) {
00328 |       const int64_t row = item[0] / n, col = item[0] % n;
00329 |       float acc = 0;
00330 |       for (int64_t inner = 0; inner < k; ++inner) {
00331 |         const float value = static_cast<float>(ah[row * k + inner]);
00332 |         if (value == 0.0f) continue;  // same signed-zero behavior as CPU
00333 |         const auto* tile = packed + ((inner / 16) * (n / 16) + col / 16) * (32 * bits);
00334 |         const int t = exl3::Fragment(inner % 16, col % 16);
00335 |         acc += value * exl3::Decode(exl3::Codeword(tile, bits, t), cb);
00336 |       }
00337 |       result[item[0]] = acc;
00338 |       });
00339 |       RecordProfileEvent(q, "exl3_reference_gemm", event);
00340 |     }
00341 |     Had(q, out, raw, nullptr, &svh, kInvSqrt128, "exl3_output_hadamard");
00342 |   };
00343 |   // Reuse the same bounded buffer as prefill. Decode graphs retain its address
00344 |   // and the backend orders all eager/graph users across queues. Larger eager
00345 |   // shapes or tight budgets keep the original temporary-buffer fallback.
00346 |   constexpr size_t workspace_bytes = 32 * 1024 * 1024;
```

### E19 — Producer primitive/scratch scopes need native ownership

**Source:** `EXL3/exl3-qualified-source/csrc/exl3_ops.sycl`  
**SHA-256:** `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`

Lines 630–670:

```text
00630 |     int MB = M <= 4 ? 4 : (M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64)));
00631 | #elif defined(EXL3_DPAS_MB24) && defined(EXL3_DPAS_MB48)
00632 |     int MB = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : (M <= 32 ? 32 : (M <= 40 ? 40 : (M <= 48 ? 48 : 64)))));
00633 | #elif defined(EXL3_DPAS_MB24)
00634 |     int MB = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : (M <= 32 ? 32 : 64)));
00635 | #else
00636 |     int MB = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64));
00637 | #endif
00638 |     int tk = Kdim / 16;
00639 |     int P = 4, rps = (tk + P - 1) / P; P = (tk + rps - 1) / rps;
00640 |     auto part = at::zeros({P, M, N}, xh_blocked.options().dtype(at::kFloat));
00641 |     bool ok = false;
00642 |     if (K == 4 && cb == 2) ok = use_vec ? dispatch_mr<4, 2>(MR, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q)
00643 |                                         : dispatch_dpas<4, 2>(MB, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q);
00644 |     if (K == 6 && cb == 2) ok = use_vec ? dispatch_mr<6, 2>(MR, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q)
00645 |                                         : dispatch_dpas<6, 2>(MB, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q);
00646 |     TORCH_CHECK(ok);
00647 |     (void)S;
00648 |     return part.sum(0);
00649 | }
00650 | 
00651 | // Prefill helpers: row-major input Hadamard (GEMM operand) and fp16 output Hadamard.
00652 | void exl3_had_in_rm(at::Tensor x, at::Tensor suh, at::Tensor xh) {
00653 |     TORCH_CHECK(xh.scalar_type() == at::kHalf && xh.is_contiguous() && xh.size(1) == x.size(0));
00654 |     auto& q = queue_of(x);
00655 |     if (x.scalar_type() == at::kHalf) launch_had_in<fp16>(x, suh, xh, q, true);
00656 |     else if (x.scalar_type() == at::kBFloat16) launch_had_in<sycl::ext::oneapi::bfloat16>(x, suh, xh, q, true);
00657 |     else TORCH_CHECK(false, "x dtype");
00658 | }
00659 | 
00660 | void exl3_had_out_h(at::Tensor y, at::Tensor svh, at::Tensor out) {
00661 |     TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous() && y.sizes() == out.sizes());
00662 |     auto& q = queue_of(y);
00663 |     if (out.scalar_type() == at::kHalf) launch_had_out<fp16, fp16>(y, 1, svh, out, q);
00664 |     else if (out.scalar_type() == at::kBFloat16) launch_had_out<sycl::ext::oneapi::bfloat16, fp16>(y, 1, svh, out, q);
00665 |     else TORCH_CHECK(false, "out dtype");
00666 | }
00667 | 
00668 | // ---- flash-attention forward (prefill), fp16, head_dim 256, GQA; returns O and natural-log LSE
00669 | void exl3_fa_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse, bool causal, double scale) {
00670 |     TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf);
```

Lines 690–704:

```text
00690 | //   M <= small_m_max : fused decode-in-GEMM (vector / DPAS)
00691 | //   otherwise        : row-major input Hadamard -> reconstruct W_inner slices -> oneDNN GEMM -> output Hadamard
00692 | 
00693 | static at::Tensor& weight_scratch(const at::Device& dev, int64_t numel) {
00694 |     static std::vector<at::Tensor> bufs(16);
00695 |     auto& b = bufs.at(dev.index());
00696 |     if (!b.defined() || b.numel() < numel)
00697 |         b = at::empty({numel}, at::TensorOptions().dtype(at::kHalf).device(dev));
00698 |     return b;
00699 | }
00700 | 
00701 | at::Tensor exl3_linear(at::Tensor x, at::Tensor trellis, at::Tensor suh, at::Tensor svh, at::Tensor shard_of_nb,
00702 |                        std::vector<int64_t> group_bounds, int64_t K, int64_t cb, int64_t small_m_max,
00703 |                        int64_t slice_n) {
00704 |     auto shape = x.sizes().vec();
```

### E20 — Independent host output-copy test scope

**Source:** `EXL3/exl3-qualified-source/tests/test_m04_output_copy.py`  
**SHA-256:** `00dd695539660e817bcb71efedf2029f7505f4acf668184cf9c361cc35add6b1`

Lines 1–35:

```text
00001 | """Exercise the actual run() output path with a deterministic stand-in kernel."""
00002 | import unittest
00003 | from unittest.mock import patch
00004 | import torch
00005 | from exl3xpu import shared_kv_verify as m04
00006 | 
00007 | 
00008 | class CopyContract(unittest.TestCase):
00009 |     def test_native_output_and_no_output_preserve_all_half_bitpatterns(self):
00010 |         for batch in (1, 2, 4):
00011 |             for rows in (2, 3, 4, 5):
00012 |                 with self.subTest(batch=batch, rows=rows):
00013 |                     count=batch*rows*24*256
00014 |                     bits=((torch.arange(count,dtype=torch.int64)%65536)-32768).to(torch.int16)
00015 |                     y=bits.view(torch.float16).reshape(batch,rows*24,256)
00016 |                     expected=m04.unpacked_output(y,batch,rows)
00017 |                     def kernel(*args):
00018 |                         args[8].copy_(y)
00019 |                     for has_out in (True, False):
00020 |                         q=torch.zeros((batch*rows,24,256),dtype=torch.float16)
00021 |                         out=torch.empty_like(q) if has_out else None
00022 |                         d=dict(q=q,k=q,v=q,block_table=torch.zeros((batch,1),dtype=torch.int32),
00023 |                                cu_seqlens_q=torch.arange(batch+1,dtype=torch.int32),
00024 |                                seqused_k=torch.ones(batch,dtype=torch.int32),
00025 |                                k_descale=torch.ones(1),v_descale=torch.ones(1),
00026 |                                max_seqlen_q=rows,max_seqlen_k=4096,out=out)
00027 |                         with patch.object(torch.ops.b70_exl3_attention,'shared_kv_verify_out',kernel,create=True):
00028 |                             actual=m04.run(d)
00029 |                         self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
00030 |                         if has_out:
00031 |                             self.assertIs(actual,out)
00032 |                             self.assertEqual(actual.data_ptr(),out.data_ptr())
00033 | 
00034 | 
00035 | if __name__=='__main__':unittest.main()
```

### E21 — Independent audit results (not a new GPU run)

The following report was recomputed from archive payloads and original per-wave power JSONs. `same_worker` checks the original recorded worker ID against the corresponding independent worker receipt, not a later live worker. Reference-case phase numbers are archival observations.

```json
{
  "manifest": {
    "count": 1867,
    "mismatches": []
  },
  "guard_files": {
    "count": 13,
    "mismatches": []
  },
  "cpp_files": {
    "count": 147,
    "mismatches": []
  },
  "policy_sha256": "1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839",
  "power": [
    {
      "cap_w": 150,
      "waves": 70,
      "scenarios": 20,
      "requests": 124,
      "tokens": 126976,
      "seconds": 3002.905145410041,
      "wave_minutes": 50.04841909016735,
      "energy_j": 450591.828126,
      "energy_wh": 125.16439670166667,
      "weighted_mean_w": 150.0519684462003,
      "matrix_tps": 42.28438590345873,
      "tokens_per_joule": 0.2817982752330196,
      "tokens_per_wh": 1014.4737908388705,
      "preemptions": 0.0,
      "excess_prefill": 0.0,
      "all_finish_length": true,
      "all_outputs1024": true,
      "same_image": true,
      "same_worker": true,
      "raw_hash_matches": true,
      "started_at": "2026-10-02T18:14:34.578003+02:00",
      "sum_native_prefill_s": 1511.1459931649151,
      "sum_native_decode_request_s": 3384.331533261953,
      "all_published_prefill_decode_wall_medians_match": true,
      "reference_cases": {
        "phase-4k-c1": {
          "prefill_tps_median": 1857.9723659403157,
          "decode_tps_median": 45.82650503012299,
          "ttft_s_median": 2.2124200419930276,
          "fully_overlapped_decode_tps": 46.264324837486264
        },
        "phase-32k-c1": {
          "prefill_tps_median": 1637.052564465122,
          "decode_tps_median": 48.20014906489941,
          "ttft_s_median": 20.0436122980027,
          "fully_overlapped_decode_tps": 46.69232083421526
        },
        "phase-128k-c1": {
          "prefill_tps_median": 1069.4548297686094,
          "decode_tps_median": 33.25461429656206,
          "ttft_s_median": 122.7152665949834,
          "fully_overlapped_decode_tps": 34.4543468831132
        },
        "concurrency-4k-c4": {
          "prefill_tps_median": 838.0359873393934,
          "decode_tps_median": 33.79300132603084,
          "ttft_s_median": 5.963049661993864,
          "fully_overlapped_decode_tps": 146.88592816308298
        }
      }
    },
    {
      "cap_w": 180,
      "waves": 70,
      "scenarios": 20,
      "requests": 124,
      "tokens": 126976,
      "seconds": 2500.5744521731103,
      "wave_minutes": 41.67624086955184,
      "energy_j": 450361.54272800003,
      "energy_wh": 125.10042853555557,
      "weighted_mean_w": 180.10323281381037,
      "matrix_tps": 50.7787320188176,
      "tokens_per_joule": 0.28194236841552056,
      "tokens_per_wh": 1014.992526295874,
      "preemptions": 0.0,
      "excess_prefill": 0.0,
      "all_finish_length": true,
      "all_outputs1024": true,
      "same_image": true,
      "same_worker": true,
      "raw_hash_matches": true,
      "started_at": "2026-10-02T13:46:47.069638+02:00",
      "sum_native_prefill_s": 1259.0094530759961,
      "sum_native_decode_request_s": 2803.243303393974,
      "all_published_prefill_decode_wall_medians_match": true,
      "reference_cases": {
        "phase-4k-c1": {
          "prefill_tps_median": 2241.3797289916142,
          "decode_tps_median": 57.13718599670426,
          "ttft_s_median": 1.8316276450059377,
          "fully_overlapped_decode_tps": 57.63418680984851
        },
        "phase-32k-c1": {
          "prefill_tps_median": 1977.653997047513,
          "decode_tps_median": 54.50002666502254,
          "ttft_s_median": 16.595009754993953,
          "fully_overlapped_decode_tps": 54.232790370059476
        },
        "phase-128k-c1": {
          "prefill_tps_median": 1262.549916766922,
          "decode_tps_median": 39.92785132812526,
          "ttft_s_median": 103.9844079829927,
          "fully_overlapped_decode_tps": 41.417401868862555
        },
        "concurrency-4k-c4": {
          "prefill_tps_median": 1024.5571601897282,
          "decode_tps_median": 41.69827795718012,
          "ttft_s_median": 4.609792751492932,
          "fully_overlapped_decode_tps": 180.06958692151267
        }
      }
    },
    {
      "cap_w": 230,
      "waves": 70,
      "scenarios": 20,
      "requests": 124,
      "tokens": 126976,
      "seconds": 2146.0716213020496,
      "wave_minutes": 35.76786035503416,
      "energy_j": 493786.345944,
      "energy_wh": 137.16287387333333,
      "weighted_mean_w": 230.08847470077134,
      "matrix_tps": 59.1667112782387,
      "tokens_per_joule": 0.25714765311554455,
      "tokens_per_wh": 925.7315512159604,
      "preemptions": 0.0,
      "excess_prefill": 0.0,
      "all_finish_length": true,
      "all_outputs1024": true,
      "same_image": true,
      "same_worker": true,
      "raw_hash_matches": true,
      "started_at": "2026-10-02T16:24:51.152341+02:00",
      "sum_native_prefill_s": 1082.1420711828105,
      "sum_native_decode_request_s": 2407.779981472064,
      "all_published_prefill_decode_wall_medians_match": true,
      "reference_cases": {
        "phase-4k-c1": {
          "prefill_tps_median": 2615.2080456916988,
          "decode_tps_median": 66.79356275738908,
          "ttft_s_median": 1.5720174899906851,
          "fully_overlapped_decode_tps": 69.16269294521551
        },
        "phase-32k-c1": {
          "prefill_tps_median": 2309.9892159017995,
          "decode_tps_median": 65.33383010522155,
          "ttft_s_median": 14.219453439989593,
          "fully_overlapped_decode_tps": 64.94817114268875
        },
        "phase-128k-c1": {
          "prefill_tps_median": 1466.2604621030996,
          "decode_tps_median": 45.86066811606659,
          "ttft_s_median": 89.53207004998694,
          "fully_overlapped_decode_tps": 47.58688309806499
        },
        "concurrency-4k-c4": {
          "prefill_tps_median": 1205.8643126357122,
          "decode_tps_median": 47.41373616731683,
          "ttft_s_median": 4.151673418004066,
          "fully_overlapped_decode_tps": 204.73492579493575
        }
      }
    },
    {
      "cap_w": 275,
      "waves": 70,
      "scenarios": 20,
      "requests": 124,
      "tokens": 126976,
      "seconds": 2033.590073335101,
      "wave_minutes": 33.89316788891835,
      "energy_j": 559212.69531,
      "energy_wh": 155.33685980833332,
      "weighted_mean_w": 274.98791553053144,
      "matrix_tps": 62.43932917697545,
      "tokens_per_joule": 0.22706208400653485,
      "tokens_per_wh": 817.4235024235255,
      "preemptions": 0.0,
      "excess_prefill": 0.0,
      "all_finish_length": true,
      "all_outputs1024": true,
      "same_image": true,
      "same_worker": true,
      "raw_hash_matches": true,
      "started_at": "2026-10-02T17:14:07.780958+02:00",
      "sum_native_prefill_s": 1011.9246991029358,
      "sum_native_decode_request_s": 2294.627861665009,
      "all_published_prefill_decode_wall_medians_match": true,
      "reference_cases": {
        "phase-4k-c1": {
          "prefill_tps_median": 2791.515956899824,
          "decode_tps_median": 69.45982700575246,
          "ttft_s_median": 1.4745875200023875,
          "fully_overlapped_decode_tps": 71.56079662944592
        },
        "phase-32k-c1": {
          "prefill_tps_median": 2473.1345034573,
          "decode_tps_median": 65.13276993022532,
          "ttft_s_median": 13.277795399015304,
          "fully_overlapped_decode_tps": 64.79142632220038
        },
        "phase-128k-c1": {
          "prefill_tps_median": 1562.62802334814,
          "decode_tps_median": 47.99028580236665,
          "ttft_s_median": 84.03904395899735,
          "fully_overlapped_decode_tps": 49.70185153178362
        },
        "concurrency-4k-c4": {
          "prefill_tps_median": 1300.495567371334,
          "decode_tps_median": 50.89190952016081,
          "ttft_s_median": 3.868737113996758,
          "fully_overlapped_decode_tps": 218.9949433975482
        }
      }
    }
  ],
  "power_request_markers_counts_agree": true
}
```

### E22 — Host tests run during this review

These checks ran locally without a GPU. The C4 test substitutes a deterministic host kernel to check the actual packing/output-copy code; it does not execute native M04.

`AUDIT/host_inventory_tests.log`

```text
test_conflicting_codebook_rejected (test_checkpoint_inventory.InventoryTests.test_conflicting_codebook_rejected) ... ok
test_duplicate_canonical_rejected (test_checkpoint_inventory.InventoryTests.test_duplicate_canonical_rejected) ... ok
test_empty_inventory_rejected (test_checkpoint_inventory.InventoryTests.test_empty_inventory_rejected) ... ok
test_exl3_rejects_gptq_environment (test_checkpoint_inventory.InventoryTests.test_exl3_rejects_gptq_environment) ... ok
test_int8_gate_preserves_threshold_and_rejects_nonfinite (test_checkpoint_inventory.InventoryTests.test_int8_gate_preserves_threshold_and_rejects_nonfinite) ... ok
test_missing_component_rejected (test_checkpoint_inventory.InventoryTests.test_missing_component_rejected) ... ok
test_missing_qc_mtp_is_discovered (test_checkpoint_inventory.InventoryTests.test_missing_qc_mtp_is_discovered) ... ok
test_missing_trellis_rejected (test_checkpoint_inventory.InventoryTests.test_missing_trellis_rejected) ... ok
test_normalization (test_checkpoint_inventory.InventoryTests.test_normalization) ... ok
test_other_quant_format_rejected (test_checkpoint_inventory.InventoryTests.test_other_quant_format_rejected) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.001s

OK
```

`AUDIT/host_copy_tests.log`

```text
test_native_output_and_no_output_preserve_all_half_bitpatterns (test_m04_output_copy.CopyContract.test_native_output_and_no_output_preserve_all_half_bitpatterns) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.017s

OK
```

`AUDIT/host_cache_test.log`

```text
PASS bounded LRU, hot hits, async retirement, pressure-only wait, compile failure
```

### E23 — Missing assets and externally sourced context

The original checkpoints, full replay tensors, compiled libraries, complete upstream trees, the full local C++ repository and the old EXL3 branch are not all in the ZIPs. Implementation must reuse their recorded revisions/hashes and fail clearly on absence. No claim in this plan depends on unseen branch contents.

The uploaded source and live receipts are authoritative for this review. General/current oneDNN or EXL3 documentation is supplementary, not permission to change the pinned 3.13/2026.1.1 producer implementation. The native plan intentionally does not assume current documentation APIs exist unchanged in that pin.

---

**End of handoff. Begin with §1, then stop at each milestone gate.**
