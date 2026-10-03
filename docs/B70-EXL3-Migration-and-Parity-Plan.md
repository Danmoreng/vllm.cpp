# B70 native C++: EXL3 migration and parity plan

**Decision date:** 2026-10-02.  
**Merged native base:** `3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6` (`b70-gptq-int4`).  
**Historical review base:** `c1342199e09bd4c2c018c4694330152d283bf7ad`; retained only for the GPTQ diagnostic reference and original review excerpts.  
**New production oracle:** qualified EXL3 source `c59d9442aba8610188837e37724600f1517d7335`, vLLM 0.30.0, production EXL3 v2.  
**Status:** active local implementation plan, reconciled with the main merge on 2026-10-02. S0a/S0b artifacts, the local projection gate and actual pinned-worker KV/Conv/GDN/attention layout contract are verified in `docs/b70-exl3`; S0 deliverables are complete within that scope. Native loader/SmallM/model parity and S1–S6 remain incomplete.  
**Purpose:** one actionable handoff for the next Codex session, with the original evidence appendix archived alongside it. This is a migration plan, not a claim that native EXL3 already matches production.

## 0. Start here on the merged checkout

This is the active B70 plan. Follow the local [AGENTS.md](../AGENTS.md): one assistant, current branch, one small implementation step followed by its focused check. No new branch, worktree, PR, process campaign or automatic commit. Historical plans and their next-task instructions are references only. No power change or broad benchmark campaign is authorized.

**Developer update, 2026-10-02:** continue all S0–S6 steps incrementally with exclusive use of the B70. Stop the production `b70-qwen38-vllm.service` and leave it stopped during development. No other vLLM instance may run on the card alongside the instance being built/tested. Focused GPU replays, tests and isolated pinned-oracle captures are authorized without another service-window approval. Run the oracle and native GPU processes sequentially; the former production image remains an immutable reference, not a service that must stay available. Preserve existing code/data and the current branch. This supersedes the original production-preservation requirement throughout this plan.

### 0.1 What the merge changes

| Item | Observed at this reconciliation | Consequence |
|---|---|---|
| Native branch | `b70-gptq-int4`, HEAD `3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6` | Continue here; record actual HEAD again next session, without resetting it. |
| Main integration | Second parent/main `9d96b162c8c3dfa8f0143932962e445b68fe3a8b`; first parent `c1342199e09bd4c2c018c4694330152d283bf7ad` | 351 main commits plus the merge have been integrated locally; nothing was pushed. |
| Old EXL3 branch | `b70-sycl-qwen38-exl3` tip and merge-base are both `2e1f5fbda6579c23dc63ea3faf0c69067ebbe19a`; `HEAD...b70-sycl-qwen38-exl3` is `489 0` | All commits from that branch are already ancestors of HEAD. Do not merge/cherry-pick it again or restore old file versions over newer code. |
| New shared EXL3 detection | `974c3b7f8` adds `include/vllm/model_executor/layers/quantization/exl3_checkpoint.h`, updates the DeepSeek-V4 loader to accept `mul1`, and adds `test_exl3_checkpoint` | Reuse `IsExl3Checkpoint`/`Exl3QuantConfig` where appropriate. They detect metadata, not tensor validity or FP16 execution. The MiMo-V2 loader work does not qualify Qwen/B70. |
| Qwen EXL3 loader and XPU kernels | `qwen3_5_dense_weights.cpp`, `qwen3_5_mtp.cpp`, `dense_exl3_linear.cpp` and `src/vt/xpu/` have no diff between the historical review base and merged HEAD | The reviewed precision/dispatch limitations remain; upstream's EXL3 changes do not replace S0–S6. |
| Shared Qwen execution | The merge combines `DenseDev`/`ActDType` with upstream step tracing and adds tiled GDN out-projection handling | Recheck these actual call paths when building the execution map. A clean textual merge is not a model-parity receipt. |
| Merge validation | Only `qwen3_5.cpp` received a host `c++ -fsyntax-only` check with `VLLM_CPP_XPU`, C++20, `-Wall -Wextra -Werror`: exit 0 | No complete SYCL build, link, GPU test or whole-model parity has been run for this merged base. |

Static checks still locate the gaps in:

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp::ResolveQwen3_5DensePrecision` and `qwen3_5.cpp::DenseDev`: FP16 policy is selected only for GPTQ.
- `include/vllm/model_executor/layers/quantization/exl3.h`: EXL3 gate/up activations are explicitly BF16; Qwen EXL3 Q/K/V and other call sites also need an output-dtype audit.
- `src/vt/xpu/xpu_exl3_strategy.h::MeasuredStrategy` and `xpu_exl3.cpp::AutomaticStrategy`: the measured domain requires F32 output; callers remap BF16, not FP16. `xpu_exl3_prefill.cpp` is FP16 panel reconstruction, not the producer's W8A8 route.
- `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`: the draft prefill computes logits before sampled-row selection; `GreedySampleDraft` copies full FP32 logits to the host.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp::ForwardQwen3_5Dense`: `hidden_tap` returns through the eager path before graph selection.
- `src/vllm/v1/worker/gpu/runner.cpp`: recurrent prefix snapshots explicitly require non-speculative execution.

These are source observations, not newly reproduced numerical failures. Inspect current symbols rather than using the original appendix's line numbers as current offsets.

### 0.2 Local assets found, with verification boundaries

The following are discovery locations, not substitutes for the immutable identities in §3:

| Asset | Local location / observation | Still required in S0 |
|---|---|---|
| Pinned EXL3 donor | `/home/sebastian/LocalLLM/exl3xpu-review`, HEAD `c59d9442aba8610188837e37724600f1517d7335`; tracked files clean when inspected | Verify the relevant file hashes and guards before porting. Other `exl3xpu*` checkouts have different HEADs. |
| Serving repository | `/home/sebastian/LocalLLM/intel-b70-qwen38-vllm` contains commit `ed47c5b84614f5654440355b12fa61b6eea8ee5e`; current HEAD was `906be5bbcdd7756b514b32e3043fb1060bf867cd` | Read the pinned commit or captured runtime, not arbitrary current working-tree bytes. Do not switch that service repository. |
| Review archives | `/home/sebastian/LocalLLM/review-bundles/exl3-current-and-cpp-20261002/` contains the EXL3 production/power ZIP, the `c1342199e` C++ ZIP and supplemental handoff/identity files | Verify payload hashes as needed; archive presence is not a repeated audit. |
| Current model mount | `/home/sebastian/.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw` contains config, index and two safetensors shards; mounted at `/models/checkpoint` | Verify provenance/revision, headers, all 409 modules including 8 MTP modules, marker values and packed-byte identity. Directory naming is not revision proof. |
| Production container | Read-only Docker inspection returned image `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918` for `b70-qwen38-vllm` | Historical identity/mount evidence only. Keep the production service stopped; use this pinned image for isolated oracle work when needed, sequentially with native GPU work. |
| Historical GPTQ fixtures | `docs/bench-evidence/b70-gptq-fp8-parity-20260927/`, `tools/gptq_reference/`, local `B70_GPTQ_INT4_Plan/evidence/` | Preserve paths and recorded geometry. Determine which large captures are actually available before requesting a new capture. |
| Prior build metadata | `/home/sebastian/LocalLLM/b70-main-integration-build/cmake/` contains a CMake cache and compilation database with container paths `/work`, `/build/cmake`, `/opt/intel/...` | Resolve the original isolated build environment before reusing it. Host `icpx` was absent at the recorded path; existing objects are not merged-source validation. |

`tools/b70_inventory.py` still hard-codes the **old** Mia-AiLab model/revision and a 401 text/head comparison. It must not issue a new-model identity receipt unchanged. Reuse its safetensors parsing where suitable, but parameterize identity/expectations with focused host tests as a separate small change. Distinguish 401 target text/head modules from 409 total modules including eight MTP modules; count actual headers rather than relabeling a constant.

### 0.3 Archive and next-session entry point

- [Original Pro proposal and evidence E01–E23](archive/b70/2026-10-02-Pro-EXL3-Migration-Plan-original.md) is retained byte-for-byte. Its branch-creation instruction and historical next tasks are superseded here.
- [Old native EXL3 implementation plan](archive/b70/B70-SYCL-Qwen38-EXL3-Implementation-Plan.md) is archived. Its Mia-AiLab 3.5-bpw/BF16 milestones are historical, not this migration's acceptance contract.
- Local `B70_GPTQ_INT4_Plan/` and `B70_GPTQ_b6db_Review/` are marked historical in place so tools, evidence and relative references remain usable. Original ZIPs and tracked benchmark receipts remain unchanged. GPTQ remains a diagnostic reference only.

Original S0a entry instruction (S0a artifacts now exist; continue with S0b under the exclusive-GPU update above):

> Read AGENTS.md and docs/B70-EXL3-Migration-and-Parity-Plan.md. Continue on b70-gptq-int4. Start only S0a: verify the pinned local references, inventory requirements and current EXL3 execution boundaries, then write REFERENCE_MANIFEST.json and EXECUTION_MAP.md with explicit verified/pending states. If inventory tooling needs adaptation, make that one small change and run its focused host test before continuing. Do not start S1, GPU inference, service changes or benchmarks. Report the S0a result and the next S0b replay step.

## 1. Decision and immediate next task

**Move the native project to the current EXL3 checkpoint now. Do not require a complete GPTQ performance/release campaign first. Build on the current merged C++ branch; the old EXL3 branch is already contained in its history.**

The merged C++ checkout already contains EXL3 model loaders, target/MTP integration and registered XPU operators. The missing work is therefore **production-compatible execution, qualification and optimization**, not inventing EXL3 support from zero. Its existing EXL3 paths still differ from the new Python producer in precision, projection grouping and kernel dispatch. [E06–E11]

Keep GPTQ as a frozen **diagnostic/regression reference**, not a second production objective. A shared state, ownership, mask or sampler defect must be fixed before the affected infrastructure is reused. A discrepancy isolated to the GPTQ projection implementation must remain documented, but need not block a separately qualified EXL3 engine. Never mark the old GPTQ gate as passed merely because EXL3 passes.

### First concrete tasks for Codex — S0a, then S0b

Do one small step at a time. The documentation reconciliation is preparation, not completion of S0.

**S0a — reference and execution contract (next session's first scope):**

1. Read local instructions; record actual HEAD/status without resetting or stashing work. Stay on the current branch. Use the ancestry result in §0.1; repeat only if branch tips changed.
2. Produce `docs/b70-exl3/REFERENCE_MANIFEST.json` and `docs/b70-exl3/EXECUTION_MAP.md`. Record references A/B separately, local source paths and immutable pins, checkpoint inventory, FP16 boundaries, kernel dispatch/grouping, head policy and actual KV/state layouts. Identify each field as verified, receipt-derived or pending. Resolve draft sampling from the pinned config/runtime; mark it pending if only the conditional implementation is available.
3. Inspect the old inventory tool before use. If needed, adapt identity/expected counts in a separate focused host-tested change; do not silently certify the new model with its old constants. Inventory reads must not load a model onto the GPU or fully dequantize it.
4. Record existing fixture locations, hashes and missing assets. Keep old GPTQ first-GDN-to-QKV captures available for shared-code diagnosis; do not launch new GPTQ tuning or service captures.

**S0a completion:** a source-backed manifest/map, a verified header inventory or a precise account of the remaining inventory work, and explicit missing runtime/capture fields. Pending required fields block dependent replay, not independent host work. Report before moving to S0b.

**S0b — first-projection replay, after S0a prerequisites:** add a fixture/replay for one real 4-bpw MLP projection and 128-aligned first/last blocks of the 6-bpw head, initially at **M=1 and M=4**. Record exact dtypes and input/output/weight hashes. Build the changed replay target in the resolved SYCL environment. Compare packed decode, input Hadamard, GEMM and output Hadamard/scaling to local pinned captures, reporting the first differing stage. Keep an explicitly failing FP16 route diagnostic rather than substituting BF16 or another checkpoint.

The B70 is reserved for this migration by the developer. Keep the production worker stopped, use existing captures first and run required GPU/model replays under §6's sequential isolation rule. No further service-window approval is required for these focused steps.

**S0 completion:** executable replay with a correctly attributed pass/fail and the required execution contract. Missing captures/weights or an unresolved build are reported as blocked checks, never fabricated parity. Use the frozen GPTQ stage replay only when necessary to distinguish shared normalization/state defects from GPTQ projection math.

## 2. Supplied Pro review: historical evidence and its limits

### 2.1 Work reported by the supplied review

This section preserves findings reported in the supplied Pro proposal, not tests rerun during this documentation update. Local reconciliation performed Git/source inspection and read-only asset discovery only. The original reviewer reports unpacking two archives and checking:

- **1,867 EXL3 payload entries:** manifest SHA-256 and byte size, with zero mismatches.
- **13 declared active runtime-source dependencies:** included bytes match the expected guards. The union of the two active guard declarations was also inspected, rather than assuming a partial directory was complete.
- **147 native C++ source files:** match the supplied verification of the tracked `c1342199e` checkout, with zero mismatches. The archive is still a curated subset, not the complete repository.
- The production-policy hash and the original four-cap raw measurement files, worker identities, counts and published prefill/decode/wall medians.
- Locally rerun **10 host inventory tests**, the **host C4 output-copy test** with all its batch/row/output subcases, and the compiled **host bounded-cache policy test**. All passed. These are CPU/host checks, not new B70 numerical or performance tests.

The supplied Pro review reports no native model build, GPU inference, current live-container inspection or new BF16 inference. The limited local image/mount inspection in §0.2 does not extend its numerical qualification. The eight live runtime artifact hashes are supported by the included read-only live receipt; omitted binaries cannot be independently rehashed from this ZIP. Large Q/K/V/logit captures and some task fixtures are also omitted. Supplied GPU pass receipts are therefore reviewed evidence, not independently reproduced GPU results. [E01–E05, E20]

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

The supplied review reports independently recomputing the following values from original per-wave wall time and card-energy counters:

| Cap | Sum measured wave time | Card energy | Matrix output throughput | Tokens/joule |
|---:|---:|---:|---:|---:|
| 150 W | 50.05 min | 125.16 Wh | 42.28 tokens/s | 0.2818 |
| **180 W** | **41.68 min** | **125.10 Wh** | **50.78 tokens/s** | **0.2819** |
| 230 W | 35.77 min | 137.16 Wh | 59.17 tokens/s | 0.2571 |
| 275 W | 33.89 min | 155.34 Wh | 62.44 tokens/s | 0.2271 |

Use `sum(output_tokens) / sum(wave_wall_seconds)` and `sum(output_tokens) / sum(card_joules)`, not the average of scenario rates. These figures include prefill, decode and request overhead. They exclude warmup, worker startup and the separate isolated 64K-prefix probe; they are **card energy, not whole-system energy**, and **matrix throughput, not decode speed**. Concurrent per-request native decode seconds overlap and must not be added to create elapsed server time. [E02]

Keep **180 W** for the initial native comparison. Here, 150 W took 20.09% longer with essentially identical energy (about 0.051% higher); higher caps traded additional energy for time. The non-random sequence was 180 → 230 → 275 → 150 W, with no fresh repeated 180 W thermal control and no comparable 180 W thermal series. This supports a stable project baseline, not a universal efficiency optimum. Do not reopen power tuning during the port.

Useful current 180 W serving anchors, not native parity proofs:

| Python production workload | Producer-reported prefill rate | Producer-reported request-weighted decode rate | Other metric |
|---|---:|---:|---|
| 4K C1 | 2,241.38 tokens/s | 57.14 tokens/s | TTFT 1.832 s |
| 32K C1 | 1,977.65 tokens/s | 54.50 tokens/s | TTFT 16.595 s |
| 128K C1 | 1,262.55 tokens/s | 39.93 tokens/s | TTFT 103.984 s |
| 4K C4 | not comparable to C1 prefill | 41.70 tokens/s, median request-weighted rate across waves | **180.07 tokens/s aggregate fully overlapped decode** |

These are Python EXL3 production sampled-MTP3 measurements, not native vllm.cpp measurements. Different continuations and acceptance can change their work per emitted token. A fresh small matched replay is required before attributing a difference to native kernels.

## 3. Freeze the correct references

### Reference A — historical GPTQ diagnostic oracle

- Historical native reference HEAD (not current checkout): `c1342199e09bd4c2c018c4694330152d283bf7ad`.
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

The producer has conditional draft sampling: probabilistic sampling when a draft-logit buffer is configured, otherwise argmax. Fresh pinned-image config inspection and `EngineArgs.create_engine_config` resolve the launch's MTP3 option to **greedy draft sampling / standard rejection**, as recorded with the config-source hash in `docs/b70-exl3/REFERENCE_MANIFEST.json`. The subset/global-ID and buffer/state contracts still require native qualification. Target temperature 1.0 is not evidence for draft sampling. [E14]

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

**Deliver:** §1’s S0a/S0b, one manifest containing separate immutable reference A/B records, the actual route/precision map, recorded ancestry finding and single-projection fixtures. Header-based inventory and replay qualification remain pending until performed; this plan update is not an S0 pass. Record selected driver/compiler/library bindings and configuration-derived sampling defaults.

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
8. **Operational isolation:** keep the production service stopped throughout development. No other vLLM instance may run on the B70 alongside the instance being built/tested. Do not run Python and native models concurrently to “pair” them; use existing captures first, otherwise run isolated pinned-oracle captures followed by native replay. Exclusive GPU use and focused tests are already authorized by the developer; do not restore the production service automatically. Compilation/host tests must also respect 32 GB host RAM.

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

**Keep deferred:** complete GPTQ performance parity; new MTP-depth/vocabulary sweeps; full-vocabulary draft expansion; GPTQ W4A8 or its 196,608 ceiling transplanted into EXL3; power-cap sweeps; TP>1; C16 optimization; large-prefill graph capture; new quantization algorithms; vision/video porting before native text parity.

Do not re-run the 124-request serving matrix at each milestone. Do not turn historical plans into instructions to restart old experiments. Do not expand the quality scope endlessly: use the bounded tests above and add a case only for a concrete failure or missing production capability.

The native EXL3 text path is ready to promote when:

- The exact EXL3 checkpoint, 6-bpw target head and 65,536-entry MTP3 draft are loaded and traced; Python/Torch/Triton are not inference dependencies.
- Matched-state eager/graph/operator and MTP commit/RNG tests pass; no known shared correctness defect is hidden by a different quantization.
- Moderate C4, prefix reuse, cancellation/recovery and the 262K total-token boundary fit GPU/host budgets without unbounded resource retention.
- The fixed representative serving measurements are reproducible and accurately labeled. Any remaining speed gap is explicit; no historical GPTQ percentage is relabeled as EXL3 parity.
- One native EXL3 production policy is documented. Internal shape-specific safe kernel selection is allowed; silently loading GPTQ or silently changing precision is not. The Python image/policy remains the frozen reference. Its production service stays stopped during development; deployment is a separate explicit decision.

Reuse the current C++ model/state/serving infrastructure, replace or extend the EXL3 execution seams to match the now-qualified producer, and optimize the same-checkpoint EXL3 path. Do the few shared correctness checks that protect that reuse; do not spend another campaign finishing a weight format the user no longer runs in production.

## 9. Compact machine-readable milestone contract

The following is a planning contract, not an executable command list. Paths under `docs/b70-exl3` are proposed output paths in the native repository.

```json
{
  "schema": 1,
  "decision": "migrate_native_exl3_now_reuse_current_cpp",
  "date": "2026-10-02",
  "cpp_base": "3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6",
  "historical_cpp_reference": "c1342199e09bd4c2c018c4694330152d283bf7ad",
  "merged_main": "9d96b162c8c3dfa8f0143932962e445b68fe3a8b",
  "branch": "b70-gptq-int4",
  "next_step": "S1_native_loader_and_FP16_vertical_slice",
  "migration_status": "not_yet_qualified",
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
    "preserve_existing_production_worker": false,
    "exclusive_gpu_for_migration": true,
    "production_service_must_be_stopped": true,
    "other_vllm_instances_on_gpu": 0,
    "focused_gpu_tests_authorized": true,
    "oracle_and_native_gpu_work_sequential": true,
    "automatic_production_restart": false
  },
  "milestones": [
    {
      "id": "S0",
      "deliverable": "S0a immutable references and execution map; S0b real projection replay",
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

## 10. Evidence appendix and provenance

Evidence labels E01–E23 refer to §10 of the [archived original Pro proposal](archive/b70/2026-10-02-Pro-EXL3-Migration-Plan-original.md#10-evidence-appendix-and-source-boundaries). Multiple excerpts under one evidence number belong to the same topic. Their C++ hashes/line numbers describe `c1342199e`, not automatically the merged base. Inspect current symbols and separate unchanged files from changed call sites.

The archive's `EXL3/`, `CPP/` and `AUDIT/` roots are review-package roots, not paths created in this checkout. The local review-bundle location is in §0.2. None of the original review's reported host/GPU results have been relabeled as newly executed here.

Original uploaded Markdown SHA-256: `08a4b275303217796a946f4d8f763a4f26d8987fe8d4a7569ca8cf4662b4d398`.

The original plan reconciliation updated documentation and navigation only. Subsequent S0a/S0b work is recorded in `docs/b70-exl3` and the focused test outputs; it does not qualify the whole native engine. Under the developer's exclusive-GPU update above, `b70-qwen38-vllm.service` has been stopped and must remain stopped during this goal. No automatic tests are required for this documentation-only operating-policy edit. S0/S1 must establish their own focused build and replay results.
