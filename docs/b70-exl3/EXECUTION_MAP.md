# S0: frozen references, execution map and projection replay

Observed 2026-10-02 at native HEAD `3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6`, branch `b70-gptq-int4`. The initial S0a source/asset inspection made no product/service changes. Subsequent S0b results are recorded in §7 and the actual isolated-worker layout contract in §8 below. Existing working-tree changes were preserved; neither stage qualifies the whole native engine.

[REFERENCE_MANIFEST.json](REFERENCE_MANIFEST.json) records absolute local paths, immutable pins, hashes, all 409 module descriptors and missing fields. `verified` means inspected local source/data in this session, within the stated scope; `receipt_derived` means a frozen receipt reports the value; `pending` means dependent replay must not assume it. A source observation does not prove that a runtime route ran.

## 1. Reference identities

| Reference | Identity and local source | Evidence state |
|---|---|---|
| A: GPTQ diagnostic | Native `c1342199e09bd4c2c018c4694330152d283bf7ad`; model `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`; Python `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | Receipt-derived from historical plan/baseline; never compare its raw logits against EXL3. |
| B: EXL3 donor | `/home/sebastian/LocalLLM/exl3xpu-review`, `c59d9442aba8610188837e37724600f1517d7335` | Verified HEAD and clean tracked source. Relevant math, loader, guard and subset files match pinned Git blobs and the qualified-source ZIP entries. Hashes are in the manifest. |
| B: serving policy | `/home/sebastian/LocalLLM/intel-b70-qwen38-vllm`, pinned `ed47c5b84614f5654440355b12fa61b6eea8ee5e` | Read with `git show`; current HEAD is `906be5bbcdd7756b514b32e3043fb1060bf867cd`. Pinned policy hash verified as `1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839`. |
| B: production image | `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`; parent `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737` | Receipt-derived from the read-only snapshot at `2026-10-02T17:58:44.492089+00:00`; no fresh worker inspection or inference in this step. |
| B: checkpoint | `turboderp/Qwen3.8-27B-exl3` at `113cf7ab958054860e43fb7f3063b1af19171095`, local `/home/sebastian/.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw` | Headers, metadata and markers verified in S0a. Both complete shard SHA256 hashes subsequently match the pinned revision's download-metadata LFS hashes (§7). |

The old EXL3 branch tip remains `2e1f5fbda6579c23dc63ea3faf0c69067ebbe19a`, and native HEAD remains the plan's merged base. The plan's ancestry finding therefore applies without repeating the calculation: its commits are already included. There is no merge/cherry-pick prerequisite.

The union of `SOURCE_HASHES` in pinned donor `target_runtime.py` and `attention_dispatch.py` contains 13 entries. All 13 captured `live-runtime/installed-vllm` files were rehashed and match those guards. This verifies captured source closure, not live binary/driver identity or every runtime dependency. Installed and workspace MTP/GDN files differ in the capture; use the installed files named in the manifest for the execution reference.

## 2. Header inventory and representation

The host inspection passed index/header names and shard assignment, tensor dtype/shape/byte spans, non-overlap, index byte total, embedded/standalone quantization agreement, every declared `tensor_storage` descriptor, donor canonical inventory, per-module dimensions/scales and all 409 mul1 markers.

| Item | Verified observation |
|---|---|
| Shards / tensors | 2 shards; 2,426 tensors; 16,860,508,228 stored tensor bytes; no unclaimed data bytes or unclassified tensors |
| EXL3 modules | 409 total = 400 target text + 1 target head + 8 MTP; all body/MTP projections 4 bpw, head 6 bpw |
| Target head | K=5,120, N=248,320; trellis in shard 2 at data offsets `[6301652288,7255201088]` (relative to tensor data section) |
| Components | trellis `I16 [K/16,N/16,16*bits]`; `suh F16 [K]`; `svh F16 [N]`; `mul1 I32 []` |
| Marker bits | All 409 scalar reads have unsigned pattern `2212286765` |
| Metadata omissions | 516 tensors absent from quantization storage metadata; the index and headers remain authoritative, including all eight MTP modules |
| Remainder | Embedding is BF16; text has 353 BF16 tensors and MTP has 7. Model config declares `bfloat16`; captured serving launch overrides execution to `float16`. Storage dtype alone is not the producer execution contract. |

The eight MTP modules are `mtp.fc`, `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj`. The complete header-derived family counts and 409 module names/K/N/bit widths/trellis spans are in the manifest.

The existing [inventory tool](../../tools/b70_inventory.py) was inspected and left unchanged. Its `checkpoint_inventory`, `add_source_pins` and `verify_checkpoint` paths still contain historical pins/expectations and must not certify this checkpoint. This inspection used only its model-independent `inventory()` header parser and the pinned, dependency-free donor `checkpoint.inventory()`; explicit descriptor and four-byte marker checks completed the host inspection. No third-party runtime was imported, no model was loaded and no dense weights were reconstructed. A reusable CLI adaptation remains a separate focused host-tested change if needed.

S0a initially recorded only expected LFS hashes. Subsequent full-file SHA256 reads match `b1e7fcc53a2bb211460ebb65e329cdb7a5af8bd819dcbcd89e06f0ccc041e590` and `c6d72f51b0cd99510239b264fef37bc26c950e6843f35421ce4830d4f87f8c05` (§7). Selected replay packed/scaling bytes and inputs additionally have per-tensor hashes. This verifies payload identity; native loader/mapping qualification remains S1 work.

The native byte trellis view `[K/16,N/16,32*bits]` can describe the same I16 storage only through a bit-preserving view. Header/marker validation has not yet tested unpacking, first/last tiles or Hadamard/GEMM arithmetic.

## 3. Precision and projection map

The donor paths below are relative to the pinned EXL3 source; captured vLLM paths are relative to the bundle's `live-runtime/installed-vllm`. Native paths link to this checkout. Both sets of inspected source hashes are recorded in the manifest.

| Boundary | B: producer contract / source observation | Current native source observation |
|---|---|---|
| Execution dtype | Captured launch `--dtype float16`; EXL3 `ops.py`/`csrc/exl3_ops.sycl` allocate output with input dtype. BF16 remainder is loaded into the chosen model dtype. Runtime per-boundary captures pending. | The target loader records an explicit EXL3 F16 policy (§10), independently preserving F32 recurrence. DenseDev applies it to paged/graph consumers (§12) and all four dense unpaged/pooling entries (§13). Synthetic eager entries are checked; MTP and graph arithmetic remain to audit. |
| Embedding / norm / residual | FP16 execution required by policy; exact intermediate rounding to capture. | The target remainder is loaded directly as F16 (§10). The synthetic paged hybrid trace (§12) observes F16 embedding, residual and final/decoder norms. Whole-model loading and producer rounding parity remain unverified. |
| Full-attention QKV | `vllm_plugin.py::process_weights_after_loading` maintains source groups and their `suh`, then block-to-group `shard_of_nb`; QKV groups are separate source transforms. Q projection includes the output gate. | ProjectFullAttnQkv retains separate source transforms in separate calls with scoped F16 output. Scoped SmallM projections now use the producer kernel (§14). Actual merged QKV storage/source-map integration and real-model arithmetic remain pending. |
| GDN QKVZ / BA | Producer fused QKVZ expects groups `[(0,1,2),(3,)]`: QKV share their source transform; Z has its own. BA uses model-dtype dense storage. GDN accumulation/state must remain distinct from model dtype. | QKV and Z remain separate calls through producer SmallM at scoped M≤128 (§14). F16 input/gate/output, F16 BA and F32 recurrence are observed in the synthetic trace. Actual merged QKVZ remains pending. Prefill Conv scratch is F32; producer GDN arithmetic is not qualified. |
| MLP | Producer gate/up grouping keeps independent source `suh`; output remains model dtype. | Exl3MlpGateUpMethod and EXL3 down follow explicit activation dtype, retaining BF16 when unspecified (§11–12). Synthetic model trace observes F16. Grouped producer projection and fused SwiGLU arithmetic are not yet qualified. |
| Linear seam | Group-aware packed SmallM/W8A8 producer interface; each 128-output block selects its input transform. | [Exl3MatmulD](../../include/vllm/model_executor/models/dense_attn_block.h), line 336: accepts F16/F32/BF16 outputs; non-F16 input is cast, or cast inside the XPU fused path; Hadamard scratch F16. Merely permitting FP16 here does not change model boundaries or kernel selection. |
| Target head / hidden tap | Full 6-bpw 248,320-row head. Select requested hidden rows before evaluating the head. | DenseForwardLayers taps post-final-norm hidden before requested-row gathering. Scoped EXL3 head now writes F16 then widens to F32 sampler logits. Synthetic P4 final-row head requests are witnessed as M1 (§12); full-vocabulary real-model head parity remains pending. |
| MTP feedback / head | Captured `qwen3_5_mtp.py` returns post-final-norm hidden. Donor draft head computes selected 128-token blocks and scatters into global logits with `-inf` elsewhere. Exact feedback-vs-logit row selection remains a runtime capture field. | EXL3 MTP projections are loaded; dense MTP model shares the target EXL3 head, and `ComputeLogits` requests F32 over full vocabulary. [proposer](../../src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp) computes all prefill logits before sampled-row selection, copies full FP32 logits to host and runs greedy selection. |

Grouping must preserve source transforms, not just concatenate packed weights. The donor constructs one `suh` per source group, boundaries in output columns and an I32 `shard_of_nb` entry per 128-column block. Header validity does not certify a fused runtime grouping. The pinned-worker loader report in §8 now records actual producer groups/bounds; native grouped mapping remains an S1 check.

## 4. Kernel and attention dispatch

| Area | B: pinned producer | Native today |
|---|---|---|
| SmallM | `ops.py` default `SMALL_M_MAX=128`; supported `csrc/exl3_ops.sycl::exl3_linear` dispatches SmallM first at M≤128. Physical flattened rows decide the route, including graph padding. | [XPU automatic strategy](../../src/vt/xpu/xpu_exl3.cpp), line 274, remaps BF16 output to F32, but leaves F16 unchanged. [MeasuredStrategy](../../src/vt/xpu/xpu_exl3_strategy.h), line 19, requires a historical device/driver/compiler/kernel domain and F32 output. F16 therefore returns Packed under automatic selection. This historical strategy still governs legacy/diagnostic Exl3Gemm calls. Scoped FP16 model calls at M≤128 now use the registered grouped producer SmallM operator (§14): the synthetic model trace witnesses129 calls, no Packed operator selections. Actual merged model groups and large-M W8A8 remain pending. |
| LargeM | With `EXL3_INT8_PREFILL=1`, mul1 4/6-bpw larger calls use rotated W8A8. Static weight bound `3.453125`; activation row scales and reconstruction/GEMM/output Hadamard preserve the producer's rounding. Padded GEMM rows are multiples of 256. | [xpu_exl3_prefill.cpp](../../src/vt/xpu/xpu_exl3_prefill.cpp) is FP16 panel reconstruction/matrix work, not this W8A8 algorithm. Existing strategy thresholds include historical M≥128 prefill/M≥512 all-rows cases. They are not the new 128/129 production contract. |
| Reconstruction memory | In the active oneDNN branch, `exl3_ops.sycl` lines 761–770 allocates `y [Ms,N]` F16 and `w [K,ng]` I8 for each whole source group. `slice_n=16384` bounds only the FP16 reconstruction fallback. | Memory/ownership equivalence and selected native queue/library bindings are pending; do not claim whole-group W8A8 reconstruction is slice-bounded. |
| Attention prefill | Guarded FP16 Q, static FP8 E4M3 K/V/scalar F32 scales; Q≥64, exact K≥4096 through 262144, query bucket 256, exact K without key padding. Bottom-right causality. | Native paged attention/FP8 writer and optional Xe2 verifier exist. Identical-input operator and whole-model EXL3 precision/state qualification have not run. |
| Verification / copy | Policy M04 uniform Q2–Q5/C1–C4. C4 supplied-output path uses direct permuted-view copy; C1–C3 and no-output preserve the old route. | Existing native verifier is an implementation reference to qualify, not a new MTP3 pass. Q5 is compatibility scope, not a request for MTP4. |
| Graph / prefix | FULL_DECODE_ONLY, sizes `[1,2,4,8,12,16,24,32,40,48,56,64]`; producer supports recurrent prefix with MTP3. | [ForwardQwen3_5Dense](../../src/vllm/model_executor/models/qwen3_5_dense.cpp), line 209: hidden tap returns through eager path before graph selection. [runner](../../src/vllm/v1/worker/gpu/runner.cpp), line 762: recurrent prefix snapshots explicitly require non-speculative execution. |

The native method also contains an M>144 reconstruction seam gated by operator registration in `dense_attn_block.h`; its comment describes a CUDA reconstruction route. This is separate from the XPU strategy above and is not evidence of producer W8A8 on B70. The registered native XPU paths are retained as source references; no strategy override or kernel was executed here.

## 5. Cache, state, sampling and ownership boundaries

**Source constraints, now supplemented by actual runtime views in §8:** donor `attention_dispatch.py::prefill_eligible` admits contiguous FP16 Q `[tokens,24,256]`, FP8 E4M3 K/V `[pages,page_tokens,4,256]`, page_tokens divisible by 64, matching K/V strides and legal non-overlapping physical strides. It permits padding; do not infer contiguous page strides from logical shape. It requires I32 cumulative query positions, active lengths and block tables, with scalar/broadcast F32 scales. The actual EXL3 capture in §8 supplies page_tokens, physical strides, offsets, scales and active P128 metadata. The observed EXL3 page1600 comes from this worker, independently of any historical GPTQ geometry.

Native `FlashAttentionBackend::get_kv_cache_shape` in [backend.cpp](../../src/vllm/v1/attention/backend.cpp), line 248, returns `[pages,2,page_tokens,heads,head_size]`. Runner allocation obtains page size, KV dtype, FP8 interpretation and scales from `AttentionSpec`, then exposes K/V views. Recurrent state allocation follows `MambaSpec.shapes/dtypes` in order, prepends slot dimension and publishes `states[0]` as Conv and `states[1]` as SSM. It can carry more state tensors; do not truncate it to an assumed two-buffer production receipt.

The verified checkpoint config has 64 target layers (48 GDN / 16 full attention), 16 GDN key heads, 48 value heads, key/value dim 128, Conv kernel 4 and `mamba_ssm_dtype=float32`. Config-derived recurrent matrix geometry is `[48,128,128]` (3 MiB in F32 per slot/layer), not an observed slot allocation. Captured producer `QwenGatedDeltaNetAttention.get_state_shape` delegates to `MambaStateShapeCalculator` with `num_spec`; actual Conv/SSM views and the initialized MTP3 cache specification are now captured in §8. Accepted-prefix speculative state transitions and prefix snapshots remain S3/S5 qualification work. Native `ResolveMambaSsmCacheDType` honors float32 separately from activation/Conv dtype. These facts must not be turned into an unobserved runtime state layout.

**Draft subset and configured sampling verified:** the pinned `draft_vocab.json` contains 512 unique ordered blocks of 128 tokens, within 248,320 target rows, giving 65,536 draft IDs. Its metadata `vocab=248077` is not the physical head width. The manifest stores the ordered blocks and SHA256 of expanded global IDs encoded as little-endian I32. Producer `_build_draft_head` slices whole trellis/scale blocks and retains this order; `_patch_mtp_draft_logits` writes `-inf` for unsupported target IDs. Current native draft computes the shared full target head and does not yet implement this subset contract.

Captured `v1/worker/gpu/spec_decode/speculator.py` lines 173–177 allocates draft-logit storage only for `draft_sample_method="probabilistic"`; otherwise `sample_draft` uses argmax. Fresh inspection of the immutable production image's `config/speculative.py` (SHA256 `9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19`, line 588) finds the default **greedy**. `EngineArgs.create_engine_config` on the pinned local model, FP16/FP8, MTP3, context262144, prefix/align cache, sequences16, scheduled4096 and GPU fraction0.965 resolves **greedy draft / standard rejection**. The isolated pinned worker in §8 independently confirms greedy/standard, MTP3 and `draft_logits=None`. Target temperature was not used to infer it.

The producer's partition capacity 64 is per thread/queue, with completed-LRU eviction and a victim wait only when all entries are busy. Native strategy memoization's 512-entry map is a different cache. Queue/context identity, last-consumer events, retained panel/output/metadata ownership and graph-held references still need implementation-specific qualification. The native proposer currently synchronizes during its full-logit host copy; removing that wait later requires an explicit owner/event contract.

## 6. Existing assets and S0b prerequisites

The manifest hashes the two local review ZIPs and selected relevant payload entries, not a newly rerun whole review. GPTQ token fixtures and W4/W5 documents were hashed in `docs/bench-evidence/b70-gptq-fp8-parity-20260927/`; `tools/gptq_reference/` and `B70_GPTQ_INT4_Plan/evidence/` remain available. The small historical W4A16 safetensors fixture exists (2,716,073 bytes), but reading it returned **Permission denied**; its hash is pending. No permissions were changed.

All eight known W4/W5 raw capture directories (`b70_{cpp,python}_spec_stages`, `b70_spec_probe_{cpp,python}`, `b70_m04_real_replay`, `b70_cpp_m04_inputs`, `b70_{python,cpp}_prequant`) are absent at their recorded `/tmp` paths. First-three-GDN-to-first-QKV diagnostic inputs therefore need a preserved copy before their replay. Historical TV failures remain open at 0.02445/0.02329 versus 0.02.

EXL3 `target-quality-stages-v1/raw-capture-manifest.json` identifies `/home/sebastian/LocalLLM/intel-b70-qwen38-vllm-onednn/benchmark-results/exl3-quality-target-stages-v2`: all 160 declared arrays are present. Their large payloads were not rehashed. These are log-probability/NLL arrays, not identified M=1/M=4 projection-stage captures. No new capture was requested from the running service.

Native build metadata exists at `/home/sebastian/LocalLLM/b70-main-integration-build/cmake`. Its cache/compilation-database hashes are recorded. The cache refers to `/work`, `/tla` and `/opt/intel/oneapi/compiler/2026.1/bin/icpx`; that compiler is absent on the host and `icpx` is not on PATH. The donor's pinned Torch-2.13 build receipt names DPC++ 2026.1.1, oneDNN enabled and its exact flags/library hash; these are receipt-derived, not a resolved native build environment. Existing objects do not validate this merged source.

The original proposed S0b preparation/replay is now executed as recorded below. The developer's subsequent 2026-10-02 instruction grants exclusive B70 use: keep production stopped, run isolated oracle/native GPU work sequentially and continue focused tests without another service-window approval. Earlier S0a observations remain historical evidence.

S0a's host inspection (`PYTHONDONTWRITEBYTECODE=1 python3 /tmp/b70_s0a_collect.py`, a local collection script) exited 0: 409 modules, eight MTP, all 409 marker values, all 13 captured guards, pinned policy/donor bytes and subset mapping consistent. At that inspection, JSON/counts, archive SHA256 sidecars, native source hashes and map links agreed. This documentation-only step required no compile or automatic unit suite. Current S0b source hashes/results follow; §8 closes the required runtime-layout fields for the S0 contract.

## 7. S0b executed projection replay

The [replay tools](../../tools/exl3_reference/README.md) extract real packed fixtures with deterministic FP16 M1/M4 inputs. Captures and per-tensor hashes live in `/home/sebastian/LocalLLM/b70-exl3-fixtures/s0b`; exact files, SHA256 values, source hashes and build/device receipts are indexed in `REFERENCE_MANIFEST.json`. The original v1 diagnostic captures remain intact; `_oracle_v2`, `_native_v2` and `_comparison_v2` are the current production-stage evidence.

| Real projection | Geometry | Producer splits M1 / M4 | Production-output relative norm M1 / M4 |
|---|---|---|---|
| Layer0 MLP gate, 4 bpw | K5120 / N17408, full projection | 8 / 6 | 4.8636383e-4 / 3.5339231e-4 |
| Head first block, 6 bpw | K5120, columns0:128 of248320 | 320 / 320 | 5.9777726e-4 / 3.2131485e-4 |
| Head last block, 6 bpw | K5120, columns248192:248320 of248320 | 320 / 320 | 7.1748723e-4 / 3.5719053e-4 |

**Result:** all six local projection gates pass strict relative norm `<2e-3`, with finite numerical stages. Packed decode and actual input-Hadamard bytes match exactly. The first bit difference is GEMM: compare the native raw F32 output with an increasing-split CPU F32 sum of the producer's actual F32 split buffers. The latter is explicitly a derived sum of directly captured parts, rather than a separately exported binary sum. Maximum production-output absolute error is 0.001953125. This is an attributed local pass with arithmetic differences, not bit-exact end-to-end parity.

The original `_C.so` and `ops.py` hashes match the pinned runtime. `TorchDispatchMode` retains nested allocation tensors while redispatching the original XPU `linear`; every observed output matches an ordinary call byte-for-byte. Actual blocked input agrees exactly with the registered row-major helper. `exl3_gemm_raw`'s fixed four-split result is retained separately as a diagnostic and never substituted for the production split buffers.

The isolated output-Hadamard probe uses identical saved FP16 operands/scales on both sides. It exposes the native extra FP16 rounding before `svh`; producer `HadOutKernel` multiplies in F32 before the final FP16 store. The producer M1 `GemvKernel` also uses FP16 tile accumulators before F32 accumulation, unlike native increasing-K F32 arithmetic. These source differences remain for S1's qualified SmallM port even though this bounded gate passes.

Native traces show **F16 → auto/packed → F16** in every case. A diagnostic `Exl3GemmReplayKernel` exposes its selected packed/reference leaf's raw buffer in caller-owned storage, without changing dispatch. Its results equal the normal native kernel's output/input-Hadamard bytes in all six cases; five invalid-storage cases each are rejected without writes. An explicit fused selection is refused rather than silently replaced. This standalone target compiles current native leaf, queue, allocator and reader sources; it does not qualify operator-wrapper registration, loader, full head, model execution or performance.

Build: local builder image `ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca`, oneAPI2026.1.1, C++20, `-O3 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt`; build/link exit0. `ldd` confirms no Torch/oneDNN dependency. Native runtime observes B70 device57891, driver1.17.39758+10, runtime1.17. The existing Copy registrar received `[[maybe_unused]]` to satisfy this compiler's warning check. Full-engine oneDNN/SYCL-TLA bindings remain separate pending prerequisites.

Focused checks: exporter6 host tests, capture5 host tests and comparison4 metric tests pass (each exit0); six native GPU replays, three current comparison commands and fused-route refusal pass (exit0). Both complete model shards were streamed through SHA256 without dequantization: `b1e7fcc53a2bb211460ebb65e329cdb7a5af8bd819dcbcd89e06f0ccc041e590` and `c6d72f51b0cd99510239b264fef37bc26c950e6843f35421ce4830d4f87f8c05`, matching the download metadata at the pinned revision.

**S0 runtime follow-up:** completed in §8. Native loader/mapping, producer SmallM and full-engine block/model/state checks remain S1 work; the local packed projection gate does not qualify those routes.


## 8. Actual pinned-worker layout contract

The isolated pinned image initialized `XPUModelRunnerV2`, ran one synthetic **P128/O1** request and shut down cleanly (exit0). Production stayed inactive/dead, MainPID0; no other vLLM instance ran. The image `_C.so`/`ops.py`, all 13 installed guards and checkpoint metadata were verified before loading. The profile SHA256 `737c6dfbfd86b40dfd8c2786452d147e916a8ae8a1fd67fdb2ad42649dd764d0` also matches the pinned donor's `migration-target-optimized.yaml`. HTTP-only arguments were excluded from the direct worker API; the reasoning config was converted to its required typed configuration. No numerical kernel or argument was replaced.

Capture: `/home/sebastian/LocalLLM/b70-exl3-fixtures/s0-runtime/runtime_layout_v2.json`, 1,666,860 bytes, SHA256 `8f692b4782a039ebe0bc6449982a98677f2c81eba21e4acfde21f15616511681`. The original `runtime_layout.json` and the exact input reference-manifest snapshot are retained. The current manifest hashes the capture, source helpers, worker log and loader report. The first startup attempt failed before weights loaded because the direct API rejected a dictionary reasoning config; the corrected first layout capture and this completed argument-view capture both exited0.

The read-only worker-extension RPC records actual bound caches, and a prefill hook records active request/state metadata. A temporary observer delegates the original `flash_attn_varlen_func` unchanged while describing its actual Q/K/V arguments for all 16 target layers; it is removed after capture. Ten focused host tests pass (exit0), including alias/offset validation, bounded metadata copies, broadcast scales, poisoned unused block columns and retained active slot mappings. No Q/K/V or recurrent-state payloads were downloaded.

| Actual object | Observed shape / dtype | Physical strides in elements / offset |
|---|---|---|
| Conv, representative first GDN layer | `[201,6,10240]` F16 | `[1638400,10240,1]`; offset0 |
| SSM, same layer | `[201,48,128,128]` F32 | `[819200,16384,128,1]`; offset30720 F32 elements, i.e.122880 bytes after Conv |
| Bound raw attention cache | `[201,4,1600,512]` U8 | `[3276800,512,2048,1]`; layer bank offset |
| Actual attention Q and output | `[128,24,256]` F16 | `[6144,256,1]`; offset0 |
| Actual attention K/V | `[201,1600,4,256]` E4M3FN | `[3276800,2048,512,1]`; K starts at layer bank offset, V at K+256 bytes |
| Actual K/V descales | `[1,4]` F32 | `[0,0]`; each broadcasts a single owning F32 scalar with value1 |

There are four cache groups: three sets of16 GDN layers, then16 target full-attention layers plus one MTP full-attention layer. **All 65 bound layer caches share one owning storage of11,196,825,600 bytes**, with physical block stride3,276,800 bytes and layer-bank stride658,636,800 bytes. Different groups use different allocated block IDs while sharing the same layer-bank offsets. Do not sum owning-storage bytes per layer/group. The earlier capture had202 pages and11,252,531,200 owning bytes; both workers selected1600 tokens per block. Capacity varies with the runtime memory budget and is not a fixed checkpoint invariant.

The actual initialized GDN cache spec records MTP3, align mode, three speculative blocks and zero prefill checkpoint blocks. The active request occupied request-state index6. GDN groups used state indices1/5/9, query starts `[0,128]` and initial-state flagsfalse. Target/draft attention metadata had lengths128, active first block ID13 and exact I64 slot mapping20800..20927 (`13*1600 + token_position`). Actual target attention calls independently reported the same lengths and block ID. Remaining block-table columns were described but never read as values.

The worker confirms FP16 model dtype, U8 storage for FP8 quantization mode1, greedy draft/standard rejection and `draft_logits=None`. Its guarded Python loader audit passed409/409 modules, including8 MTP, with exact producer shapes/groups and no missing or unexpected duplicate modules; the shared head was loaded for target and draft. This is evidence for the pinned producer, not a native loader pass. The observed post-request memory was28,817,533,952 allocated and29,068,623,872 reserved bytes; this is neither a peak-memory measurement nor native capacity qualification. The synthetic output token13 makes no quality claim.

**S0 deliverables are complete within their scope:** immutable reference/header/payload identities, source execution map, real local M1/M4 projection gate and actual initialized cache/state/attention metadata contract. The historical GPTQ TV failures remain unresolved and unrelabeled. **Next S1 step:** validate the native loader/mapping against the real checkpoint, then make FP16 boundaries explicit and port producer SmallM. Full-engine compilation, block/eager target parity, speculative accepted-prefix lifecycle and prefix snapshots remain unqualified.


## 9. S1 native shared loader and mul1 validation

`tools/exl3_reference/check_loader.cpp` builds the current `SafetensorsFile`,
`LoadExl3` and production mmap/weight-owner implementation in a focused host
executable. The real pinned two-shard checkpoint check passes409 modules,
including8 MTP, codebook2 and per-tensor 4/6-bpw geometry (exit0). Every native
packed/sign-scale view borrows exactly the complete source pointer/span; total
mapped weight bytes13,343,720,448, without dense expansion or a second packed
copy. All weights remain valid after destruction of the files/index, and the
mapping expires after the last weight owner is destroyed. The existing complete
S0 shard hashes remain the payload identity prerequisite; this check does not
rehash the entire model.

The first checker found that the shared loader selected mul1 by presence/dtype
but did not validate its multiplier. `LoadExl3` now accepts only the exact
`0x83DCD12D` I32 multiplier, in scalar or historical `[1]` form, with exactly four
bytes and a non-null payload. The read is alignment-safe. A focused direct
loader unit test passes4 cases/11 assertions, including unaligned valid storage,
wrong multiplier, malformed shapes/byte counts/null pointer/dtype and competing
markers (exit0). The existing native-loader positive fixture now writes the
real multiplier rather than arbitrary fill bytes. The real409-module check was
rebuilt and passed after this product change.

Builder/compiler: the pinned local image
`ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca`,
IntelLLVM2026.1.1, host C++20, `-O2 -Wall -Wextra -Werror -fno-fast-math
-ffp-contract=off`. Dead-section elimination excludes unrelated model forwards
from this standalone diagnostic. The marker target initially required CMake
regeneration; one rerun also used a mismatched container build path and was
refused. Both corrected focused runs passed. The manifest records the final
binary/source/log hashes and precise scope.

Negative duplicate-shard rejection belongs to the diagnostic index; it does
not prove that the complete Qwen loader refuses duplicates. **S1 remains
incomplete:** complete model-loader mapping/upload, explicit FP16 remainder and
activation/output boundaries, typed grouped SmallM, full-engine build and
block/eager P128 D1/D64 parity remain required. No model/state parity or
production SmallM performance is claimed from this host check.


## 10. S1 FP16 target parameter loading

The shared `dense_loaders::LoadF16Direct` now preserves F16 bytes and borrows
mapped F16 source spans with their owning lifetime. BF16/F32 sources convert
directly to F16, avoiding the previous F16→BF16 precision loss and an
intermediate BF16 rounding. Rank, dimensions, count overflow, byte span and
shape-preserving reshapes are checked before reading. The existing GPTQ F16
reader delegates to this primitive after its existing strict F16/configured
shape check. `LoadMergedF16RawNK` preserves listed row order for the dense BA
parameter; packed EXL3 weights still use the existing verbatim reader.

The Qwen target loader marks `exl3_checkpoint`, resolves explicit F16 storage
for model activations/dense weights/auto KV/Conv and F32 recurrent state, and
loads EXL3 decoder/QK/gated norms, Conv and BA as F16. Its embedding and final
norm call sites select the same reader. A BF16 exported dtype does not override
the pinned runtime's explicit FP16 policy. Ordinary dense/GPTQ policy behavior
is covered by focused policy cases. The current A_log/dt_bias owners remain F32
computation inputs; matching the producer's complete gate arithmetic is still
required in the block/state replay.

Focused checks in the pinned IntelLLVM2026.1.1 builder:

- `dense_f16_loader_test`: **5 cases, 26 assertions, exit0**. Known half-bit
  fixtures catch BF16 rounding; unaligned BF16/F32 conversions, mapped-owner
  lifetime, malformed shapes/spans and mixed-dtype B→A merge are checked.
- `qwen_exl3_precision_test`: **2 cases, 11 assertions, exit0**. Explicit EXL3
  F16 policy despite BF16 export, ordinary BF16/GPTQ F16 behavior, competing
  formats and unsupported recurrent-state dtype are checked.
- `qwen_exl3_parameter_check /models/checkpoint NEW_REPORT`: **64 current native
  per-layer target loads, 48 GDN/16 attention, exit0**. All loaded decoder/QK/
  gated norms and Conv values match the expected direct-F16 source conversion;
  BA retains exact stored half bits in B→A row order. This calls the actual
  `LoadQwen3_5DenseLayer` implementation, without model forwards or GPU access.

The per-layer checker first failed to compile with an incorrectly named index
reader. After correction, linking identified the actual FP8-config dependency;
adding `fp8_block_quant.cpp` made the focused build/check pass. These failures
ran no inference. The report is
`/home/sebastian/LocalLLM/b70-exl3-fixtures/s1/qwen_f16_parameters.json`.
Source/binary/log hashes are recorded under `S1_FP16_parameter_loading` in the
manifest. Comment-only source cleanup followed the passing checks. Earlier
S1 compile-command hashes are labeled historical because this isolated build
directory has since been regenerated; earlier binaries/reports remain available.

**Qualification boundary:** whole-model embedding/head/vision and MTP loading
have not been functionally checked by this per-layer tool. `DenseDev`, several
EXL3 projection/MLP output requests and runner cache defaults still need their
explicit F16 execution wiring. MTP remainder readers remain BF16. No full-engine
link, GPU/model/state or grouped SmallM parity is claimed. **Next step:** wire
these native activation/output/cache boundaries to the stored policy, run a
focused compile/functional check, then qualify the complete loader and producer
SmallM/block/eager target routes. S1–S6 remain incomplete.

## 11. S1 FP16 gate/up execution and cache planning

`Exl3MlpGateUpMethod` now applies `Dev.activation_dtype` to gate, up and
SwiGLU output, retaining BF16 when unspecified. The existing operator rounds
SiLU through the gate dtype before the product. This arithmetic has not yet
been independently compared to the pinned producer's fused SwiGLU.

The regular root-CMake XPU/oneDNN engine library and selected test targets
build with pinned IntelLLVM 2026.1.1 and oneDNN 3.13. Optional Xe2 targets are
disabled; no optimized-policy or model qualification follows from this build.
The compiler library path must precede the image's older Unified Runtime
loader. An unused, unreferenced region-budget constant was removed after the
regular `qwen3_5.cpp` compile refused it under `-Werror`.

`test_exl3_linear_method`: **8 cases, 1,565 assertions, exit 0**. Seven cases
exercise existing CPU behavior. The new case runs the actual registered B70
path with synthetic 4-bpw mul1 weights at K=N=128 and M=1/M=4, verifying F16
projection/output boundaries, exact current SwiGLU rounding, finite/nonzero
values and legacy BF16 output. Its first CPU attempt was refused by the public
operator's XPU-only F16 output contract; that contract was preserved. This is
a dtype/plumbing check, not independent producer SmallM or checkpoint parity.
The linked libraries include native SYCL/oneDNN, without Python/Torch.

`MakeQwen3_5KVCacheSpec` now resolves the declared EXL3 precision through the
same policy as the target loader: F16 Conv, F32 recurrence and F16 unquantized
auto KV, including draft KV. Explicit F32 diagnostic KV remains selectable;
legacy GDN dtype overrides do not override the scoped F16 policy. Runner
selection of production FP8 storage remains separate from this auto default.

`test_model_registry --test-case='Qwen3.5 KV-cache spec:*'`: **2 cases,
62 assertions, exit 0**; 24 unrelated cases were not selected. Checks cover
BF16-exported EXL3 config, exact unpadded Conv/SSM shapes and byte counts at
num_spec=0/3, attention/draft auto dtype, unsupported recurrent dtype,
ordinary BF16 and GPTQ F16 state behavior. No GPU allocation, physical packed
layout, speculative commit or model arithmetic was exercised by this check.
Metadata-free/partial EXL3 cache planning is not qualified.

Sources, binaries, compiler settings and logs are recorded under
`reference_B.S1_FP16_gate_up_and_cache_planning` in the manifest. Production
was verified inactive/dead with MainPID=0 after the tests, and remains stopped.
**Next:** `DenseDev` and remaining EXL3 projection/head output boundaries,
then focused functional checks. Runner FP8 allocation/layout, MTP remainder,
grouped producer SmallM and block/eager P128 D1/D64 parity remain open.

## 12. S1 paged hybrid FP16 execution

DenseDev now enables the explicit EXL3 F16 model policy on XPU. Full-attention
Q/K/V/output, GDN output and MLP down preserve it, as do merged/sliced/separate
BA outputs. Legacy callers without a scoped policy retain their previous
BF16/F32 behavior. The target EXL3 head writes F16 before widening selected
logits to F32; the existing pre-head row gather is retained. Paged validation
now checks EXL3 cache types as well as GPTQ: F16/F32/E4M3 attention KV, F16 Conv
and F32 recurrent state.

`test_xpu_qwen_exl3_fp16` builds and runs through the actual native model and
registered XPU operators: **1 case, two BA subcases, 1,066 assertions, exit 0**.
The deterministic synthetic H128/I128/V128 model has one GDN and one full
attention block, 4-bpw mul1 body and 6-bpw head, F16 remainder, F32 A_log/dt_bias,
P4 then D1, and two cache/state slots with active slot1. Checks cover F16 tapped
hidden, finite/nonzero selected logits exactly matching a separately invoked
F16 head, untouched inactive SSM slot, updated active SSM slot and BF16 Conv
cache rejection. A wrong ForwardLogits field name initially failed the new
test compile; correction preceded all functional checks.

The trace audit observes **206 tensor rows, 146 native-XPU provider selections,
zero BF16 tensor arguments and zero CPU-reference selections**. All60 EXL3
calls (52 body/8 head, including selected-row head comparisons) use F16 input
and output and the existing **Packed** leaf. All head calls are M1 despite P4.
BA and model norm/embedding outputs are F16, while GDN recurrence state is F32.
The native prefill's Conv history scratch is F32 before F16 cache storage; its
same-input arithmetic still needs comparison to the producer.

Sources, binary, exact logs and trace audit are recorded under
`reference_B.S1_FP16_paged_hybrid_execution`. Earlier source/compile-database
hashes affected by this step are labeled historical observations. Optional
Xe2 targets remain disabled in this build. Production was observed
inactive/dead/MainPID0 after the test and remains stopped.

**Qualification boundary:** synthetic P4/D1 is not real-checkpoint, producer,
full-model P128/D64, FP8-layout, MTP or graph parity. The observed Packed route
is diagnostic; producer SmallM is not ported. The alternate unpaged/pooling
entry-point gap observed here has since been addressed within the synthetic
scope of §13. Typed grouped producer SmallM and real block/eager target
qualification remain required. S1–S6 remain incomplete.


## 13. S1 FP16 dense entry points and unpaged attention

ForwardDense, ForwardDenseHidden, ForwardDenseLastLogits and paged ForwardHidden
now resolve DenseDev instead of constructing an unscoped context. Unpaged
full-attention keeps Q/K/V and attention output in the selected F16 dtype;
legacy callers retain their F32 attention path. The pooling gather buffer now
remains owned through the subsequent CastF32 submission. Previously the
borrowed gathered view outlived its owner before the cast.

The first extended B70 check passed the paged/pooling subcases (1,582
assertions), then refused the unpaged call because no XPU Attention provider
was registered. The new native adapter borrows contiguous K/V as a single
page and calls the existing native PagedAttention, preserving causal/GQA and
output-alias behavior without a KV copy or quadratic score allocation. Four
I32 metadata values are held in owned scratch through the call. This is an
eager diagnostic entry; temporary-metadata graph capture is not qualified.
The public Attention wrapper admits F16 output only on XPU.

Focused checks in the regular IntelLLVM 2026.1.1 / oneDNN 3.13 build:

- `test_xpu_qwen_exl3_fp16`: **2 cases, 3,891 assertions, exit 0**. The prior
  H128 GDN/full-attention P4/D1 test now checks pooling rows {3,0} against the
  separately owned F16 tap. A new case exercises all three unpaged APIs,
  finite F16-grid hidden/logits, exact last/full-logits correspondence and
  selected logits matching a separate F16 head call.
- `test_xpu_attention --test-case='XPU unpaged attention:*'`: **1 case,
  206 assertions, exit 0**, eight unrelated cases unselected. Independent CPU
  F32 accumulation is compared at T5, Hq6/Hkv2, D7/D256, causal/noncausal,
  all F32/BF16/F16 input/output combinations and query/output aliasing. The
  CPU F16-output refusal remains intact; portable reference-tier hits are zero.
  The new test initially used F32 rtol/atol 2e-6 and failed on a roughly 1e-5
  CPU/device reduction difference. Its final F32 tolerance is 2e-5, matching
  the existing paged test; F16 uses 0.001 and BF16 0.008. These are local
  dtype comparisons, not changes to the plan's producer parity gates.
  A later zero-shape test construction was refused by Tensor::Contiguous
  after 204 passing assertions; that unsupported construction was replaced
  by the CPU F16-output contract check before the final passing run.
- `test_qwen27_dense_forward --test-case='qwen27 dense forward: finite*'`:
  **1 case, 242 assertions, exit 0**, eleven unrelated cases unselected.
  This checks the existing CPU/BF16 dense model's finite/deterministic logits.

The extended model trace has **446 tensor rows, 318 native-XPU provider
selections, zero BF16 arguments and zero CPU-reference selections**. All128
EXL3 calls (117 body / 11 head) use F16 input/output and Packed. All three
unpaged Attention calls have F16 output/Q/K/V. Ten head calls use M1;
one M4 call belongs to the explicit full-logits diagnostic API at tiny V128.
Selected-final-row APIs continue to gather before head evaluation.

Exact source/binary/log hashes and failed-run receipts are recorded under
`reference_B.S1_FP16_dense_entrypoints`. Changed earlier source and build-path
binary hashes are labeled historical observations. Production remains
inactive/dead/MainPID0. Optional Xe2 targets remain disabled.

**Qualification boundary:** this closes the tested FP16 entry/gather wiring,
not real-checkpoint model or producer arithmetic parity. Packed remains the
diagnostic leaf. Full loader/upload, Conv/SwiGLU/recurrent rounding, FP8
physical layout, MTP and graphs remain open. **Next:** implement the typed
grouped-linear VT API and port pinned producer SmallM 4/6-bpw kernels with
separate source transforms; qualify one real block and then the eager target
at P128/D1 and P128/D64. S1–S6 remain incomplete.


## 14. S1 producer SmallM port and single-projection model wiring

The registered typed `vt::Exl3GroupedLinear` accepts packed trellis bytes,
F16 input/output, F16 `suh[S,K]`, F16 `svh[N]`, and I32 `shard_of_nb[N/128]`.
Each whole 128-output block chooses its own source input transform. Blocked
input scratch `[S,K/16,Mp,16]` and ordered F32 split-K partials `[P,M,N]`
are caller-owned. Writable overlaps, invalid layouts and out-of-range source
IDs are refused. Physical M129 is refused by this SmallM implementation;
S2 must supply the distinct W8A8 route rather than hiding it behind Packed.

The ESIMD header is an unchanged copy of `csrc/exl3_esimd.h` at
`c59d9442aba8610188837e37724600f1517d7335`, SHA-256
`aabdb13eddbcf7387dac2716b26e1005259a0a658d4b7d8b0e6b338499d0fdc6`.
MIT license and attribution are retained in `third_party/exl3xpu`. Native VT
queue/registration/event wrappers replace ATen allocation and stream wrappers;
no Torch runtime is linked. The donor's `-ffast-math` applies only to this
kernel translation unit; unrelated native arithmetic retains its flags.

The port matches default mul1 4/6-bpw dispatch: ESIMD vector at M≤2, DPAS
above it through physical M128, MB8/16/24/32/40/48/64, padded rows zeroed,
thread targets 1024/1408/2048, NT8/4/2 by route, and GRF256 at MB40/48/64.
Fused single-kernel opt-in and alternate experimental donor flags are not
selected. Input products/transform round at F16; ordered split reduction and
output Hadamard/scales run F32 before the final F16 store.

`test_xpu_exl3_smallm`: **2 cases, 222 assertions, exit 0**. Ten synthetic
two-group cases cover 4/6bpw at M1/4/12/16/128 with nonidentical transforms,
nonmonotonic map[1,0,1], poisoned DPAS padding and negative metadata/alias
checks. Their maximum relative norm against separately transformed CPU-order
blocks is **0.000552425**, below the unchanged 0.002 gate.

Six real single-group calls reuse the hash-verified S0b captures: K5120/N17408
4-bpw MLP gate, plus the first and last N128 blocks of the 6-bpw head, each at
M1/M4. **Input Hadamard, actual F32 split-K Partials and F16 outputs are all
bit-exact against the pinned producer captures; relative norms are zero.**
MLP gate uses P8/Mp1 then P6/Mp8; both head slices use P320/Mp1 then P320/Mp8.
This is not a full-vocabulary head or grouped real-family qualification.
The trace witnesses16 successful SmallM calls (5 GEMV/11 DPAS); two additional
provider/tensor rows belong to refused alias/source-ID calls. All18 XPU
provider selections use the native grouped operator; none use the portable
CPU reference tier. MB24/32/40/48 and vector M2 compile but were not run here.

Scoped XPU F16 `Exl3MatmulD` projections at M≤128 now invoke this operator
as one source group. CPU/legacy callers remain on their existing interface.
The model's QKV, QKVZ and gate/up are still separate calls; true merged packed
storage and distinct source-map ownership are the next implementation step.

`test_xpu_qwen_exl3_fp16`: **2 cases, 4,287 assertions, exit 0** after wiring.
The synthetic H128 GDN/full-attention P4/D1/pooling and unpaged paths witness
**129 SmallM calls (93 DPAS/36 GEMV), zero old Exl3Gemm selections, zero BF16
arguments and zero CPU-reference selections**. Every call currently has S1.
`test_exl3_linear_method`: **8 cases, 1,565 assertions, exit 0**, preserving
seven legacy CPU cases and the F16 gate/up boundary/SwiGLU check against direct
SmallM calls. This does not independently qualify producer SwiGLU arithmetic.

The first model check failed70 exact comparisons between full M4 logits and
the selected M1 head. The pinned GEMV has tile-local F16 accumulation while
DPAS uses F32, so cross-route bit identity was a Packed-era test assumption.
The corrected test compares each route exactly with an independently invoked
head at the same physical M, checks all512 full-logit elements, and retains
the plan's relative-norm gate between M1/M4. The observed head relative norm
is **0.000443042 < 0.002**. No kernel math or plan threshold was changed.
The initial port compile also exposed const-pointer list deduction, unused
shared-header NaN constants under donor fast-math, and deprecated SYCL property
launch syntax. All were corrected before the passing GPU checks.

Exact source/binary/build/link/trace and failed-check receipts are in
`reference_B.S1_producer_smallm`. Earlier overwritten source/compile-database
and build-path binary hashes are historical observations. Production remains
inactive/dead/MainPID0; native GPU jobs finished before this receipt.

**Remaining:** immutable source-map ownership/reuse, actual merged
QKV/QKVZ/gate-up, real grouped families and full-vocabulary head, full
loader/upload, one real block and complete eager P128/D1 and P128/D64.
Eager source-ID validation currently incurs a metadata readback/wait; remove
that cost through a qualified immutable-map contract without shortening its
lifetime. Graph/multi-queue lifetime and performance are not qualified.
M>128 model calls still use the historical diagnostic route; S2 W8A8, FP8,
MTP and S3–S6 remain open. **Next:** merge actual gate/up storage and its
separate transforms through this seam, then QKVZ/QKV and real-block replay.


## 15. S1 model-owned packed Gate/Up group

The scoped XPU FP16 SmallM MLP now submits one two-source projection.
`MergeExl3Weights` concatenates output tiles separately for every K tile,
retains both F16 input transforms, concatenates output transforms and owns
an I32 map of complete 128-column output blocks. A lazy group in
`DenseMlpWeights` survives per-call factory methods. Its four GPU operands
are uploaded once; upload completion precedes releasing the merged host
staging. Checkpoint sources remain intact, and the grouped method does not
build separate gate/up GPU residents. Legacy callers keep their split path.

Focused regular-engine build: exit0. `test_exl3_linear_method`: **9 cases,
1,717 assertions, exit0**. Three-source unequal-width byte checks cover4/6bpw
and ownership after moves/source destruction; malformed widths, transforms,
codebooks, bit widths, byte spans and overflow refuse. Synthetic Gate/Up
M1/M4 matches separate SmallM plus the existing SiLU-through-F16/multiply
rounding bit-exactly. The test checks one projection call and reused packed/
source-map resident pointers; merged host staging is absent after upload.

`test_xpu_qwen_exl3_fp16`: **2 cases, 4,287 assertions, exit0**. Its paged
hybrid P4/D1, pooling and unpaged entries now witness **111 SmallM calls**,
including **18 two-source Gate/Up groups** (14 M4,4 M1), versus129 separate
calls previously. Trace has301 native provider/tensor rows, zero old
Exl3Gemm selections, zero BF16 tensor arguments and zero CPU-reference
selections. M1/M4 head relative norm remains0.000443042. These are synthetic
H128 model checks, not real-checkpoint or complete-target parity.

Receipts, source/build-path hashes and trace counts are recorded in
`reference_B.S1_grouped_gate_up`. Earlier changed source/binary hashes remain
historical observations; immutable oracle artifacts are unchanged.

**Remaining:** real grouped Gate/Up producer capture/replay (merged N may alter
split-K geometry), QKVZ/QKV grouping, full loader/upload/head and real block/
eager P128 D1/D64. The map is now model-owned and resident, but VT's eager
source-ID validation still reads back/waits. Graph/multi-queue lifetime and
performance have not been qualified. Production remains stopped.


## 16. S1 real full Gate/Up producer gate

Both complete checkpoint shard hashes and the immutable S0b gate fixture were
rechecked before extracting the real layer0 Up projection. New independent
`capture_grouped.py` concatenates Torch output tiles, stacks each source input
transform, builds the whole-block source map and runs the unchanged pinned
producer `_C.so`. It performs no dense reconstruction. Host guards pass
**5 tests, exit0**. Existing single-projection capture tooling is unchanged.

The producer captures K5120/N34816/S2/4bpw at **M1/4/12/16/128**. Each ordinary
output is bit-exact with its allocation-observed invocation. Actual splits
are **4/3/3/3/1**, padded rows **1/8/16/16/128**. The capture SHA and all24 tensor
hashes/geometry were checked before the sequential native replay.

Focused regular engine build: exit0.
`test_xpu_exl3_smallm '--test-case=*real grouped*'`: **1 case,112 assertions,
exit0**; the other2 cases were deliberately filtered. Current shared loader
and native packed merger exactly match the independently merged producer
operands. Both active source Hadamards, actual F32 split Partials and F16
outputs are **bit-exact at all five M**; relative norm and maximum absolute
error are0, outputs finite and nonzero. Poisoned native padded rows become0.
The actual model-resident grouped wrapper gives the same exact outputs, drops
merged host staging after first upload and reuses its resident across all M.
Separate source trellis GPU residents remain absent.

Trace witnesses **10 native grouped calls, all S2/4bpw/mul1** (2 GEMV,8 DPAS),
zero BF16 arguments, zero CPU-reference/old Exl3Gemm selections. No thresholds
were changed. Source/capture/build/binary/test/trace identities are recorded in
`reference_B.S1_real_grouped_gate_up`; overwritten test source/build-path
binary hashes are historical observations. Immutable S0b artifacts are intact.

**Remaining:** QKVZ/QKV grouped wiring and real captures, independent producer
SwiGLU/entire MLP, full head/upload, real block and eager P128 D1/D64. This real
grouped-linear pass does not qualify complete model/state arithmetic or
performance. VT still reads back the resident map for validation. Production
is inactive/dead/MainPID0; both sequential GPU jobs finished with exit0.


## 17. S1 model-owned QKVZ and QKV wiring

Scoped XPU FP16 SmallM now builds/reuses model-owned QKVZ S2 and QKV S3
packed residents with every source input transform retained. GDN borrows
row-strided mixed/Z views from one output owner through Conv/gated norm.
Full attention borrows packed views for stride-aware consumers and makes
exact native Copy outputs for contiguous consumers. Its packed owner stays
alive alongside all views/copies. No original source residents are built by
these actual model calls. Legacy/unscoped/large-M execution is unchanged.

Focused regular engine build exit0. `test_xpu_qwen_exl3_fp16`: **3 cases,
4,357 assertions, exit0**. The new direct case checks both groups at M1/M4
against each separate source, including all rows/columns of native strided
Copy and resident reuse. The hybrid paged P4/D1 (merged/split BA), pooling
and unpaged cases pass with explicit group/source-residency assertions.

The actual model queues witness **84 SmallM calls**, versus111 before this
step:9 QKV S3,9 QKVZ S2,18 Gate/Up S2,48 single-source calls. Both new families
use M4 seven times and M1 twice. Nine native Copy calls materialize contiguous
Q/K/V; other model QKV consumers use packed views. The separate diagnostic
case adds14 calls, making98 total. All298 provider/tensor rows are native,
with zero BF16 arguments, CPU-reference or old Exl3Gemm selections. Head
M1/M4 relative norm remains0.000443042. Thresholds were unchanged.

Receipts/current source/build-path binary hashes are in
`reference_B.S1_grouped_qkvz_qkv`. Changed earlier source/binary hashes remain
historical; immutable producer captures are intact.

**Remaining:** real QKVZ/QKV producer capture/replay, full loader/upload/head,
producer nonlinear and real block/state comparison, eager P128 D1/D64 and
all S2-S6 gates. The map is resident but still read back for validation;
graph/multi-queue lifetime/performance are not qualified. This synthetic
model check does not qualify the complete pinned checkpoint. Production
is inactive/dead/MainPID0 and the focused GPU job finished with exit0.


## 18. S1 real QKVZ and QKV producer gates

The complete checkpoint shard containing these five sources was rehashed
against the pinned payload identity before extracting full projections. The
unchanged grouped capture tool/original producer binary recorded layer0 QKVZ
(K5120,N16384,S2) and layer3 QKV (K5120,N14336,S3), both4bpw, at
**M1/4/12/16/128** without dense reconstruction. Ordinary and allocation-
observed producer outputs match exactly. QKVZ splits are8/6/6/6/2; QKV
splits10/7/7/7/3; padded rows1/8/16/16/128. Each capture SHA and all24 tensor
geometry/hash receipts were verified before the native job.

Focused regular engine build exit0.
`test_xpu_exl3_smallm '--test-case=*real attention groups*'`: **1 case,
248 assertions, exit0**, with3 other cases deliberately filtered. Current
shared loader mapping lifetime is exercised after each source file handle
is destroyed. The native packed/transforms/map exactly match the independent
producer merge. All active source Hadamards, actual F32 split Partials and
F16 outputs are **bit-exact at all five M for both families**. Relative norm
and maximum absolute error are0; outputs finite/nonzero. Poisoned padded
rows become0.

The actual model translation-unit seam initializes its own empty resident
cache and returns the same exact outputs. First-upload completion precedes
host staging release; packed resident is reused across all M. Separate
source GPU residents remain absent. Trace witnesses **20 native SmallM calls**
(10 S2,10 S3;4 GEMV,16 DPAS), zero BF16 arguments, CPU-reference/old Exl3Gemm
selections. No thresholds changed. Receipts/current source/build-path hashes
are in `reference_B.S1_real_qkvz_qkv`; immutable earlier captures are intact.

**Remaining:** complete6bpw head, complete loader/upload, independent producer
nonlinear and real block/state stages, eager P128 D1/D64 and all S2-S6 gates.
These grouped-linear gates do not establish complete model/state parity or
performance. Resident source-map validation still reads back/waits.
Production remains inactive/dead/MainPID0; both oracle jobs and native replay
were sequential and finished with exit0.

## 19. S1 complete 6bpw target head

The complete second checkpoint shard was rehashed against its pinned identity
before extracting all target `lm_head` columns. The explicit single-source
capture mode requires exactly one complete source and avoids an extra host
packed concatenation. Its focused host guard test passes **6 tests, exit0**.
The unchanged pinned producer captures K5120/N248320/6bpw at M1/4/12/16/128,
with splits1 throughout and padded rows1/8/16/16/128. Ordinary and allocation-
observed outputs match bit-for-bit. Capture identity/runtime/tool pins and
all24 tensor geometry/hash receipts were verified before native execution.

Focused regular engine compile/link exit0.
`test_xpu_exl3_smallm '--test-case=*full 6bpw head*'`: **1 case,
125 assertions, exit0**;4 unrelated cases deliberately filtered. The actual
shared native loader preserves packed/transforms/map bytes. All953,548,800
packed GPU bytes match the source after bounded64MiB chunk readback. The test
uses one packed GPU resident; actual model linear calls reuse it, with the
dense head owner remaining empty.

Active input Hadamards, actual F32 split parts and F16 outputs are **bit-exact
at all five M** through both direct VT and the actual model linear seam.
Relative norm and maximum absolute error are0; outputs finite/nonzero.
Poisoned padded rows become0. Trace witnesses **10 native SmallM calls**,
2 GEMV/8 DPAS, groups1/bits6/splits1 throughout, zero BF16 arguments,
CPU-reference or old Exl3Gemm selections. No thresholds changed.

Receipts/current mutable source/tool/build-path binary hashes are recorded in
`reference_B.S1_full_target_head`. Earlier observed hashes remain historical;
immutable captures retain their original identities. Production remains
inactive/dead/MainPID0; oracle and native jobs were sequential and exit0.

**Remaining:** complete native model loader/GPU upload, independent producer
nonlinear and real block/state stages, eager P128 D1/D64 and all S2-S6 gates.
This complete-head projection check does not qualify whole-model/state parity
or performance. Resident source-map validation still reads back/waits.

## 20. S1 real Gate/Up SwiGLU eager boundary

The isolated `capture_swiglu.py` verifies the immutable real layer0 grouped
capture and all its tensor identities, installed13 active guards and checkpoint
metadata. Exact pinned profile EngineArgs resolves `custom_ops=['none']` and
`SiluAndMul.forward_native`, without creating a model worker. The original
eager method (native compilation disabled for this arithmetic gate) consumes
the immutable real Gate/Up outputs at M1/4/12/16/128. Its SiLU narrows through
FP16 before multiplication; the unrounded FP32 SiLU diagnostic differs at every
M. Installed activation/model/CustomOp source hashes and profile are recorded.
The receipt's installed XPU library hash is provenance, not evidence that this
selected native method invokes that library. All10 capture tensors were
geometry/hash-checked before native execution.

Focused compile/link exit0.
`test_xpu_exl3_smallm '--test-case=*real Gate/Up SwiGLU*'`: **1 case,
105 assertions, exit0**,5 unrelated cases filtered. Direct native SiLU and
actual model GateUp (projection+activation) match the producer **bit-exactly
at all five M**, relative norm/max error0, finite/nonzero outputs. The model
seam lazily initializes/reuses its grouped resident; original source GPU
residents stay absent and no dense Gate/Up owner is populated. Existing native
activation arithmetic needed no change and no thresholds changed.

Trace witnesses5 native grouped SmallM and10 native SiluAndMul calls, zero
BF16 arguments, CPU-reference or old Exl3Gemm selections. Receipts/current
mutable source/build-path identities are in `reference_B.S1_real_swiglu`;
prior observed identities/captures remain preserved. Oracle/native GPU jobs
were sequential and finished exit0.

**Remaining:** down projection/full MLP, real normalization/Conv/GDN/attention
and block/state comparisons, complete model loader/upload, eager P128 D1/D64
and all S2-S6 gates. Compiled activation fusion/graphs and performance are not
qualified by this isolated eager arithmetic comparison.

## 21. S1 actual layer0 P128/D1 capture and FP16 Gemma residual norm

Both complete checkpoint payload hashes were revalidated before one isolated
worker capture. `capture_block.py` uses the pinned image/profile with explicit
S1 overrides: no MTP, compilation or graphs; `custom_ops=['none']` preserves
the resolved native arithmetic. It runs one ordinary128-token prompt/two-token
greedy continuation, resets prefix cache, then observes the same request with
bounded read-only layer0 module hooks. Both greedy outputs are **[13,198]**.
This observer check is not bitwise whole-model parity or performance.

The capture records **44 tensors**, all identity/geometry/hash-verified before
native execution: real hidden/residual/norm, QKVZ/BA, core/gate/gated norm,
out projection, Gate/Up/SiLU/down, layer outputs and active Conv/FP32 SSM states
for P128 and D1. Cold initial state is ignored and explicitly not read or
invented as zeros. Both P128-after state tensors equal D1-before **bit-for-bit**.
Logical active Conv layout is[3,10240], SSM[48,128,128]. Native numerical mixer,
Conv and SSM replay remains pending; collecting these values is not that gate.

Installed `GemmaRMSNorm.forward_native` calls the producer IR
`fused_add_rms_norm`: it normalizes the **unrounded FP32 x+residual sum** while
returning the new residual narrowed separately. The native XPU FP16 Gemma leaf
previously normalized the stored/reloaded FP16 residual. A new narrow/wide
(2/5120-column) witness, including output aliases, fails before the fix:
**2 cases,89 assertions,8 failures, exit1**. Captured real norm comparison
already passed its numerical band, but exposed186,652 differing P128 post-norm
half values and2,368 at D1; a local tolerance pass did not excuse the wrong
rounding boundary.

The FP16 Gemma leaf now keeps both original operands through reduction and
normalizes their FP32 sum, storing the narrowed residual in the output pass.
No extra scratch/kernel is introduced. Other dtype/weight modes retain their
existing contract. Focused compile/link exit0. Unchanged focused test:
**2 cases,89 assertions,exit0**. New residual values are exact at both steps.
Actual normalization results (relative norm/max absolute/differing halves):

| Stage | Relative norm | Maximum error | Differing FP16 values |
| --- | ---: | ---: | ---: |
| P128 input norm | 2.79498e-6 | 0.000976562 | 17 |
| P128 post-attention norm | 8.61468e-6 | 0.000488281 | 44 |
| D1 input norm | 0 | 0 | 0 |
| D1 post-attention norm | 0 | 0 | 0 |

These remaining P128 differences are explicit; norms are not all bit-exact.
The fixed producer IR norm tolerance remains rtol0.002/atol0.01 and relative
norm<0.002. No threshold changed after the failure.

Focused affected synthetic paged hybrid model regression (merged/split BA):
**1 case,1,602 assertions,exit0**. Its trace shows56 native SmallM calls and30
native RMSNorm calls. Existing RMSNorm CPU-agreement/alias regression:
**1 case,63,928 assertions,exit0**,8 other cases filtered. Norm witness/capture
trace has10 native RMSNorm calls. Both traced jobs have zero BF16 arguments,
CPU-reference or old Exl3Gemm selections. No full suites were run.

Receipts/current source/build-path identities and preserved pre-fix failure
are in `reference_B.S1_real_block_capture_and_Gemma_norm`; immutable earlier
captures remain intact. Production is inactive/dead/MainPID0; oracle and
native GPU jobs were sequential and are terminal. An initial test-only compile
failure (missing declaration/same-line doctest names) was corrected before
the pre-fix functional comparison; both build logs remain preserved.

**Next:** native replay of the captured GDN mixer, active Conv/FP32 recurrence
and full MLP/block on identical inputs, then complete loader/GPU upload and
eager P128 D1/D64. S1 remains in progress and all S2-S6 gates remain required.

## 22. Real layer0 mixer/state and full MLP replay — bounded mixed result

The reusable `RunGdnBlockPaged` wrapper now accepts an optional explicit model
activation dtype. Real EXL3 replay passes F16; legacy callers retain their
previous default. Native slot4, cold-state flags, positions and Conv history
geometry match the captured step. Inactive slots have nonzero witnesses.

Focused compile/link succeeded after a test-fixture construction error was
corrected (both build logs preserved). Mixer/MLP replay: **2 cases,1 pass,
1 failure;446 assertions,445 pass,1 failure;exit1**. The full layer0 MLP
Gate/Up -> SwiGLU -> Down is bit-exact at both captured P128 and D1 inputs.
This is a component replay, not full decoder-block execution.

P128 GDN mixer relative norm0.000151254/max absolute0.0078125 satisfies its
fixed band. Conv cache is exact and inactive state slots remain unchanged.
FP32 SSM relative norm0.000565168/max absolute0.0196743 fails the existing
pointwise rtol1e-4/atol1e-5 check. GDN D1 was not reached after that fatal
failure. No tolerance changed. Trace has6 SmallM calls and zero BF16 tensor
arguments, CPU-reference selections or old Exl3Gemm calls.

## 23. Original GDN split capture and explicit FP16 prefill preparation

`capture_gdn.py` invokes the original fused `gdn_attention` and original split
`causal_conv1d_non_spec` / `gated_delta_rule_non_spec` from the pinned image.
Both independent executions consume the immutable real QKVZ/BA operands,
carry their own state from P128 into D1 and exactly reproduce the full-worker
Core, Z, Conv and SSM endpoints: **all8 endpoints exact,exit0**. Slot4 and the
original first-dimension cache strides are retained; inactive capacity is
bounded to5. No original math is replaced. All28 captured tensor geometries
and hashes were checked. Capture SHA:
`e64a2da6f89e9c0945e32730e4232255a41e44c1f6b15c3f8d873d9f09adb44c`.

The actual operator lives in `_xpu_C.abi3.so` with Xe2 helpers in
`libgdn_attn_kernels_xe_2.so`; both binary hashes are recorded. The local
release/0.1.15.4 source (ddf336d86e3c8602888572a3502f951abd51df12) is a
version-matched source reference; exact correspondence of source and wheel
is still not asserted. Python FLA gating is not the actual original XPU route.

The failed native replay's mixed QKV input is byte-exact. V is also exact;
normalized Q, after applying/narrowing the legacy separate scale, differs
in92,132 values (relative norm0.000396869). The first visible difference is
therefore in the Conv/normalization preparation, rather than projections.
The producer-style preparation normalizes FP32 Conv output and folds the
Q scale into the inverse norm before its FP16 store.

A typed `GdnPostConvArgs` now provides an explicit XPU FP16-prefill mode;
legacy `L2NormArgs` callers convert to the unchanged default mode. The new
mode uses four contiguous FP32 features per lane,16-lane subgroup partials,
scaled Q before storage, FP32 beta and `log(1+exp)` softplus. Other backend
signatures were adjusted mechanically; only CPU/XPU are compiled by this
focused target. The new numerical mode is rejected outside XPU. It is a
prototype and **is not wired into actual model execution**.

Both focused builds exited0. Exact prep comparison before explicit FMA:
**Q16/K6 differing FP16 values, V exact, all6,144 beta values exact;exit1**.
Explicit contracted sum-of-squares comparison: **Q15/K5 differing values,
V/Beta exact;1 case failed,6,154 assertions with6,152 pass and2 failures;
exit1**. Exact equality remains the requirement; no threshold was relaxed.
G math and full recurrence are not qualified by this prep test. Failing logs
are retained; later diagnostics use new output names.

Affected default paged hybrid regression: **1 case,1,602 assertions,exit0**.
Its trace has56 native SmallM and30 RMSNorm selections. Both prep jobs and
the regression have zero BF16 tensor arguments and CPU-reference selections.

Current identities, captures, failed checks and preserved historical source/
build hashes are in `reference_B.S1_real_GDN_MLP_and_prefill_prep`.
Production remains inactive/MainPID0; oracle and native jobs were sequential
and are terminal. **S1 remains incomplete; S2-S6 remain required.** Next:
isolate the remaining FP32 Conv/SiLU versus Q/K normalization rounding,
qualify the original prefill recurrence and native P128->D1 state, then the
real complete block and full eager P128 D1/D64.

## 24. Actual Conv binary and Q rounding — Q exact, K still failing

Direct embedded BMG ELF metadata from the pinned `_xpu_C.abi3.so` corrects
the earlier subgroup attribution: actual half/Width4/Tile8/ReorderTrue Conv
uses SIMD32 without a required-subgroup attribute. The separate recurrence
helper uses SIMD16. Both producers identify oneAPI2026.1.0; the native
builder is2026.1.1. Exact release-source/wheel correspondence remains unproven.
Original profiled fused/split replay again reproduces all8 worker endpoints.

The original Conv ISA computes Q inverse with SQRT(sum+eps), SQRT(D), MUL,
then INV. Multiplying two RSQRT results preserves a different rounding
boundary. A Torch-free original tiled-body diagnostic initially had Q14,
K/V/Z/Conv-history0 differences; expressing the observed Q inverse makes
all five outputs exact, exit0. This is one bounded cold P128 probe, not
model qualification. Its source assumes four norm scratch subgroups while
the actual SIMD32 launch has two; do not transplant uninitialized extra slots.

The native prototype now uses SIMD32 and explicit native Q root/reciprocal
operations. The binary Q quadratsum order and an independent FP32 replay
support starting with feature1, contracting feature0, then2 and3. That order
reproduces all262,144 Q values in the alternate original F32-input diagnostic.
The alternate dtype route is not a substitute for the original F16 gate.
No enumerated K order qualified; no tolerance or root-expression workaround
was adopted. Host `sycl::half` conversion in one external FP32 probe gave
misleading half counts; independent IEEE half conversion yields Q14/K0.

Focused native build/link: exit0. Preserved native comparisons:

| Version | Q differences | K differences | V differences | Result |
| --- | ---: | ---: | ---: | --- |
| SIMD32 before Q correction | 17 | 2 | 0 | exit1,6152/6154 assertions pass |
| Binary Q inverse before Q order | 4 | 15 | 0 | exit1,6152/6154 assertions pass |
| Current binary Q inverse and Q order | 0 | 15 | 0 | exit1,6153/6154 assertions pass |

All6,144 beta values remain exact. The current trace has zero BF16 tensor
arguments and CPU-reference selections. This mode is still a prototype,
not wired into real model execution; g and recurrence remain unqualified.
The previous default regression pass predates this source/build.

All failed probes/builds and current source/build identities are preserved in
`reference_B.S1_GDN_Conv_binary_and_Q_rounding`. Production remains stopped.
**Next:** resolve K, independently qualify original FP16 recurrence and native
P128->D1 state, then the real block and full eager model. S1 remains incomplete;
all S2-S6 gates remain required.

## 25. Uniform roots and original recurrence — bounded exact replays

The native prep test can now use existing `VT_DUMP_ACT` to capture its
FP32 Conv stage, refusing an existing receipt path. Focused build exit0;
unchanged exact test initially still fails K15 (6153/6154 assertions).
The instrumented original tiled body retains exact original F32 Q/K endpoints
and exposes unnormalized Q/K/V: all1,310,720 native FP32 Conv values are exact.
The first external raw dump attempt wrote empty files from untouched shared
USM; v2 copies to host storage and checks I/O, preserving the failed files.

Both original Q/K quadratsums use feature order1,0,2,3. Direct native replay
has exact sums and Q inverse, but650 K inverses differ. Native subgroup-leader
root evaluation plus broadcast reproduces the original uniform scalar ISA:
all Q/K sums, inverses and262,144-element F32 outputs are exact. This is an
observed B70/compiler execution difference; the original F16 gate remains
the required native test.

The explicit native prep mode now uses that order and leader/broadcast.
Focused compile/link exit0; unchanged original F16 operand/output comparison:
**1 case,6154 assertions,all pass,exit0**. Q/K/V and6144 beta values are exact.
Affected default paged hybrid regression: **1 case,1602 assertions,exit0**.
Both traces have zero BF16 tensor arguments and CPU-reference selections.
The prep prototype is still not wired into actual model execution; its g
output has not been independently qualified by this test.

A separate Torch-free original recurrence-body replay uses immutable real
Q/K/V/B/A,191-row padding,checked layer0 A_log/dt_bias,slot4,F32 state stride
819200 and the original cold-state flag. Existing local SYCL-TLA archive
headers are read-only diagnostic dependencies. Their declared87f685 commit
is not newly verified against Git; the main engine pin guard was not bypassed
and Xe2 was not enabled in that engine build. Unused benchmark includes and
Torch host glue are omitted; original kernel math is retained.

Initial component replay fails: Core300 half differences,SSM15594 float
differences,relative norm1.76714e-5,max0.00414562;44 state values fail the fixed
pointwise band. Gate prefix616 differences;inactive state is exact. Replaying
only recurrence stages on identical captured gate prefixes gives Core/SSM
bit-exact,exit0. That stage replay does not qualify gate preparation.

Actual producer Prepare ISA contracts softplus*A_log_exp into the local
prefix sum and contracts the reverse-prefix subtraction. Making those FMA
boundaries explicit in the external original-body diagnostic reproduces
**gate prefix,Core,FP32 SSM and inactive slots exactly,exit0** from the original
raw A input, without substituting the reference prefix. This is still one
bounded P128 component replay, not native model integration or D1 continuity.
Missing unused header,Torch host-helper and SPIR-V-extension build failures
are retained alongside the corrected builds and all successful/failed runs.

Current source/build identities and these receipts are recorded in
`reference_B.S1_GDN_uniform_inverse_and_original_recurrence`. Production
remains stopped. **Next:** integrate original raw-gate/FMA prefix and FP16
recurrence with genuine pinned dependency identity; then native P128->D1,
the full real block and complete eager target. S1 and all S2-S6 remain required.

## 26. Native raw-gate integration and the first real BA difference

The genuine SYCL-TLA commit/tree objects now verify914 local source/header
files against87f6850680a580654b9ea2c80dbc01aeb36ad231, zero mismatches.
The original archive is unchanged. The separate read-only reference contains
only matched files and genuine detached Git metadata; the main CMake pin
guard was not bypassed. Independent GDN configure/compile both exit0, with
Xe2 GDN enabled and unrelated FP8 prefill disabled.

The explicit native `GdnPrefillRawGate` operator accepts already-scaled FP16
Q, FP16 K/V/raw-A, FP32 beta/A_log/dt_bias and caller-owned FP32 state.
It packs the producer's191-row capacity, rounds bias to the original FP16
boundary and applies the verified prefix FMAs. Its kernel namespace is
distinct from the older scaled-gate experiment; it uses the completion-aware
native workspace and records both output and state writes. The component
test uses immutable original operands, real strided BA A, and its own zeroed
state, without substituting the captured prefix or state.

Strict global math gives Core2912/state41692 differences; contraction alone
gives3749/40247. The pinned icpx driver confirms that its original default
also enables approximate functions, unsafe/reciprocal math, signed-zero and
denormal behavior. Source-scoped `-fp-model=fast` reproduces those original
frontend options; unrelated operators remain strict. Focused build exit0;
native component **1case/11assertions, all pass, Core/state bit-exact, exit0**.
Failed builds and comparisons are retained, including compiler termination
under parallel8GiB and the successful serial12GiB build.

Actual paged EXL3 pure P128 now retains FP32 Conv through qualified prep and
uses the raw-gate recurrence with existing state gather/zero/scatter. The
unchanged real continuation test still fails the strict FP32 state band;
**D1 has not been reached**. Stage capture localizes the first difference to
the dense BA projection: A20/B21 half values differ, while mixed QKV,
all1,310,720 FP32 Conv values, and normalized Q/K/V are exact.

Selecting the existing native dense oneDNN FP16 operator for EXL3 BA reduces
this to A3/B3 differences. Mixer remains within its original band, relative
4.97493e-5/max0.0078125. State relative1.74627e-5/max0.00263786 still fails
rtol1e-4/atol1e-5; **1case,413/414assertions,exit1**. This route uses dense
weights, not GPTQ quantization. The affected default paged regression passes
**1case/1602assertions,exit0**; trace records native RawGate and dense-FP16
selection, with zero BF16 tensor arguments or CPU-reference selections.

An isolated original Torch2.13.0+xpu replay identifies producer oneDNN3.12,
while native uses3.13. Plain merged/split and N-contiguous diagnostic BA
replays also differ from the immutable original P128 worker output; D1 BA
matches. Installed source selects UnquantizedLinearMethod -> F.linear,
with conditional N-contiguous relayout disabled by the observed defaults.
These observations do not establish which live-worker operand/layout or
execution setting explains P128. Do not replace the original worker fixture
with the easier isolated result or weaken the state gate.

Receipts and current identities: `S1_GDN_raw_gate_native_integration` in the
manifest. Production remains inactive, all GPU jobs terminal. **Next:**
observe actual worker BA input/weight bits, layouts and selected implementation
with a same-state replay; resolve BA, then native P128->D1, complete real block
and eager target. S1 and all S2-S6 remain incomplete.

## 27. Original live-worker BA repeat and native FP16 D1 integration

The additional pinned-worker observer captures the actual BA input, weight,
layouts and implementation before replaying the unmodified quant method on
the same live operands. Input is F16 [128,5120], strides [5120,1], and equals
the same capture's input-norm output. Weight is F16 [96,5120], strides
[5120,1], and its bytes equal the checkpoint's B-then-A weights exactly
(SHA256 e5dbd32b6b1adb82dc5803a7239b9ae005ed621af21a89984512a111022760e9).
UnquantizedLinearMethod selects default_unquantized_gemm -> F.linear, with
deterministic algorithms disabled. Capture exits0; ordinary/observed greedy
IDs both remain [13,198].

P128 BA differs5 half values on the same live-operand replay and7 on a
contiguous clone replay; both D1 replays have zero differences. The new
original capture differs8 P128 BA values from the immutable original capture,
despite exact input-norm and QKVZ values. Original-original P128 state has
relative1.746171136e-5/max0.00263786316 and fails122 values at the existing
rtol1e-4/atol1e-5. Thus the original worker itself is not bit-stable at BA and
does not always satisfy this frozen pointwise state gate. No capture or
tolerance has been replaced, and full native P128 qualification remains open.

The separate Torch-free original P128->D1 body replay produces zero numerical
differences in P128 state, D1 core/state and inactive slots, exit0. The native
XPU PackedDecode now incorporates that original SG32 body: raw F16 Conv Q/K
are normalized internally in F32, gates/update/read remain F32, and core is
stored F16. Selection is limited to the qualified C1 EXL3 F16 geometry
Hk16/Hv48/D128/Kw4 with F32 state. The BF16 selector is unchanged for other
backends. P128 and D1 reserve identical completion-owned workspace capacity,
including decode-first calls; state graph writes and metadata are recorded.

Focused native build exits0. The product P128->D1 chain uses its own computed
P128 state, strided BA views and a poisoned physical packed-row tail:
**1case/19assertions pass, zero numerical core/state/inactive differences,
exit0**. Trace records RawGate, StateScatter and PackedDecode, with zero
BF16 arguments or CPU-reference selections.

An independent actual model D1 test restores the original Conv/SSM starting
state and runs projections, Conv, packed recurrence, gated norm and output
projection: **1case/26assertions pass, mixer/Conv/SSM values exact, exit0**.
Its trace confirms native PackedDecode and dense F16 BA selection. It does
not qualify native P128 continuity. The affected default paged regression
also passes **1case/1602assertions,exit0**; all three traces contain zero BF16
arguments or CPU-reference selections.

Receipts and current source/build identities are recorded in
`reference_B.S1_GDN_FP16_D1_and_live_BA_repeat`; changed mutable records remain
historical observations. Both immutable original capture hashes are
rechecked unchanged. **Next:** calibrate original same-checkpoint/route
reproducibility and qualify full real P128/block/eager-target execution per
plan section6, retaining the failing strict continuation gate. S1 and all
S2-S6 remain incomplete. Production stays stopped.

## 28. Complete eager native P128/D1 target and original reproducibility

The bounded original-target observer delegates the existing compute_logits
exactly once, preserving its input/result objects. It adds the actual gathered
F16 [1,5120] head input and full [1,248320] logits to existing layer0 captures.
Three cold-prefix P128/D1 repeats run in one pinned eager worker, with MTP and
graphs disabled. Five focused host tests pass, including delegation identity,
known TV/KL distributions and unchanged pointwise-state failures.

Original capture exits0 and shuts down before native GPU work. Ordinary and
all observed greedy IDs are [13,198]. Across all six original logit comparisons,
maxTV0.0026584763/maxKL0.0000223777/top10overlap10. Final head inputs are not
bit-stable; some original-original layer0 state values also fail the existing
strict pointwise band. This is a bounded same-worker baseline, not permission
to weaken an existing gate or a universal reproducibility claim.

The new focused product-loader/target test loads all64 real layers and the
full6bpw248320-column head, uses F16 activations/F32 GDN state/E4M3 KV with
page1600 and unit scales, and executes P128 then D1 from its own persistent
states. Its D1 input is the original first token13, which also equals the
native P128 greedy token. Unwritten KV capacity is poisoned. No captured
recurrent state is substituted into the native target.

Initial test compile errors are retained; corrected focused build exits0.
Native target **1case/195assertions pass,exit0**. Against original repeats0/1/2:

| Phase | TV | KL | Top10 overlap | Native greedy ID |
| --- | --- | --- | --- | --- |
| P128 | .00203102 / .00206428 / .00192130 | .0000139761 / .0000142226 / .0000120002 | 10 / 10 / 10 | 13 |
| D1 | .00398878 / .00338876 / .00321305 | .0000512979 / .0000386444 / .0000348535 | 10 / 10 / 10 | 198 |

These satisfy the plan's existing TV.02/KL.002/top10>=9 investigation thresholds.
The trace records514 native grouped linears,48 original RawGate prefills,
48 PackedDecode calls and32 FP8 cache writes. Both full-head calls have output
[1,248320], confirming final-row gather before head evaluation; there is no
M128 head/logit temporary. Zero BF16 tensor arguments, CPU-reference or old
scalar Exl3Gemm selections are observed. This demonstrates the complete eager
native target in this bounded P128/D1 case, not a server/performance release.

Reference logits, native raw F32 logits, immutable source snapshots, build/test
logs and trace identities are recorded in
`reference_B.S1_eager_target_P128_D1_and_reproducibility`. The older strict P128
block/state continuation gate remains failing and unchanged. **Next:** extend
bounded original/native target continuation through D64; finish matched-input
block/attention/FP8 writer/state and other remaining S1 checks before S2-S6.
S1 remains incomplete. Production stays stopped; both GPU jobs are terminal.

## 29. Complete D64 execution and directly witnessed reference variability

The bounded target observer now supports P128 plus64 decode heads. It removes
the existing layer0 block hooks after D1, retaining later head observations
without pretending to capture later block states. New captures are separate;
all earlier fixtures remain unchanged. Nine focused host tests pass, covering
D64 bounds, trace-token validation, full logit rows and actual prefix witnesses.

The first original D64 capture exits0 with65 full-vocabulary rows. Ordinary and
observed greedy sequences differ later; both are retained. The native test uses
the observed sequence, gathering the final prompt row before the head and
continuing through all64 decodes from its own persistent states. Focused build
exits0; **1case/804of807assertions pass,3fail,exit1**. Every native greedy choice
matches the initial observed original sequence on that prefix. Failed triggers:

| Step | TV | KL | Top10 overlap | Failed threshold |
| --- | --- | --- | --- | --- |
| D11 | .0227991 | .00194376 | 10 | TV |
| D29 | .0442485 | .0584163 | 9 | TV and KL |

Trace records65 full-head calls, all [1,248320],48 original RawGate prefills,
3072 PackedDecode calls and zero BF16 arguments, CPU-reference or scalar
Exl3Gemm selections. All65 raw native F32 logit rows are preserved. This is
complete bounded execution, not a passing D64 numerical gate or performance
measurement. The original strict P128 GDN state failure also remains unchanged.

The pinned original engine already supports enable_trace_replay with
SamplingParams.trace_decode_token_ids: it overwrites sampled IDs while leaving
model logits/logprobs unchanged. Three exploratory replays first retain the
same emitted sequence; their results and installed API source are preserved.
Equal emitted IDs alone are insufficient to prove the model's actual inputs.

The strengthened observer therefore copies actual embedding token IDs and all
three position axes for every step. Three further original trace replays exit0,
and all65 actual prefixes/positions match in each. Cold initial GDN state is
explicitly unconsumed. Physical request slots are4/1/4 across repeats: this is
the same cold logical prefix, not identical full physical-state restoration.

Even with directly verified tokens/positions, the195 original-original logit
comparisons have maxTV.1352963609/maxKL.0868865655/minTop10overlap9, with9
investigation triggers. D11/D29 also fail in these original comparisons. The
host comparison of the existing native rows against these three directly
witnessed references completes195 comparisons and retains8 triggers. Report
generation exits0; it does not replace the failing exit1 native functional test.
No threshold is enlarged and no reference capture is substituted to green it.

Source snapshots, all captures, the65 native rows, build/test logs, installed
trace API and host comparison hashes are recorded in
`reference_B.S1_D64_target_and_actual_prefix_reference_variability`. **Next:**
qualify the first real full-attention block and FP8 writes on identical original
operands, then resolve remaining state/D64 investigations using matched-stage
and physical-state replays. S1 and all S2-S6 remain incomplete. Production
stays stopped; all current GPU and comparison jobs are terminal.

## 30. Real layer3 attention capture and identical-input FP8/core qualification

The new bounded observer captures48 actual original layer3 tensors at P128/D1:
input norm, merged QKV, Q/K norm, post-RoPE Q/K, V/gate, ungated attention,
gated attention, mixer/projection outputs and active cache bytes/metadata.
Five focused host tests pass, covering single original delegation/result
identity, invalid metadata, physical nonzero blocks and page crossings.
An initial capture exits1 because the observer used `prefix` instead of the
installed Attention's `layer_name`; its log is retained. Corrected capture
exits0, ordinary/observed IDs both[13,198], unfused QK/RoPE route observed.

The actual uint8 cache view is[212,4,1600,512], strides
[3276800,512,2048,1]. K/V are interleaved within each head; active physical
block is1 and both per-tensor scales are1. Only128/129 written logical rows
are gathered after validating actual block table/slots/positions. No cold or
unused cache contents are captured. P128-after equals D1-before exactly, and
D1 leaves the existing128 rows unchanged.

The native replay initially exits1 at the head-contiguous API guard before
arithmetic. XPU FP8-write/paged-read validation now admits bounded interleaved
head views; other providers retain their prior contract. The XPU writer and
DPAS prefill reader now address heads through actual head strides. Existing
contiguous layout arithmetic is preserved.

Focused build exits0. The unchanged-original-input replay uses captured
post-RoPE Q/K/V, actual slots/block/lengths and its own persistent poisoned
physical cache. **1case/37assertions pass,exit0**. P128 and D1 both have zero
different K bytes, V bytes and inactive capacity bytes. Native attention
max errors are0.001953125/0.0009765625, with zero failures at the unchanged
rtol0.01/atol0.003 band. Trace selects native prefill at128 rows and split
decode at1 row; two FP8 writers/two paged readers, zero BF16 arguments or
CPU-reference selections.

Three existing focused regressions exercise raw KV writes, E4M3 conversion
and appended-query paged attention: **3cases/2275assertions pass,exit0**.
These include existing contiguous pages and padded/duplicate slots; they are
not a full suite. Receipts and immutable source snapshots are recorded in
`reference_B.S1_attention3_identical_input_FP8_and_core`.

This qualifies the bounded writer/attention core on identical original
operands. It does not yet qualify native QKV->QK norm->RoPE or the complete
model-owned full-attention mixer, whose cache wrapper currently uses its own
NHD layout. **Next:** compare those model stages against this new reference,
then resolve the remaining GDN-state/D64 gates. S1 and all S2-S6 remain
incomplete. Production remains stopped; current GPU jobs are terminal.

## 31. Actual model-owned attention mixer and first differing preamble

A thin owning full-attention seam exposes the existing private
BuildFullAttnStepDevInputs/MaybeBuildAttnCosSin/FullAttnBlockPaged, analogous to
the existing GDN seam. It implements no second mixer arithmetic. Scoped eager
stage dumps now capture actual grouped row-strided QKV views via VT Copy and
the existing (step,layer) manifest; they are inert when VT_DUMP_ACT is unset.

Focused build exits0. The actual real layer3 mixer is loaded from the pinned
checkpoint and runs P128/D1 from the captured input-norm boundaries, with its
own poisoned NHD native cache and persistent native P128 K/V into D1. No
original cache is restored. Positions on all3 axes, actual slots and block IDs
are checked. **1case/657assertions,652pass/5fail,exit1**; all observed stages
remain finite and no threshold is changed.

Actual native Qgate/K/V projections and gate values match the original exactly
at both steps. The first captured difference is the fused QK/RoPE preamble:

| Phase/stage | Rotated half-value differences | Unrotated differences |
| --- | --- | --- |
| P128 Q | 61654 | 16 |
| P128 K | 10377 | 8 |
| D1 Q | 503 | 0 |
| D1 K | 90 | 0 |

Q/K stay inside the fixed pointwise band, but P128 native K writes differ122
FP8 bytes; these122 differences persist at D1. V and every inactive byte
remain exact. The downstream core fails2088 P128 values and5 D1 values at
rtol0.01/atol0.003; max errors0.0246582/0.00494385. The P128 gated output
fails1 value. Complete mixer outputs meet the band and relative-norm gate
(P128 relative0.000401790/max0.0078125; D1 relative0.000251804/max0.000244141),
but these outputs do not waive the failing intermediate/core/cache checks.

The trace witnesses2 native cosine-cache producers,4 grouped linears,
2 fused preambles,2 FP8 writers,2 paged readers and2 sigmoid-gate operators,
with zero BF16 arguments or CPU-reference selections. Original installed
RoPE source is retained separately: its native path consumes normed FP16
Q/K and matches cosine/sine cache dtype to the query. The existing XPU fused
preamble retains F32 normalized values and coefficients through rotation.
That source difference and the concentration in rotated columns identify the
next arithmetic investigation; they do not prove which change will pass.

Source/build identities, all16 native stage dumps, manifest, failing test log,
trace and first-difference report are recorded in
`reference_B.S1_attention3_model_owned_mixer_first_difference`. Earlier original
block/GDN/attention fixture hashes remain unchanged. **Next:** independently
replay original Q/K norm and RoPE intermediates/coefficients, then reproduce
the observed FP16 rounding in the scoped native model path and rerun this
unchanged mixer/cache gate. Remaining GDN-state/D64 gates and all S2-S6 remain
open. Production remains stopped; current jobs are terminal.

## 32. FP16 RoPE operand replay and intermediate checkpoint

The passive original layer3 observer now retains actual normalized FP16 Q/K
and selected FP16 cos/sin at P128/D1. The original rotary method is delegated
once with its arguments/result preserved and restored after capture. A new
54-tensor fixture is kept separately; the earlier 48-tensor fixture is unchanged.
Observer host checks pass5 tests; capture exits0. Independent stdlib replay
rounds each product to FP16 before add/subtract and matches all four original
Q/K outputs bit-exactly. Its focused host checks pass3 tests,exit0.

An explicit XPU FP16-intermediate mode now reproduces these boundaries in
RopeFromCache and the fused QK-normalization/RoPE primitive; it defaults false.
The shared dense-attention norm upcast now decodes actual F16 storage correctly
in addition to BF16. The model's private upcast already handled both formats.

Focused native v4 build exits0. Test **1case/38assertions,37pass/1fail,exit1**:
RopeFromCache on captured normalized inputs and coefficients is bit-exact for
P128/D1 Q/K. Fused outputs meet the unchanged pointwise/relative gates; P128
Q/K differ16/5 half values (relative2.78508e-6/4.838e-6, max0.00195312), D1 is
exact. Generated coefficients still differ2 half values; scoped FP32 power
and reciprocal reduced the v3 count from9 but did not pass the exact gate.
The original initialization device is not directly witnessed. Failed attempts
and v1-v4 receipts are retained, without changing thresholds.

**This mode is not yet selected by the actual model.** The earlier mixer
5 failures and GDN-state/D64 failures remain open; scalar fused and legacy
SG16 regressions have not been rerun for this checkpoint. Source snapshots,
fixture and receipts are recorded in
`reference_B.S1_attention3_FP16_RoPE_checkpoint`. The developer requested a
commit/push of the intermediate implementation. **Next:** resolve generated
coefficients, select the qualified mode in scoped EXL3 model execution and
rerun the unchanged mixer/cache gate. S1 and S2-S6 remain incomplete.

## 33. Qualified bounded FP16 RoPE cache and primitive

Production systemd was inactive after resumption, but its container/worker
was running. The explicitly authorized container stop completed before GPU
work; no production worker remains. Pinned Torch regeneration on explicit
CPU/XPU gives9/0 half-coefficient differences against the unchanged new
fixture. This is a value comparison, not a direct initialization-device witness.

Native v5 diagnostics identify the2 differences at (position65,column3) and
(121,3). Ordinary division widened to F64 (v6) still fails. Explicit Intel
round-to-nearest `fdiv_rn` at the scoped producer F32 reciprocal boundary
resolves them. **v7 focused build exit0; native auto1case/38assertions pass,
exit0**, with zero different half coefficients at positions0-128. Same-operand
rotation remains byte-exact. Fused P128 Q/K retain16/5 half differences within
the fixed band; D1 Q/K are exact. Scalar variant also passes38 assertions,
but its normalization produces more nonexact half values within the fixed band.

Two existing default SG16 preamble/cached-RoPE regressions pass115 assertions,
exit0. All failed attempts, bounded Torch/SYCL probe sources/results and
qualified source snapshots are retained in
`reference_B.S1_attention3_FP16_RoPE_qualified_P128_D1`. No frozen gate changed.
**Next:** select this mode in scoped native EXL3 model attention and rerun the
unchanged model-owned mixer/cache test. Long positions, GDN state, D64 and
remaining S1/S2-S6 are not qualified by these bounded checks.

## 34. Model FP16 RoPE wiring and remaining K normalization byte

FullAttnRopeArgs now selects producer boundaries only for XPU F16 activation
and an EXL3-declared checkpoint. Cache build/refill and paged/unpaged consumers
share it. The unfused path also consumes rounded coefficients. Its first D1
check exposed absolute positions indexing a per-step selected cache; row-index
lookup fixes that exception. The temporary unfused row-index allocation/upload
is not graph/lifetime/performance qualification and remains later work.

Focused model build exits0. The frozen old48-tensor fixture, original input-norm
values and own poisoned native cache remain unchanged. Model-owned P128/D1
now passes **655/657 assertions,exit1**: all original Q/K/core/gated/mixer
numerical bands pass; K-byte differences fall122->1 at P128 and persist into
D1. V and inactive capacity remain exact. Q/K differ19/9 P128 half values and
are exact at D1. The unfused route after index correction also passes655/657,
retaining the same1 K byte. Two synthetic paged/unpaged model regressions
pass4303 assertions,exit0, before the subsequent v8/v9 inverse-root variants.

The K difference is unrotated (row116,head3,channel213): native1.4375 lies
on an E4M3 midpoint (byte60), original1.4365234375 rounds to captured byte59.
Pinned Torch IR source uses mean of squares then rsqrt. On identical old
captured QKV, the literal Torch expression reproduces every original K-norm
half exactly; replacing rsqrt with1/sqrt differs9 halves including this point.
Native ordinary/approximate SYCL rsqrt variants v8/v9 do not remove the byte.
A bounded reconstruction of the SG16 reduction gives mean2.6682863235473633
versus original Torch2.6682865619659424. This probe is not instrumentation of
the actual kernel; it narrows the next reduction/root investigation.

Final v9 combined primitive/mixer run: **2cases,693/695 assertions,exit1**.
Primitive38/38 still passes, mixer retains2 K-byte failures. Trace has zero
BF16 arguments/CPU reference selections. All failures and stage dumps are
retained in `reference_B.S1_attention3_model_FP16_RoPE_and_remaining_K_byte`.
Old/new full original captures differ upstream beginning at row100 and are
not substituted for one another. **Next:** reproduce original mean/root
rounding on identical QKV and resolve that byte in fused/unfused execution.
Then rerun real target/state/D64 gates; S1 and S2-S6 remain incomplete.
Production remains stopped; all current GPU/build processes are terminal.


## 35. Source-matched Q/K mean and actual inverse-root checkpoint

The pinned Torch XPU D256 reduction uses four adjacent registers per virtual
lane and an ascending-offset subgroup tree. The scoped FP16 fused preamble now
reproduces that order, including the different active-vector grouping at small
output counts. Explicit F32 Gemma-weight addition and multiplication boundaries
are retained. The legacy mode keeps its existing arithmetic.

A bounded literal Torch replay on the unchanged old QKV reproduces all original
normalized half values. The corresponding host reduction and standalone native
helper replay match the recorded means; the corrected standalone native replay
also matches all512 K inverses and131072 normalized half values. Its first
attempt wrote an empty output before host staging was added; that receipt is
retained as failed evidence. These helper replays do not qualify the actual mixer.

Optional VT_XPU_ATTN_NORM_PROBE instrumentation now observes the actual fused
kernel. With it unset there is no observer allocation or explicit event wait.
The final v13 observer writes F32 [T,28,3] values (mean,mean+epsilon,inverse),
with24 Q heads followed by4 K heads per row. Both mean and variance match the
same-input Torch replay exactly in all3612 heads at P128/D1. Inverses still
differ in891 Q/171 K heads at P128 and10 Q/1 K heads at D1. At the remaining
K-byte point, mean2.6682865619659424 and variance2.668287515640259 match;
native inverse0.6121864318847656 differs from original0.6121863722801208.
The standalone helper produces the original inverse on that same input. The
cause of the different actual-kernel root result is not yet established.

Focused v10/v11 builds exit0; combined primitive/mixer checks each pass
693/695 assertions,exit1. The initial v12 observer build failed on a const SYCL
event; v12b corrected that compile error. Final v13 focused build exits0 and
actual mixer test passes **655/657 assertions,exit1**. All fixed numerical bands
pass; the unchanged exact K-cache checks still fail once at P128 and once at D1.
V and inactive capacity remain exact. No threshold or frozen fixture is changed.
The source-matched mean has not yet been integrated into generic unfused RMS.
Final scalar/legacy regressions and whole-target/state/D64 reruns are pending.

The developer requested this interrupted work be committed and pushed as a
checkpoint. Sources, failed attempts and final observations are recorded in
`reference_B.S1_attention3_source_matched_mean_and_actual_inverse_checkpoint`.
Production is inactive and no native/oracle job remains running. **Next:** isolate
the actual-kernel inverse-root difference on identical variance inputs, resolve
the remaining K byte, qualify fused/scalar/unfused paths, then rerun the real
target/state/D64 gates. S1 and all S2-S6 remain incomplete.


## 36. Qualified fused and scalar Q/K, RoPE and model-owned KV bytes

Changing native rsqrt to regular SYCL rsqrt alone (v14b) still passes655/657,
exit1, with the same actual inverse differences. The first v14 invocation
ended before norm observation because its dump directory had not been created;
that failed receipt is retained separately. LLVM text emission is unavailable
in this installed Intel compiler; saved device bitcode translated to SPIR-V
text is retained instead. Failed diagnostic invocations are also retained.

Separate compile-time producer/legacy kernel instantiations (v15) resolve the
actual inverse difference. The observed3612 means, variances and inverses all
match the same-input bounded Torch replay exactly. The old frozen48-tensor
model-owned mixer test passes **657/657 assertions,exit0**. P128/D1 Q/K become
bit-exact, both active K/V caches and inactive capacity exact; all unchanged
numerical core/gated/mixer bands pass. This shows the effective repair, not a
proof of a particular backend optimization or compiler defect.

The v15b default run without VT_XPU_ATTN_NORM_PROBE passes695/695,exit0.
Scalar initially fails5/695: only rotated Q/K differ, with121 K bytes retained
into D1. Its actual means/variances/inverses are also exact. Saved scalar
SPIR-V lowers rotation products to Half FMul, unlike the passing SG16 F32
products. Explicit F32 round-to-nearest multiplication before each scalar F16
product narrowing restores the original boundary (v16).

Final focused v16 build exits0. Default and scalar runs each pass
**2cases/695assertions,exit0**, with the norm observer unset. The separate
54-tensor primitive fixture and old48-tensor mixer fixture remain unchanged;
all P128/D1 Q/K half values and bounded positions0-128 coefficients are exact
in each test's own reference. Active K/V writes and poisoned unused capacity
are exact at both phases. All16 native model stage dumps are byte-identical
between default and scalar. Two existing SG16/cached-RoPE cases pass
**115/115 assertions,exit0** in both default and scalar selections against the
rebuilt current library.

Source/build identities, successful bounded runs and all failed attempts are
recorded in `reference_B.S1_attention3_fused_scalar_exact_QK_and_KV`.
**Next:** use the qualified reduction/root boundaries for scoped unfused Q/K
RMS and rerun its unchanged model-owned mixer/cache gate. The temporary
unfused selected-row allocation still needs later graph/lifetime work. Whole
native target/state/D64 have not been rerun since the RoPE wiring; their prior
failures remain open. S1 and S2-S6 are incomplete. Production remains stopped;
all jobs from this step are terminal.


## 37. Qualified scoped unfused Q/K normalization

RmsNormArgs has an explicit default-false qk_fp16 selection. The standalone
VT wrapper accepts it only on XPU at D256, with F16/F32 inputs, F16 output,
F32 Gemma weights and no residual. Model paged/unpaged Q/K calls select it
only when the existing EXL3/XPU/F16 boundary is active and Dh is256. Other
head widths retain their existing norm selection; unsupported explicit requests
are rejected instead of silently using another precision/backend contract.

The qualified D256 mean and explicit F32 multiply helpers now live in one
private XPU header shared by fused attention and standalone RMS. The new
standalone producer kernel has its own rsqrt body, separate from legacy1/sqrt,
and narrows normalized outputs into F16. B70 uses its SG16 implementation;
its no-SG16 scalar fallback has not been exercised on this card. Generic
residual/other-width RMS arithmetic is unchanged.

Focused v1 and final v2 builds exit0. Final own-cache P128/D1 unfused mixer
passes **1case/657assertions,exit0** on the unchanged48-tensor fixture. Every
Q/K half, active K/V byte and inactive-capacity byte is exact; numerical
core/gated/mixer bands pass. The trace witnesses4 standalone RMS selections,
2 cached rotations and zero fused preamble selections, BF16 tensor arguments
or CPU-reference selections. All16 v1 unfused/fused native model stage dumps
are bit-identical. The final v2 width eligibility restricts the new selection
to the qualified geometry; internal standalone means/inverses are not separately
instrumented by this gate.

After sharing the helper, v1 fused auto and scalar each pass2cases/695assertions,
exit0, and final v2 fused auto repeats695/695,exit0. The existing generic RMS
case (weights/Gemma, residuals and aliases) passes **63928/63928 assertions,
exit0**. No frozen tolerance/fixture or earlier failed receipt was changed.
This qualifies bounded attention paths, not the complete target or S1.

Source/build identities and focused results are retained in
`reference_B.S1_attention3_unfused_QK_FP16_qualified`. **Next:** rerun the
complete native64-layer target P128/D1 and D64 gates after these attention
changes, then resolve the remaining full block/state requirements. The unfused
selected-row scratch/upload still needs later graph/lifetime/performance
qualification. S1 and S2-S6 remain incomplete; production remains stopped.


## 38. Whole-target rerun after Q/K/RoPE qualification

The current final v2 binary executes the complete64-layer target with F16
activations, FP32 GDN state, FP8 KV and the full6-bpw248320 head. It keeps its
own native continuation/cache; original state is never restored. The norm
observer and stage dumps are unset, and MTP/graphs are disabled by this eager
functional seam. New output prefixes preserve all earlier logits/failed tests.

P128/D1 against the unchanged3 original repeats now passes
**1case/197assertions,exit0**. Greedy IDs13/198 match all repeats; top10 overlap
is10 in all6 comparisons. MaxTV0.00370011 and maxKL0.0000469401 remain inside
TV0.02/KL0.002. Trace witnesses514 grouped linears,48 raw-gate prefill calls,
48 packed decodes,32 FP8 writers and2 full-head M1 calls. BF16 arguments,
CPU-reference and old scalar-packed selections are zero.

P128/D64 completes all65 steps but fails its unchanged distribution gate:
**1case/801of807assertions,exit1**. All65 native greedy IDs equal the original
observed sequence, and top10 overlap is at least9. Both TV and KL fail at
D24 (0.0315285/0.00285727), D27 (0.0343527/0.0025539) and D29
(0.223781/0.169353). The earlier D11 failure is absent in this run; the larger
D29 failure is retained, not replaced or attributed to variability as a pass.
Trace witnesses65 full-head M1 calls,48 raw-gate prefill and3072 packed decode
calls, with zero BF16/CPU-reference/old scalar-packed selections.

The existing comparator also checks all65 current rows against the3 earlier
original repeats with directly observed input prefixes and all3 position axes.
It revalidates those witnesses and immutable capture hashes. Report generation
exits0, but **10of195 comparisons retain investigation triggers**: D11/repeat1,
D14/repeat2, D24/repeats0/1, D27/all3 and D29/all3. MaxTV0.2057348624 and
maxKL0.1627901091 remain failing. This additional comparison does not replace
or waive the exit1 native test. Previously measured original-original variance
and strict state failures remain separate evidence, not qualification.

Results, all65 new bounded logit rows, traces and source/build identities are
recorded in `reference_B.S1_target_after_QK_RoPE_qualification`.
**Next:** replay full-width Gemma normalization and BA/state boundaries on
identical frozen operands, identify the first remaining mean/root/block
arithmetic difference, then rerun the unchanged block/state/D64 gates after a
source-backed repair. Complete S1 and then S2-S6; none is marked complete here.
Production remains stopped and all jobs from this step are terminal.


## 39. Exact bounded D5120 Gemma norm; BA/state and D64 remain failing

The literal pinned Torch eager Gemma expression reproduces all4 frozen layer0
P128/D1 input/post norms and both post-norm residual outputs exactly. It uses
the checkpoint weight converted to F16, then F32 weight+1, unrounded F32
input+residual, mean of F32 squares and rsqrt before F16 output narrowing.
This is a same-input replay, not observation of live original-worker internals.

The pinned contiguous vec4 reduction accumulates4 registers per virtual lane,
folds those registers left-to-right, halves group-x lanes down to32, then uses
ascending subgroup offsets. B70's maxWG1024 and min/maxSG16/32 determine a
virtual width1024 divided by the largest power of two no greater than
min(rows,32). At P128, width32 plus an F32 reciprocal factor reproduces all256
replay means exactly. MeanOps projects by multiplication with F32(1/5120);
division instead differs18 input/24 post means. D1's2 means coincide for
width256/512/1024, so its width1024 is source/device-derived, not uniquely
identified by those2 values.

The new separate producer kernel emulates this tree with physical SG16,
uses rsqrt and explicit F32 residual/square/Gemma multiplication boundaries,
and returns the independently narrowed residual. Selection is scoped to
Gemma D5120 F16 input/output/weight and optional F16 residual, maxWG1024,
minSG16/maxSG32. Other geometries keep generic RMS; this step does not qualify
those fallbacks. No public RMS argument or model policy changed.

Focused v1/v2/v3 builds exit0. Initial captured norm run passes40/40;
final2 Gemma cases pass **113/113 assertions,exit0**, including separate output,
output/input aliases and output/residual aliases. All4 captured norm outputs
have zero differing half words; residuals remain exact where still visible.
The existing generic RMS regression passes **63928/63928,exit0**; its F32
outputs/BF16 residuals do not exercise the new path. Internal native D5120
means/inverses are not separately observed by these output checks. M4/12/16,
other layer operands, performance and graph/multi-queue behavior remain open.
Initial v1 used an unsupported trace environment variable and produced no
trace; finalv3 uses VT_OP_PROVIDER_TRACE. An initial source command exits1
because its second grep finds no MeanOps in ReduceOps.h; the actual
SharedReduceOps MeanOps fragment is preserved separately.

Both whole-target runs keep native cache/state and use the original captured
input-token prefix, full64 layers/full248320 head, with MTP/graphs disabled.
They are not autonomous native-greedy continuation. P128/D1 passes197/197
in both runs. Finalv2 maxTV0.0039302/maxKL0.0000485418, top10all10, greedy13/198.
Earlier v1 maxTV0.00311897/maxKL0.0000323448 is retained, not substituted.

D64 remains failed in both runs. Initialv1: **803/807,exit1**; D27 TV0.0294933
fails while KL0.00196467 passes; D29 TV0.247458/KL0.180265/top10overlap8 fails.
Greedy differs atD27 (native74455/reference248045). Finalv2 against the final
v3 binary: **802/807,exit1**; D11 TV0.023854/KL0.00201852 fails, D29
TV0.281446/KL0.217539/top10overlap8 fails. Greedy differs atD7. The source/device
eligibility guard does not establish a cause for this variation. All65 steps
execute in each run; neither failed result is waived by the exact norm.
Final traces have65 full-head M1 calls and zero BF16 arguments, CPU-reference
or old scalar-packed selections. Independent same-prefix/3-position-axis
witness comparison retains11/195 triggers forv1 and10/195 forfinalv2. Final
maxTV0.2633897193/maxKL0.2041601301 remains failing; report generation's exit0
is not a parity pass. Frozen thresholds and all earlier captures are unchanged.

The unchanged own-state GDN seam starts from original input_norm_output,
bypassing the repaired full-width norm. Current rerun passes **413/414,exit1**:
P128 mixer relative4.97236e-5, P128 state relative1.74616e-5/maxerror0.00263786,
16379 differing F32 values. Strict elementwise state REQUIRE fails atindex360512
(head22); D1 never executes. The below-threshold relative norm does not waive
the unchanged rtol1e-4/atol1e-5 elementwise failure. Bounded stage dumps show
same-input BA differs in3 A and3 B half words. The B head22,row74 projection
is native-0.33056640625 versus original-0.330810546875; this points to the next
projection/gate/state investigation, not proven causality for D64.

Source snapshots, both complete logit sets, traces, failed receipts and focused
results are recorded in
`reference_B.S1_Gemma5120_source_matched_norm_and_remaining_BA_state`.
**Next:** inspect the same-input original/native oneDNN BA rounding and replay
its affected raw gates/state without changing thresholds. Qualify remaining
Gemma row counts and S1 requirements before S2-S6. Production remains stopped;
all GPU/build processes from this step are terminal. No commit/push is made
for this new step without another developer request.


## 40. Same-input BA variability and causal P128 state attribution

Literal pinned Torch F.linear with unchanged frozen P128/D1 norm inputs and
byte-identical actual worker/checkpoint BA weights (B then A, SHA256
e5dbd32b6b1adb82dc5803a7239b9ae005ed621af21a89984512a111022760e9) completes,
exit0. Verbose identifies original oneDNN3.12; native uses the pinned3.13.
Original P128 repeats differ8/7/3 half values from the frozen capture and
5/7/6 from one another. All3 D1 repeats are exact. Split/F32 diagnostics
retain10/14 P128 differences; neither replaces original F16 production math.
All3 original repeats give the captured native B22,row74 value rather than
the frozen original value. This strengthens the earlier live-operand variance
observation; it does not qualify the failing strict state gate.

A standalone native oneDNN3.13 probe reproduces VT's F16 descriptors, strict
accumulation and user-scratchpad recipe. Build/run exit0. Default P128 repeats
differ6/5/12 halves from original and9/9/10 from one another. Deterministic
selection differs13 halves in each repeat, with zero pair differences, but
still misses B22. D1 is exact in both selections. The source selector rejects
kParallel catalogue candidates when deterministic is set; the physical selected
subkernel is not separately observed here. This is not actual VT instrumentation.
Its SYCL queue has no profiling, so verbose time0 fields are not measurements.
Deterministic selection is not adopted as a product repair.

Exact rational dot products of the6 differing FP16 operand pairs are a
mathematical diagnostic. Correct rounding matches native at5 points and frozen
original atB22,row74. That exact sum is-0.3306885194615461, only4.2899046e-8
below the F16 midpoint-0.3306884765625. This shows rounding sensitivity; it
does not require producer GEMM to use exact rational arithmetic.

A bounded original fused GDN replay varies only BA, keeping QKVZ, weights,
positions and cold-state contract fixed. The unchanged original BA baseline
reproduces original core/state exactly, exit0. Using the actual native6-half
BA difference gives1934 differing core halves and16379 differing F32 state
words, relative1.7461556855e-5/max0.00263786316. It fails122 unchanged pointwise
state bands, all head22, firstindex360512. Using only the native B22,row74
value with otherwise original BA produces the same counts/first state value.
Replacing that single value in the diagnostic native BA restores exact
original core/state despite the5 remaining BA half differences.

The actual native P128 core dump and original fused GDN core from the same
actual native BA are byte-identical in all786432 halves. Both SHA256 values
are7ebe5cf7cc828fe3a0f80e01ca7b201e5d820df7f30c329419101d1657af40eb.
Thus the current bounded recurrence/core discrepancy is causally attributed
to the BA operand, not unexplained downstream core arithmetic. Actual native
state is not separately dumped for byte comparison in this step. A same-input
original Torch BA repeat also produces220 strict state failures in3 heads
when replayed by the original GDN body. Original variability is retained,
not a replacement capture, relaxed threshold or whole-target parity claim.

No BA selector/arithmetic or product B22 substitution is implemented. New
root-owned safetensors initially had mode600; only those receipt read permissions
were corrected to644 so host hashing/comparison could complete. Bytes remain
unchanged. All identities/results are recorded in
`reference_B.S1_BA_same_input_reproducibility_and_state_causality`.
**Next:** qualify remaining Gemma M4/12/16 geometries and continue same-input
block/D64 localization; S1 and S2-S6 remain open. Production remains inactive,
and all jobs from this step are terminal. No commit/push is made for this step.


## 41. Recovery R01: first autonomous native C1 output

The developer adopted `docs/B70_CPP_EXL3_IMPLEMENTATION_FIRST_RECOVERY_PLAN_EN.md` as the active R01–R11 plan. The original S0–S6 plan remains scope/evidence. `RECOVERY_STATUS.json` separates implemented, locally tested, target-qualified and serving-qualified.

R01 now uses one shared host-testable feedback driver and the existing full native model entry point. A rendered natural P128 prompt is encoded again by the native tokenizer; subsequent inputs are the native C++ greedy choices from the full 248320/6bpw target head. C1, no MTP/graph/prefix reuse. KV allocation derives from prompt/output budget and actual physical1600-token page size; GDN has one active slot. Each of two requests explicitly resets all caches/states. Only layer0 Conv/FP32 state is exported at initial/prefill/final boundaries.

Focused tests: host3cases/35assertions; nativeGPU1case/366assertions, both exit0. Both cold requests emit11IDs includingEOS and the same text: `There are 12 tomato plants in total.<|im_end|>`. This proves functional native feedback on this prompt, not target/state parity. Layer0 FP32 state hashes differ between cold repeats although emitted IDs and Conv exports match; numerical repeatability is not claimed. Trace has no BF16 arguments, CPU reference or old scalar-packed selections.

Exact source/executable/build/prompt/trace/export identities: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r01-control-v1/native-receipt-v1.json`. The strict `RunRealEagerTarget` body is unchanged; no new D64 replay was run. S1 D64/strict-state failures remain open. Remaining R01: report gate/accounting and real64-token output-cap check; then bounded same-input state work R02 and independently testable W8A8 R03.


## 42. Recovery R01 complete: output-cap64 and explicit comparison gates

The same native harness passes578/578 assertions for a second natural P128 prompt requesting a story: both cold requests emit exactly64IDs and stop at `output_limit`, with identical output IDs. The real EOS case remains366/366 and host feedback35/35. No production/quality/performance release is claimed.

`compare_target.py` now defaults to `--gate`: threshold failures write `investigation_required` and return1. Explicit `--report-only` preserves those failures while returning0 for report creation; JSON records both gate/process exit codes. Passing bounded logits leave whole-target/state status pending. Twelve focused host tests pass. Applying the current CLI to existing native D64 Gemma v2 logits and three directly witnessed producer repeats returns1 with195comparisons/10failed rows, byte-equivalent metric values to the prior report. No GPU D64 rerun or changed thresholds/captures.

Receipt and exact retained source snapshot: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r01-outcomes-v1/receipt.json`. R01 is complete as a functional development delivery. R02 starts with actual native final-state export, separately collected D1 and native VT B/A repeatability; numerical release blockers remain open.


## 43. Recovery R02: actual final-state attribution closed locally

The focused native test now exports all five physical Conv/FP32 state slots before frozen comparisons can abort. Its independent matched-state D1 case executes even when P128 fails. Current frozen run: two cases, one pass/one fail,463/464assertions,exit1; first strict P128 state failure nowindex34008. D1 mixer/state relative error0. These are retained current observations, not replacements for old failures.

Pinned original GDN receives the current actual native BA matrix in full, unchanged QKVZ (native mixed inputs checked byte-exact) and identical logical initial values across active/inactive slots using producer physical strides. Actual native core786432halves, all Conv slots and all5x786432FP32state elements match original bitwise. Untouched slots remain unchanged in both implementations. Matched D1 Conv/state exports also match the frozen original bitwise. This closes the previously missing actual final-state comparison on these bounded operands; D64 is not explained or qualified.

Three actual VT MatmulDenseF16 calls per shape, same F16[96,5120]weight/strides: P128 pair differences3/6/7halves; D1 all0. Diagnostic test30/30,exit0 classifies variability and finite execution; it does not assert P128 repeatability. Captured selected implementation `jit:gemm:any`, scratch2261120 forM128 and0 forM1. Profiling recorded selection, with no speed claim.

Exact source/binary/build/fixture/run/export identities and original same-input script: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-state-v1/receipt.json`. R02 still needs Gemma5120 short/odd rows, aliases and second-layer operands. Independent W8A8 remains the next implementation item, without reopening this already attributed local consumer or hiding frozen qualification failures.


## 44. Recovery R02 complete locally: required short-row norms

Pinned Torch on real layer0/layer3 operands verifies the existing P128 captures before deriving20 operator cases: M3/4/12/16 and physicalM4/logicalM3, each with/without residual. Native contiguous independent/output-input/output-residual alias variants match active output and residual stores bitwise; poisoned inactive rows do not contaminate active rows. Noncontiguous tensors are explicitly refused by the public API before any mutation; strided arithmetic is not claimed. Focused native test821/821,exit0. Earlier failed test/build attempts remain in the receipt directory.

The Gemma5120 tree uses physical subgroup16 while emulating pinned Torch logical subgroup32/virtual width1024 and its row-count-dependent reduction. Native capability checks are eligibility guards, not cross-device bit-equivalence proof. No new arithmetic or fast-math changes were needed in this step. Exact frozen fixture/source/build/binary/log/trace identities: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-norm-rows-v1/receipt.json`.

R02 is complete as bounded same-input state and required-norm delivery. D64 and strict frozen full-target state failures remain release blockers. Next implementation is R03 typed true grouped EXL3 W8A8, beginning with real GDN-QKVZ M129/M256; no production restart or commit/push.


## 45. Recovery R03: first actual grouped native W8A8 family

A separately registered typed `Exl3GroupedW8A8` operator uses the pinned donor INT8 Hadamard/row-quantization and packed 4/6bpw reconstruction kernels. Shared pinned oneDNN3.13 executes signed INT8 row-major matmul with static3.453125/127 weight scale and per-row activation-scale binary post-op, rounds to FP16, then applies the donor output Hadamard/SV transform. M129–4096 is explicit; existing SmallM M<=128 dispatch is unchanged. No Packed/BF16/GPTQ alias or FP16-reconstruction fallback.

Caller owns one aligned byte workspace and an I8[K,128]panel, used sequentially on the same in-order queue; existing queue-owned oneDNN user scratch is accounted separately. First GDN-QKVZ K5120/N16384/S2 uses11012160workspace bytes and655360temporary weight bytes, rather than caching full groups/model reconstructions. Zero and padded activation rows have INT8zero/scale1. Native refuses NaN/Inf and overflow at either FP16 transform boundary before output writes; the donor float-to-INT8 conversion has no defined nonfinite contract. Initial route is eager; graph/performance qualification is not claimed.

Frozen pinned original references use actual packed GDN-QKVZ groups and real P128 norm rows repeated to M129/M256, with explicit zero/near-zero rows; these are same-input operator operands, not a real long-prompt trajectory. Allocation observation is checked against an ordinary original call. Native focused2cases/100assertions,exit0: actual INT8 activation, FP32 scales, last reconstructed panel, FP16 intermediate and final output all bit-exact. M128 control is exact; large/small/large workspace transitions and poisoned padding pass. Earlier instrumentation/compile/link failures remain in the receipt.

Exact source/build/binary/frozen fixture/log/trace identities and snapshots: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r03-w8a8-gdn-v1/receipt.json`. R03 remains in progress: one real M4096 operator call and attention/remaining MLP/down/selected6bpw head family coverage are next; full-target integration is R05.


## 46. Recovery R03 complete locally: 4K and required projection families

One real GDN-QKVZ M4096 operator call passes18/18,exit0; actual INT8/row-scale/last-weight-panel bytes and all FP16 intermediate/output words match original exactly. It allocates176193600workspace bytes plus655360weight-panel bytes. This is cyclic real short-input operator evidence, not a complete 4K target/continuation run. Focused shared oneDNN packed/dense and repaired obsolete test-call coverage passes3cases/1277assertions,exit0. Retained first 4K attempt used the previous binary after the additional regression target failed compilation; it fails missing M129 rather than being a qualification pass. Freshly linked4K run passes.

Additional original/native M128/129/256 fixtures: actual Layer3 attention QKV S3, Layer0 MLP gate/up S2 and down K17408/S1, and selected first/last128 columns of the6bpw target head. First4family tests58/58 each; final last-head60/60, all exit0 and bit-exact at every recorded INT8/FP32/F16 stage. Down uses the donor16-thread quantizer rather than the8-thread K5120 path. Temporary weights are only I8[K,128]:655360 or2228224bytes. Head operator inputs are real Layer0 norm operands of the correct width, not actual final-layer head inputs. Model head evaluation remains full248320/6bpw, selected-row-only.

A focused GDN test also swaps the first/last complete packed128-column blocks, SV and sourceIDs to create a noncontiguous map, requiring the exact corresponding original output-byte permutation. It passes66/66,exit0 including invalid-map rejection before output writes and unchanged normal reuse cases. No new product arithmetic is changed for this extra coverage.

Receipts/snapshots: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r03-w8a8-4k-v1/receipt.json` and `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r03-w8a8-families-v1/receipt.json`. R03 is complete as a locally tested native operator delivery; target/serving qualification remains false, with target integration scheduled R05. Next is R04 variable producer GDN and one stable, accounted reservation before decode or prefill first allocation.


## 47. Recovery R04 complete locally: variable C1 producer and stable reservation

`PlanGdnFp16C1` defines exact logical1–4096 lengths, donor physicalT+63 capacity, aligned Q/K/V/A/W/U/rawA/beta/metadata regions. Maximum4K layout214538112bytes fits the explicitly chosen230686720byte (220MiB) reservation. Backend atomically accounts one shared serialized reservation on the first native-GDN consumer, including decode-first. Completion waits remain. Native raw-gate producer uses the actual logical offsets/length, zeroes the full physical tail, and consumes the caller-prepared zero or gathered persistent FP32 state. EXL3 C1 model Conv FP32/post-conv/raw-gate selection is generalized; full large-M model execution awaits R05 linear/attention integration.

Focused layout test193/193; unchanged real P128/D1 raw-gate chain19/19, both exit0 and exact. Frozen original15case reference: coldP127/P128/P129/P256/P4096, each followed by one append3 prefill from its nonzero state and D1 decode. Each original fused/split endpoint must match bitwise before capture is accepted. Native end-to-end Conv/post-conv/raw-gate chain264/264,exit0: Q/K/V/beta/core/Conv state/FP32 state exact at every phase, four inactive slots unchanged. No captured state substitution; native cache carries each continuation. Preliminary zero decode and all subsequent short/large/append/decode calls assert the same230686720byte workspace capacity. Irregular tails and4K->short reuse are covered.

Source/build/binary/fixture/test identities and snapshots: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r04-variable-gdn-v1/receipt.json`. Initial oracle capture failed only while packing differently shaped decode Q/K/V tensors; the corrected row packing precedes the accepted frozen capture. R04 is a locally qualified C1 operator delivery; full-target/serving/graph/maximum-context qualification is still pending. Next R05 integrates the already-tested true grouped W8A8 path into the target and performs a native4K autonomous smoke before extending exact-K attention.

## 48. Recovery R05 partial: real autonomous4K and native CLI invocation

The dense linear seam now dispatches M129–4096 to the typed W8A8 operator with caller-owned scratch/panel retirement. Attention QKV, GDN QKVZ and MLP gate/up retain their source transforms in model-owned groups at large M; SmallM arithmetic is unchanged. Grouped real seam19/19; single-down seam plus planner/operator125/125; synthetic grouped-vs-independent QKVZ/QKV at M1/M4/M12979/79, all exit0 and bit-exact.

Existing `vllm-cli` accepts `--prompt-file`, `--max-model-len`, `--max-num-batched-tokens`, `--num-blocks` through the unchanged public C ABI. Six focused argument cases pass. A realP129/O16 invocation executes the scheduler/runner/model and returns fluent native text, exit0. Its first explicit4-block pool is correctly refused by hybrid admission; eight blocks support the4352-token configured length. That CLI call used the earlier separate-projection large-M implementation; subsequent source now selects grouped large-M projections.

Real rendered natural P129/P512/P4096 prompts are independently reencoded by the native tokenizer. Each autonomous O64 test runs all64layers, all248320target logits, native-owned48GDN and16FP8KV caches, and two zero-state resets. Each passes579/579,exit0 with identical64token output IDs. No captured continuations/states are supplied. Initial4K attention uses native Xe2 prefill; decode uses native split. New oneDNN integration remains separately in progress.

Development timings (model forward/full-row D2H/greedy and dispatch logging, excluding resets/state exports): resident-weight P1291.086s/118.82prompt-tok/s; P5121.342s/381.64; P40966.463s/633.74. Decode is2.91–2.98tok/s across these runs. First prefills11.20/11.39/15.91s include lazy uploads/JIT and are reported separately. These are functional-development measurements, not a matched serving performance qualification. Bitwise FP32 state repeatability and frozen D64 qualification are not asserted by identical output IDs.

Exact eager source/build/binary/prompt/run identities: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r05-target-v1/receipt-eager-v1.json`. R05 is incomplete pending original exact-K attention agreement, intended-route target evidence, chunked32K and output-transition coverage. Target/serving qualification stays false.

## 49. Recovery R05 local exact-K oneDNN attention

The pinned fused SDPA graph uses the existing native oneDNN device engine/queue stream and explicit accounted user scratchpad. FP8 K/V are gathered one KV head at a time with actual physical page/token/head strides; half query groups are padded at the front to multiples256, while K keeps its exact GPU length. Bottom-right visibility is `j <= L-Q+r`. Runtime inverse attention scale, K/V scales and negative-infinity inputs remain owned through completed graph execution and native output copy. Neither host maximum lengths nor unused poisoned cache rows enter the mask. Completed eager calls permit bounded compiled-partition eviction; owned cache and library cache capacities are16. Decode and ineligible signatures retain native alternatives.

Synthetic physical1600-page boundaries1599/1600/1601, Q3/Q129, unit/nonunit KV scales, two runtime attention scales and unequal C4 Q8/7/6/5 pass87/87,exit0 against native CPU mathematical reference within the existing F16 attention budget (observed relative errors about0.00029). C4 strided destinations receive explicit native contiguous-result copy, leaving head-tail padding unchanged. Public XPU attention admits only nonoverlapping supported destination strides; other backends retain their previous contract.

Frozen original oneDNN using repeated real Layer3 Q/FP8 operands and the actual interleaved hybrid strides `[3276800,2048,512,1]`: (K,Q)=(1599,3),(1600,129),(1601,256),(4096,4096), including nonunit scales and reversed pages. All four native outputs match every original F16 word;26/26,exit0. The default pinned B70 auto route now chooses this leaf for eligible large-M queries. Default-route original cases plus18new lengths and first-length reuse after bounded eviction pass84/84,exit0; the cache remains at most16entries and the reused result is exact. Profile witnesses require four actual SDPA executions per C1 operator case, excluding a silent fallback pass.

Receipt/source/build/binary/fixture identities: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r05-attention-v1/receipt-v1.json`. Failed const-event/doctest compile attempts and the first oracle's packaging-only missing metadata argument remain recorded. This qualifies the local attention operator and copy semantics; full-target, full-model C4/MTP and graph qualification are separate. Intended-route4K and chunked32K autonomous evidence remain the immediate R05 work.

## 50. Recovery R05 complete functionally: intended-route4K and public32K

The full autonomous target with the new default oneDNN prefill route passes579/579,exit0, with identical64 output IDs after two resets. Cold P4096 prefill12.064s; resident-weight prefill4.025s/1017.63prompt-tok/s. Decode2.909/2.961tok/s. These retain the development timing scope from section48; dispatch tracing is enabled and a matched serving comparison is not claimed.

Public `vllm-cli` P4096/O64 exits0 and returns exactly the private autonomous driver's text, finish_reason=length,34.130s total generate time. Public P32768/O256 also exits0, length,137.347s total: eight scheduler-owned4096-token chunks, then256 native-selected output tokens. Trace records16 SDPA executions per chunk with exact K4096/8192/12288/16384/20480/24576/28672/32768; compiled partitions remain bounded at16. FP16 grouped W8A8 and full248320/6bpw target head are used. The reported1.864 emitted tokens per total second includes cold setup/prefill and is not decode throughput. The32K pool has48 blocks and max_model_len33024; no maximum-context or prefix/MTP claim follows.

Receipt: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r05-target-v1/receipt-complete-v1.json`, linking frozen source/build identities and both public logs/prompts. R05 is complete as a bounded eager functional delivery. Target/serving qualification remains false; P4096/D64 teacher-forced qualification is conditional on repair of the retained shared S1 numerical blockers. R06 begins with scoped FP16 draft norms, followed by the exact compact draft subset, selected-row head dataflow and native global-token selection. Production stays inactive; no commit/push.

## 51. Recovery R06 functional compact draft; numerical gate remains open

The production loader resolves XPU dense EXL3 from actual FC storage and converts all seven draft norms directly to FP16. Two focused cases pass1104/1104. First run retained a stale CPU fixture missing its declared FP32 target recurrence; that fixture declaration is corrected, not the precision guard. Generic/GPTQ/MoE policy is unchanged.

The compact builder selects exactly512 complete128-column blocks, preserving the supplied production order, all packed6bpw bytes and SV, with65536 stable global IDs. Invalid count/range/duplicates/types/metadata are refused. Three focused host cases45/45 include the actual production list and a permuted map. The public XPU MTP loader requires `EXL3_DRAFT_VOCAB` and exact artifact SHA256 `b4eadc088059190983fe0498af11864f5aaa2eaf2ec58ae9864f0715634d313d`. Historical corpus `vocab=248077` is not an extra mask; full blocks remain within the actual248320 target configuration, as in the donor.

Draft ComputeLogits now explicitly uses FP16 SmallM and rounds to FP16 before F32 compact logits. Proposer gathers selected hidden rows before the head while retaining the whole forward for KV/feedback. A native XPU mapped argmax selects the lowest global ID on ties, downloads only chosen I32 IDs and retains completion waits. Selection6/6, three native trace calls; focused CPU shifted-prefill proposer4/4 and depth3 control20/20, all exit0. The target's full248320 head is not replaced or expanded.

One original worker P128 MTP1 observation delegates every original call. Ordinary and observed IDs `[13,198]` match; resolved draft_sample_method=greedy, rejection_sample_method=standard. It records actual128x5120 target/draft hidden, selected row127, one compact head and global ID map. This model returns one normalized tensor for both draft logits and feedback. First root model hooks missed forward boundaries; second observer used an obsolete Attention.prefix attribute; both failures and executed source snapshots are retained. Successful observer wraps the actual speculator `_run_model` and reads Attention.layer_name.

Native identical-head-input call matches all65536 logits bitwise and selects the same global ID. Full native draft produces the same global argmax, but **numerical test exits1**: hidden relative0.000533097 with34 pointwise failures; compact logits relative0.000699084 with77 failures. The unchanged bands are relative<0.002 and absolute0.003+relative0.01 per value. A nonfatal collector preserves those bands/process failure while allowing independent head checks. One bounded attention-route hypothesis changes only native prefill to exact-K oneDNN: still exit1, hidden29/logits45 pointwise failures. No default promotion, relaxed threshold or original-value substitution.

Receipts: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r06-draft-norm-v1/receipt-v1.json`, `r06-compact-head-v1/receipt-builder-v1.json`, `r06-compact-head-v1/receipt-dataflow-v1.json`. Native F32 boundary exports, source/build/binary/trace/oracle identities and failed attempts are retained. R06 functional implementation is delivered; complete/numerical/target/serving qualification stays false. Next is public native MTP1 smoke and dependency-ready R07 transaction work, retaining R06 and S1 numerical failures. Production remains stopped; no commit/push.


## 52. Recovery R07: public eager MTP1/MTP3 and shortened Conv rollback

Public CLI C1 greedy math prompt26/O16 passes with k0/k1/k3, exit0; same output text. MTP1 proposed8/accepted8; MTP3 proposed15/accepted13. These are short functional observations, not a matched performance comparison.

A focused production-loader engine test retains raw global IDs and counters, reuses one engine for two requests, and compares against the native spec-OFF baseline. Each64-token request is identical for k0/k1/k3; k0 test14/14, k1 21/21, k3 23/23, all exit0. Real MTP3 per-request proposed depths18/18/18, accepted17/15/15,19 proposer calls and38 completed draft decode forwards. Its two-request acceptance histogram is count0:2,count1:4,count2:0,count3:30. Native operator tests cover the missing count2; no original-target numerical qualification follows from native self-agreement.

A new independent one-token Conv recurrence test exposes stale rejected history after a shortened verification: first k1/zero accepted drafts fails valid history and the following output, exit1. Pinned causal_conv1d.py:845,1221 uses effective state width=(taps-1)+(query_length-1), not physical speculative capacity. XPU and CPU reference now retain taps-2 history elements then append the current provisional tokens, leaving the spare tail untouched. CUDA is unchanged and untested in this XPU delivery. Focused Conv tests and actual27B FP32 GDN snapshot rollback for k1/k3 accepted0..k pass4cases453/453,exit0, including inactive-slot sentinels. Existing GDN MTP4 test uses the same extracted helper but was not run here. Two older post-conv test calls were updated to the explicit prefill=false argument so the focused test target compiles.

The post-fix real k3 engine remains23/23. Added lifecycle checks pass90/90,exit0: limits1..5 emit exact baseline prefixes; stop string ends after11 IDs, stop token after5; synchronous cancel after verification/proposal and subsequent reuse emits the same64 baseline IDs. This tests safe engine step boundaries, not interruption of a queued GPU operation. Actual forced EOS and target sampling/processor behavior remain R07 work. The greedy verifier currently does not call the filter helper that sampled verification already uses; this is the next concrete gap to close.

Receipt/source/build/binary/raw IDs/logs: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r07-mtp1-v1/receipt-v1.json` (SHA256 2e913cf02e8a6f2548d1f495533a566fcda2c286ca0a31d3e70a3cf2b21fc6ae). The shared MTP3 trace appends initial/post-fix/lifecycle processes; logs/results are separate. Red executable was not frozen before rebuild; final tested binaries and50 source files are frozen. R07 remains partial, target/serving false; S1 and R06 numerical failures stay open. Production inactive; no commit/push.


## 53. Recovery R07: greedy target filters and actual EOS

A focused real engine test restricts tokens to271 and the actual checkpoint EOS248044, adds EOS bias100 and min_tokens3. Before repair, two MTP3 requests ignore filters during verification, output eight normal tokens and finish length:17assertions/4failures,exit1. The red executable and executed sources are frozen.

Greedy verify now clones only verification logits when allowed tokens/bias/min_tokens are active, applies the existing request/depth-aware filter helper, and uses those same rows for main-queue and copy-queue verification. Original forward logits stay intact; filtered storage is retained and synchronised before retirement. No full prefill-head temporary. The copy-queue branch is updated but was not independently exercised by this eager test.

Greedy k0 passes15/15 and k1/k3 each17/17; sampled k0 passes15/15 and k3 passes17/17, all exit0. Each arm reuses the engine twice and outputs exactly[271,271,271,248044], finish stop. Sampled tests explicitly use temperature0.7/topk20/topp0.8/minp0.05/seed931. This forced point-mass case proves API/filter/min-length/EOS behavior, not nontrivial random distribution or RNG equivalence. Test-only temperature instrumentation was added after initial green k0/k3; their earlier sources/binaries are distinguished in the receipt.

Receipt: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r07-sampling-v1/receipt-v1.json`. Bad words/penalties/custom processors remain missing for greedy verification and explicitly refused for sampled verification; provisional-history-aware support is next, followed by nontrivial sampled/RNG checks and R08. R07 stays partial, target/serving false; numerical failures remain unchanged. Production stopped; no commit/push.


## 54. Recovery R07: full history-aware target processors and MTP3 RNG

The shared verification processor pipeline now covers allowed tokens, bad words, min-tokens, bias, registered host callbacks and penalties in ordinary Sampler order. Each logical row receives committed output history plus exactly the preceding provisional draft inputs, excluding the anchor and future drafts. Structural metadata is validated before writes; min-token positions avoid uint64 overflow. Builtins remain native device operations. Explicit user callbacks preserve the ordinary sampler's host staging contract. Sampled verification no longer refuses these processors; speculative logprobs remain unsupported.

Focused GPU pipeline against independent serial ordinary CPU Sampler histories for combined MTP1/MTP3 rows passes41/41,exit0. Callback-observed input logits match bitwise, final logits within the existing sampling comparison budget. Invalid row offsets/draft length/history/penalty vectors/request keys preserve input bytes.

Public real advanced processor/EOS tests: greedy k1/k3 and sampled k3 each19/19, sampled spec-OFF17/17; two reused requests each emit[271,1206,271,248044], exit0. Real penalties/bad words/callback are active, and callback histories are retained. Controlled selections are point-mass tests, not general sampled parity. Repeated greedy k3 requests have differing proposal totals9/6 despite identical final IDs; no draft-proposal repeatability claim.

Nontrivial native MTP3 one-hot rejection tests2048request seed/position identities. Target distributions by depth are[.2,.3,.5],[.1,.2,.7],[.25,.15,.6],[.4,.35,.25], with token2 proposed. Accepted0/1/2/3 counts1033/319/289/407, corrected draws preserve target mass on unproposed tokens0/1, and bonus tokens follow their target row. Per-depth chi-square0.268228/1.85503/2.32232/5.69638 is below the existing fixed-corpus threshold24. Reverse request order and condensed subset preserve all outputs/counts bitwise.30489/30489,exit0. This is operator RNG evidence; full-model C4 identity remains R08. First test build failed on two same-line doctest CAPTURE declarations; corrected source rebuilt successfully.

Receipt/source/retained final binaries/builds/results: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r07-processors-v1/receipt-v1.json`. R07 C1 functional sampling delivery is available; numerical/target/serving qualification stays open. Next is dependency-ready R08 genuine simultaneous C2/C4, unequal lengths, request turnover and isolation. Production inactive; no commit/push.

## 55. Recovery R08: genuine simultaneous eager C4/C2

A focused public-engine test submits four unequal natural prompts before the first step, then reuses the engine for two requests. First C4 forward has actual67 tokens and offsets[0,26,37,54,67]. One request ends by EOS after5 IDs; the others hit distinct8/16/23 output caps. Spec-OFF passes127/127, MTP3 passes84/84, both exit0. All six completed raw-ID sequences exactly match their native spec-OFF controls.

MTP3 C4 runs9 steps with request counts[4,4,4,3,2,2,1,1,1] and logical full-head rows[4,16,16,12,8,8,4,4,4], proposing54 drafts. C2 runs4 steps with counts[2,2,2,1], head rows[2,8,8,4], proposing15 drafts. Trace records full248320-token and compact65536-token heads; padded projection rows are distinguished from logical verification rows.

Receipt/source/frozen tested executable/build/logs/results: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r08-batch-v1/receipt-v1.json`. This is native greedy functional batch evidence, not original numerical qualification. Groups C4 then C2 are serialized; continuous1->4->2->1, cancellation/replacement, state sentinels and page crossings remain next. Graphs/prefix disabled. R08 partial; target/serving false and S1/R06 failures preserved. Production inactive; no commit/push.

## 56. Recovery R08: continuous turnover and one-token admission reset

The public-engine test starts one48-token request, admits three unequal requests during decode, aborts one after a completed step, and admits a different one-token prompt into its freed slot. It follows stable request IDs and compact bases through reorder/condense, proves ordered1->4->2->1, and poisons unused Conv/FP32 SSM slots in the first/last GDN layers before C4 admission and during the final C1 tail.

Initial off892/892 passes, but MTP3 fails470/471,exit1: replacement IDs differ. A one-token fresh prompt is classified decode, bypassing prefill has_initial_state reset, and a newly assigned recurrent slot retained the prior owner's bytes. Runner admission now zeros all published states and all k+1 slots on assignment to a new identity, before pinned-prefix restoration. Existing owners retain their states; unused slots are not cleared merely on release.

Rebuilt off894/894 and MTP3 473/473 pass,exit0. All four completed raw-ID sequences agree exactly, including the one-token replacement also matching its standalone control. MTP3 request counts[1,4,4,4,3,3,2,2,2,2,1,1,1,1,1,1]; mixed spec/prefill is witnessed. Stable bases remain unique; replacement reuses cancelled base12; no later cancelled-ID output. First/last-layer unused-state sentinels remain bit-exact during tail. These are completed-step cancellation checks, not in-flight interruption.

Red/green executables and sources, build logs and exact runs: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r08-lifecycle-v1/receipt-v1.json`. R08 functional lifecycle delivery passes; page-boundary integration is next. Prefix/graphs disabled, greedy controls only, target/serving false and S1/R06 numerical gates unchanged. Production inactive; no commit/push.

## 57. Recovery R08: simultaneous exact page-boundary continuation

Two C2 groups use real tokenized passage prefixes of[1599,1601] and[1600,1599] tokens, native feedback and8 emitted IDs each. Every active token's write slot is checked against its own physical block table and absolute position; no across-request duplicate write slots are allowed. Positions1599 and1600 are observed, and MTP3 must witness8 logical C2 verification rows.

The initial bounded16-block pool passes off12898/12898 but serializes MTP3, whose12907/12909,exit1 fails only both genuine-C2 witnesses; all four outputs still equal off. Increasing the focused fixture pool to32 (already used for short C4) gives off12898/12898 and MTP3 12903/12903,exit0, with genuine C2 and exact raw-ID agreement for all four outputs. This adjusts admission capacity, not numerical thresholds or model arithmetic.

Both pool16/pool32 executables, sources, build/run commands, logs and actual prompt IDs are frozen in `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r08-page-v1/receipt-v1.json`. Batch/lifecycle/page functional R08 delivery is available. Greedy controls only; original numerical target and serving gates remain false. Next R09 is compatible prefix identity and joint recurrent/draft state protocol, retaining the explicit unsupported guard until that protocol is tested. Production inactive; no commit/push.

## 58. Recovery R09: shifted MTP prefix identity

The request hasher gains an explicit opt-in MTP boundary policy. Draft KV at the final position depends on the next token embedding, so the key includes that next token under a versioned extra-key tag; a full block with no known next token is deferred. Existing MM/LoRA/salt fields and parent chaining are retained. Ordinary hashing remains default and unchanged.

Focused MTP key tests plus existing ordinary incremental byte-level control pass2cases30/30,exit0. They cover different next tokens, unchanged later-token effects on an earlier block, append vs cold construction, deferred block completion, one-token pages and salt/LoRA separation. Sources, tested binary, build and commands are frozen in `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-prefix-hash-v1/receipt-v1.json`.

This delivery is the key primitive; integration remains pending and both existing prefix/spec guards were retained at the tested snapshot. Next is joint target/recurrent/draft publication/restoration and bounded real cold/warm MTP3 checks before long-context runs. Numerical target/serving qualification remains false. Production inactive; no commit/push.

## 59. Recovery R09: first joint native MTP3 prefix protocol

Loader/runner now admit prefix speculation only with the native MTP model; other methods remain refused. MTP requests use the tested next-token boundary keys. Recurrent snapshot publication moves after completed MTP proposal/prefill, with a completion wait covering target and shifted draft KV. Shared target/draft prefix pages retain the engine's page identities; Conv/FP32 SSM snapshots are restored to the reset new owner's private slot. No cached proposal, sampling history or mutable running state is imported.

Real eager MTP3 no-cache control passes56/56; prefix run55/55,exit0. P3201 requests start cold at0, repeat/append(P3217) restore3200, a changed following token1600 correctly starts cold at0, and the original prefix again restores3200. All five8-token raw-ID completions exactly match their independent no-cache controls; original prefix survives the alternate suffix. Actual publish/restore traces are retained. Target prefill chunks1600 align snapshot boundaries; no speed claim or original numerical qualification.

Source/build/tested executable/raw prompts/commands/logs: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-prefix-native-v1/receipt-v1.json`. This is the first bounded native prefix/MTP3 protocol delivery, not R09 completion. Partial-hit/cancel/eviction and measured resources precede32K/64K/128K/exact261120+1024 capability. S1/R06 gates remain open, target/serving false. Production inactive; no commit/push.

## 60. Recovery R09: partial hits, snapshot LRU, cancellation and short resources

Extended native MTP3 tests add a1600-token partial hit, three new prefix identities that evict the original from the four-entry recurrent snapshot cache, original-prompt recomputation, and completed-step cancel/slot reuse after warm restore. Cold280/280 and prefix257/257 pass,exit0. All ten raw-ID controls match exactly; prefix starting positions[0,3200,3200,0,3200,1600,0,0,0,0]. The cancelled owner never emits again; subsequent output matches the original. Initial cold test260/261,exit1 incorrectly required a completion from an incomplete prefill chunk; the corrected test expects no output there. Failed executable/source/log retained.

Prefix backend-owned device peak21,973,124,196B (20.46GiB); Linux host peak RSS13,581,844,480B (12.65GiB). Live backend bytes rise by1,353,540B on the first append shape, then remain21,958,837,284B for requests2..9. Engine release leaves947,295,772B in persistent backend resources and no pinned bytes; this is not a zero-allocation or repeated-engine leak-free claim. GPU high-water excludes driver/library-owned allocations; driver free is sampled at step boundaries. Fixed native GDN workspace230,686,720B (220MiB), graph bytes0.

Receipt/executables/sources/raw outputs/resources/commands: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-prefix-lifecycle-v1/receipt-v1.json`. Short prefix protocol/lifecycle is functional, long-context and admission remain pending. Next real32K cold/warm/append, then64K reuse/eviction/cancel and memory/page accounting before128K/exact262144. Numerical/target/serving gates remain open. Production inactive; no commit/push.

## 61. Recovery R09: real32K MTP3 cold/warm/append

The focused prefix harness now accepts explicit prompt lengths through65536 for separate bounded runs; larger admission is not yet claimed. Real tokenized repeated passages P32768/P32768/P32784 with native MTP3 and O8 each pass cold303/303 and prefix147/147,exit0. Prefix first positions[0,32000,32000], steps23/3/3 versus cold23/23/23. All three raw-ID outputs exactly agree. Prefill chunks1600 align snapshot boundaries; appended16 prompt tokens are actual input tokens.

Prefix backend-owned GPU peak22,967,503,784B (21.39GiB), host peak RSS13,598,646,272B (12.67GiB). Released backend live1,080,515,936B, pinned0, graph0. The fused oneDNN SDPA trace reports scratch descriptor0 throughout; no hypothetical full score-tensor allocation is inferred. Backend peak excludes driver/library-owned allocations; step-boundary driver free is retained. No timing score or numerical qualification.

Initial build failed on an unparenthesized doctest boolean expression. A previous binary was launched too early, stopped with exit137 and excluded; stale-named artifacts retained. Corrected build-v2 succeeds before both counted32K runs. Frozen source/executable/build/commands/results: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-prefix-32k-v1/receipt-v1.json`. Next64K reuse/eviction/cancel, then admission/page accounting before128K/exact261120+1024. S1/R06 target/serving gates remain open. Production inactive; no commit/push.

## 62. Recovery R09: real64K MTP3 reuse, eviction and cancel

P65536/P65552 natural repeated-passage controls use native MTP3, FP8 KV, O8 each and139-block test pools. Cold914/914 and prefix599/599 pass,exit0. Prefix first positions[0,64000,64000,0,0]: genuine64K-near warm reuse/append, a distinct prefix, then recomputation after the original's snapshot/page cache is displaced. All five raw-ID outputs exactly equal their cold controls. A subsequent warm cancel restores64000, emits no later cancelled-ID output, and fresh reuse emits the original8 IDs.

Prefix backend-owned GPU peak26,213,530,872B (24.41GiB), Linux host peak RSS13,592,801,280B (12.66GiB); released backend live945,934,000B, pinned0, graph0. Actual per-step driver free and live ownership are retained. Fused oneDNN scratch descriptors remain0. These backend GPU metrics exclude driver/library-owned allocations; no unmeasured total-driver peak or performance qualification is claimed.

Frozen executable/source/build/commands/raw IDs/resources/traces: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-prefix-64k-v1/receipt-v1.json`. R09 functional32K/64K prefix/lifecycle delivery is available;128K and exact261120+1024 remain. Inspect shared target/draft page-table admission accounting before those runs, then R10 graph ownership and R11 final screen. Original S1/R06 numerical gates and target/serving false remain. Production inactive; no commit/push.

## 63. Recovery R09: shared native MTP page allocation

EXL3 MTP now registers its separately stored draft KV layer in the target FA group, matching the page IDs actually consumed by the proposer. The loader opts in only for MTP; other checkpoint/method topologies retain their separate draft group. Resolved names count every target/draft storage layer in the byte budget. The runner excludes the additional draft layer from the target membership mask and allocates its own KV buffer against the target geometry.

Focused configuration2cases36/36 and actual MTP3 prefix lifecycle261/261 pass,exit0. All ten O8 raw-ID outputs equal retained cold controls; starts[0,3200,3200,0,3200,1600,0,0,0,0], cancel/restore reuse passes. Fixed64-block test storage remains unchanged; the savings are in duplicate global-ID reservation, not omitted KV bytes. Host configuration in this first receipt is synthetic48-layer16FA/32GDN; the native pinned model has64 layers,16FA/48GDN plus its draft. Initial build was deliberately stopped,exit137, before testing to limit the option to MTP; no stale executable run. Completed build-v2 exits0.

Frozen sources/executables/commands/raw controls: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-shared-pages-v1/receipt-v1.json`. Numerical/target/serving gates remain open; no commit/push.

## 64. Recovery R09: bounded recurrent admission and retirement

The shared EXL3 MTP topology now selects the existing aligned Mamba allocator. Request-owned native Conv/FP32 SSM storage is unchanged. Page bookkeeping retains k+1 current identities plus one transition identity, while the logical block-table row remains full-length and null-padded. Non-shared configurations retain mode none.

Focused host3cases1524/1524,exit0 verifies the pinned64-layer geometry's byte count, startup refusal below170 global blocks, every allocation through262144 positions, at most five live recurrent identities, second maximum request rejection, and free/readmission. Pool accounting includes164 target page identities, five recurrent identities and one null block; all17 actual FA/draft buffers are charged,55,705,600B per global block. This is metadata allocation evidence, not262K inference or1024 emitted tokens. Initial test61/62,exit1 wrongly applied waiting-only full-prompt reservation at every running chunk; corrected to the scheduler's actual calls without changing thresholds. Failed executable/source/log retained.

Actual prefix MTP3 lifecycle261/261 and genuine C4/C2 control84/84 pass,exit0. All ten prefix outputs and six batch outputs exactly match retained cold/off controls. Prefix backend GPU peak21,973,124,196B; host RSS peak13,574,574,080B, released backend947,295,772B,pinned0,graph0. GPU backend metrics exclude driver/library allocations. Frozen evidence: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-recurrent-admission-v1/receipt-v1.json`. Next actual128K and exact261120+1024 native capacity. S1/R06 qualification failures remain open; production stays inactive; no commit/push.

## 65. Recovery R09: actual128K MTP3 capacity

A dedicated single-request capacity harness uses max_model_len262144,180 global blocks, four configured recurrent request slots, FP8 KV, native MTP3 and recurrent prefix publication. Actual tokenized natural repeated passage P131072/O8 completes in84 engine steps,786/786 assertions,exit0; six drafts proposed and eight native-selected emitted IDs retained. This proves one long request, not four simultaneous maximum contexts. Initial compile failed on a nonexistent StepInputs accessor; corrected build exits0 before any GPU run.

Backend GPU peak30,443,559,840B (28.35GiB), Linux host RSS peak13,586,153,472B (12.65GiB). Released backend945,934,504B,pinned0,graph0; actual step-boundary driver-free observations retained. Backend peaks exclude driver/library-owned transient allocation. No numerical/oracle or matched-performance qualification is inferred. Frozen source/executable/build/commands/input IDs/output IDs/resources: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-capacity-128k-v1/receipt-v1.json`. Next exact261120+1024 native MTP3 capacity, thenR10/R11. Original S1/R06 failures remain open; production inactive; no commit/push.

## 66. Recovery R09: exact261120+1024 native maximum context

The same successfully built capacity executable completes actual P261120 and exactly1024 emitted output tokens:262144 total,420 engine steps,767 native draft proposals,4318/4318 assertions,exit0. Native MTP3, FP8 KV, full target/compact draft contract and recurrent prefix publication remain enabled; four recurrent request slots are provisioned with a180-block global pool. Highest target input position262142 is within the context. This is one long request with greedy ignore-EOS capacity output, not four concurrent maximum requests or a quality/oracle benchmark.

Backend GPU peak30,608,586,304B (28.51GiB), Linux host RSS peak13,595,750,400B (12.66GiB). Engine release leaves983,960,968B in backend resources,pinned0,graph0. Live device and driver-free samples are retained across all420 steps; backend peak excludes driver/library-owned transient allocation. Complete input/output IDs, trace, resources, command and frozen parent executable/source identities: `/home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r09-capacity-262k-v1/receipt-v1.json`.

R09 eager functional prefix/lifecycle/admission/maximum-context delivery is demonstrated. S1/R06 numerical failures and target/serving qualification remain open; no thresholds changed. NextR10 persistent graph hidden/logit outputs and ownership/replay controls, thenR11 final screen. Production remains inactive; only chat UI remains in Docker; no commit/push.

## 67. Recovery R10: paired graph hidden/logit output ownership

Dense graph slots now hold a persistent normalized FP16 MTP hidden tap and logits. Holding either returned output retains both buffers and a one-shot producer readiness event; a live output lease retires the captured generation before its destinations are replaced. The proposer waits on the producer event before consuming the hidden tensor. CPU capture-routing/ownership controls pass5cases89/89,exit0, including held outputs after graph destruction and retirement/re-capture. Initial88/89 exposed the sticky captured() diagnostic; it now reflects live captured slots.

Frozen component source/executable/build/red and green controls: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-graph-output-v1/receipt-v1.json. CPU events have no native handle; this is not real SYCL replay or cross-queue qualification. Public hidden-tap graph dispatch and real native controls follow. Existing S1/R06 numerical gates stay open, target/serving false; production inactive.

## 68. Recovery R10: public native dispatch and metadata preflight attribution

The public dense route passes MTP hidden taps into supported decode/speculative graphs while retaining eager prefill/mixed fallback. A fresh native MTP3 eager C1 control passes23/23,exit0: two64-token requests exactly match the existing no-MTP raw-ID baseline. The real graph arm refuses its first capture,exit1, before returning captured logits: metadata is being written inside capture. Both initial and first corrected graph executables/logs are retained, without claiming replay success.

Per-layer step-local RoPE row indices moved into persistent StepDevInputs. Host ownership/routing controls remain89/89,exit0. Backend refusal diagnostics now identify the affected validation; the next real run narrows the remaining refusal to EXL3 SmallM shard_of_nb. Single-projection routing maps were zero-filled on every call. The current correction stores the immutable generated map with the weight, and adds a focused real single-projection graph replay/changed-input control. Compilation/testing of this correction is pending; a compile API typo is retained in build-v6.log and no stale executable was launched after it. XPU already mandates persistent graph input staging; its existing policy remains unchanged.

Sources/executable identities, commands and incremental logs are under /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-native-c1-v1. R10 remains partial; real native capture, state semantics, C2/C4 and GPU lifetime cases remain unqualified. Target/serving false, existing numerical thresholds unchanged; production inactive, no commit/push.

## 69. Recovery R10: real C1 capture/replay functional control passes

Generated one-group EXL3 routing metadata is now owned with its projection and initialized once outside capture. A focused real four-bit MLP-down projection test passes31/31,exit0: captured M128 output is bit-identical to eager and the frozen reference, the map address is stable, and a second replay observes changed zero input. The shared StepDevInputs RoPE row indices remain immutable and persist across capture. The backend preflight guard is unchanged apart from a diagnostic identifying the affected check.

Corrected build-v7/v8 complete before counted tests; no stale binaries. Final host ownership/routing controls pass5cases89/89. Same frozen build native MTP3 C1 eager23/23 and graph27/27 pass,exit0. Two requests emit64 IDs each, exactly equal to eager and the prior ordinary control. Proposal/acceptance counters also agree:54 proposed/47 accepted each. The graph arm records two genuine SYCL segments and replays16 times by request0 and34 cumulatively by request1; eager records zero. Backend graph device bytes8192 are reported without inferring a total driver/library footprint or a serving speed.

Frozen sources/executables, command lines, raw outputs and failed preflight attempts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-native-c1-v1/receipt-v1.json. This completes the bounded native C1 functional graph component, not R10 state/lifecycle qualification. Next is all-layer Conv/F32 SSM comparison, C2/C4 supported forms and GPU queue/output/cache lifetime cases. Original S1/R06 numerical gates remain open and unchanged; target/serving false. Production inactive; no commit/push.

## 70. Recovery R10: full C1 GDN state snapshots expose repeat variability

The native C1 test optionally copies every Conv/F32 SSM row in all48 GDN layers after each of its two completed synchronous MTP3 requests. Both arms use the same frozen executable and independent read-only probe queue; snapshots are diagnostic outputs, never runtime inputs. Eager1180/1180 passes. Graph2656/3104,448 failures,exit1 fails exact-byte row equality while retaining exact64-token IDs, counters and all36 speculative transaction traces. Actual two captures/34replays remain witnessed.

The first differing final state is GDN layer2 SSM, max absolute1.8067657947540283e-7, relativeL2 2.0658236797990362e-7. The identical eager request repeated in the same engine also differs there, and its second result exactly matches graph in this layer. Later layers amplify differences; last SSM graph/eager relativeL2 .0028929475093169062. This rules out attributing the observed final-state difference solely to graph dispatch; it does not resolve the earliest arithmetic cause or qualify states. Existing strict byte test and original numerical bands remain unchanged.

Frozen snapshots/source/executable/commands/failed assertions and CPU-only diagnostic: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-c1-state-v1/receipt-v1.json. Full C1 state qualification remains open alongside S1/R06. Next dependency-ready work is supported C2/C4 native capture and lifecycle witnesses; no broad repeated D64 campaign follows this read-only evidence. Production inactive; no commit/push.

## 71. Recovery R10: supported C4/C2 native graph witnesses pass

The focused batch test has a separate matched graph-control mode with output caps16/20/24/31 and ignore_eos=true in both arms, allowing genuine C4 verification to outlive both cold ring slots. Historical R08 caps/EOS behavior remain the default. Per-step capture/replay counters are recorded; each group must witness a replay while the full requested concurrency and all four MTP verification rows/request are active, rather than merely a later C1 tail.

Same frozen build eager112/112 and graph119/119 pass,exit0. All six raw-ID completions match: C4 outputs16/20/24/31, C2 outputs16/20. Full C4 replay steps3/4/5 and full C2 replay steps2/3/4/5 are observed. Total six captures/thirteen replays; the platform graph batch limit remains4. This is functional C4/C2 dispatch and reuse evidence, not EOS/in-flight/cancellation qualification or a performance score.

Source/executable/commands/raw outputs/logs: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-batch-graph-v1/receipt-v1.json. Full C1 state comparison remains failed and preserved with native eager repeat variability; S1/R06 gates unchanged. Next is bounded same-input FP16 BA repeatability attribution and remaining GPU output/queue/cache lifecycle cases. Target/serving false, production inactive; no commit/push.

## 72. Recovery R10/R02: deterministic dense FP16 BA closes native C1 state control

The same-input BA operator had previously varied3/6/7 half values at P128. Dense FP16 oneDNN descriptors now request deterministic reductions; GPTQ4 and EXL3 W8A8 descriptors retain their prior policy. The focused actual P128/D1 BA test requires zero differing halves across all three repeat pairs, passes36/36,exit0, and observes jit:gemm:any/scratch descriptor0. This is native repeatability, not oracle qualification. Source/binary/logs: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-ba-deterministic-v1/receipt-v1.json.

The newly linked full native C1 MTP3 eager1180/1180 and graph3104/3104 pass,exit0. All96 Conv/SSM tensors across48layers/all4slots are bit-identical between the two eager requests, and both graph requests'192 tensors/768rows exactly match their eager controls. Both64-token raw-ID sequences remain exact, with two actual captures/thirty-four replays. The earlier448-row failure and its source are preserved; only the native dense-F16 reduction policy changed in the product.

Full-state source/executable/snapshots/commands: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-c1-state-deterministic-v1/receipt-v1.json. This closes the bounded native C1 state/control regression, not original S1/R06 target/draft numerical gates or R10 in-flight ownership. Next reconfirm C2/C4 on the arithmetic repair, then supported GPU event/output/cache/cancellation cases. No timing qualification; production inactive, no commit/push.

## 73. Recovery R10: C4/C2 reconfirmed after dense FP16 repair

The exact C1 state-qualified deterministic candidate binary is reused for matched C4/C2 controls. Eager112/112 and graph119/119 pass,exit0. All six outputs16/20/24/31 and16/20 exactly match; both full C4 and full C2 replay witnesses remain true. Source/executable identity is the C1 deterministic receipt, and independent batch logs/commands/results are frozen in /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-batch-graph-deterministic-v1/receipt-v1.json.

These are supported batch functional controls after the arithmetic change. C1 all48-layer/all4-slot state equivalence is established separately; full C4 states, original oracle numerical gates, in-flight output/event/cancellation/cache cases remain open. Next real GPU paired lease/queue/retirement tests; graph batch limit4 unchanged. Production inactive; no commit/push.

## 74. Recovery R10: real GPU paired outputs survive retirement and engine destruction

Read-only runner output accessors let the focused public engine test retain actual outputs. Hidden-only and logits-only owners each retain the paired FP16 hidden/F32 logits storage and producer event. Two independent consumer queues read all bytes after every subsequent native step and after engine destruction;341/341 assertions pass,exit0. Both64-token outputs are exactly equal; retained generations force retirement/new destinations and re-capture, with2captures/16replays. The sampler has already completed its queue when public step() returns, so this does not claim pending cancellation or arbitrary early release of asynchronous consumers.

Frozen sources/executable/commands/raw result/build and prelaunch attempts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-gpu-output-lifetime-v1/receipt-v1.json. Next direct supported forward pending/event cancellation and remaining graph lifecycle/cache/prefix controls. Original S1/R06 gates remain open, target/serving false. Production inactive; no commit/push.

## 75. Recovery R10: pending native graph forward cancellation and event consumers

The focused test uses the production Scheduler/GPUModelRunner execute_model/sample_tokens split over the real model. Completed warmup steps sample/propose/update normally using only native IDs. The next actual graph replay returns without sampling; its producer event is observed incomplete both before and immediately after scheduler abort. An independent consumer waits that one-shot producer event and enqueues device copies while the retained pair stays owned. A replacement request resets/reuses the native owner and emits the exact16 control IDs. All paired bytes remain exact after replacement and runner/graph destruction;76/76 assertions pass,exit0.

Frozen source/executable/commands/provider trace/raw pending witnesses: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-pending-cancel-v1/receipt-v1.json. This qualifies the supported abandoned-forward/retained-consumer case, not arbitrary release before a consumer finishes. Public C4/C2 EOS/mixed graph lifecycle and prefix/cache controls follow; original S1/R06 numerical gates remain open. Target/serving false, production inactive; no commit/push.

## 76. Recovery R10: public graph batch lifecycle and eager transition

The public lifecycle test records actual captures/replays, checks mixed-prefill steps remain eager, retains early EOS and unused-slot sentinels, and compares cancelled-owner replacement with an independent native control. Eager497/497 and graph489/489 pass,exit0. Ordered1/4/2/1 concurrency, EOS after5 tokens, all raw IDs/finish reasons, exact sentinel bytes and cancelled-owner reuse agree. Four genuine captures/sixteen replays are observed. A separate26-token native prefill between already-active graph phases proves graph->eager->graph, with12 new output IDs exact across arms.

The original one-token Hello reset case remains covered. v1 incorrectly expected that shape to be eager; graph482/484,exit1 and source/binary/logs are retained. It legitimately takes the single-row graph path, so v2 adds a real multi-token prefill instead of weakening the transition assertion or altering thresholds. Frozen evidence: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-public-lifecycle-v1/receipt-v1.json. Next graph prefix cache/reuse/eviction/cancel and full supported batch GDN states, thenR11. Original S1/R06 numerical gates remain open, target/serving false. Production inactive; no commit/push.

## 77. Recovery R10: graph prefix warm/partial hits, LRU eviction and cancellation reuse

Matched native prefix controls now use16 outputs only in the explicit graph-screen mode; legacy8-output mode remains. Eager396/396 and graph357/357 pass,exit0. All ten16-ID outputs and the16-ID continuation after cancellation agree exactly. First positions0/3200/3200/0/3200/1600/0/0/0/0 witness full/partial hits and LRU miss; cancellation uses3200. Every request has genuine graph replays (2/4/4/5/5/5/5/5/5/5); four captures/fifty-one replays total.

Graph backend peak22,009,251,150B, host RSS13,567,651,840B. After engine release backend983,391,958B, graph0,pinned0; library/driver transients are not inferred from these counters. Frozen source/executable/commands/outputs/provider trace: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-prefix-graph-v1/receipt-v1.json. Next all48-layer/all16-slot C4/C2 state comparison, thenR11. Existing S1/R06 qualification remains open, target/serving false. Production inactive; no commit/push.

## 78. Recovery R10: full native C4/C2 Conv and FP32 SSM states agree

The batch control optionally snapshots every allocated recurrent row after each completed group. Same frozen candidate eager1270/1270 and graph7805/7805 pass,exit0. All192 Conv/F32 SSM tensors across48 GDN layers/all16 slots in C4 and C2 are bit-identical:3072 row checks, zero differing bytes, independent frozen SHA256 equality. All six raw-ID completions, finish reasons and proposal counts agree. Full C4 replay steps3/4/5 and C2 steps2/3/4/5 remain witnessed, with6captures/13replays total. Snapshots never enter runtime execution.

Source/executable/commands/state files and hashes: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-batch-state-v1/receipt-v1.json. Together with C1 full states, paired output retirement/destruction/two queues, true pending cancellation/event consumers, public EOS/mixed/reset/sentinels/graph-eager-graph and prefix LRU/cancel controls, this completes the bounded supported R10 functional integration. R10 implemented/operator-tested true; original S1/R06 gates, target/serving qualification and overall completion remain open. NextR11 frozen practical capability screen and honest attributed serving measurements. Production inactive; no commit/push.

The public dense-graph header now documents supported uniform verification, paired MTP ownership/readiness, the requirement to retain a carrier until cross-queue reads finish, and the separate aux/non-MTP slot-view contract. Documentation-only correction after the recorded state test; no automatic test needed, and the frozen test receipt remains identified by its original source snapshot. R11 will build/freeze the current source candidate.

## 79. Recovery R11: frozen practical driver and first executable code task

The public native engine driver uses one fixed262144/1600/C4/180-block MTP3/FP8/prefix configuration and the pinned chat template with thinking explicitlyfalse. Eight held-out tasks are frozen before inference: four executable Python functions, two60872-token retrieval questions and two schema-constrained JSON/tool records. The semantic checker executes code in a resource-bounded child with restricted definitions/builtins, checks outputs/exceptions/unchanged inputs and retains malformed/truncated output failures. Six positive/negative grader controls pass.

First native merge-interval task79/79 passes,exit0; P98/O106 terminates atEOS and its four held-out execution checks pass without mutation. The partial grader deliberately exits1 because only1/8 tasks were selected. CPU-only pinned tokenizer confirms all98 first native input IDs exactly; retrieval bounds pass. Build optional-max_tokens typo and BatchEncoding diagnostic mistakes are retained, not claimed as model failures. Frozen source/executable/tasks/commands/results: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-capability-v1/receipt-first-code-v1.json. Full suite is running on the same candidate in b70-r11-capability-v1, tools session13985; continue polling that live handle, never restart solely on an observation timeout. Step wall times are labeled full engine steps, not target-forward timing or serving speed parity. R11/target/serving and original S1/R06 qualification remain open. Production inactive; no commit/push.

## 80. Recovery R11: full practical screen passes6/8, structured MTP errors exposed

All8 frozen tasks ran on the same native candidate. The request-completion harness341/341 exits0, which is not a quality success. Semantic validation exits1: four code functions/all20 held-out execution cases and both60872-token retrieval questions pass exactly; both schema/tool requests terminate error with incomplete JSON. Grader v2 has7/7 positive/negative controls and explicitly rejects error finish reasons; the original frozen grader/results are preserved.

Scheduler.cpp1320 assigns FINISHED_ERROR when the structured FSM rejects returned IDs. Source attribution shows assemble_sample_logits still calls apply_grammar_bitmask with an empty spec-token map under a stale T0-only assumption, while failed requests actually have four verification rows. The manager also retains T0 row-capacity/draft-advance assumptions. Next correct full structured/MTP plumbing and test original failed tasks, preserving their inputs/limits/constraints; no switch to unconstrained output or MTP0. Frozen suite/raw answers/IDs/semantic checks/provider trace/source: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-capability-v1/receipt-suite-v1.json. Suite process/container is terminal; only chat UI remains. R11, original S1/R06 and target/serving qualification remain open. Performance comparisons follow feature correction; production inactive, no commit/push.

## 81. Recovery R11: structured MTP/EOS integration closes frozen capability screen

Scheduler now keeps only each grammar-valid draft prefix without mutating its real state, including async trim/validate/pad. Manager grows masks for all draft/bonus rows and rolls back actual successful advancements on either completion or failure. Runner retains scheduled spec row counts and masks every expanded logit row in dense request order. Focused scheduler/manager55/55 and mixed mask144/144 pass,exit0. First real native ticket now produces complete valid JSON but still terminates error;33/33 completion assertions are not a quality pass. This retained v1 exposes the second defect: backend only knows tokenizer.json EOS, while Engine InputProcessor resolves model/chat/generation EOS. The lazy factory now shares that resolved set; focused primary/secondary EOS15/15 passes,exit0.

Both original schema tasks then finish stop and pass strict held-out validation with MTP3 and unchanged schemas/budgets. Selected single-task graders intentionally exit1 because each1/8 result is incomplete. Full same frozen v2 engine suite391/391 passes,exit0; semantic8/8 passes,exit0: four functions/all20 execution cases, both60872-token retrievals and both JSON/tool records. Every prompt ID matches the failed original suite; all six previously passing raw-ID completions remain exact. Schema outputs29/79tokens, accepted/proposed21/22 and58/58. After engine release graph bytes0. Async validation/padding is unit-tested; the GPU capability screen uses synchronous scheduler/runner.

Frozen v1 failure/v2 sources, builds, executables, original tasks, commands, answers, provider traces and semantic outcomes: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-structured-mtp-v1/receipt-v1.json. R11 practical capability component is complete; wholeR11 measured optimization, original S1/R06 gates and target/serving qualification remain open. Next native serving timing/cost instrumentation and planned like-for-like sequential producer comparison; these capability step times are not performance qualification. Production inactive; no test/oracle container remains, no commit/push.

## 82. Recovery R11: native serving timestamps, bounded profiling and MTP cache boundary repair

The frozen native serving harness records actual emitted chunks, arrival/first-output/final timestamps, public MTP cycle wall times, proposed/accepted counts, actual scheduled positions and capture/replay deltas. Vendor-free optional target/draft queue brackets submit nothing when disabled; focused off/host/device-host controls pass68/68,71/71,81/81. The metric tool counts aggregate output in one common overlap interval and explicitly distinguishes diagnostic profiles from serving measurements.

The initial exact C1P4096/O1024 run crashes in BlockPool::cache_full_blocks; GDB on the frozen old binary localizes the missing shifted MTP boundary hash. A tiny BS4 regression reproduces SIGSEGV after2 assertions. Cache publication now waits until both full computed page and committed next-token hash exist; the pool rejects missing hashes/allocations before mutation. Focused two-case29/29 passes,exit0. MTP3, prefix caching, budgets and hash policy remain unchanged. Corrected frozen v2 C1 passes2049/2049,exit0:1024 actual IDs,621/1209 accepted/proposed,2 captures/401 replays. Decode6.48217 emitted tokens/s,TTFT15.92221s,TPOT154.26925ms,E2E173.73968s. This cold first request includes native lazy kernel/graph warmup; model loading is excluded. It is one run, not a matched Python comparison or three-repeat qualification.

The separate profile initially hits the fixed host record cap and aborts; retained failure is not a result. Harness-only per-cycle draining after each cycle timestamp bounds pending diagnostics without raising caps or changing the unprofiled arm. Frozen v3 profile passes2049/2049,exit0 and emits exactly the same1024 IDs. Its403 decode cycles average target queue span358.69250ms and draft33.66268ms; queue spans include host gaps and nested stages are not additive. The two real eager warmup cycles3/4 expose129 Gemma5120 norms each, totaling301.48938/295.35261ms of371.05510/361.11719ms target spans. Graph profiling exposes aggregate graph compute, not individual captured-node costs. This identifies a bounded norm optimization opportunity without guessing a GEMM bottleneck.

Initial C4 identical prompts generate all four256-token outputs but fail three cold-position assertions at3200: prefix caching legitimately reuses earlier requests. Preserve that failure. Four natural vegetable variants, encoded on CPU by the pinned tokenizer with intact chat template and exactly4096 tokens each, have distinct first pages. With the same binary and unchanged strict cold assertions, C4 passes2353/2353,exit0; all four start0. Actual common overlap contains951 emitted tokens at19.74668 aggregate tokens/s, E2E138.29716s,569/1359 accepted/proposed,5 captures/110 replays. Neither case changes the original numerical bands.

Frozen sources/binaries/workloads, failed builds/crashes, raw profiles, timestamps, source-identified cost split and commands: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-v1.json. Next scoped compile-time norm geometry specialization and immediate original identical-operand/alias checks, then actual native remeasurement and remaining32K/mixed/finalC4 plus sequential pinned producer. Current performance candidate has not rerun the practical screen; historical8/8 is retained under its own source. R11/target/serving remain incomplete; production inactive, no commit/push.

## 83. Recovery R11: native FP32 Gemma boundaries reduce measured C1 cycle cost

The geometry-only specialization passes821/821 original controls and2049/2049 native C1, with exact1024 IDs, but measures6.47424 versus6.48217 decode tokens/s. It provides no demonstrated gain in this single focused trial and is removed from product source. Frozen source/binary/result/commands remain in norm-specialization-rejected-v1.json; this is not a speedup.

The next scoped hypothesis replaces external explicit-RN calls only inside the eligible FP16 Gemma5120 path with native FP32 adds/multiplies. Local clang contract(off) and existing -fno-fast-math/-ffp-contract=off preserve separate F32 boundaries; precise division/sqrt flags remain. The original runtime row geometry, reduction tree, reciprocal factor, residual/output stores and shared Q/K helper are unchanged. Short/odd M3/M4/M12/M16, aliases/padding, layer0/layer3 controls pass821/821,exit0. Original P128/D1 input/post-norm plus aliases pass64/64,exit0; all four normalized outputs and residuals are byte-exact. The full target P128/D1 comparison against all three fixed producer repeats passes197/197,exit0. An initial existing default export-name refusal159/160,exit1 is retained; the retry uses a fresh output prefix on the separate receipt mount and never overwrites originals.

Frozen v5 C1P4096/O1024 passes2049/2049,exit0, with every1024 raw ID and every per-cycle proposed/accepted count exact to the unprofiled baseline. Same406 cycles,621/1209 accepted/proposed and2 captures/401 replays. Decode27.92915 emitted tokens/s,TTFT14.70534s,TPOT35.80489ms,E2E51.33377s. The observed single-trial decode ratio is4.30861; neither this nor the earlier baseline is a three-repeat or Python parity qualification. Model load is excluded, lazy native startup remains included. Backend peak30,349,646,306B,host RSS13,582,405,632B; backend graph bytes0 after release.

Separate same-candidate profiling also passes2049/2049,exit0 and emits identical IDs. Mean decode target queue span83.36417ms and draft8.41948ms, versus358.69250/33.66268ms before. The actual first two eager decode cycles'129 norm kernels total24.12636/20.42032ms, versus301.48938/295.35261ms. These diagnostic nested/queue intervals are not added to unprofiled serving throughput. The whole implementation is still unqualified: same frozen v5 D64 validation fails3 assertions,804/807,exit1. D11 TV.0221232, D29 TV.287664/KL.227609, minimum top10 overlap9; ancillary greedy mismatchD27. All original source failures and fixed TV.02/KL.002/top10>=9 bands remain. The greater D29 discrepancy is not explained away as harmless rounding; strict state and R06 draft/full-target qualification were not rerun or transferred.

Source/compiler flags/executables/fixtures, original controls, raw logits, new/old cost splits, successful and failed runs: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-norm-native-f32-v5.json. Next pinned producer timing with exact workload-v2 IDs and actual launch/cache/warmup receipts, remaining native32K/mixed/finalC4 cases, small sequential three-repeat comparison and current-source practical screen. Historical8/8 belongs to its earlier frozen source. R11/target/serving remain incomplete; production inactive, only chat UI remains, no commit/push.

## 84. Recovery R11: matched warmup, pinned producer timing and current-source screen

The native harness optionally runs the exact case with O32 per request, using native feedback and the same arrival policy, then clears prefix state before scoring. Frozen v6 changes only this harness relative to the v5 norm product. Incremental build exit0; warm C1 passes2071/2071 and final C4 passes7794/7794,exit0, with cold first positions0 and exactly1024 output IDs/request. C1 IDs remain exact to un-warmed v5. The shared report's producer branch passes its focused positive/negative controls; it records actual chunks and supplied proposal statistics without relabeling frontend batches as GPU MTP cycles.

The pinned offline Python public-engine tool verifies checkpoint/image/plugin/subset and resolves actual worker/configuration receipts. Same FP16/FP32 GDN/FP8 policy/MTP3, max262144, batch1600,C4,pool180 and synchronous FULL_DECODE_ONLY capture sizes1/2/4/8/12/16. Exact original prompts, budgets and O32 warmup/reset are shared. Initial config misses the parent b70_inventory module; corrected frozen root mount passes configuration. The first real C1 completes warmup but refuses an incorrect physical-dtype assertion. Pinned FP8 cache policy uses torch.uint8 storage; v2 explicitly checks both, retains the failed v1 and completes actual C1/C4,exit0. No threshold or model behavior is relaxed.

First matched C1: native27.77448 versus producer55.74871 emitted decode tokens/s; TTFT4.88902 versus1.94625s; E2E41.72142 versus20.29647s. Native621/1209, producer612/1239 accepted/proposed. Both emit1024 real IDs from the same4096 prompt; their free continuations differ. This is one trial each, not numerical/quality parity or three-repeat qualification. Constructor startup and real O32 warmup are excluded; prefix hits are forbidden during scoring.

First submitted C4 final case: native133.62631s versus producer76.16722s E2E for4096 total output tokens,30.65265 versus53.77642 E2E tokens/s. Native common decode overlap contains3791 tokens at66.42725 aggregate tokens/s. Producer first/final times show four sequential decode phases with no common overlap, so its C4 overlap metric and the matched simultaneous-decode ratio are null. Do not compare native overlap throughput with producer E2E throughput. Native TTFT5.70256/33.17424/51.57680/71.42678s; producer1.95132/19.86078/39.21542/59.46450s. All prompt IDs match; all four free output sequences differ. Read-only pinned source supports an attribution: engine-core cache-block negotiation chooses the actual1600-token groups; align splitting rounds a fresh smaller first chunk to zero when a running decode has consumed part of the1600 budget, then defers waiting admission. This is source-based inference consistent with timestamps, not an instrumented scheduler trace. The fixed comparison configuration is retained.

The remaining native constructor-only representative cases pass on v5: P32768/O256633/633,exit0, TTFT49.27112s/decode12.74296 tokens/s/E2E69.28233s; mixed P128/O256 plus P32768/O2561058/1058,exit0,512 emitted tokens/E2E224.04384s. The mixed trace witnesses21 actual short-decode/long-prefill cycles; the long request's relative TTFT190.86383s remains explicit. Its27.37338 tokens/s common overlap does not hide slow prefill. These cases have no explicit request warmup and are not compared with the newly warmed Python cases.

Current frozen v6 practical screen passes391/391 and semantic8/8,exit0: four functions/all20 held-out cases, both60872-token retrievals and both structured/tool records. All8 prompt IDs and raw completions are exactly equal to the historical successful candidate; all finish stop. Peak backend30,461,556,091B; graph bytes0 after release. Historical evidence remains source-identified separately.

Frozen source/build/workload identities, commands, failed attempts, actual config/allocations, outputs and comparisons: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-serving-matched-v6.json. The planned small sequential three-repeat C1/C4 comparison is next; C1 native repeat2 is running on the same frozen candidate. Original S1/R06 and v5-product D64 failure804/807 remain release blockers, with unchanged bands. R11/target/serving false; goal active, production inactive,180W unchanged, no commit/push.

## 85. Recovery R11: planned three-repeat C1/C4 comparison complete

Frozen native v6 and pinned producer v2 run sequentially, three unprofiled trials per arm/case at180W, with identical case prompt IDs, output caps, MTP3/subset, FP16/FP32 GDN/FP8 contract and O32 request warmup followed by prefix reset. No performance trace is enabled. Every native C1 passes2071/2071; every native C4 passes7794/7794; all six producer runs finish length with exactly1024 tokens per request,exit0. Configuration/images/tool identities and actual proposal counters are recorded. No variant continuation is injected.

C1 native decode27.77448/27.62786/27.81462 emitted tokens/s, median27.77448; producer55.74871/63.98214/64.01968, median63.98214. Median native/producer ratio.43410 describes these actual autonomous outputs, not same-continuation or numerical parity. Native IDs and621/1209 counters are identical in all trials. Producer repeat1 differs from repeat2/3 beginning at output index48 (39666 versus6761); repeat2/3 match exactly. Producer612/1239 then666/1074 accepted/proposed. Same prompt, tool/image and overrides are verified. This variation is retained rather than explained away as harmless rounding or converted into a confidence interval. Native median TTFT4.88902s; producer1.95314s.

C4 E2E native133.62631/133.72834/133.77234s, median133.72834; producer76.16722/76.25775/76.38876s, median76.25775, for4096 actual output tokens each run. Median E2E throughput ratio.57024. Native aggregate common-overlap decode66.42725/66.43523/66.44828 tokens/s; producer common overlap remains null in all three trials. Never compare these native overlap rates with producer E2E throughput or label the producer frontend batches GPU cycles. Each arm repeats all four own raw ID sequences exactly; cross-arm sequences differ. The same fixed1600-token batch budget constrains the pinned producer's align admission; these results do not claim its best possible production configuration.

Immutable six-run receipts/raws/logs/source/commands: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-producer-serving-v1/comparison-c1-three-repeat-v1.json and comparison-c4-three-repeat-v1.json. Both original first-trial comparisons remain intact. This completes the planned small three-repeat comparison component only. Warm matched32K/mixed producer comparison and numerical target/state/draft release qualification remain open. Separate native mixed short-decode/long-prefill profiling is now running on v6 with constructor-only startup matching its unprofiled v5 baseline; no cost result is claimed yet. R11/target/serving false, goal active; production inactive,180W unchanged, no commit/push.

## 86. Recovery R11: mixed profiling narrows synchronous readback/queue waits

Separate constructor-only native v6 mixed profiling passes1059/1059,exit0; both256-token raw sequences exactly match the unprofiled v5 baseline. The extra assertion is the optional warmup-environment validation; no workload cap, arrival policy, MTP3, prefix rule or numerical threshold changes. Actual135 cycles include21 mixed target steps6–26, short positions>=128 and long positions<32768. This diagnostic's239.25740s E2E includes profiling/draining and is not serving throughput.

Device target brackets for those21 mixed steps total190.81499s; enclosing host target brackets independently total190.78906s. Host staged_D2H_wait totals161.58968s across33642 calls (84.695% of those host target spans). A representative step7 has9.02602s host target duration and8.25143s staged readback wait. These waits include previously queued GPU work; they are neither isolated transfer bandwidth nor exclusive CPU compute. Metadata validation16.08277s, metadata_D2H15.35598s and named validation-error intervals overlap and must not be summed. The next small code change is profile-only copy byte/caller attribution at this existing staged-copy seam, preserving its ordinary safety/ordering, followed by one narrowed readback/metadata repair and immediate operation/state checks.

The kernel trace does not establish GDN arithmetic as the dominant cost. Across the same21 steps it witnesses the chunked GDN path, including11,856 pair prepares/inverses and24,672 cross/output/state events. W8A8 validation/reconstruction/output stages and oneDNN stream brackets are recorded separately. Queue/library brackets include host gaps; nested spans are not additive. A source-only suspicion about mixed GDN is superseded by the actual synchronization attribution, without changing arithmetic or disabling validation.

For precision clarity the common cache contract is Conv history FP16 and recurrent SSM FP32, in addition to FP16 model boundaries and FP8 KV policy. The producer's actual MambaSpec dtype pair is[torch.float16,torch.float32]; pinned dtype-calculator source confirms(Conv,SSM) order. Native all-state snapshot controls likewise require Conv F16/SSM F32. This is no new qualification or cache-format change. Physical group allocation/layout and the open cross-stack numerical failures retain their separate evidence.

Frozen source/commands,12 unprofiled comparison runs, failed attempts, profile/raw-ID checks and clock-separated cost reports: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-three-repeat-mixed-cost-v6.json, mixed-cost-report-v6.json and mixed-host-cost-report-v6.json. Existing status now records completed small comparison and mixed cost capture; warm matched32K/mixed producer timing and original S1/R06/D64 qualification remain open. R11/target/serving false, goal active. GPU cases are terminal, production inactive;180W unchanged, no commit/push.

## 87. Recovery R11: copy byte/caller attribution localizes W8A8 mapping waits

Profile-only staged-copy records now carry actual chunk bytes and caller return PCs; ordinary copies, submission/completion ordering and staging lifetime stay unchanged. Incremental focused build exit0. Real4MiB+37-byte roundtrip and existing span/operation controls pass off72/72,host92/92,device-host117/117 assertions,exit0. Byte sums, both actual chunks, calling PCs and unchanged data are checked. Native raw exports preserve these diagnostic fields.

Frozen v7 constructor-only mixed profile passes1059/1059,exit0; both256-token prompt/output ID sequences are exactly equal to unprofiled v5. Observed135 cycles again contain21 actual mixed steps6–26. Enclosing host target spans total189.95489s; staged D2H waits160.87910s across33642 calls. Observed process maps identify the non-PIE executable; addr2line resolves the return PCs against that frozen executable. W8A8 shard_of_nb readback at0x4a918b,160 bytes,2688 calls accounts for136.74704s; larger mapping chunks and other validation waits are recorded separately. Release symbols resolve function rather than precise source line; source inspection distinguishes the three W8A8 Copy calls.

These host waits include previously queued work, and eliding mapping readback could move the wait into required svh/input validation. No isolated bandwidth or speedup is claimed. Next small diagnostic records the existing memcpy device events, adding no commands, to distinguish copy execution from preceding queue work before a runtime repair. Numerical thresholds and all failure receipts remain unchanged.

Immutable source/build/binaries/commands/controls/raw/maps/caller report: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-copy-attribution-v7.json and mixed-copy-caller-report-v7.json. New v7 profiling source has no inherited full practical8/8; canonical v6 screen and three-repeat measurements retain their own source identity. R11/target/serving remain false; goal active, no commit/push.

## 88. Recovery R11: actual memcpy events exclude a shard-copy bandwidth bottleneck

Frozen v8 records existing staged H2D/D2H memcpy events, without adding commands or changing waits/staging/arithmetic. Initial compile refuses const SYCL event.wait_and_throw; mutable handle correction builds successfully. The failed log remains immutable. Real multi-chunk copy controls require nonzero device timestamps, same-queue nonoverlapping copy intervals and exact roundtrip bytes: off74/74,host94/94,device-host146/146,all exit0. The existing norm/BA timestamp probe now explicitly includes its output readback event.

Same constructor-only mixed case passes1059/1059,exit0; both256 IDs exactly match unprofiled v5 and21 mixed cycles remain6–26. Per-queue/per-direction ordinal pairing checks equal40594 D2H and10900 H2D host/device record counts and ordered synchronous device intervals. Host/device epochs are never subtracted. For2688 W8A8 shard-map160B copies, host wait totals136.62663s while the copies' device intervals total4.592927ms. Actual target host spans189.88124s and diagnostic E2E238.71719s. This excludes those small copies themselves as the measured transfer bottleneck and supersedes any inferred gain from eliding mapping readbacks. Required validation/lifetime checks remain.

The preceding elementwise nonlinear and row-copy commands are not yet individually recorded in the original profile. Next small instrumentation records their existing events, then focused operator tests and this same mixed case, before any math/layout change. Frozen source/build/failure/control/raw/command evidence: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-copy-events-v8.json and mixed-copy-event-report-v8.json. No current-v88/8 or numerical qualification is inherited; canonical v6 results remain distinct. Whole R11 and numerical/serving qualification remain open; goal active, production inactive,180W unchanged, no commit/push.

## 89. Recovery R11: measured duplicate-row scatter is the dominant mixed kernel

Frozen v9 records already-submitted copy/add/nonlinearity/gather/scatter device events with unchanged commands and arithmetic. Focused original dtype/stride/alias/index controls and profile probes pass5841/5841 off,5861/5861 host,5913/5913 device-host,all exit0. Same constructor-only mixed profile passes1059/1059,exit0; both256-token outputs exactly match unprofiled v5. Actual21 mixed target queue spans total190.28021s. IndexCopy accounts for137.36283s over2016 GPU kernel events, whereas SiLU totals1.48858s over1344 events. This supersedes the source-only nonlinearity suspicion.

Source inspection identifies the actual repeated work: each scatter column searches every later source index to preserve last-write-wins for duplicate destination rows. Next scoped product change moves that check to one GPU workgroup per source row, uses a collective any-later result, then copies the row's columns. No new allocation, readback or host wait is introduced; existing index validation and WithOutput alias preservation stay in place. The developer explicitly prefers GPU-resident dataflow and avoiding host wait loops; measured producer work must still be completed safely. Immediate tests cover wide1600x5120 and ragged37x129 actual FP16 data, unsorted duplicate IDs, aliased snapshots, untouched destination rows and negative/out-of-range refusal before mutation, followed by selected unprofiled mixed serving remeasurement.

Frozen diagnostics/source/build/commands/controls/raw/cost report: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-elementwise-events-v9.json and mixed-elementwise-cost-report-v9.json. Numerical thresholds/failure artifacts unchanged; no new current full practical/target qualification. R11/serving incomplete, goal active, no commit/push.

## 90. Recovery R11: GPU row-group last-write-wins removes the measured mixed scatter bottleneck

Product v10 changes only IndexCopy with more128 source rows. One GPU workgroup searches later duplicate IDs collectively once for its source row, then distributes the row's byte copying across256 lanes. Existing small-row/decode launch, metadata validation and WithOutput alias snapshot stay in place. No new temporary allocation, readback, host wait or floating-point operation is introduced. Both unique and duplicate mappings preserve exact last-write-wins.

Incremental focused builds exit0. Original dtype/stride/alias/index/profile controls plus new ragged37x129, optimized-boundary129x257 and wide1600x5120 controls pass5895/5895 off,5915/5915 host,5967/5967 device-host,all exit0. Actual GDN core is[T,48,128]; a test-only v11 extension with1600 such source rows passes72/72,exit0 on the same frozen product kernel. Every destination byte, untouched row and nonaliased source is checked; aliased inputs use their original snapshot. Real graph capture/replay recomputes changed unsorted duplicates, and negative/out-of-range indices fail before output mutation.

Same unprofiled constructor-only mixed workload passes1059/1059,exit0. Both prompts/all512 emitted raw IDs and all135 cycles' positions, emitted/proposed/accepted counters and graph deltas exactly match v5. E2E224.04384->84.47434s, observed single-trial ratio2.65221. Long request relative TTFT190.86383->51.30697s; common decode overlap323 tokens at27.59495 emitted tok/s. Same21 actual mixed steps6–26,322/576 accepted/proposed,4 captures/109 replays. Peak backend30,467,049,480B unchanged,host RSS13,549,924,352B; graph bytes0 after release. This is one same-continuation before/after, not three-repeat or producer/quality/numerical qualification.

Separate profiled v10 mixed run also passes1059/1059,exit0 with exact IDs. Old2016 scatter kernels total137.36283s; new1008 large-row workgroup kernels total191.978433ms plus unchanged1008 small kernels12.687621ms, aggregate204.666054ms. Actual21 target queue spans190.28021->51.13898s. Device-copy/host waits remain independently paired: small DMA intervals do not justify eliding guards. Profile E2E100.00090s is diagnostic only. The reduction confirms a GPU kernel bottleneck rather than transfer bandwidth or presumed SiLU/GDN arithmetic.

Frozen sources/builds/controls/commands/raws/comparison/metrics/costs: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-index-copy-optimization-v10.json. Current runtime candidate is v10; extra rank3 controls change tests only. Canonical v6 three-repeat and practical8/8 remain separately identified; no new current full practical/target qualification is inherited. Next current-source practical evidence and matched O32-warmup/reset32K/mixed native/producer cases, then one measured remaining GPU-resident metadata/wait opportunity. Original S1/R06/D64 failures and bands remain explicit. Whole R11/target/serving incomplete, goal active; production inactive,180W unchanged, no commit/push.

## 91. Recovery R11: current scatter candidate passes the unchanged practical8-task screen

Same frozen product v10 capability run passes391/391,exit0; unchanged semantic validator passes8/8,exit0: four code functions/all20 held-out cases, both60872-token retrieval tasks, strict ticket/tool schemas. All8 original prompt IDs and autonomous raw output sequences are exactly equal to canonical v6; all finish stop. Graph bytes0 after release. This is current-source practical evidence and does not transfer original target/state/draft numerical qualification or unmeasured performance.

The initial start accidentally retained the old provider trace filename. That own test is aborted(exit137), its full appended trace preserved separately, and the original35,847,296-byte v6 prefix restored only after verifying SHA256 d069d4f232b0ff0c4410be82fc57810b1d12e46df7fa451f436dc23c527495eb. Host restoration initially refuses root-owned output permissions; CPU-only no-device helper performs the same verified restoration. The retry uses separate raw/log/trace files and completes normally. Neither abort nor failed restoration is counted as a passed test.

Source/binary/commands/current answers/validator/provider trace and failed-attempt/recovery artifacts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-capability-index-copy-v10.json. Next matched O32 warmup/reset32K and mixed native/producer cases, at unchanged180W and sequential GPU ownership. Whole R11 and numerical/serving release qualification remain open, goal active; production inactive, no commit/push.

## 92. Recovery R11: actual matched warm32K native/producer case measured

Frozen current native v10 runs P32768/O256 after identical real-request O32 warmup then successful prefix reset. Native653/653,exit0; exactly256 outputs finish length,first scheduled position0,all prompt/output IDs exact to constructor-only v5. TTFT39.98335s,TPOT79.15018ms,decode12.63421 emitted tokens/s,E2E60.16680s;148/327 accepted/proposed. The un-warmed v5 result retains its own startup state and is not used as a matched producer comparison.

Sequential pinned producer v2 uses unchanged image/tool/checkpoint/subset/FP16 Conv-FP32 SSM/FP8 policy/MTP3/max262144/C4/batch1600/pool180 and actual0-prefix-hit guard after its O32/reset. Actual producer exit0:256 outputs finish length,TTFT18.57784s,TPOT20.22534ms,decode49.44291 emitted tokens/s,E2E23.73536s;153/315 accepted/proposed. Every prompt ID and cap matches. Cross-arm free IDs differ starting index0=56(46040 versus70269),retained explicitly. This is one trial per arm at unchanged180W, not numerical/quality/same-continuation or three-repeat qualification.

Frozen raws/metric/source identities: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-producer-serving-v1/comparison-32k-warm-v10-v2.json. Current warm mixed native run is next,then sequential producer. Repaired scatter also serves native overlapped C4 prefills; the final small C4 comparison must receive current v10 measurements rather than transfer v6 timings. Current practical8/8 remains source-identified; original S1/R06/D64 release gates remain open. Whole R11/target/serving false,goal active,production inactive,no commit/push.

## 93. Recovery R11: matched warm mixed submitted workload completes with explicit admission difference

Current native v10 warm mixed case passes1091/1091,exit0 after40 real O32 warmup cycles and successful prefix reset; first scoring positions0. All512 prompt/output IDs exactly match un-warmed current v10 and the original v5 outputs. Same135 cycles,21 actual mixed steps6–26 and322/576 accepted/proposed. E2E70.79778s,shortTTFT.20633s,long relativeTTFT48.71200s; common decode overlap323 tokens at27.36299 emitted tokens/s. Constructor startup/request warmup are excluded,scoring prefix state cold.

Pinned producer v2 actual same workload and O32/reset completes exit0 with512 emitted outputs,0cached prompt tokens,all finish length,320/585 accepted/proposed. E2E27.68211s,shortTTFT.17469s,long relativeTTFT22.19185s. Its short request finishes4.09047s before long first output; common decode overlap and matched simultaneous-decode ratio are null. Public arrival remains triggered by actual short>=16 emitted tokens,not injected IDs; output delivery intervals do not prove exact GPU scheduler cycles. The earlier pinned align/block1600 source inference remains consistent with deferred long prefill admission; no new scheduler-cycle capture or best-production-config claim.

All actual prompt IDs, caps, sampling and fixed comparison configuration match. Cross-arm autonomous outputs differ: short first output index0=0(26104 vs479),long index0=51(10345 vs7854). Differences are retained,not explained as harmless rounding or counted as numerical/quality parity. This completes the one-trial representative matched warm32K/mixed component only. C4 overlap uses the newly repaired scatter too; current-source three-repeat C4 measurement starts next rather than inherit canonical v6 timings.

Frozen sources/commands/initialized worker allocations/raws/metrics/comparisons: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-warm-long-comparison-v10-v2.json and /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-producer-serving-v1/comparison-mixed-warm-v10-v2.json. Current practical8/8 is recorded,original S1/R06/D64 numerical gates remain open. WholeR11/target/serving false,goal active,production inactive,180W unchanged,no commit/push.


## 94. Recovery R11: current scatter candidate three-repeat C4 measurement

Frozen product v10 independently runs all three unprofiled warm C4 P4096/O1024 trials after per-request O32 warmup and prefix reset. Each passes7794/7794,exit0. Every4x1024 raw output/prompt ID and all405 cycle positions/proposal/acceptance/logical-token/graph-delta fields are exact to the original v6 control. E2E84.821714179/84.706619327/85.049605669s, median84.821714179s versus original native median133.728335310s: descriptive1.576581x reduction in time ratio. Native common decode66.658994/66.768751/66.483336tok/s, median66.658994tok/s.

The unchanged source/image/config/power-identified pinned producer three trials remain76.167219/76.257753/76.388756s, median76.257753s. Native E2E throughput ratio0.899036; native elapsed time is11.23% larger. Producer has no common four-request decode interval, so no matched simultaneous decode ratio is reported. Cross-arm free IDs differ; no quality/numerical parity or optimal producer settings claim. All prior native/producer trials remain frozen.

Comparison with source/binary/raw/log/report hashes and launch commands: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-producer-serving-v1/comparison-c4-three-repeat-v10-v2.json. Current-source practical8/8 and warm32K/mixed were measured separately. Next bounded product step combines GPU SVH/input finite checks in existing W8A8 workspace and one readback, retaining errors/output invariants; focused operator checks run first. Numerical S1/R06/D64 and full target/serving qualification remain open. Production inactive; no commit/push.


## 95. Recovery R11: W8A8 finite checks stay on GPU with one combined readback

The product replaces the separate serial SVH metadata check/allocation with parallel ESIMD128-element SVH blocks. SVH/input flags share spare words in the existing64B scale region and one8B readback. SVH error priority, raw input checks, mapping bounds and output-before-error invariants remain. This removes one metadata allocation, readback/wait and retirement drain per large-M call; mapping readback and finite-result synchronization remain. No completely wait-free or large-M graph qualification claim.

Expanded NaN/Inf/overflow controls exposed an existing finite-contract hole. Initial merge v12 and diagnostic v13 fail104/106,exit1. Actual original R03 wrapper/identical ValidateRows control v14 also fails104/106,exit1 at FP16 product overflow and output mutation. Explicit pre-narrow F32 integer-bit bounds fix product overflow, then v16 exposes the second Hadamard boundary107/109,exit1. Both pre-conversion checks now compare absolute bits against0x477ff000, the65520 round-to-nearest FP16 overflow boundary, before preserving the half bit checks. The donor input quantization, GEMM and output transform themselves are unchanged; no original operands are substituted. The initial const ESIMD view build error v15 is retained. Post-conversion checks were insufficient in the observed fast-math build; no compiler-IR causal claim.

Frozen v17 focused real grouped4-bpw GDN controls198/198 and6-bpw head controls192/192 pass,exit0. Captured quantized activations/scales/last packed panel remain exact; F16 intermediates/final outputs report zero differing halves; M128 boundary, padding, noncontiguous group routing and reuse remain covered. NaN/positive/negativeInf SVH first/last blocks and input, combined-invalid error priority, both finite-operand overflow boundaries, unchanged output on rejection and valid reuse after failed calls pass. No CPU reference route selected.

Source/binary/build/fixture/control identities, launch commands and failed variants: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-w8a8-validation-operator-v17.json. No v17 serving/capability timing or numerical qualification is transferred from v10. User explicitly requested checkpoint commit/push; follow with current-source mixed serving/exact-ID cycle comparison and separate cost profiling, then bounded numerical target/draft localization. Original S1/R06/D64 and target/serving gates remain open; production inactive.


## 96. Recovery R11: W8A8 combined validation reduces real mixed elapsed time

User-requested checkpoint2255ebd125fb886cf3a24f75aa3d4be97a9d0924 was committed and pushed to origin/b70-gptq-int4 with89 product/test/plan files; local review archives stayed untracked. Subsequent v17 warm mixedP128/P32768 O256 each passes1091/1091,exit0. All512 prompt/output IDs and135 per-cycle proposals/acceptance/positions/logical-token/graph-delta fields are exact to warm v10. E2E66.315986722s versus70.797778877s, descriptive single-trial6.33% elapsed-time reduction/1.06758 time ratio. Long-relative TTFT44.285965840s versus48.712001103s; common323-token decode27.439764tok/s. Backend peak30,521,661,843B, host peak13,549,580,288B, released graph0; both remain within32GiB. No speed/quality parity claim.

Separate constructor-only profile1059/1059,exit0 has exact IDs/counters to constructor-only v10. Warmup changes graph capture/replay deltas, so profile compares its own matching-startup baseline. Actual21 mixed target device/host spans46.827711253/46.801007874s versus51.138975524/51.112992671s. W8A8 finite readbacks10752 separate4B ->5376 combined8B; host waits9.382191 ->4.371213s, including prior GPU work. Input validation device time4.333216s versus3.888795s includes the stronger overflow guards. Selected D2H host24.532076s, actual copy43.776234ms. Complete per-queue/direction synchronous-copy counts/ordering are verified; nested spans are not additive and diagnostic intervals are not serving throughput.

Source/build/commands/raw/log/metrics/profile/cost: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-w8a8-validation-serving-v17.json. CPU-only stdlib inspection of the pinned producer excludes one route hypothesis: supported M<=small_m_max returns SmallM before the later M>16 int8 branch; default128/no image or pinned-profile override, and plugin passes that value. The native M128 boundary remains unchanged; this source check does not explain actual output-ID or D29 differences.

Next bounded numerical step: execute the full6-bpw248320-token head on preserved original D29 hidden operands/logits, separating head arithmetic from upstream own-state trajectory. No captured operands enter product inference. Original S1 strict-state/R06/D64 bands and failed receipts remain. Current v17 full practical/final representative timing/numerical qualification still open. Production inactive.


## 97. Recovery R02/R11: D29 identical-input full head is byte-exact

A focused checkpoint full6-bpw248320-token head test uses only preserved original final-normalized F16 hidden at P128/D1/D29. All248320 F16 logits in each phase equal the actual original capture byte-for-byte.34/34 assertions pass,exit0, native provider/reference-tier0. Selected checkpoint trellis/scale/mul1 payloads, captured operands/output hashes, source/build/binary/commands and the initial missing-include compile failure are frozen in /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-head-v2.json. These captured operands enter the isolated operator test only, never product inference.

A CPU-only audit reproduces the retained v5 D29 failureTV0.2876643873/KL0.2276087431. Most probability mass changes involve248046 (native0.412409 versus original0.137800),248068 (0.554957 versus0.805473) and248045 (0.005356 versus0.042027). Argmax248068 matches, which does not waive the large distribution error. Logit differences1.21875/-0.25/-1.9375 are observed, without causal rounding attribution. Same-input head equality narrows this case upstream of the final head.

Next bounded collector retains own native state and the original teacher-forced prefix only throughD29, exports actual final-normalized hidden for P128/D1/D11/D27/D29, and keeps all original probability checks/failures. The existing fullD64 test is not shortened or replaced. Then identify the first diverging decoder block; no broad state-dump campaign or relaxed numerical bands. Production inactive; pushed checkpoint2255ebd125 remains available, subsequent diagnostic/tracking changes are local.


## 98. Recovery R02/R11: actual own-state final hidden localizes D29 jump

The new separate bounded diagnostic executes P128+29 teacher-forced forwards with native-owned state, preserving the original fullD64 test. It exports only final-normalized last-row F16 hidden at P128/D1/D11/D27/D29 and30 logit rows. Collection completes but the numerical test fails505/508,exit1: D11 TV0.0221232, D29 TV0.287664/KL0.227609, unchanged gates; D27 ancillary greedy mismatch persists. Every30 current logit vector is byte-exact to the preserved failing v5 trajectory, so the diagnostic tap does not alter these logits.

Actual final-hidden relativeL2/max-absolute-error: P128 .00266035/.046875; D1 .00347129/.0625; D11 .01701007/.375; D27 .01551853/.087890625; D29 .96992083/24.1953125. Original D29 RMS1.6367653, errorRMS1.5875328,5119/5120 half values differ. These are diagnostic metrics without a new acceptance threshold. The large D29 jump upstream of the already same-input-exact head is not labeled benign rounding.

Source/binary/build/commands/actual token/position/slot layouts, all selected hidden and logit hashes, provider/reference-tier0 and explicit failed gates: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-hidden-v3.json. Next narrow per-block operands at the actual failing step against observed original boundaries and identify the first divergent operation, preserving reference repeat variation separately. No broad state-dump campaign, no captured-value product substitution, no relaxed gates. Production inactive; subsequent diagnostics/tracking remain local after pushed2255ebd125.

## 99. Recovery R02: D29-only block boundaries and independent original control

A scoped programmatic dump selection preserves all forward ordinals but rejects unselected steps before device download/compaction. Default all-step diagnostics are preserved; serving dumps remain disabled. Focused writer controls47/47 pass, buildexit0. Actual own-state P128+29 run remains505/508,exit1 with the same three numerical failures. Exactly786 blobs/7,336,960B contain onlyD29; all30 logits are byte-exact to v3 and every64 input-norm direct/recheck read is byte-exact. Initial copied-binary permission126 is retained, corrected before GPU inference.

Pinned original D29 selected-block capture and separate no-block control each exit0, with actual token and three-axis position witnesses matching the frozen prefix. Block0 boundaries are byte-exact. First nonzero D29 difference occurs at block1: input norm exact, post-attention norm relativeL2 .000456841 and MLP delta .000842956. This identifies a boundary, not the causal GDN operation or preceding state history.

New selected original final hidden differs from native by relativeL2 .02563366; native-vs-new logitTV .02838168 remains an investigation trigger. New original-vs-legacy final hidden relativeL2 .96087751. Independent no-block original hasTV .00784438 versus selected original, butTV .25237636 versus legacy original. Separately retained matched-prefix D64 original repeats already differ atD29 byTV .02307226/.13529636/.11339447. No-block control shows the large legacy discrepancy is not exclusive to the new block-hook arm; separate launches and different warm generation lengths do not establish a single causal variability mechanism. No numerical envelope or acceptance threshold changes, and no inherited operator relative-norm gate is applied to diagnostic boundary metrics.

Source/binary/commands/import and permission failures/raw original/native captures/comparisons: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-blocks-v6.json. The initial missing frozen inventory import exit1 is retained and corrected before GPU inference. Next bound actual D29 block1 GDN active states/operands and replay identical inputs, distinguishing local arithmetic from preceding own-state drift. Production inactive, no competing vLLM; checkpoint2255ebd125 remains pushed and subsequent diagnostics remain local.

## 100. Recovery R02: D29 layer1 identical-state GDN is exact; own initial SSM differs

Pinned original bounded D29 layer1 detail captures actual FP16 Conv[3,10240], FP32 SSM[48,128,128], raw BA/QKVZ, gated norm and mixer outputs with active metadata/slot3. The native isolated layer1 consumer uses the same input, transposes Conv into native[5,10240,3] slot4 and retains exact FP32 SSM.43/43 assertions pass,exit0: all5120 mixer values and all786432 active SSM values exact, Conv exact, four inactive slots unchanged. Independent CPU raw-byte/transpose checks confirm both initial and final state equality. Initial v8 fixture-dtype string/enum compile error retained; v9 correction built,exit0. Receipt /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-layer1-v9.json. Original v5/v7 block0 is exact; first original-vs-original differences also occur at layer1 post-attention norm/MLP with identical input norm.

Native own-state capture before/after actual D29 remains521/524,exit1 with the same three numerical failures, all30 logits byte-exact to v4. Actual layer1 input norm and all30720 Conv values before/after equal the detailed original; four inactive slots unchanged. Actual initial SSM differs786088/786432 values, relativeL2 .0003537498886, maxabs .0002025067806. Final relativeL2 .0003421720014. Actual mixer relativeL2 .0001437404885; no new acceptance band. Thus D29 local consumer has exact identical-state evidence while its actual starting SSM already differs. The earlier operation/state history causing that difference remains open. Receipt /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-own-layer1-v10.json.

Next narrow layer1 P128/D1 state and arithmetic under identical operands before more full-target diagnosis. Captured states remain confined to isolated tests; no product substitution, threshold relaxation or target/serving qualification. Production inactive. User-requested2255ebd125 checkpoint pushed; later diagnostic/tracking changes remain local.

## 101. Recovery R02: initial layer1 state difference is projection-dependent

Actual own-input layer1 P128 fails426/427,exit1 at the unchanged strict SSM band; its fatal check prevents D1. Under identical original projections, P128 core and all786432 active FP32 SSM values are exact; independent matched-state D1 mixer/SSM are exact. The two separately reported controls pass59/59,exit0. Actual native BA three repeats pass37/37 with zero differing halves at P128/D1. Original default BA P128 repeat pairs differ4/6/6 halves, while original deterministic K/N-contiguous variants equal native byte-for-byte. Actual full-model original BA weights are F16[96,5120],strides[5120,1].

Independent integer-dot audit covers20 observed mismatch positions plus32 seeded held-out scalars. Native/deterministic outputs match exact nearest-half rounding40/52, observed full worker45/52, default repeats50/50/46; every selected output is within one half encoding. This does not establish native higher accuracy, an all-matrix bound or target qualification. A separately named full original D29 deterministic-BA capture still yields native TV.0761893/KL.0120133. No reference promotion or release threshold change.

## 102. Recovery R02: first layer0 prefill discrepancy is gated norm

Scoped layer selection retains serving/default dump behavior. Writer controls59/59 pass,exit0. Two bounded own-state P128/D1 model runs each pass218/218; all native parent logits and outer stages are byte-exact before/after nested observer enablement. At layer0 P128, input norm and all786432 GDN core halves match the separately identified original deterministic-BA diagnostic. Gated norm differs49 halves(relativeL2 1.381828e-6,maxabs .0000610352), mixer10642. D1 gated norm/mixer/outer boundaries are exact; its packed path does not expose core. The first comparison's missing D1-core error is retained; corrected report explicitly marks it unobserved.

Frozen source/build/binary/commands/original/native artifacts and retained failures: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-prefill-attribution-v18.json. Next isolated same-input gated-norm consumer and narrowly scoped arithmetic repair. Original D64/strict-state/draft failures remain open; no target/serving qualification transfer. Production inactive, goal active; subsequent changes remain local after pushed2255ebd125.

## 103. Recovery R02: actual FP16 gated norm and first prefill block become exact

Pinned original isolated static/split gated norm reproduces actual captured P128/D1 endpoints byte-for-byte. Its F32 mean is reproduced for all6144/48 rows with Torch's non-vectorized D128 virtual32 reduction. Baseline native fails71/75,exit1 with49 differing P128 halves in every gate-stride/alias layout, D1 exact. Reduction/rsqrt variant retains46; explicit rounding throughout worsens919/7 and is not promoted. A typed ordinary-division variant retains40/0 and its F32 product equals the standalone no-precision-division arm byte-for-byte. The standalone explicitly rounded SiLU division reproduces every original mean/inverse/activation/normalized/weighted/product F32 value. These observations do not establish compiler-IR causality or justify global math flags.

Final product selects one SG16 kernel for F16 core/gate, F16/F32 norm weights, width128, >=32 rows and SiLU. It implements the original virtual32 reduction/rsqrt/F32 multiplication boundaries and explicitly rounded SiLU division. Generic modes remain on their previous route. No new host readback or wait is introduced. Actual model trace proves weightF16 and gate token stride16384; the first F32-weight-only passing79/79 control did not execute this route in the model and is not transferred as integration evidence.

Expanded same-input test passes183/183,exit0: all16 P128/D1 x weight-type x actual padded-stride x alias combinations have zero differing halves; both weight types also produce byte-exact original F32 products. Final bounded own-state model passes218/218,exit0 against the unchanged three legacy logit references. Layer0 P128 and D1 input norm, gated norm, mixer, post-attention norm and MLP output now all equal the separately identified deterministic-BA original byte-for-byte; P128 core remains exact. D1 maxTV versus legacy .00351968. Shared generic post-conv/gated-norm and F16 strided-alias controls pass2cases,120/120,exit0. Capture-selection/policy controls pass3tests. Missing directory/dependency/filter/export and exploratory arithmetic failures are retained and not counted as passes.

Source/build/binary/fixtures/commands/raws/comparisons/failed attempts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-gated-norm-v41.json. This closes the localized first-layer prefill discrepancy, not the original full-target D64/state/draft gates. No reference promotion, post-fix D29 or serving-speed/capability qualification. Next bounded current-source own-state D29 comparison, then the next divergent operation or independent qualification. Goal active, production inactive; subsequent product/test/diagnostic changes remain local after pushed2255ebd125.

## 104. Recovery R02: repaired GDN closes first three D29 blocks; attention3 is next

Current-source native-owned P128+29 bounded run fails518/524,exit1 at unchanged original gates: D11 TV.022471, D24 TV.022253, D27 TV.0230159, D29 TV.264349/KL.194832/top10=8. Six failed assertions versus the prior three are retained; no overall numerical qualification or fullD64 claim. Original failed runs and references remain unchanged.

Against separately identified original deterministic-BA diagnostic, all observed D29 boundaries in GDN blocks0/1/2 are now byte-exact. First nonzero boundary is full-attention block3: input norm exact, post-attention norm1923 differing halves/5120, relativeL2 .0003161497,maxabs .000244140625. All64 input-norm direct/recheck reads equal. Final-hidden relativeL2 .1443897; whole D29 TV.05414384/KL.00726958/top10=9 still triggers investigation. The earlier controlled diagnostic TV.0761893 and all legacy comparisons remain separately identified; none becomes a release envelope.

Frozen current binary/command/dumps/native layer1 initial/final states/all30 logits/selected hidden/comparison/failed gates: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-gated-v42.json. No captured-state inference substitution. Next observe original attention3 D29 Q/K/V, gate, rotated operands and active FP8 KV, then identical-input/state consumer attribution before changing arithmetic. Target/serving false, goal active, production inactive. Pushed2255ebd125 checkpoint remains available; subsequent kernel/test/diagnostic/tracking changes remain local.

## 105. Recovery R02: exact attention operands isolate core and missing FP16 sigmoid boundary

The bounded original attention3 D29 observer delegates original execution and copies only initialized KV rows:156 before,157 after. All487 parent tensors and ordinary IDs equal the separately identified deterministic-BA original v16 byte-for-byte. Native observer controls preserve all30 v42 logits; the bounded numerical run remains529/535,exit1 with six retained failures. Actual input norm, raw Q/gate/K/V, rotated Q/K and every initialized FP8 K/V byte match. First difference is the native split-K attention core:1104/6144 halves, relativeL2 .00015414294,maxabs .0009765625. Its arithmetic remains open; exact operands do not qualify the output.

An isolated original-input attention-gate regression initially fails40/46,exit1: P128204805,D11560,D291603 differing halves in each alias mode. Independent exact-half arithmetic reproduces every original endpoint only when sigmoid is narrowed to FP16 before multiplying. Product XPU F16 attention/output now preserves this boundary using explicitly rounded division; the F32 gate input is preserved and other admitted dtype combinations retain their route. One GPU pointwise kernel, no new host wait/readback, no global math flags or captured-value product substitution.

Focused build exits0. Repaired same-input P128/D1/D29 with separate/in-place output passes46/46,exit0 with zero differing halves. Shared F16 attention-preamble case passes49/49 and generic elementwise case passes5550/5550, each one actual case,exit0. The preceding GDN repair retains its recorded183/183 isolated,120/120 shared and218/218 bounded-model evidence; no current-source full-target, draft, capability or serving-speed qualification is inferred.

Source/build/binary/commands/fixtures/raw observers/failed baseline and independent audit: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-attention3-gate-v46.json. User requested a current-branch checkpoint commit/push. Next run bounded repaired native-owned P128+29 against unchanged numerical gates, then isolate remaining attention-core arithmetic. Target/serving false, goal active, production inactive.

## 106. Recovery R02: pushed checkpoint and same-input D29 core isolate tiled FP16 probabilities

User-requested current-branch checkpoint7f7ebc5bd4293edab721716819e283f4a211028a was committed and pushed to origin/b70-gptq-int4; remote identity verified. Subsequent bounded native-owned P128+29 run retains six frozen numerical failures529/535,exit1. Legacy D29 TV.270911/KL.202450/top10=8; no overall improvement or qualification claim. Against separately identified deterministic-BA original, D29 TV.06060780/KL.00851823/top10=9 versus prior .05414384/.00726958/9; final-hidden relativeL2 .14265758 versus .14438968. All30 logit rows change after the intended arithmetic change; this is not an observer-invariance test.

Actual attention3 Q/K/V/gate/RoPE and all initialized KV bytes remain exact. Core remains1104/6144 differing halves. Corrected gating reduces2267 to1011 differences and maxabs .0001220703125 to .000030517578125; mixer differences3279 to2632. A new isolated D29 consumer reproduces precisely1104 differences in both original interleaved and native planar layouts; cache unchanged and reference-tier0.14/16 assertions,exit1: the two strict output-equality failures remain visible.

Pinned local XPU source inspection identifies FP16 unnormalized probabilities before the P*V matrix product, four SGs of16 keys within64-key tiles, native exp2 and FP32 cross-SG normalization. Independent CPU exact-dot/rounded-once approximation gives1103 differences for full F32 arithmetic,1803 for normalized-half probabilities and113 for the original tiled unnormalized-half boundaries. This narrows the next implementation; it does not reproduce actual DPAS/reduction/transcendentals exactly or certify a new GPU route.

Current/failed runs, source/binary/commands/raws/controlled comparison and independent audit: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-attention-core-D29-v49.json. Next port the bounded short-decode producer arithmetic onto the native GPU queue with completion-owned workspace, then immediately exercise the isolated D29 test. Preserve all target/draft/serving gates. Goal active, production inactive; new consumer regression/tracking changes are local after pushed7f7ebc5bd.

## 107. Recovery R02: short C1 XMX decode becomes exact; P128 prefill remains open

A separate Torch-free C1 translation unit consumes native Q/output and planar or original interleaved E4M3 pages directly. The admitted pinned-B70 F16 Hq24/Hkv4/D256, unit scales, page1600/1664, max_seq_len1–960 route preserves original one-split XMX arithmetic and FP16 unnormalized probabilities. Existing completion-owned16MiB workspace is reused; no new host wait/readback, Q packing or scale copy. Original math options are scoped to this new translation unit. Optional packed verification retains its prior options/routes. Three unmodified BSD-noticed helpers were recovered from existing local pinned TLA Git objects; unused utility includes were removed. Initial missing-include builds and intermediate failed arithmetic controls are retained.

Inactive V tail operands are masked before DPAS, preventing zero-probability times NaN padding from poisoning output. This does not claim physical tail memory is never loaded. Focused build exits0. Expanded same-input D29 controls pass60/60,exit0: all16 page-size x KV-layout x padding x output-alias combinations have zero differing halves/nonfinite outputs/cache writes, reference-tier0. Original P128/D1 error-band control passes37/37,exit0. New strict same-input P128/D1 control fails42/43,exit1: P1281542/786432 differing halves, maxabs .001953125; D1 exact. Existing bands are unchanged, and P128 bit equality is not claimed.

Bounded own-state P128+29 fails528/535,exit1 with seven unchanged-contract numerical assertions. Actual D29 attention3 input/Q/K/V/gate/RoPE/core/gated/mixer and all initialized KV bytes now equal the separately identified original diagnostic byte-for-byte. Whole-model numerical qualification remains open: legacy D29 TV .0459077/KL .0422017 improves from .270911/.202450, but the separately controlled diagnostic TV worsens .06060780 to .17098873 and final-hidden relativeL2 .14265758 to .39353775. Therefore no overall quality-improvement claim is made. Full D64, draft/state gates and current-source graph selector/capture threshold qualification remain open; no current performance remeasurement.

Source/build/binaries/commands/local-helper provenance/raws/retained failures: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-short-XMX-v58.json. Next narrow existing native P128 prefill base2 exponential/scaled FMA/reduction boundaries and immediately rerun strict P128/D1, then bounded own-state integration. User authorizes regular tested checkpoint commits and immediate pushes. Goal active, production inactive; no milestone/target/serving qualification transfer.

## 108. Recovery R02: scoped P128 base2 softmax becomes bit-exact

The previous native short prefill already used XMX QK/PV and FP16 unnormalized probabilities. Its observed remaining1542/786432 output-half differences are closed by preserving the original raw-score maximum/log2(e) scaling, native exp2 and scaled-score-minus-maximum FMA boundaries. The arithmetic change is admitted only for single-probability F16 C1 P128/max_seq_len128, Hq24/Hkv4/D256, unit E4M3 scales, page1600/1664, causal/no window/no softcap. Other shape/dtype/mode arithmetic, existing reductions and output normalization remain unchanged. No global math options or additional host wait/readback.

Initial focused build exits0; strict P128/D1 tests pass43/43 for default Q64 and43/43 for explicit Q32. Expanded final tests include both1600/1664 page sizes with original logical slots remapped into the selected physical page geometry. Final focused build exits0; Q64 and Q32 each pass86/86,exit0 with every P128/D1 FP16 output bit-exact, active FP8 cache exact and inactive poison unchanged. The test-helper constexpr/runtime compile error is retained, corrected before the final build.

First own-state model attempt stops at D29 because the dump directory was not prepared;502 assertions/499 passes/exit1 and partial outputs are retained. Retry uses new prepared directories and finishes530/535,exit1 with five strict numerical failures. Legacy D29 TV .0234147/KL .0083951/top10=10; actual attention3 D29 observed input/Q/K/V/gate/RoPE/core/gated/mixer and initialized KV remain bit-exact to the separate controlled original. The whole controlled diagnostic worsens: TV .17098873 to .21191660, KL .08034247 to .14201886, final-hidden relativeL2 .39353775 to .61631918. Thus local arithmetic parity is established, but no overall quality improvement, full D64/state/draft qualification or current performance claim follows.

Source/build/binaries/commands/raw comparisons/failed attempts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-prefill-base2-v64.json. The previous decode step51b235a748810f4ed242392928dffdf29efcf2b6 was committed, pushed and remote-verified under the user's regular checkpoint instruction. Next bound the first remaining divergent block/state after repaired attention3 on current P128/D1 with identical original inputs, and qualify current-source graph policy thresholds before transferring serving/performance evidence. Goal active, production inactive; milestones and target/serving flags remain unchanged.

## 109. Recovery R10: current short-decode graph crossing preserves full GDN states

The new short C1 XMX policy changes host dispatch at max_seq_len960. Device lengths are refreshed on graph replay, but host dispatch does not rerun. A new actual public-engine test uses a naturally encoded956-token prompt,12 native-selected outputs, no MTP, C1/one GDN slot, FP16 activations/FP8 KV/full248320 target head. It visits959/960/961/962 and records graph execution on both sides, then snapshots all48 GDN layers' Conv/F32 SSM tensors. Initial4-block test configuration fails admission before inference;8 blocks admit the same2048 maximum. The failed attempt remains retained.

Before repair, eager passes602/602,exit0; graph fails901/991,exit1. Both emit the same12 raw IDs, but90 of96 state tensors differ byte-for-byte. This is a concrete graph/eager state failure masked by fluent identical output. The shared host short-decode bound now controls both kernel admission and per-slot dense graph retirement. Crossing the bound uses the existing graph/persistent-input retirement path, then recaptures the currently selected kernel. Other-device dispatch and eager arithmetic are unchanged; no new host polling loop/readback was introduced.

Focused compile succeeds. Same frozen candidate eager602/602 and graph991/991,each exit0: exact prompt/all12 output IDs/all96 Conv+F32 SSM tensors,4 captures/7 graph executions, genuine subsequent replays above the boundary, released graph device bytes0. Independent raw-file comparison also confirms candidate eager versus prior eager all96 tensors and12 IDs exact. Existing current-source whole-target530/535 numerical failure is not erased, and earlier broad MTP/batch/prefix graph evidence is not transferred to this changed candidate.

Source/build/binaries/commands/raw states/failed baseline/independent comparison: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r10-short-decode-boundary-v1/receipt-v3.json. User's regular tested checkpoint commit/push instruction applies. Next return to bounded numerical attribution after the repaired attention3 block; retain strict target/state/draft gates and later current-source broader graph/capability/performance qualification. Goal active, production inactive; target/serving flags unchanged.

## 110. Recovery R02: first observed remaining D29 drift is GDN21 state history

Current7cf5d0b190880365e189fcdf0cbc0bed0fb6489e binary runs one bounded own-state P128+29 with all D29 substage boundaries.503/508,exit1 retains the five original numerical failures; full D64 is not shortened or qualified. All30 logits equal v61 byte-for-byte and all64 input-norm direct/recheck reads equal. Among192 observed input-norm/post-attention-norm/MLP boundaries, blocks0–20 are bit-exact to the separately identified v43 deterministic-BA original diagnostic. First observed difference is layer21 post-attention norm526/5120 halves, relativeL2 .0001394144,maxabs .0009765625; layer21 input norm is exact. This is an observed-boundary localization, not full state or release qualification.

The existing bounded original observer and native isolated GDN consumer now admit selected GDN21 while preserving GDN1/default keys. Focused CPU selection test1/1,exit0; native focused compileexit0. New pinned original GDN21 captureexit0 preserves all487 parent tensors and ordinary IDs byte-for-byte to v43. The initial missing inventory import is retained and repaired in a new frozen bundle using the existing pinned helper; no dependency/weight download. Captured original states are used only in isolated tests, never substituted into product inference.

Native same-input/same-state GDN21 passes44/44,exit0: mixer all5120 halves exact, active Conv and all786432 active SSM values byte-exact before/after by independent raw checks; all four inactive slots unchanged. The generalized old GDN1 control also passes44/44,exit0. First GDN21 attempt lacks a prepared stream dump directory and aborts before comparison; separate retry preserves this failure and passes with fresh prepared paths.

Actual own-state selected GDN21 capture remains522/527,exit1 with the same five whole-model failures; all30 logits byte-exact to the prior all-layer observer. Actual input norm, BA A/B and all30720 Conv values before/after are exact. Initial SSM already differs424403/786432 values, relativeL2 .0000439123,maxabs .0000574142; final relativeL2 .0000434515. Gated output103 halves/mixer1612/post-norm526/MLP3827 differ. No new acceptance band or overall numerical improvement claim. This distinguishes an exact local consumer from its earlier divergent own-state history.

Source/build/binaries/commands/original/native captures/raw comparisons/retained failures: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-gdn21-attribution-v72.json. No product arithmetic changes in this step. Next bound GDN21 P128/D1 initial/final SSM and actual operands under the same original protocol, then isolate producer/history arithmetic. Regular tested checkpoint commit/push authorized; goal active, production inactive, target/serving flags remain false.

## 111. Recovery R02: GDN21 state drift already exists after P128

Optional read-only early GDN21 observer preserves all496 original parent tensors byte-for-byte to v67 and emits a separately identified P128/D1 history file. Initialized P128-after and D1-before Conv/SSM are byte-exact within the original; cold unconsumed prefill state is not read. Default captures and original D64 qualification remain unchanged. CPU admission controls pass2/2 and selected replay-weight control1/1,exit0; focused native buildexit0.

Native early collector retains the original D64 prefix and stops after D1 solely for diagnosis. One case230/230,exit0; both parent logits equal v65 byte-for-byte, input-norm direct/recheck unchanged. All30720 active Conv halves match at P128-after/D1-before/D1-after. P128-after SSM differs429708/786432 F32 values, relativeL2 .0000991247,maxabs .0001051128; D1-after differs450993,relativeL2 .0000693055. P128 input norm already differs59/655360 halves, first differing token48; BA B/A differ3/7 halves and core65118. D1 input norm and BA are exact but gated333/mixer2559 halves differ. Therefore this is earlier history/upstream attribution, not proof that the GDN21 producer is wrong.

Frozen sources/build/binary/commands/original/native captures/comparison: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-gdn21-history-v76.json. No product arithmetic, inference state substitution, reference promotion, overall quality or speed claim. Five previously recorded D29 whole-model failures/full D64/strict-state/draft gates remain open. Next bound earlier P128 block/norm boundaries through layer21, then isolate the first operation on identical operands. Regular tested checkpoint commit/push authorized; goal active, production inactive, target/serving false.

## 112. Recovery R02: first P128 discrepancy is block20 MLP output

The GDN21 history observer now also copies131 bounded original P128 block/norm boundaries through layer21. One CPU selection test passes,exit0; default D29 selection/counts remain unchanged. Focused native compileexit0. New pinned original captureexit0 preserves all496 v73 main tensors and all46 GDN21 history tensors byte-for-byte. The first raw metadata counter matched two p128_logits keys as well; the receipt records the corrected131 block-boundary count explicitly, without altering the raw report.

Native one-case/two-forward collector observes all64 P128/D1 substages,199/199,exit0; no full D64 qualification is shortened. Both parent logits remain byte-exact to v75, and all22 P128 input-norm direct/recheck triples agree. Among66 compared original/native input-norm/post-attention-norm/MLP boundaries through21, the first discrepancy is block20 MLP output320/655360 halves, relativeL2 .00000498916,maxabs .000244140625, starting at token48. Every compared boundary through block19 plus block20 input/post-attention norms is bit-exact. Block21 input norm59/core/history differences therefore already have a preceding divergent MLP boundary; this does not establish that GDN21 arithmetic is wrong.

Frozen sources/build/binary/commands/raw captures/comparison: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-prefill-boundaries-v80.json. Prior checkpoint3ee03b702448cd48f184feb06e8ea8e31481bfd0 committed/pushed/remote-verified. No product arithmetic, reference promotion, overall numerical qualification or current speed remeasurement. Next observe original block20 P128 gate/up, SwiGLU and down boundaries, then identical-operand native consumer attribution with actual weights. Five original D29 failures/full D64/strict-state/draft gates remain open; goal active, production inactive, target/serving false. Regular tested checkpoint commit/push authorized.

## 113. Recovery R02: one FP16 SwiGLU midpoint caused block20 and GDN21 history drift

Pinned original P128 MLP20 observer copies gate/up, SwiGLU and down boundaries without replacing operations; all627 previous main tensors and46 GDN21 history tensors remain byte-exact to v77. Native identical-input baseline39/42,exit1: gate/up all4456448 halves exact, isolated/chain SwiGLU each one differing half, isolated down655360 halves exact, chain down320 differences byte-equal to the actual model v79. The scalar is token48/column16742, gate-2.724609375/up-.346923828125, native .058197021484375 versus original .058135986328125. Independent F64 SiLU narrowed before up multiplication reproduces the original endpoint; this scalar audit is not an all-operand accuracy proof.

Product merged SiluAndMul keeps eager FP16 materialization and uses explicitly rounded F32 division only for F16 input/output. Other dtype arithmetic remains unchanged; no new host wait/readback/global math flags. Focused compile succeeds. Identical-input repaired MLP42/42,exit0, all five stage raw files bit-exact. Actual model P128/D1 GDN21 collector230/230,exit0: all observed stages and30720 Conv/all786432 active SSM values at P128-after/D1-before/D1-after now byte-exact; all four inactive slots equal the prior candidate and P128-after/D1-before continuity is exact. Both logits change as expected after product arithmetic; observer invariance is not claimed.

Final real-midpoint packed-alias/strided-rejection control35/35 and existing generic elementwise5550/5550,each one case,exit0. Initial const-download and repeated-CAPTURE compile failures are retained and corrected. The first alias diagnostic6 assertions then exit1 uses an unsupported strided target; final control uses supported contiguous packed overlap and verifies public strided rejection leaves operands unchanged, without broadening the op contract.

Frozen source/build/binaries/commands/parent/raw/state/comparisons/failed attempts: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-swiglu-repair-v95.json. No new references/thresholds, whole D64/state/draft qualification, current D29, graph/capability or speed claim. Next bounded repaired own-state P128+29 against the unchanged gates/controlled original, then next divergent operation or full D64 qualification. Regular tested checkpoint commit/push authorized; goal active, production inactive, target/serving false.

## 114. Recovery R02: controlled P128+29 logits and all D29 boundaries are bit-exact

Pushed/remote-verified product checkpoint40673658b5bbf2cdeaba6dc1c36873dd1b0c7bd6 runs the bounded native-owned P128+29 case504/508,exit1. Four unchanged frozen failures remain D27 TV.0305754/KL.00212581 and D29 TV.212752/KL.163144/top10=9. Legacy D29 worsens from .0234147/.0083951; no default-producer overall quality improvement or qualification is claimed.

Against the separately identified controlled deterministic-BA original v81, every one of30 full248320 logit rows is bit-exact and every one of192 observed D29 input-norm/post-attention-norm/MLP boundaries is bit-exact. All64 direct/recheck input-norm reads agree; all656 dumps are step29 only. Controlled D29 TV/KL are0/top10=10, compared with prior .21191660/.14201886. This closes the localized native arithmetic drift for this protocol/prefix while retaining the original acceptance/reference separately. The first comparison script refuses an old output filename without overwriting it; corrected fresh-name script v97 completes.

Frozen native command/raws/scripts/retained frozen failures: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-D29-swiglu-v98.json. Controlled reference is not promoted; full default D64/strict-state/draft/serving/performance remain unqualified. Next full D64 validation with unchanged frozen gates and an independently pinned controlled original full-D64 capture.

## 115. Recovery R02: full controlled D64 has65 bit-exact full-vocabulary rows

Full existing D64 qualification now optionally writes complete outputs to the existing external diagnostic-output directory; references stay read-only. No step shortening, state substitution, sampling/prefix/threshold change. Focused buildexit0. Pinned original controlled D64 captureexit0 validates all65 real input witnesses against the original frozen65-token trace; its first30 logit rows equal v81 byte-for-byte. Folder's historical D29 prefix does not describe its actual64 decode steps, which are recorded in command/metadata.

Native full P128+64 one-case replay804/808,exit1 retains exactly four frozen failures at D27/D29. Separately controlled original/native compare65/65 full248320 F32-widened logit rows bit-exact, maxTV0/maxKL0, no first nonzero row. Native first30 rows equal the bounded v94 run byte-for-byte, demonstrating complete continuation rather than a shortened passing subset. Default producer/reference gates remain unchanged and are not marked passed; captured controlled values are never consumed by product inference. This is C1/eager/no-MTP/P128+64 protocol evidence, not all-prefix, long-context, graph/batch/draft or serving parity.

Frozen sources/build/binary/commands/original/native outputs/comparison: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r02-D29-head-v1/receipt-controlled-D64-v103.json. Current performance not remeasured. Following the recovery stopping rule, next current-source isolated R06 compact draft numerical gate on identical original target-hidden/state/operands. A proposed reference-contract correction still requires independent held-out evidence; no silent reference promotion. Goal active, production inactive, milestones C/D/E and target/serving qualification remain open. Regular tested checkpoint commit/push authorized.

## 116. Recovery R06 qualification: existing compact draft becomes bit-exact on current source

R06 functionality was already implemented; this is the bounded numerical requalification after shared target/GDN/attention/SwiGLU repairs, not a restart of earlier implementation. Focused relink of current a62a87d4903a905709cf99cca48fe886a8ac0360 test succeeds. Existing unchanged isolated P128 draft/compact-head case passes131644/131644,exit0. All655360 draft-hidden values,65536 compact logits and the independent identical-input65536 head values equal the same immutable original capture byte-for-byte after F32 widening. Exact65536 token-map and mapped global argmax checks pass; full target head is never constructed in this bounded draft test, reference-tier0.

Input target-hidden/shifted IDs/positions are the unchanged original operands; native owns FP8 page1600 KV with observed physical block/slot mapping and poison initialization. No reference or relative/pointwise budget changes. Earlier34 hidden/77 logit pointwise failures and alternative oneDNN29/45 failures remain in immutable old receipts; current zero-error comparison supersedes their scope without erasing them. This does not qualify all MTP3 iterations, graph/batch/prefix contexts or default target/state gates.

Frozen current source/build/binary/command/input/subset/raw comparison: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r06-compact-head-v1/receipt-current-numerical-v3.json. Next R11 post-repair frozen-candidate practical screen, then current like-for-like serving costs/performance rather than another implementation pass over R06. User's review question is addressed by preparing current performance/cost evidence before a new optimization review. No current speed remeasurement; overall milestones C/D/E/serving qualification remain open, production inactive, goal active. Regular tested checkpoint commit/push authorized.

## 117. Recovery R11: current post-numerical-repair practical screen passes

Frozen current product code (a62a87d4903a905709cf99cca48fe886a8ac0360; docs-only checkpoint43808e788408d7ca2a2242ac35c92a1461fed3c1) passes the existing native public-engine practical screen393/393,one case,exit0. The unchanged semantic grader passes8/8,exit0: four executable functions/all20 held-out execution cases, two60872-token retrieval tasks and two structured/tool records. All8 actual prompt and emitted-token sequences equal the historical passing v10 run; all finish stop. MTP3, GPU graphs, prefix reuse and FP8 KV are enabled with the existing262144 context/C4-capacity/batch1600/page180 configuration. Requests are sequential, so this does not establish simultaneous C4 behavior or timing.

Observed backend device peak30461556091B; graph device bytes0 after release. Binary provenance is the focused R06 relink and its immutable build receipt. Trace-enabled native wall times are diagnostic only, not scored performance. No code change, reference promotion, threshold change, full state/lifecycle qualification or current Python speed-parity claim. Production remains inactive; only unrelated chat UI is running.

Frozen binary/source identities/commands/manifest/grader/outputs/traces/results: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-capability-post-numerics-v1.json (9702B,SHA25634bb399f112c531b2f1a6222ba8896b7c17280d01c67fb8f0400df15fd2decb1). Current bounded controlled D64 and isolated draft numerical passes remain separately identified; four unchanged frozen D64 failures and broader strict-state/reference-contract gates remain open.

Next use the existing small matched serving cases and a separate current cost profile to choose one measured optimization. Record prefill/TTFT,target decode,MTP cycle and accepted/proposed/emitted tokens, overlap and end-to-end separately. Prior11.23percent C4 end-to-end and old32K comparisons belong to their earlier source identities. Pro review becomes useful with current source-identified measurements and explicit remaining gates; R11 is the final phase, not proof that performance work is nearly finished. Regular tested checkpoint commit/push authorized; goal active, milestones C/D/E and target/serving qualification remain open.

## 118. Recovery R11: fresh post-numerics C1 comparison and graph/eager cost attribution

Current frozen post-numerics binary completes existing C1P4096/O1024 MTP3 warmup-O32/prefix-reset test1975/1975,one case,exit0. Native unprofiled TTFT4.83705611s,34.82772ms chunk-observed TPOT,28.71276031 emitted decode tok/s,E2E40.46584006s; accepted642/proposed1152. Fresh original pinned producer same workload/config/cold-prefix/sampler/subset/180W exits0: TTFT1.948671716s,TPOT15.78921ms,63.33439783 emitted decode tok/s,E2E18.10104495s,accepted664/proposed1083. Native throughput ratio.45335/E2E elapsed ratio2.23555 are descriptive one-trial own-continuation comparisons; first output mismatch index53. No identical-continuation, three-repeat or numerical/serving parity claim.

Separate graph profile1975/1975,exit0 preserves all1024 raw IDs,387 cycles and every non-timing cycle field. Mean384 decode spans target85.54129ms/draft8.44231ms; graph compute85.11465ms is opaque. Separate eager profile1975/1975,exit0 has same final1024 IDs,387 cycles and total accepted/proposed, but acceptance/emission differ in cycles379/380 (3/4 then0/1 versus2/3 then1/2); second start position5104 versus5103. This remains explicit integrated numerical/graph evidence, not all-cycle/state equivalence. First receipt construction correctly fails its overly broad eager all-cycle equality assertion before writing outputs; failure is retained and v2 reports actual differences.

Eager diagnostic target decode mean90.03512ms: EXL3 SmallM DPAS28.04702ms,Gemma5120 norm20.52673ms,attention split partial14.89328ms,GDN spec WG10.34824ms per cycle. Stage sums may overlap and are not added across nested spans. Three prefill target spans average1599.61063ms; W8A8 stream675.4525ms,merged SiLU200.4775ms,validation180.5076ms,reconstruction120.1456ms per chunk. These are native costs, not a producer operator comparison or isolated transfer throughput. Host staged waits include preceding queued GPU work; do not call them copy bandwidth or proven removable CPU overhead.

Frozen binary/build/candidate/protocol/source identities/commands/raws/reports/cost script/failed receipt: /home/sebastian/LocalLLM/b70-exl3-fixtures/recovery/r11-serving-timing-v1/receipt-c1-post-numerics-v2.json (16410B,SHA2564edf469a61100c3a7bb5fe5f9283befa11407b77592851071639917a6f45ae0f). Native unprofiled backend peak30369570051B/host RSS13581381632B,graph release0; production inactive, no other vLLM GPU instance. Product code unchanged; focused measured C1 progress is not whole R11 completion.

Next bound the two late graph/eager MTP differences and test one independent cost-backed short-row Gemma5120 work-distribution optimization without changing the original reduction tree/F32 boundaries. Earlier rejected compile-time-width-only specialization is a different attempt and is not repeated. Current no-MTP/32K/C4 and final small three-repeat comparisons remain pending; a comprehensive Pro package should include them and explicit frozen D64/state/reference-contract gates. Regular tested checkpoint commit/push authorized; goal active, target/serving and milestones C/D/E remain open.
