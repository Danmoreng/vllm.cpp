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
