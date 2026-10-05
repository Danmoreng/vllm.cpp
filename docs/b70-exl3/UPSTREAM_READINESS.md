# XPU/EXL3 integration and contribution dependencies

The active plan is
[`B70_EXL3_UPSTREAM_READINESS_AND_PERFORMANCE_PLAN_3803ad4_EN.md`](../B70_EXL3_UPSTREAM_READINESS_AND_PERFORMANCE_PLAN_3803ad4_EN.md).
The developer requested Main integration first, followed by the portable U0
slice. The closed P0–P7 plan and its original receipts remain historical.

The native baseline is `3803ad4a6b9974f4afd1b6c9ad3e333d5e9b6ff8`.
Merge `3dfa3c3193c5950bf284229441e81f26b99d499c` integrates upstream
`e67a071a649814e102a91145a0b1b424884f16da` without performance changes.
The local checkpoint ref is `refs/b70-checkpoints/pre-upstream-3803ad4a6`.
Frozen binaries, commands and measurements are external artifacts; their U1
receipt identity is indexed in `RECOVERY_STATUS.json`. Original records are
unchanged. The baseline binary SHA-256 is
`4d47b46e24c7f6c99e11b534370177361f1207e141f539b7aa63f0e68ba8468c`.

Local Git inventories from common ancestor
`9d96b162c8c3dfa8f0143932962e445b68fe3a8b` found 486 changed paths on the
preserved branch, 107 upstream paths and 10 overlapping paths. These totals
include documentation and generated files. The overlap concerns Qwen dense
loaders/headers, NVFP4 ownership, CMake/test registration, Vulkan narrowing
casts, model registration and the local agent policy. The latter stays local
and must be excluded from a future public contribution.

| Contribution surface | Dependencies and current evidence |
|---|---|
| XPU substrate and callable operators | VT queue/allocation/provider integration, SYCL compiler, synthetic tests; ordinary CPU configuration must remain independent of oneAPI. |
| EXL3/Qwen target | Grouped SmallM/W8A8, oneDNN capability, loader and FP16 boundaries; needs the substrate and explicit external model/fixture inputs. Standalone projection replay has a smaller dependency set. |
| MTP, graphs, state and prefix | Compact draft and target execution, page1600 metadata, completion ownership and full recurrent snapshots; needs target integration and separate lifecycle/reference qualification. |

The merge's native smoke preserves all 256 output IDs, 113 non-time cycles,
complete lifecycle results and tracked device peak. GDN snapshot/route and
affected host tests pass. The single short timing pair shows about 0.2% change;
it is a merge sanity check, not a new Python score. The initial full CPU build
exposed pre-existing GCC16 shared-pointer atomic deprecation and GCC12
array-bounds failures. U2 replaces the deprecated publication with a copyable
C++20 atomic shared-owner cache. GCC16 now builds the CPU library and selected
targets; five focused CPU tests pass (5,019 assertions, one existing registry
case skipped). The SYCL build and SmallM/W8A8 ownership tests pass (199/76
assertions); MTP lifecycle passes 483 assertions with its complete payload
unchanged from the merge baseline. GCC12 was not rerun. The U2 cache receipt is
indexed in `RECOVERY_STATUS.json`; this repair makes no performance claim.
Affected Vulkan source compiles and CUDA host syntax checks pass at integration;
the affected CUDA Qwen dense host syntax check also passes after this repair.
CUDA device build/link/runtime and HIP remain unverified.

U0 supplies generated host fixtures and an explicit artifact-root entry point
in `tools/exl3_reference/capture_projection.py::main`. Host tests require no
model, GPU, oracle image or third-party Python package. See the
[tool usage](../../tools/exl3_reference/README.md) for required failures versus
optional external skips. Missing required qualification inputs never pass.

U2 names the shared optional dependency `VLLM_CPP_XPU_ONEDNN`; the old
`VLLM_CPP_XPU_GPTQ4` CMake option remains a compatibility alias. Both produce
identical compile commands and keep oneDNN at 3.13.0. Both OFF removes the
capability without leaving a sticky cache value. CPU checks pass without XPU
or oneDNN, and the oneDNN-disabled XPU backend smoke/operator compile passes.
The new option alone builds and passes W8A8 ownership (76), oneDNN attention
(74), isolated dense FP16 (199) and MTP lifecycle (483) assertions; the latter's
complete payload remains exact. The disabled-stub build failure and initial
order-dependent dense-test failure remain in the indexed external receipt.
The corrected dense case also passes with its preceding packed diagnostic
(1,232 assertions together). No performance or qualification gate is promoted.
See [XPU build options](../XPU.md#shared-onednn-capability) for the interface.

The SmallM test split preserves all 140 original assertion expressions. Its
three synthetic cases pass 288 assertions without external inputs; the isolated
context-exit sentinel passes two assertions and completes worker teardown.
Six external CTest cases separately skip missing optional data and fail in the
required mode (actual CTest exit8). Corrupt present payloads remain failures.
The generated host admission test passes five methods without oneAPI/GPU.
The real Gate/Up case passes 112 assertions with unchanged complete byte checks;
other external cases are not newly numerically qualified. See
[SmallM input/outcome controls](../XPU.md#smallm-test-inputs-and-outcomes).

Root `NOTICE` now indexes the B70 donor adaptations and external SYCL-TLA/oneDNN
dependencies. The GDN directory carries the unchanged Apache-2.0 donor license;
individual BSD notices remain in their headers. Current source pins, dependency
build commands and route-specific admission scope are in
[XPU source dependencies](../XPU.md#current-source-dependencies-and-xe2-scope).
The notices step changes no GPU code and needs no automatic test. A subsequent
fresh Git archive of `0293084ce`, without a personal `.env`, configures and builds
the CPU library and selected targets with GCC16.2.1 (624 steps). All four focused
CTest cases pass: cache ownership, panel planning, external artifact admission
and direct upload (1,063 C++ assertions plus five host admission methods).
All GPU backends, oneDNN, downloads, server and diarization are explicitly OFF;
the linked host test has no SYCL/oneDNN dependency. The receipt is indexed in
`RECOVERY_STATUS.json`. This is an executed clean CPU build, not a fresh oneDNN
dependency build, a full suite or GPU qualification. The previously executed
required/optional SmallM input checks remain the separate GPU-test admission
evidence. A model and oneAPI were not needed by this CPU slice.

F1's separate eager C1/C4 profile now labels each device/host record with its
engine cycle. The focused native target builds; C1/C4 pass 130/451 assertions.
Adding labels preserves all 32/128 emitted IDs and 17/25 complete non-time cycle
records against the preceding unlabeled runs. The executed geometry isolates
672 C1/Q4 and 288 C4/Q4 producer/recurrence pairs across all 48 GDN layers,
excluding mixed prefill/decode cycles. Graph captures/replays are zero.

| Pure speculative chain | C1/Q4 | C4/Q4 |
|---|---:|---:|
| Median Post-Conv kernel | 4.271 us | 4.271 us |
| Median typed-SLM recurrence | 136.667 us | 404.479 us |
| Median queue gap | 15.729 us | 26.146 us |
| Producer GPU work / profiled decode-cycle wall | 0.397% | 0.259% |

The queue gap includes required metadata validation/readback; it is not a
removable transfer measurement. These are eager diagnostic results, not graph
attribution or a refreshed serving/Python score. F1 stops at its bounded
precondition because the measured producer work offers little demonstrated
complete-path potential. No fused variant was implemented or timed, and no
graph speed upper bound is claimed. Existing typed-SLM arithmetic and state
storage remain unchanged. F2 evaluates final-consumer event leases first,
followed separately by direct checked-preparation consumption.
The immutable profile/binary/command receipt is indexed in `RECOVERY_STATUS.json`.

F2's first candidate used a final in-order queue event for the private pools,
explicit cross-queue dependencies, retirement before growth and synchronous
exception cleanup. The pinned oneDNN stream copies that same native queue.
Focused candidate checks passed: synthetic lifetime (290), four real projection
cases (92 each), public failure boundaries at M129/M1600 (1,552/7,699) and
private-budget fallback (15 assertions). Complete xq/scales, FP16 intermediates,
public scratch and output checks remain exact; queued distinct inputs and
immediate result retirement pass. The real operand fixtures and their cyclic
P128 expansion are declared in the receipt; QKVZ was tested at M129 only.

| Complete call, fixed panel1024 | Fence median ms | Event median ms | Rate change |
|---|---:|---:|---:|
| Gate/Up M129 | 1.3990 | 1.3931 | +0.42% |
| Gate/Up M896 | 2.6624 | 2.6908 | -1.05% |
| Gate/Up M1600 | 3.9240 | 3.9240 | +0.00% |
| QKVZ M129 | 0.8362 | 0.8293 | +0.83% |

One separate unprofiled native P4096/O32 pair, after O32 warmup/prefix reset,
passes 130 assertions per arm with identical 32 IDs and all 17 non-time cycles.
Three-chunk TTFT is 2,320.6 versus 2,318.6 ms; tracked device peak is identical
at 30,326,149,443 bytes. This single short pair does not demonstrate a useful
gain and is not a refreshed original/U6 score. F2 rejects this first candidate;
the baseline backend and original projection test are restored byte-exact.
The complete rejected source capsule, frozen binaries, commands and the initial
test namespace compile failure remain in the indexed external receipt.
Broader large-QKVZ/integrated-state qualification was not pursued after rejection.

The retained `test_xpu_w8a8_workspaces` is a hermetic regression for pool readers,
growth, callback exceptions and queue retirement, requiring a visible GPU but no
model/oracle data. It builds and passes 289 assertions on the restored baseline.
No event-lease runtime option remains in the product. Continue only the second
F2 candidate: direct checked-preparation consumption, measured independently.

F2's second and final candidate was initially measured with
`VT_XPU_W8A8_DIRECT_PREPARE=1` (default0 at that checkpoint). Only the model-owned certificate path
uses a compact workspace and reads the checked private activation buffer
directly. Its ownership scope encloses every oneDNN GEMM and the output
Hadamard; the existing completion fences and lock order remain. Public callers
still receive complete published scratch, and allocation refusal retains the
checked generic path. No per-call persistent allocation or panel retuning was
introduced.

Focused checks pass: ownership/two host queues (76), Gate/Up M129/M896/M1600 and
QKVZ M129 (104 each), public failure boundaries (1,552/7,699), and model-private
budget refusal/reuse (17). Actual private INT8 activations/padded scales, final
reconstructed INT8 panel, active oneDNN F16 intermediate and F16 output are
byte-exact. Internal oneDNN I32 accumulators are not exported by these tests.
External cases separately skip optional missing inputs with77 and fail required
missing inputs with1 before GPU initialization. The initial compile failure is
preserved in the indexed receipt.

| Complete call, fixed panel1024 | Baseline median ms | Direct median ms | Rate change |
|---|---:|---:|---:|
| Gate/Up M129 | 1.4130 | 1.3756 | +2.72% |
| Gate/Up M896 | 2.4297 | 2.4103 | +0.81% |
| Gate/Up M1600 | 3.9745 | 3.9684 | +0.15% |
| QKVZ M129 | 0.6863 | 0.6656 | +3.11% |

One fresh unprofiled native P4096/O32 worker per arm passes130 assertions each,
preserving all32 IDs and17 complete non-time cycles after O32 warmup/reset.
TTFT is2,319.338 versus2,287.559ms; three-chunk prefill wall is2,319.208
versus2,287.442ms. Tracked peak falls from30,326,149,443 to30,307,785,027B
(18,364,416B), with shared workspace capacity148,387,904 versus130,023,488B
and unchanged preparation capacity31,202,304B. Mixed-route operator processes
retain high-water pools and cannot establish that memory reduction themselves.
This single pair is a first signal, not a refreshed Python/U6 score. The route
remained opt-in pending large QKVZ, broader lifecycle/integrated-state checks and
bounded repeat measurements; matching tokens alone does not qualify state.

The subsequent projection qualification adds pinned original QKVZ M896/M1600
captures from the same frozen P128 normalization operands. All six Gate/Up and
QKVZ cases at M129/M896/M1600 now pass111 assertions each. The test additionally
checks the actual last128-column INT8 weight block against the independent
original witness, accounting for the final panel's compact row stride. Existing
native/public byte checks, queue reuse and immediate result retirement remain.
The new capture, frozen tool/binary and commands are indexed in the projection
receipt. This closes the large-QKVZ operator check; integrated model-state,
lifecycle and repeat-performance qualification still remain.

F2 now also has a native integrated recurrent-state comparison after real4K
prefill and MTP. The optional reuse-test mode reads
`VT_B70_EXL3_ENGINE_PREFILL_WORKLOAD`, containing schema
`b70-f2-fixed-prefill-prompt-v1` and exactly4,096 `prompt_ids`; those IDs are the
consumed input in this mode. It requires MTP3 and the existing state-prefix
output, uses1600/1600/896 prefill chunks, and deterministically initializes all
four recurrent slots before first use. Ordinary short reuse is unchanged.

Two requests each emit64 IDs. Baseline/direct pass1,186/3,106 assertions, with
all192 Conv/SSM arrays (627,572,736B per completed request) byte-exact across
all48 layers and all four active/provisional slots. All non-snapshot result
fields match, including proposal/acceptance counters and two graph captures
with26/54 cumulative replays. Actual W8A8 traces show1,044 M1600 and522 M896
dispatches in each arm; all1,566 direct-arm dispatches use the prepared view.
These are completed-request native state checks, not per-token Python or
initialized FP8KV comparisons, and traced state runs are not serving scores.

The existing short lifecycle regression also passes483 assertions and its
complete payload remains exact to the U2 control: ordered1/4/2/1 turnover,
mixed prefill/speculation, EOS, cancellation-slot reuse and poisoned spares.
Its short prefills useSmallM; the separate4K state runs establish real W8A8
execution. The state receipt preserves both raw snapshot sets, frozen source,
binary and commands, including the corrected initial test-setup compile error.
F2 remained opt-in pending its bounded performance/retention decision. Q1's
integrated original state and default-reference gates remain separate.

F2 is now complete: the first event candidate remains rejected; the second
direct-consumption candidate is retained. Three fresh unprofiled C1 P4096/O1024
trials per arm, using one binary and only the direct-preparation selector,
preserve all1,024 IDs and all371 non-time cycles. Each passes1,900 assertions.
The fixed180W,24-GiB/no-swap/two-CPU envelope and O32 warmup/prefix reset match.

| Protected C1 median | Original publication | Direct consumption |
|---|---:|---:|
| TTFT ms | 2,325.617 | 2,293.791 |
| Three-chunk prefill ms | 2,325.500 | 2,293.673 |
| Decode emitted tokens/s | 51.8687 | 51.8060 |
| End-to-end ms | 22,048.540 | 22,040.571 |
| Tracked native device peak B | 30,326,149,443 | 30,307,785,027 |
| Shared W8A8 workspace B | 148,387,904 | 130,023,488 |

The retained benefit is1.37% lower TTFT and12.38% smaller shared workspace
(18,364,416B). Total tracked peak falls0.061%; decode changes-0.12% within
the observed trial ranges and end-to-end changes-0.036%. This is a modest
prefill/memory improvement, not a material decode gain or refreshed Python
parity score. Public semantics, panel1024, oneDNN3.13 and final-consumer fences
remain; no further F2 variant or panel sweep is pursued.

Direct preparation now defaultsON for eligible model-owned calls. Override0
retains the previous publication path; unsupported preparation dimensions or
budget refusal retain the checked fallback. Default activation builds5 steps
and passes ownership/public guards/queues76, budget fallback/reuse17 and a
P4096/O32 native smoke130 assertions. Without the selector, its32 IDs/all17
non-time cycles and tracked peak match the preceding explicit1 smoke; trace
also proves10 model direct calls while four public calls retain publication.
The six full-score trials precede this default-only host-selector change;
the default smoke is not a replacement for U6's remaining final matrix.
Frozen binaries, commands, resource checks and the reproducible analysis are
indexed in the performance receipt. Q1 and whole-backend admission remain open.

A clean public contribution composition,
ordinary-route admission and integrated original MTP state qualification remain
separate work. Preserve the frozen default D27/D29 failures and controlled-oracle
label; no verifier default or full-backend qualification has been promoted.

Q1 now has an original integrated C1 trace: cold P128 followed by two actual
Q4 target forwards with the original depth3 MTP drafter and rejection sampler.
The capture validates the real embedding-module token witness even when the
VLM runner passes root input_ids=None. It copies all48GDN consumed seeds and
full per-token FP32 snapshots, valid Conv windows and only initialized logical
FP8 KV rows from all16attention layers. Cold unconsumed seeds, spare SSM slots
and unwritten/rejected-tail KV capacity are excluded. Each complete arm contains
931 tensors/1704365720 bytes,432 produced SSM snapshots and96 consumed seeds.
Payload identities and256 inter-step seed/KV-prefix checks per arm pass exactly;
11 focused host admission/observer tests pass without Torch, model or GPU.

Default-v1 actually exits1 after an observer assumed root input_ids was present;
the embedding witness corrects that error. Default-v2 captures all three complete
forwards but actually exits1 at the ordinary/observed16-ID repeatability gate.
Both failures and the default-v2 full boundary payloads remain preserved.
Controlled-v3 separately enables the existing scoped deterministic BA diagnostic:
actualexit0, ordinary/observed16IDs exact, all three complete boundary captures
and initialized-state continuity checks pass. Its sequence lengths are128/132/133
with previous accepted counts1/1; this is not acceptance-diversity coverage.
These observations do not establish that the observer caused the default
trajectory difference or that native arithmetic is more accurate.

The capsule is recovery/upstream-q1-integrated-mtp-v1/receipt-v1.json,
11377 bytes,SHA2569ee24b4d13679ef172f7e6e94347ae3a6d9d2dee54d681949502007dbcf10817.
Frozen tool versions, exact commands/resources, failed exits, per-forward raw
payloads and the reproducible integrity/continuity analyzer are included there.
All workers used the pinned original image,24GiB/no swap/two CPUs; no OOM was
observed. These eager observed runs are not serving-performance scores.
Native replay against the actual tokens/accepted lengths and C4->C2->C1 remain
next. Controlled/default labels and frozen D27/D29 failures remain separate;
no ordinary verifier admission or whole-backend state parity is claimed.

The first native C1 target replay is now complete and actually fails the exact
integrated-state gate:8237 assertions,7566 passed/671 failed. Native caches start
cold with finite poison and evolve independently; only the original observed
tokens, positions and previous accepted lengths are replayed. Snapshot order
maps original13/12/11/10 to native3/2/1/0; no original states enter inference.
All931 arrays are exported/compared, including every full FP32 token snapshot
and only valid Conv windows/initialized FP8 KV. Independent raw-file SHA checks
confirm260 byte-exact arrays and671 differing arrays; all values are finite.

Prefill129/129 arrays and all128 consumed Conv/SSM/KV arrays before the firstQ4
forward are exact. The first difference is layer0 ssm_after_t0:1355510 changed
bytes,max_abs0.0031609535217285156, while layer0 Convafter is exact. The firstQ4
step differs in272/401 arrays; the next differs in399/401, including propagated
initial-state differences. This proves the first consumed seeds agree; it does
not locate the difference within Conv arithmetic, BA/post-conv preparation or
recurrence. The next bounded attribution should observe those layer0 boundaries.
Actual trace records16 prefillM128 and32 xe2_verifyM4 calls under the explicit
experimental selector. No verifier default/reference policy or kernel changed.

The two-step focused native build passes after correcting a same-line doctest
CAPTURE redefinition; the initial compile failure/source are retained. The
native worker uses24GiB/no swap/twoCPUs,actualexit1,noOOM. Full native exports
total1704280064 bytes. Capsule receipt-v2.json in the same Q1 directory is6530
bytes,SHA256b2a5c536eab06b12cc75c0df443f65aae60ddd6d040af00f7deba6fdf552851c.
It preserves the binary/source, commands/resources, all failed comparisons and
the reproducible native-payload identity analysis. This is a full-target replay
of original MTP inputs, not an autonomous native drafter/rejection comparison.
C4->C2->C1 and ordinary qualification remain open. The experimental fast route,
controlled/default labels and oldD27/D29 failures remain; independent F3/U2/U6
work is not blocked by this failed qualification gate.

The first-layer Q1 attribution now excludes incoming normalization, projection
and computed Conv as the source of this firstQ4 difference. A separate compact
original capture preserves all19 shared layer0/target-hidden tensor descriptors
from controlled-v3 and all consumed tokens/positions/accepted lengths. Its36
payload hashes pass; ordinary/observed16 IDs remain exact. The native diagnostic
retains all931 comparisons: every comparison field and all19 exported raw arrays
are identical to the preceding full native witness. Actualexit1,6414 assertions,
5743 passed/671 failed; fewer assertions reflect omitted duplicate raw-file
exports, not omitted state comparisons. Small raw exports total36,425,728B.

On the first actualQ4 step, all three native input-norm reads, mixed QKV, both BA
halves, the computed F16 Conv output and post-conv V are byte-exact to original.
The new isolated original speculative GDN tool reuses only captured original
seeds in separate operator copies, preserving actual13/12/11/10 slots, shared
cache strides/offset and poisoned spare rows with capacity bounded to14. Both
pinned fused and split operators match seven complete full-worker endpoints:
core, z, Conv and all four full FP32 token snapshots. Split Q/K/V/B/A stay
unchanged through the original recurrence. This is attribution only; no captured
original state enters native inference and no product arithmetic changes.

The remaining difference is within post-conv preparation/recurrence. The source
shows original speculative Q/K normalization in F32 local values with subgroup
reductions, while native materializes normalized/scaled Q/K in F16 before its
ascending-K recurrence. Those are concrete source differences, not proof that
either alone explains every differing state byte or that either engine is more
accurate. Keep the original FP16/materialization contract for performance work;
this diagnostic does not authorize a precision change or reference correction.

Focused native build4 steps and11 host tests pass. Original compact and split
workers actually exit0; native actually exits1 at the unchanged strict gate.
All use sequential24GiB/no-swap/two-CPU workers with no OOM. Commands, frozen
tools/sources/binary, resource measurements and independent payload/boundary
analysis are in recovery/upstream-q1-layer0-attribution-v1/receipt-v1.json,
7998 bytes,SHA2567ff863de020bc885ed328ff67f7387932ea98461832f13cea2266bc0e22eb6c1.
C4->C2->C1, default-reference admission and autonomous native MTP equivalence
remain unproven. The experimental fast route and frozen default failures stay;
independent F3/U2/U6 work can proceed without waiting for exact Q1 parity.

Original-side Q1 transition coverage is now available. A separate synchronous
eager diagnostic admits four cold P128 requests, cancels two after the first
Q4 forward and one after the next, and observes actual C4prefill/C4Q4/C2Q4/C1Q4
target batches. Original MTP3, draft generation and rejection sampling remain;
all four output maps exactly match the unobserved cancellation control. The
in-process EngineCore, max_num_seqs4/max_num_batched_tokens512 and disabled async
scheduling are explicit diagnostic overrides, not a serving/concurrency score.

The capture copies/hashes all3,316 complete arrays (6,044,934,144B), including
1,536 produced full FP32 token snapshots,336 consumed SSM seeds and only valid
Conv windows/initialized logical FP8 KV. No raw state payloads are persisted.
All896 original inter-step Conv/SSM/KV-prefix checks pass in the GPU capture;
the host analyzer independently checks metadata/descriptor coverage and file
identities, not absent raw payloads. Previous accepted lengths include1 and4;
2/3 and ragged/mixed target batches are not covered by this GPU run. Fourteen
focused host methods pass; actual original worker exit0 with24GiB/no swap/two
CPUs and no OOM. The capsule is recovery/upstream-q1-transition-v1/receipt-v1.json,
5634 bytes,SHA2563429e85f51b0f29be1712232b24f1a9a1301210ce7ab4cc17e8c4cc7fe15516c.
Native cold-cache transition replay/full-array digest comparison remains next.
This original-only evidence does not change the existing native C1 strict
state failure, default-reference decision or experimental verifier policy.

The native C4->C2->C1 full-target replay now completes all3,316 full-array
digest checks and actually fails:117,366 assertions,114,068 passed/3,298 failed.
All arrays are finite;18 hashes match and3,298 differ. Per-step exact/different
counts are4/509,8/1593,4/797 and2/399. The first difference is already the cold
C4prefill r0_l0_ssm_after_t0; all four layer0 raw Conv-history arrays match.
Those histories do not establish computed Conv-output equality. FirstQ4 seeds
are therefore already different, and later failures include propagation.

Native caches start independently poisoned and retain stable16-slot/four-page
request ownership across survivor compaction. All336 native consumed SSM hashes
match their own preceding accepted-token producers; all128 firstQ4 initialized
KV-prefix hashes match the native prefill. The four initial layer0 native SSM
states are distinct, nonzero and not untouched poison. No original state enters
native inference; original tokens/positions/accepted lengths alone are replayed.
This does not compare autonomous native draft/rejection or cancellation scheduling.

The source's existing FP16 producer-prefill eligibility requiresnp==1, so C4
does not enter that C1-specific path. Actual eager trace has48 chunkedGDN M512,
16 EXL3oneDNN attention M512 and16 verifier calls each atM16/M8/M4. This locates
a concrete capability/route difference, not its sole arithmetic cause or the
numeric magnitude of the state deviations. Both workers hash all6,044,934,144B;
raw arrays are transient. Host analysis validates coverage/reported hashes and
producer continuity, not absent raw payloads or numeric error magnitudes.

The focused build passes2 steps; native actualexit1 uses24GiB/no swap/two CPUs
with no OOM. Sources/binary, commands, failures/resources and independent
descriptor/ownership analysis are in recovery/upstream-q1-transition-v1/receipt-v2.json,
5872 bytes,SHA256eaa509fca429b119cd8c1274d7f2dfc943f8ee0378ef620edc95ad8b3cb332ff.
The bounded C1/C4transition scopes are exercised, but Q1 state qualification
fails. Preserve the experimental verifier and all default-reference failures;
no math/threshold/reference/default changes are made. Continue independent
conditional F3 cost evidence, contribution composition and the U6 final matrix.
