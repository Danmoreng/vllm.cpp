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

F3 now records actual logical target query offsets/lengths, request IDs,
sequence lengths and GDN row classification in each R11 profiling cycle,
including graph replays. These fields are profiling-only; unprofiled output,
inference arithmetic and graph policy are unchanged. The focused build passes
two steps and the small P128/O16 smoke passes116 assertions. All six target
spans are observed; five decode cycles each contain one actual graph replay.

Canonical C4 P32768/O256 and staggered4K/32K O256 profile runs pass4,386 and
2,367 assertions, respectively. All four/two requests finish at256 tokens.
Their187/135 target cycles include98/105 target graph-compute replays. Analysis
joins events to the target span by same-queue device-timestamp containment,
not by dispatch/capture counts. Both cases have zero target
`attention_reference` events and zero ragged pure-decode query-length cycles.
Nonuniform forms occur during prefill/mixed work, where oneDNN SDPA events are
visible; this is not evidence of an expensive generic ragged decode kernel.
All output IDs and all non-time cycle fields remain exact to the historical
native matrix-v3 controls. That continuity is not an original parity check.

Do not begin a ragged verifier variant from these results. The separate narrow
FA-output M4 or graph/eager anomaly investigation remains before F3 closure.
Whole graph replay times are actual, but internal graph attention-node times
are unavailable; neither capture-only events nor eager kernel times substitute
for that attribution. These profiled runs are not serving scores or U6 trials.
Both long workers use24GiB/no swap/two CPUs with no OOM; host cgroup peaks are
6,397,235,200 and4,728,975,360 bytes. Q1 remains failed/open. Frozen inputs,
source/binary, commands, complete profiles, resources and analysis are indexed
in recovery/upstream-f3-shape-profile-v1/receipt-v1.json,6540 bytes,SHA256
08cd5db5627295fe02a951503a59f5da34b83f0eaecb2944ef13a0076b5bc8bb.

F3's remaining FA-output M4 investigation is now complete and stops without a
kernel variant. A one-weight manifest selects only the historical real
target-attention output projection, M4/K6144/N5120/4bits, MB8/NT4/18splits.
The current native target links one step. Pinned original ordinary/observed
and repeat outputs are exact; all eight complete fixture payload hashes are
independently checked and their descriptors/hashes match the historical
capture. Current native unprofiled and profiled operators pass49 assertions
each, checking full input-Hadamard, FP32 partials and F16 output bytes.

Separate three-sample GPU profiles give core medians81.352us original and
81.562us native (+0.26%); sample ranges74.374–92.394us and80.834–82.293us
overlap. The historical +20.62% core anomaly is not reproduced in this narrow
single-weight state; this does not establish universal core parity. Unprofiled
public complete-call medians remain104.419us original and112.529us native,
with uncached native map validation; do not substitute core time for that cost.
This is a synthetic identical-input operator test, not current model operands.

The same native operator's eager/graph and public/certified-map checks pass173
assertions in each of separate profiled/unprofiled workers, retaining full
intermediate/output equality, correct5/4 graph-node counts and retirement.
Unprofiled certified-map eager/graph medians are94.075/97.244us; the3.169us
single-call difference is not evidence of a material compiled-kernel anomaly
or a serving gain. Internal graph node times remain unattributed. The donor
ESIMD header is unchanged/identical; no new kernel, graph policy or math change
is made, and no global tile sweep is justified by these results.

The first oracle attempt fails before GPU execution because the frozen tool
package omitted its `b70_inventory.py` dependency. The separately recorded
second attempt includes that module and passes; the failure is preserved.
All six successful workers use24GiB/no swap/two CPUs with no OOM; peaks remain
below0.8GiB. Source/binary, commands, payloads, resources and independent
analysis are in recovery/upstream-f3-faout-m4-v1/receipt-v1.json,18499 bytes,
SHA256ab7a0ccfcced0afe014b5a6f95412654350fb8c1250144efd2046740c5823903.
Together with the actual long/mixed shape profiles this closes the bounded F3
investigation. Continue remaining U2 contribution composition and U6; Q1 and
the original/default qualification failures remain open, not reclassified.

U2 now has a first independently reviewable portable infrastructure patch at
the integrated upstream pin. Its five paths contain only external-artifact
admission, the host probe/test, public usage documentation and the matching
CTest registration. The8,693-byte patch applies cleanly to pristine upstream
files; its five resulting files are byte-exact to the separately tested source
tree. The archived upstream `AGENTS.md` remains unchanged. No development
policy, historical receipts, GPTQ experiments, weights or binaries enter this
selected diff. Personal `/home/`/`/opt/` literals and a narrow set of known
credential markers are absent; this is not a comprehensive credential audit.

That clean archived upstream source, without `.git`, `.env`, model inputs or
oneAPI, configures with ordinary GCC16.2.1, builds only the host probe in two
steps and passes the single focused CTest containing five generated-input
methods. It checks availability/exit semantics only, not tensor payloads or
native model inference. The development GPU cases already call this helper;
they are not pulled into the independent host patch or newly qualified here.
See [external test admission](../EXL3_TEST_ARTIFACTS.md).

Pinned upstream contribution, agent, workflow and CI rules are retained as
reference in the external capsule. No checkers, upstream policy or workflow
are changed; no protocol compliance/commit signatures are retroactively
claimed. Public admission/attribution records and maintainer discussion of the
larger dependency split remain necessary before publication. No contribution
branch/PR/contact is created. This is a small infrastructure slice, not a
finished clean native XPU/EXL3/MTP contribution or a backend qualification.
The patch, source identities, actual apply/build/test results and rules are
indexed in recovery/upstream-u2-contribution-surface-v1/receipt-v1.json,3832
bytes,SHA2565937c05476c42f6cd8eb5781919ed627369490a7a5419c4b558b6b54d09f0ec0.
Remaining feature composition must not delay U6's bounded final matrix.

U6's primary C1 P4096/O1024/MTP3 comparison is now refreshed on source
`fd40e0b92`, frozen native binary
`c0ca1870dac2707889e02e648bd806d33d8b26a08b4fa99d8747ac21a9461ae3`.
Three fresh unprofiled engines per arm run sequentially in order N1/O1/O2/N2/N3/O3.
All native workers pass1,900 assertions; all six actual exits are0. The matched
workload/checkpoint/subset, page1600/180blocks, FP16/FP8, O32 warmup/prefix reset,
unchanged180W and24GiB/no swap/two CPUs are checked. Native direct preparation
uses its defaultON; the fast verifier is still explicitly experimental.
Original uses the existing matched serving-config overrides, compiled graphs
and no arithmetic-control override; constructor startup is excluded.

| Current C1 median | Native | Original | Native change |
|---|---:|---:|---:|
| Decode emitted tokens/s | 51.7066 | 63.9737 | -19.175% |
| TTFT ms | 2,295.125 | 1,950.013 | +17.698% |
| End-to-end s | 22.0799 | 17.9410 | +23.069% |

Native decode samples are51.9685/51.7066/51.5097; all1,024 output IDs and all371
non-time cycle fields remain exact across repeats and to F2. Each proposes1,104
draft tokens and accepts656. Original samples are55.7053/64.0197/63.9737; none
is discarded. R1 accepts612/1,239 and is byte-exact in its1,024 IDs to historical
original R3. R2 accepts666/1,074 and is exact to historical original R1; current
original R1 versus R2/R3 first differs at token48. All nine cross-arm token
comparisons are retained: native differs from original R1 at48 and R2/R3 at68.
Timing is therefore an autonomous workload comparison, not numerical/state
parity or a same-operand kernel speedup. Q1/default-reference failures remain.

The native tracked device peak is30,307,785,027 bytes and shared workspace
130,023,488 bytes; original R1 Torch peak allocated/reserved is27,960,091,648/
28,324,134,912 bytes. These accounting scopes differ. Host cgroup peaks remain
below7.4GB with no OOM/swap. Output chunk timestamps are not individual-token
latency samples; original frontend batches are not GPU target/MTP cycles.
Commands, source/tools/binary and actual scores/resources/analysis are indexed
in recovery/upstream-u6-final-matrix-v1/primary-receipt-v1.json,14998 bytes,
SHA256d4761a3fe962c00be910ab321017a2b022d9c0780cd54b44a5405fdad73ce6ba;
the cross-arm supplement primary-receipt-v2.json is794 bytes,SHA256
6861d76b2499927957c030e1d3089773328f3825770816ffbf5d1e96ca61aa9b.
Only the primary six scores are complete. MTP0, C4 overlap, long/mixed,
lifecycle/prefix, remaining contribution composition and final handoff remain.

U6's MTP0 sentinel and initial C4 pair are now complete on the same frozen
native source/binary and fixed envelope. Each case has one fresh unprofiled
worker per arm; all four actual exits are0, with no swap/OOM and host peaks
below6.6GB. No engine arithmetic, graph policy or qualification default changes.

| C1 P4096/O256 MTP0 | Native | Original | Native change |
|---|---:|---:|---:|
| Decode emitted tokens/s | 26.3201 | 29.4640 | -10.670% |
| TTFT ms | 2,239.007 | 1,909.507 | +17.256% |
| End-to-end s | 11.9274 | 10.5641 | +12.905% |

Native passes1,354 assertions and all256 IDs/non-time cycles remain exact to
the historical final MTP0 control. Both engines have zero proposed/accepted
drafts and physical page1600. Cross-arm tokens first differ at48; this is an
autonomous target-only timing sentinel, not numerical parity or a kernel gain.

C4 P4096/O1024 each passes7,786 native assertions, finishes in43.9539s and has
130.5032 aggregate emitted tokens/s over its28.4054s common decode interval
(3,707 tokens). Pure-decode target request-count histograms are4:346,3:30,2:2,
1:30. Original finishes in76.2204s but has no all-four common decode interval.
All1,524 actual speculative scheduler outputs have `num_drafts==1`, and logs
show one running/three waiting. The frozen original `SpecDecodingStats` and
`Scheduler.update_from_output` sources establish that this counter counts
speculative requests in one actual model-result update, not configured maxseqs.
Genuine original C4 overlap remains unproven: do not expand to three repeats
or derive a C4 GPU-parity percentage from the different scheduling scopes.
Cross-arm first token differences are53/3/124/12 for requests0/1/2/3.

All4,096 native IDs/all419 non-time cycles are exact to all three **final**
matrix-v3 controls. The initial host analyzer incorrectly selected older
primary controls, which differ only atcycle388's accepted counter (2 versus3)
while IDs/all other fields match. Preserve that older relation, not a claim of
full equality to it. Analyzer v1 also has an arm-loop indentation failure;
v2 fixes it, then exposes the older-control mismatch; v3 selects the actual
final matrix-v3 baseline and retains the unchanged full-cycle assertions.
Failed versions and all six historical control comparisons remain indexed;
no GPU rerun, product correction or threshold relaxation is made.

Results, commands, resources, source-counter interpretation and analysis are in
recovery/upstream-u6-final-matrix-v1/m0-c4-receipt-v1.json,7474 bytes,SHA256
a59a513bcd1532b7dd45186e323df0da62f15534c7327716c93960fde1f04aac.
Continue long/mixed, lifecycle/prefix, remaining composition and handoff;
Q1/default-reference qualification remains failed/open.

U6 long/mixed is now complete: one fresh unprofiled native and original engine
for each fixed case, all six actual exits zero. The primary frozen binary and
source are unchanged; default direct preparation, explicit experimental
verifier, page1600/180blocks, 180W, 24GiB/no-swap/two CPUs and O32 warmup followed
by prefix reset remain the launch contract. Startup is excluded from scores.
Older original controls have matching pinned identities/workload/resources,
but no independent power observation was found; their scores are not reused.

| Fixed MTP3 case | Native/original request wall seconds | Native/original common decode tokens/s |
|---|---:|---:|
| C1 P32768/O256 | 27.416956 / 24.048544 | 40.329931 / 45.607179 |
| C4 P32768/O256 each | 122.240472 / 95.675042 | 86.307817 / unavailable |
| Staggered P4096/P32768, O256 each | 35.992697 / 30.534883 | 64.992941 / unavailable |

C1 native decode is11.5711% lower, request wall14.0067% higher, and TTFT
21.093964/18.457269 seconds. This is one autonomous trial, not a new kernel
gain or same-operand comparison: current original IDs differ from its older
control at51, and native/current original at73. All three native cases retain
every historical final-control ID and non-time cycle field:256/125,
1024/187 and512/135 IDs/cycles, with633/2447/1216 passing assertions.

Original C4 and mixed have no common decode interval; all424/213 speculative
scheduler outputs count one actual speculative request. Their request-wall
differences (+27.7663%/+17.8740%) are different scheduling/continuation
comparisons, not multi-request GPU throughput parity. Native/current original
IDs differ in every request. The mixed original retains both older control
sequences; C1/C4 original variations remain visible. No extra repetitions or
qualification/default-reference promotion follows these results.

All worker host cgroup peaks are below6.7GB, with zero swap/OOM events.
The first host C4 analyzer fails by indexing an unavailable overlap as a
dictionary; v5 preserves the absent interval and passes unchanged full-data
checks. The failed v4 is retained, and no GPU run is repeated. Initial envelope
v1 incorrectly described a post-worker observation as during the worker;
v2 corrects that scope while preserving the original file. Remaining four
workers have explicit before/after power/service/container observations.

Commands, complete outputs, resources, analysis versions and identities are in
recovery/upstream-u6-final-matrix-v1/long-mixed-receipt-v1.json,58287 bytes,SHA256
b48f49bcda5653a0ea13ee132d5b1c8ea38246f714841a7ed14b2dff37b65ee0.
Continue lifecycle/prefix, remaining contribution composition and final handoff.
Q1/default-reference qualification remains failed/open; the three-repeat
primary C1 decode gap remains19.1752%.

U6 lifecycle/prefix is complete on the same final frozen binary, with three
fresh sequential workers at180W/24GiB/no-swap/two CPUs and production stopped.
Native lifecycle passes483/483 assertions and its entire JSON equals the F2
control: ordered1/4/2/1, mixed prefill/speculation, EOS, cancellation/slot reuse,
poisoned spare slots and graph/eager/graph. Its sentinel checks cover layers0/47;
they are not a new full48-layer per-token state or FP8-KV comparison.

Native P32768/O64 cold/repeat passes363/363 assertions. All128 IDs and every
non-memory JSON field equal the historical final control; both64-token outputs
are identical. First positions are0/30400, so the repeat computes2368 prompt
tokens. Graphs capture2/replay56, and after engine release graph bytes are0.
Tracked peak drops30403660867->30385296451B, the already accepted18364416B
direct-preparation reduction, not a new optimization.

The fresh original prefix worker exits0: cold/repeat64 IDs exact and30400
cached tokens; both complete sequences equal its older original control.
Native/original first differ at51 in both cold and repeat. There is no
native/original prefix speed comparison: the native test has no comparable
request timer and performs completed-step memory probes; original frontend
timestamps follow its separate O32 warmup. No new full state dumps are made.
Affected F2 same-input/projection/full-native-state checks and failed Q1
integrated original-state checks retain their separate scopes and labels.

All three workers have zero swap/OOM events and host cgroup peaks below6.7GB.
Commands, results, exact-control relations and before/after power/service
observations are indexed in
recovery/upstream-u6-final-matrix-v1/lifecycle-prefix-receipt-v1.json,24331 bytes,
SHA256 b3a803792e018ed9ea6ffb606d12412d1c21204a07c29c11642f7073b03f3399.
Remaining work is the contribution composition and reproducible final review
handoff, with quality/reference open gates preserved. No whole-backend or
performance-parity qualification is promoted.

U2 now isolates standalone development executables behind
`VLLM_CPP_XPU_DIAGNOSTICS` (default OFF, requires XPU). It gates
`b70_exl3_bench`, the two root GPTQ probes and the test-tree
`bench_gptq4_model`; five translation units, not inference operators or unit
tests. OFF retains all1538 non-probe compile commands byte-exact to the prior
native configuration. ON restores the probes; only the older test benchmark's
Git provenance macro changes with HEAD. The head probe compiles/links in two
steps. No diagnostic GPU benchmark is run, and native configuration is restored
to OFF with identical commands. CPU plus diagnostics ON rejects with exit1.

The newly documented CPU recipe configures with the option unset/default OFF,
builds its two focused targets in six steps and passes both CTest cases:
14 shared-cache assertions and five generated admission methods. This is not
a new full library build/suite or oneDNN dependency build. Earlier clean CPU
and independent admission-patch results retain their separate receipts.

[EXL3_XPU.md](../EXL3_XPU.md) is the portable current-model build/check recipe,
dependency/notices table, centralized guard-symbol index and honest capability
matrix. A literal source inventory finds149 `VT_B70_*` names: three in product
sources,146 only in tests/tools. The product top20 override, optional fallback
trace and legacy one-shot GPTQ GDN dump are classified explicitly; the latter
is inactive for ordinary EXL3 inference, not silently removed. Test/model/dump
controls do not become ordinary engine configuration. No runtime math, hardware
admission guard, reference label or measurement is changed.

The initial host command comparison fails because the still-enabled test
benchmark's Git provenance macro changed; preserve the original command sets
and failure record. Gating that remaining benchmark yields the exact1538
non-probe command comparison without ignoring arithmetic flags. Commands,
logs, source snapshots, classification and identities are in
recovery/upstream-u2-diagnostic-isolation-v1/receipt-v1.json,13636 bytes,SHA256
3feeb701d51883e896db07684b860815703a793299e96ae3b81f9930aec07dbb.
Full contribution composition and the final reproducible Pro handoff remain;
the bounded U6 functional/performance results and Q1 failed/open gates stand.
