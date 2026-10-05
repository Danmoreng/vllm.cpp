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

A clean public contribution composition, licensing review,
ordinary-route admission and integrated original MTP state qualification remain
separate work. Preserve the frozen default D27/D29 failures and controlled-oracle
label; no verifier default or full-backend qualification has been promoted.
