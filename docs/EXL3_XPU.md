# EXL3 on the native XPU backend

This is an experimental Intel SYCL/Level Zero backend. The measured model is
`turboderp/Qwen3.8-27B-exl3`, revision
`113cf7ab958054860e43fb7f3063b1af19171095`, with a 4-bpw EXL3 body, a full
6-bpw target head and a 65536-token compact draft vocabulary. Activations are
FP16, recurrent SSM state is FP32 and attention KV is FP8. Large projections
use rotated EXL3 W8A8; this is not a GPTQ W4A8 engine. Provide model files and
the draft vocabulary explicitly. Configuration and tests do not download them.

## Source build

CPU builds require no oneAPI, GPU, model or oneDNN. A focused host configuration
is:

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_XPU=OFF \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_XPU_ONEDNN=OFF -DVLLM_CPP_XPU_GPTQ4=OFF \
  -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_WITH_DIARIZATION=OFF
cmake --build build-cpu --target exl3_external_artifact_probe test_shared_ptr_cache -j2
ctest --test-dir build-cpu --output-on-failure \
  -R '^(test_exl3_external_artifacts|test_shared_ptr_cache)$'
```

For XPU, provide an Intel oneAPI compiler environment, a Level Zero driver,
a SYCL/GPU installation of exactly oneDNN 3.13.0 and the pinned SYCL-TLA source
checkout. The measured toolchain is `icpx` 2026.1.1 (20260724), IGC 2.41.5,
driver `1.17.39758+10`, Arc Pro B70 device ID57891. This identifies the tested
environment; it does not qualify other software or devices.

```sh
cmake -S . -B build-xpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_XPU=ON -DVLLM_CPP_XPU_ONEDNN=ON -DVLLM_CPP_XPU_GPTQ4=OFF \
  -Ddnnl_DIR=/path/to/onednn-install/lib/cmake/dnnl \
  -DVLLM_CPP_SYCL_TLA_DIR=/path/to/sycl-tla \
  -DVLLM_CPP_XPU_XE2_GDN=ON -DVLLM_CPP_XPU_XE2_PREFILL=OFF \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_WITH_DIARIZATION=OFF
cmake --build build-xpu --target test_xpu_backend test_xpu_exl3_smallm test_xpu_exl3_mtp -j2
```

`VLLM_CPP_XPU_GPTQ4` is a compatibility alias for the same optional oneDNN
capability. Both it and `VLLM_CPP_XPU_ONEDNN` default OFF; either ON requires
XPU. The alias does not qualify GPTQ. The current configuration builds the
GDN donor and shared attention verification/decode sources, while the separate
prefill donor option is OFF. Preserve the source-specific floating-point flags.

`VLLM_CPP_XPU_DIAGNOSTICS` defaults OFF. It explicitly enables the standalone
`b70_exl3_bench`, `b70_gptq4_probe` and `b70_gptq4_head_probe` development
executables; the GPTQ probes also require oneDNN. With tests enabled it also
enables `bench_gptq4_model`. These are diagnostic tools, not ordinary inference
dependencies or automatic qualification checks. The option requires XPU and
does not change backend/operator compile flags or disable the focused unit
tests. Legacy GPTQ benchmarking remains diagnostic only.

| Dependency | Source pin / license |
|---|---|
| EXL3 ESIMD | `c59d9442aba8610188837e37724600f1517d7335`; MIT |
| Xe2 attention | `6d92b1bfbf32767ecda8e819613eb151e70030ad`; Apache-2.0 and individual BSD-3-Clause notices |
| Xe2 GDN | `ddf336d86e3c8602888572a3502f951abd51df12`; Apache-2.0, with separate Intel BSD-3-Clause helpers |
| External SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231`; BSD-3-Clause |
| External oneDNN 3.13.0 | `0e2a5bfeef1bfbffc3137464606540233086ce9b`; Apache-2.0 |

[NOTICE](../NOTICE) and the individual source licenses remain required. Source
pins alone do not establish the provenance of an independently supplied wheel
or prebuilt library. Record the actual compiler, dependency build and binary
identity when reproducing a measurement.

## Focused model check

After providing the model and matching draft vocabulary, a native lifecycle
check can be run explicitly. The output file must not already exist:

```sh
VT_B70_EXL3_MODEL=/path/to/model EXL3_DRAFT_VOCAB=/path/to/draft_vocab.json \
VT_B70_EXL3_ENGINE_OUTPUT=/path/to/new-lifecycle.json \
VT_B70_EXL3_ENGINE_DEPTH=3 VT_B70_EXL3_ENGINE_GRAPH=1 \
VT_ASYNC_SCHED=0 VT_ASYNC_RUNNER=0 VT_XPU_GRAPH=1 \
VLLM_CPP_CUDAGRAPH=1 VLLM_CPP_DENSE_DECODE_GRAPH=1 \
VT_XPU_ATTENTION=auto VT_XPU_XE2_VERIFY=1 \
build-xpu/tests/test_xpu_exl3_mtp --test-case='*R08 lifecycle*'
```

The verifier opt-in above is experimental. The test covers native1/4/2/1,
mixed prefill/speculation, EOS, cancellation, slot reuse, poisoned spares and
graph/eager/graph. Its sentinel layers0/47 are not a full per-token reference
state proof. Missing model-test environment values exit77; a required
qualification runner must reject that outcome. This legacy model check does
not implement `EXL3_REQUIRE_ARTIFACTS`. The separate portable SmallM external
checks do implement required/optional admission, documented in
[EXL3_TEST_ARTIFACTS.md](EXL3_TEST_ARTIFACTS.md). Present invalid data and numerical
failures must not be converted into skips. Run only the selected focused check;
routine tests do not stop services or acquire resource leases.

## Capabilities and runtime admission

| Scope | Evidence / limit |
|---|---|
| XPU substrate, SmallM/W8A8 and ownership | Focused synthetic and real-operand tests; model-owned direct preparation preserves full declared intermediates and lifetime behavior. Public workspaces retain their layout. |
| Eager target, held-out P128/D1 | Two complete hybrid-state comparisons pass; this is not integrated MTP/C4/long-context state qualification. |
| Native MTP3, graphs and request lifecycle | Native regression/ownership checks pass in the tested scopes. Integrated original/native C1 and C4→C2→C1 state comparisons fail; keep the verifier experimental. |
| P32768/O64 prefix | Each implementation reproduces its own cold output with30400 cached prompt tokens. Native and original outputs differ; no comparable native/original prefix timing is established. |
| Default reference | Frozen D27/D29 TV/KL gates remain failed. A separately controlled arithmetic reference is a separate label, not a replacement golden. |
| Maximum context / sampling | 262144 is configured, not a full262K generation proof. Stochastic/RNG equivalence is unqualified. |

The guard index is the implementation authority, rather than a universal
hardware support claim: `PagedAttentionKernel` in `xpu_attention.cpp` selects
the route; `PagedAttentionXe2VerifyQueryLength` and
`PagedAttentionXe2VerifyKernel` validate C1 Q2–Q5 or uniform C2/C3/C4 Q4,
FP16/FP8 layouts, metadata, page, device, compiler and driver contracts.
Nonuniform verification does not qualify for that compiled route.
`PagedAttentionXe2DecodeKernel`, `PagedAttentionPrefillKernel`,
`GdnPrefillKernel`, `GdnSpecDecodeKernel` and the FP16 producer wrappers own
their distinct guards. The measured page protocol is1600; a source predicate
accepting another layout is not model qualification for it. Guard refusal uses
an available fallback or an explicit diagnostic error; never remove guards to
advertise another device. `VT_XPU_XE2_VERIFY=1` is not ordinary-route admission.

Direct model-owned W8A8 preparation defaults ON under its checked certificate,
K/128<=144 and64MiB preparation budget. `VT_XPU_W8A8_DIRECT_PREPARE=0` is the
diagnostic publication fallback. This retains the original completion fence;
the measured event-lease variant was rejected and is not implemented.

## Diagnostic controls

Only three `VT_B70_*` names occur in product sources at this checkpoint:

| Control | Classification / behavior |
|---|---|
| `VT_B70_FAST_TOPK20` | Runtime diagnostic override. Unset enables the existing guarded top20/top-p attempt;0 selects the general path. This does not establish sampled-RNG parity. |
| `VT_B70_FAST_TOPK_TRACE` | Optional stderr trace of the top20 fallback witness; ordinary inference leaves it unset. |
| `VT_B70_GDN_CAPTURE` | Legacy GPTQ layer0/P4096 one-shot tensor dump, not an EXL3 inference setting. Requires an explicit destination; leave unset in ordinary inference and omit local capture orchestration from a contribution. |

The other `VT_B70_*` capture/model/fixture/report names are test/tool inputs,
not engine configuration. `VT_B70_EXL3_ENGINE_*`, `VT_B70_EXL3_MODEL`, SmallM
fixture names and serving/profile controls belong to their individual test
executables or tools. Dumps can be large and do not prove in-flight safety.
`VT_XPU_PROFILE` and `VT_XPU_HOST_PROFILE` enable measurement instrumentation;
serving scores require them OFF. Do not treat a profiling score, a native
regression result or an optional external skip as reference qualification.
