# B70 GPTQ INT4 reference

This directory contains the pinned Python reference controls for the GPTQ INT4
branch. The model, image, and operators below were inspected on 2026-09-24.

## Pinned checkpoint

- Repository: mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
- Revision: a47b0c6f0d756bc394c4cc629d5b0ded1acc7001
- Local directory: ../models-legacy/Qwen3.8-27B-GPTQ-G128
- Architecture: Qwen3_5ForConditionalGeneration, text dtype float16
- Quantization: GPTQ, symmetric 4-bit, group size 128, desc_act=false,
  pack_dtype=int32, dense LM head
- Safetensors index: 2,399 tensors, 400 quantized matrices, no quantized LM
  head; all 400 qweight/scales/qzeros/g_idx headers passed the included
  local contract inspector
- qzeros: every value and every g_idx passed the full auxiliary audit with
  disk zero offset 1; disk words encode 7 and the effective symmetric runtime
  zero point is 8
- All five local safetensors shards match their Hugging Face snapshot blob
  SHA-256 names. Detailed per-shard hashes and per-tensor audit reports remain
  local under `B70_GPTQ_INT4_Plan/evidence/`; that entire plan directory is
  Git-ignored and is not part of this branch.
- In this checkout, the header and auxiliary reports are
  `B70_GPTQ_INT4_Plan/evidence/local_headers.json` and
  `B70_GPTQ_INT4_Plan/evidence/local_aux_v1.json`.

The audit reads headers and GPTQ auxiliaries. It does not validate qweight
values, scale finiteness, or model quality.

## Pinned reference runtime and route

- Docker image:
  local/b70-qwen38-vllm:q128-m04-196k-180w
- Image ID:
  sha256:76ddd6049aaf5c0f9a2da9fd62893d7344682d441f03e290205f0ce5104f78c8
- Runtime packages: vLLM 0.29.0+xpu, vLLM XPU kernels 0.1.14.1,
  PyTorch 2.13.0+xpu
- Device: one Intel Arc Pro B70, selected as XPUwNa16LinearKernel for
  FP16, uint4b8, G128, symmetric GPTQ, and no activation ordering
- That class calls torch.ops._xpu_C.int4_gemm_w4a16; its installed schema is
  Tensor A, Tensor B, Tensor? bias, Tensor B_scale, Tensor B_zp, int
  group_size, Tensor? g_idx -> Tensor

The bundle's donor excerpts describe vllm-xpu-kernels 0.1.15 source. The
installed reference is 0.1.14.1; the current XPU class selection and
operator call were inspected in that installed image. The operator fixture
runner enables oneDNN verbose output so the compiled implementation and
oneDNN build are visible in its log.

## Controls

run_no_mtp_reference.sh starts the pinned image against the local checkpoint
on loopback port 8082 by default. It omits --speculative-config and MTP
environment flags, disables prefix caching and XPU graphs, uses FP16 model
dtype, and limits the reference context to 4K. It does not build or download
anything. Stop the container with Ctrl-C when the reference session is done.

run_w4a16_fixture.sh runs one isolated B70 W4A16 operation from the actual
checkpoint. It captures a deterministic synthetic FP16 activation, the
post-load packed words/scales/runtime zero point, and the output. It does not
load the full model or claim model-level parity. Its log is saved beside the
fixture so the oneDNN verbose line and build identity remain reviewable. By
default, both files are written under the Git-ignored
`B70_GPTQ_INT4_Plan/evidence/`; the runner recreates that directory when
needed. The current captured fixture is local and is not part of this branch.

## Native GPTQ-01 adapter

The optional native adapter is built with `VLLM_CPP_XPU=ON` and
`VLLM_CPP_XPU_GPTQ4=ON`. It requires oneDNN 3.13.0 at configure time and checks
the runtime source hash `0e2a5bfeef1bfbffc3137464606540233086ce9b` when the
XPU runtime is created. The option is off by default and is rejected for a
non-XPU build.

The typed VT operations are `MatmulGptq4W4A16` and `MatmulDenseF16`:

- GPTQ W4A16 accepts contiguous FP16 activations/output, post-load packed I32
  weights `[N,K/8]`, FP16 scales `[K/group_size,N]`, and one effective I8 zero
  point. The pinned checkpoint uses `group_size=128` and zero point `8`.
- Dense F16 accepts row-major FP16 weights `[N,K]`, with optional FP16 bias;
  it covers the unquantized BA projections and dense output-head operator
  boundary.
- Both operations use the existing VT XPU queue and GPU oneDNN execution.
  There is no Python call, weight repack, or CPU fallback. Primitives, streams,
  per-queue argument bindings, and scratchpads are reused by the adapter.

The adapter began as the GPTQ-01 standalone operator probe. The checkpoint
loader and Qwen text model were added in later commits on this branch. The
original focused B70 operator test result was **2 test cases, 1,232 assertions,
0 failures**. Build the `test_xpu_gptq4` and `b70_gptq4_probe` targets in the
configured XPU build. Run the focused test with:

```sh
ctest --test-dir <xpu-build-dir> -R '^test_xpu_gptq4$' --output-on-failure
```

## GPTQ-00/01 status and performance gate

GPTQ-00 is complete: the checkpoint revision, no-MTP reference route, actual
operator selection, packed-weight auxiliaries, and representative post-load
operation fixtures were captured. Detailed captures and per-run logs remain in
the intentionally Git-ignored `B70_GPTQ_INT4_Plan/evidence/` directory.

The warm timing comparison uses the same profiling-enabled in-order SYCL
queue-marker method for native and Python, reports host enqueue time separately,
and disables oneDNN verbose instrumentation. A marker interval includes host
submission gaps; it is a comparable stream span, not a pure kernel duration.
In the recorded acceptance set, attention K (`K=5120,N=1024`, `M=1/16/256`)
was within the Python control (native/control ratios `0.978/0.964/0.861`), the
dense BA shapes were faster, and the full dense output head (`K=5120,N=248320`,
`M=1`) was effectively equal (`0.998`). The attention fused QKV (`K=5120,
N=14336`) passed at `M=1` and `M=16` (`1.083` and `1.019`). These runs had
stable primitive/engine caches and matched their captured outputs; the largest
reported absolute difference was `0.0009765625` for attention K at `M=256`.

The original fused QKV `M=256` sample (`391.562 us` native vs `306.146 us`
Python) did not satisfy the 20% limit. Ten warmup calls left the GPU at a
different operating point across alternating runs. With 768 warmup calls on
each side, identical output, and stable primitive caches, three alternating
native/Python samples were `367.083/379.063`, `365.834/375.833`, and
`371.042/377.917 us`. The native operator is within the 20% gate in these
repeat runs. Both probes now report the measured warmup GPU duration.

## Current model-level status (2026-09-26)

The native 64-layer model graph and GPTQ checkpoint path run on the pinned B70.
After the wide-row RMSNorm and F16 attention prefill changes, three 4096-token
prefill runs measured `331.882/330.941/330.840 tokens/s`; their corresponding
decode runs measured `12.59/12.55/12.52 tokens/s` for 63 forward steps. The
native figures cover graph computation. The warmed Python reference API runs,
which also include HTTP, scheduling, and sampling, measured approximately
`2.79 s` to first token and `21.6 tokens/s` decode. The scopes differ, so the
two prefill measurements are not directly interchangeable.

The native graph's focused full-model check passed 2,483,514 assertions. The
4096-token Python-oracle comparison **does not meet the planned quality gate**:
full-vocabulary KL divergence was `0.00368401` (limit `0.01`), but total
variation was `0.0248366` (limit `0.02`). The cause of that difference has not
yet been localized. This branch should be reviewed as a measured implementation
with an open model-quality issue, not as completed GPTQ-05 acceptance.
