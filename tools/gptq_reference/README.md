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
