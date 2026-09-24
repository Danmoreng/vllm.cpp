# Native XPU development

The optional SYCL/Level Zero backend provides the resource and core-operator
foundation, EXL3, Conv/GDN, attention and the text engine path from PR00–PR06 of the
[B70 implementation plan](B70-SYCL-Qwen38-EXL3-Implementation-Plan.md).
It supports the dense `Qwen3_5ForConditionalGeneration` text path with EXL3,
BF16 residual/KV storage and F32 recurrent state. Use device `auto` with
`language_model_only` enabled and speculation disabled. Kernels still prioritize
correctness; fast decode, XMX prefill and production serving parity are later steps.

## Build and focused checks

Use an Intel oneAPI compiler environment with `icpx`, SYCL headers/runtime, and
a Level Zero GPU driver. The local B70 checks used oneAPI 2026.1.1
(20260724), driver `1.17.39758+10`, device ID 57891, subgroups 16/32.
`test_xpu_backend` prints the device, compiler, driver/runtime and supported
matrix combinations as JSON. A missing GPU is an error, never a CPU substitute.

For IntelLLVM, the build explicitly disables default unsafe floating-point
reassociation and reciprocal transforms, in addition to FMA contraction.
SYCL division and square root request correctly rounded FP32 results. Disabling
FMA alone left `icpx` host reductions reordered and device division approximate;
the first real-model RMSNorm exposed a BF16 rounding-boundary difference.

```sh
cmake -S . -B build-xpu -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_XPU=ON -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF
cmake --build build-xpu --target test_xpu_backend test_xpu_ops \
  test_copy test_op_provider test_reference_tier b70_provider_trace_probe -j4
ctest --test-dir build-xpu --output-on-failure \
  -R '^(test_b70_inventory|test_b70_provider_trace|test_xpu_backend|test_xpu_no_gpu|test_xpu_ops|test_copy|test_op_provider|test_reference_tier)$'
```

`VLLM_CPP_XPU` defaults to `OFF`; CPU builds need no SYCL installation.
The no-GPU check masks Level Zero devices with `ONEAPI_DEVICE_SELECTOR=opencl:cpu`.
The backend check uses an 8 MiB process allocation budget and a tiny F32 matrix
kernel, not model weights or a long benchmark.

## Checkpoint inventory

Python 3.11 or newer is sufficient. The default command reads configuration,
index and shard headers only; it does not load tensors or initialize a runtime.
`--verify-checkpoint` additionally hashes the full shards, checks provenance,
and reads the four-byte mul1 markers. Output paths must be outside the model
directory. Shell redirection is the caller's responsibility.

```sh
python3 tools/b70_inventory.py /path/to/Qwen3.8-27B-EXL3-3.5bpw \
  --vllm-source /path/to/vllm-upstream-e126687 \
  --donor-lock /path/to/b70_ops/sources.lock.json \
  --verify-checkpoint --microbench-output /tmp/b70-matrices.json \
  > /tmp/b70-inventory.json
```

The matrix file contains text/head descriptors and M values only; it does not
execute a benchmark. Unknown tensor groups and inconsistent index/header or
quantization metadata cause a nonzero exit. Header padding is reported.

The verified checkpoint is `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` at
`19441ac874c4018295da848e250f23511361cda4`. It has 2,426 tensors and
15,338,106,948 payload bytes. All 516 tensors absent from the quantization
metadata are classified: 333 vision, 144 text and 39 MTP. MTP storage includes
7 BF16, 16 F16, 8 I16 and 8 I32 tensors. Text/head has 401 EXL3 matrices:
137 three-bit, 262 four-bit, one five-bit and one six-bit matrix. The eight
additional MTP matrices are excluded from the benchmark file. The loader plan
selects text, embedding and head only; MTP and vision remain disabled.

Source pins are emitted with actual revision, dirty state and pin availability:

| Source | Reference revision |
| --- | --- |
| vllm.cpp plan baseline | `9e63db5dd33b35e7cc57d0f0e80fe6c7d5ababa6` |
| vLLM behavior reference | `e126687a9a828d513c01a07cd69f025f27d63280` |
| vllm_xpu_kernels donor | `6d92b1bfbf32767ecda8e819613eb151e70030ad` |
| sycl_tla donor | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |

Development stays on the developer's current branch, so its revision is reported
separately from the plan baseline. The supplied export audit passed for 6,830
files; the checkpoint hashes and all 409 markers matched. Donor wheel/source
numerical parity remains unproven, and the plan's technical-report `[Lxx]` source
index was not supplied. Header verification does not establish that parity.

## Memory and operator contracts

Device allocations are device USM and are **not host-addressable**. Each GPU has
one context; every queue is in-order. Events support explicit dependencies
between queues. Free waits for all owned queues before releasing storage.
This conservative lifetime policy is deliberate and can introduce synchronization.
Transfers to/from ordinary host pointers use a bounded 4 MiB pinned buffer.
This supports unaligned, read-only checkpoint mappings without importing them
directly into Level Zero; direct import faulted on the real LM-head upload.
Known USM-to-USM copies remain asynchronous. Host staging is accounted as pinned
memory and released after the copy completes.
Auxiliary execution, graphs and compressed-state capabilities remain disabled.

`vt::xpu::GetMemoryInfo()` separates physical capacity, process budget, live device
allocations and pinned host allocations. `VT_XPU_MEMORY_BUDGET_BYTES` sets a
positive device-allocation ceiling no larger than capacity. Driver free memory
is reported only when available (`free_known`); it is never fabricated from the
process allocation count. The budget does not reserve memory against other
processes.

The registered native operations include `Copy`, `CastBf16`, `CastF16`, `CastF32`,
`Add`, `SiluAndMul`, `MoeSiluMul`, `SigmoidGateBf16`, `IndexSelect`, `IndexCopy`,
`Embedding`, `Matmul`, `MatmulBT`, `RmsNorm`, and `GreedyArgmax`.
EXL3 adds `Exl3HadR128` and `Exl3Gemm`; it does not register reconstruction.
`Copy` adds all F16/BF16/F32 conversion pairs and same-dtype bit-preserving copies
with nonnegative strides. Existing cast entry points retain their existing
dtype restrictions. Copy and overlapping XPU outputs use snapshot semantics;
negative strides and overlapping Copy destination elements are refused.

RMSNorm supports both weight and zero-centered `1 + weight`, including the
existing residual-storage rounding. Matmul accumulates in F32. Scatter uses the
last source row for duplicate destination indices. Index validation reads back
one status integer. Greedy argmax follows VT's existing CPU contract: lowest
index wins ties, a NaN in position zero selects zero, and later NaNs are ignored.
The existing sampler reads back one I64 token per row, not the full logits,
on its ordinary all-greedy path without requested logprobs.

The six B70 operator cases passed 70,367 assertions, covering mixed dtypes,
strides, tails, aliases, RMSNorm variants, BA dimensions, invalid indices and
greedy vocabulary size 248,320. These are operator/sampler integration tests;
the opt-in checkpoint check below exercises actual full-model reachability.

## EXL3 reference kernels

The native path supports bits 1–8 and all three existing codebooks, including
the checkpoint's `mul1`. It applies FP16 input scaling, Had128, sequential F32
accumulation over packed weights, then output Had128 and F16/F32 storage.
Only activation scratch `[M,N]` is allocated; there is no decoded model copy.
The GEMM currently synchronizes when releasing this scratch. Shape tuning,
split-K and XMX are later performance work.

```sh
cmake --build build-xpu --target test_xpu_exl3_decode test_xpu_exl3 \
  test_xpu_exl3_checkpoint -j4
ctest --test-dir build-xpu --output-on-failure \
  -R '^(test_xpu_exl3_decode|test_xpu_exl3)$'
VT_B70_MODEL_DIR=/path/to/Qwen3.8-27B-EXL3-3.5bpw \
  VT_OP_PROVIDER_TRACE=/tmp/exl3-checkpoint.jsonl \
  build-xpu/tests/test_xpu_exl3_checkpoint
```

The decoder checks all 65,536 codewords for each codebook, cyclic windows for
bits 1–8 and tile permutation (204,800 assertions). Hadamard/GEMM checks cover
FP16 scaling, in-place input scratch, M=1/2/17, and F16 versus F32-to-BF16 output
(247 assertions). The checkpoint test executes all 11 real `(bits,K,N)` families
at full dimensions, including `lm_head [5120,248320]`, and compares first,
middle and last 128-column output blocks against independent CPU GEMMs
(1,047 assertions). These B70 comparisons passed bitwise; they are not a
whole-model or performance acceptance claim. Without `VT_B70_MODEL_DIR`, the
checkpoint test explicitly exits with skip code 77.

## Conv/GDN reference kernels

The native operators are `CausalConv1dFwd`, `CausalConv1dUpdate`, `GdnPostConv`,
`GdnPrefill`, `GdnDecode`, `RmsNormGated`, `GdnStateGather` and `GdnStateScatter`.
Conv history contains raw inputs. Recurrence state stays F32; Q/K arrive already
normalized and g/beta already transformed. Indexed decode permits negative
null slots and refuses out-of-range or duplicate active slots before updates.
Gather/scatter also supports the existing BF16-cache-to-F32-working-state
contract, without advertising compressed in-place recurrence.

Prefill runs the sequential reference recurrence. The existing CPU BF16 path
defaults to a chunked algorithm with different intermediate rounding; the
comparison explicitly sets `VT_GDN_CHUNKED=0` for its CPU oracle. The native XPU
path currently remains sequential regardless of that variable. Chunked GDN is
PR09 work, and donor chunked-wheel numerical parity is not claimed here.

```sh
cmake --build build-xpu --target test_xpu_gdn -j4
VT_OP_PROVIDER_TRACE=/tmp/xpu-gdn.jsonl build-xpu/tests/test_xpu_gdn
```

The seven cases cover 10,240 convolution channels, kernel width four, lengths
1/2/3/4/63/64/65 and empty varlen rows, padded input/gate strides, Hv/Hk=3,
actual 16/48-head geometry, permuted/null slots and invalid metadata. Outputs
and every final state element are compared to CPU. Full prefill versus split
prefill plus decode is bitwise equal for both convolution and recurrence.
Conv state and gather/scatter comparisons are exact; transcendental operations
use per-element absolute/relative bounds (BF16 relative 0.008, F32 up to 2e-5,
absolute 1e-6). No complete-model reachability or performance claim is implied.

## Attention and BF16 KV cache

The native operators are `AttnGateSplit`, `RopeNeox`, `RopeCosSinCache`,
`RopeFromCache`, `ReshapeAndCache` and `PagedAttention`; Q/K normalization uses
the existing `RmsNorm`. The preamble remains unfused. No FA2 capability is
advertised, so the existing generic FP32-query model path keeps its precision.
KV writes copy raw bits (including NaN payloads), use each tensor's strides,
skip negative slots and preserve the CPU's last-write behavior for repeated slots.

Paged attention uses F32 dot-product reduction and online softmax, with no
materialized scores matrix. It supports GQA, appended-query causal alignment,
local windows, logit soft-capping, float query dtypes, and F32/BF16 output.
Device metadata is checked before cache reads or writes; inactive request rows
may contain unused metadata. This baseline synchronizes for metadata validation;
it is not a graph-capture or tuned-performance implementation. FP8 KV and
multi-axis vision RoPE remain later plan steps.

```sh
cmake --build build-xpu --target test_xpu_attention -j4
VT_OP_PROVIDER_TRACE=/tmp/xpu-attention.jsonl build-xpu/tests/test_xpu_attention
```

Five cases compare against CPU at 24 Q / 4 KV heads, D=256, scale=1/16,
rotary width 64, batches 1/4, M=1 and M>1, page sizes 3/16, permuted pages,
padded inputs, strided NHD cache views and block tables. They also exercise
head-size tails, inactive requests, output aliases and invalid metadata.
Attention's F32 bounds are relative 1e-4 plus absolute 5e-6; BF16 bounds are
relative 0.008 plus absolute 2e-5. The online reduction changes rounding relative
to CPU's sequential dot product and three-pass softmax.

Legacy RoPE uses B70 FP64 frequencies/trigonometry before F32 rotation, as the
CPU path does. The primary cache mode rounds the power and reciprocal to F32,
then computes the position-scaled angle in F32. Device FP64 intermediates keep
the rounded inverse frequency accurate. Its comparison also checks a separate
scalar oracle (relative/absolute 1e-6); the CPU-library comparison allows
absolute 1e-5 for observed float-math differences. Other RoPE comparisons use
relative 4e-6 plus absolute 1e-6 (BF16 relative 0.008).

## Text engine integration

The XPU platform selects its own `XPU_ATTN` backend with NHD paged KV, without
advertising FA2, FP8 cache, graphs, vision or MTP support. The model checks native
operator availability before selecting fused attention or packed GDN decode.
BF16 convolution history uses native gather into F32 working state, update, then
scatter back to BF16; the recurrent state remains F32. All forward tensor work
stays on the GPU. Tokenization, scheduling and one-token greedy readback remain
host work.

```sh
cmake --build build-xpu --target test_xpu_platform test_xpu_qwen_checkpoint \
  test_xpu_qwen_parity -j4
build-xpu/tests/test_xpu_platform
mkdir -p /tmp/b70-text-activations
VT_B70_MODEL_DIR=/path/to/Qwen3.8-27B-EXL3-3.5bpw \
  VT_DUMP_ACT=/tmp/b70-text-activations \
  VT_OP_PROVIDER_TRACE=/tmp/b70-text-trace.jsonl \
  build-xpu/tests/test_xpu_qwen_checkpoint
VT_B70_MODEL_DIR=/path/to/Qwen3.8-27B-EXL3-3.5bpw \
  VT_B70_XPU_ACTS=/tmp/b70-text-activations \
  build-xpu/tests/test_xpu_qwen_parity
```

Use fresh dump/trace paths: manifests and trace records append. The default
checkpoint test runs the pinned tokenizer and all 64 layers twice, with five
input tokens and 65 greedy output tokens per request (64 decode steps). It checks
repeat output, no counted CPU reference fallbacks, resident GPU weights and
stable live GPU/host RSS after warmup. `VT_B70_MAX_TOKENS` and `VT_B70_REPEATS`
allow shorter development probes; those do not satisfy the default acceptance.
The parity test uses the same weight bytes in a four-layer CPU prefix (three GDN
layers and one full-attention layer) and the CPU sequential GDN oracle. It also
isolates the first RMSNorm with real input/weight values. This comparison is not
a full-model Python-vLLM or donor-wheel quality qualification.

The prefix comparison requires exact embedding values, first input RMSNorm,
first EXL3 QKV projection, convolution and post-convolution Q/V. The first
nonlinear GDN stages use relative RMS error at most `1/16384`; first-layer
outputs and residual streams use `1/512`; subsequent projection outputs use
`1/128` (one BF16 relative fraction step). Every stage also checks finiteness and
peak absolute error at most `1e-5 + max(abs(reference))/64`. This is a separate
two-step peak budget for the accumulated outputs, not per-element bit identity.
The initial universal 0.2% RMS limit failed on chained BF16 projections and was
replaced with these explicit stage contracts after isolating the exact stem and
rechecking the unchanged EXL3/GDN/attention operator gates. The observed maxima
over the four-layer prefix are 0.526% RMS in a projection and 0.090% in the
residual stream; float nonlinearities still introduce rounding differences.

On the local B70, the final repeated 65-token check passed all 27 assertions
with identical text and zero counted CPU reference fallbacks (initialization
also recorded zero). Live device allocations stayed at 14,397,796,028 bytes;
host RSS was 323,104,768 then 325,066,752 bytes. The trace contains 52,130 EXL3
calls (401 per forward), including 130 real six-bit head calls with trellis
shape `[320,15520,192]` and output `[1,248320]`. The two complete requests took
169.59 and 168.188 seconds with tracing enabled; these totals include prefill
and are not a separate decode throughput metric. The four-layer reference
comparison and real-input RMSNorm check passed 1,464 assertions together.

For a short timing diagnostic, set `VT_B70_TIMING=1` on the checkpoint test and
unset both dump variables and the provider trace. Each shape is warmed first,
then measured twice with no prefix caching, batch one, greedy sampling and no
MTP. Client prefill is input tokens / time to first token. Decode is remaining
output tokens / time from first to last token. The five-token prompt generates
17 tokens (16 decode steps); the 32-token diagnostic generates two. These short
inputs do not substitute for a production context-length sweep. In particular,
the productive Python engine uses another checkpoint, GPTQ INT4, FP8 KV and MTP4.

## EXL3 decode and small-M optimization (PR07)

The regular inference path keeps the PR03 arithmetic contract: sequential FP32
accumulation, no contraction, and the original Had128 rounding order. Aligned
codebook-2 weights at 3/4/5/6 bits use compile-time width specialization and
32-bit packed-word loads. The optional fused variant reuses each decoded
weight over four input rows (one for M=1), retains accumulators in registers,
and performs the output Had128 in 512 or 2,048 bytes of SLM. It removes the
`4*M*N` global intermediate and its allocation/free synchronization. Input
Had128 scratch remains `2*M*K`; no decoded weight matrix is materialized.
Output overlapping input scratch uses the unfused path. Unaligned weights,
other codebooks and widths retain the original reference implementation.
There are four packed and eight fused specialized GEMM binaries, independent
of the number of checkpoint matrices. Split-K is not enabled: these kernels
retain the scalar reduction order.

`VT_XPU_EXL3_STRATEGY=auto|reference|packed|fused` is sampled at first use.
The default selects measured winners separately by bits, K, N, M and output
dtype. A mutex-protected cache is bounded to 512 entries; its key also includes
device ID, driver, runtime, compiler and `exl3-packed-fused-v1`. Only the measured
B70/driver/compiler domain uses the fusion table. Other domains, F16 outputs
and unmeasured shapes use packed specialization where eligible. The table is
an offline tuning result, not a runtime search or a persistent disk cache.
Head6 and MLP3 have independent entries; M=1 Head6 keeps the packed kernel.

Selection used two independent five-sample medians for each strategy and each
real projection family, after JIT and at least 200 ms of warmup per family.
The limits were fixed before selection: at least 10% mean improvement, and no
fused run median more than 5% slower than either packed run median. All 11
families (401 projections, including the six-bit head) were measured at
M=1/2/4/5/8/16/20. Weighted operator times below sum each family's median times
its occurrence count. They include both transforms, synchronization and scratch
lifetime, exclude weight upload, and are **not full-model timings or pure device
kernel timestamps**. The selected column is calculated from those independent
measurements, rather than a single mixed-policy run.

| M | Packed (ms) | All fused (ms) | Selected (ms) |
|---:|---:|---:|---:|
| 1 | 395.76 | 346.74 | 338.19 |
| 2 | 487.48 | 555.74 | 431.55 |
| 4 | 667.18 | 750.89 | 532.02 |
| 5 | 714.99 | 794.99 | 591.41 |
| 8 | 1,226.06 | 926.33 | 927.59 |
| 16 | 2,130.32 | 1,232.11 | 1,211.65 |
| 20 | 2,724.96 | 1,299.85 | 1,280.67 |

Reproduce an operator measurement with the opt-in tool:

```sh
cmake --build build-xpu --target b70_exl3_bench test_xpu_exl3 -j4
build-xpu/tests/test_xpu_exl3
VT_XPU_EXL3_STRATEGY=fused build-xpu/tests/test_xpu_exl3
VT_XPU_EXL3_STRATEGY=auto \
  build-xpu/b70_exl3_bench /path/to/Qwen3.8-27B-EXL3-3.5bpw 5 5
```

The tool uses real packed weights, deterministic F16 input, F32 output and a
single GPU queue. It records hardware/toolchain identity, warmup count, all
samples, selected strategy and GEMM workspace. After timing, it compares the
first, middle and last 128-column blocks at full K against the CPU oracle,
for every input row and family. Normal strategies must be bit-exact. CPU
oracle calls are explicit validation, not inference fallbacks.

The final mixed-policy run checked 236,544 real output elements over all 77
family/M combinations with zero differing bits. Its weighted times for
M=1/2/4/5/8/16/20 were 337.83/431.54/528.41/586.92/921.22/1,202.70/1,267.83 ms.
No family regressed by more than 0.1% against its previous packed mean; the
largest difference occurred in an unchanged packed family. The focused
unit test passed four cases and 1,676 assertions with both automatic selection
and forced fusion, including F16 output, M=17 tails, unaligned input and output
aliasing, as well as cache identity and bounded growth.

A benchmark-only `VT_XPU_EXL3_STRATEGY=matrix` probes FP16 `joint_matrix` with
FP32 accumulation (1x64x32 for M=1, 8x16x16 otherwise, subgroup 16, 128-thread
cooperative SLM staging). This implementation is linked only into the benchmark;
it is not registered as an inference operator or considered by the strategy
cache. Its isolated weighted times were 806.67 ms at M=1, 374.41 ms at M=5 and
943.53 ms at M=20. The changed FP32 reduction order fails the original bit-exact
tests; sampled real-weight relative RMS error was at most 2.55e-6. The probe
checks a predeclared diagnostic budget of RMS <=3e-5 and peak absolute error
<=`2e-6 + 3e-4*max(abs(reference))`. This budget does not relax regular inference
gates or establish full-model quality. Matrix-instruction disassembly, register
pressure/spill analysis and inference qualification remain outstanding for the
prefill optimization; no XMX inference speedup is claimed here.

Local measurements used the pinned EXL3 checkpoint, the PR06 base commit
`3050cea77` plus the PR07 changes, Intel Arc Pro B70 (device 57891), oneAPI
2026.1.1.20260724, Level Zero runtime 1.17, driver 1.17.39758+10 and the unchanged
180 W power cap. The engine used batch one, BF16 KV, greedy sampling, no prefix
cache, no graphs and no MTP. Warmed client measurements repeated twice:

| Engine path | Decode, 5-token input (tok/s) | TPOT (ms) | Prefill, 32-token input (tok/s) |
|---|---:|---:|---:|
| PR06 scalar reference | 0.3917 | 2,553 | 1.814–1.815 |
| PR07 packed | 1.2155 | 822.7 | 7.233–7.241 |
| PR07 selected | 1.3132 | 761.5 | 7.263–7.269 |

The final engine check also passed all 27 assertions across two 65-token
requests. Generated text matched both the previous PR06 output and the other
request. Live GPU allocations stayed at 14,397,796,028 bytes and host RSS at
403,705,856 bytes in both rounds, with zero counted CPU reference fallbacks.
Request totals were 51.47 and 49.75 seconds; they include prefill and are not
used as a separate decode-rate measurement.

These are short diagnostics under the definitions above, not a 4K/long-context
benchmark. The sibling project's no-MTP GPTQ/FP8-KV baseline at 4K and one
request is 31.84 native decode tok/s and 1,558.62 native prefill compute tok/s
(`2026-09-22-vllm-030-no-mtp/summary.json`). Quantization, context, KV dtype and
prefill timing boundaries differ. The current short decode rate is about 24x
below that target; the production target is not achieved. Prefill XMX, chunked
GDN, tuned attention and serving remain separate plan steps.

## Opt-in provider trace

```sh
VT_OP_PROVIDER_TRACE=/tmp/xpu-ops.jsonl build-xpu/tests/test_xpu_ops
```

The variable is sampled at first use. JSONL is appended, so use a fresh file per
run. Selection rows include operation, device, provider, per-provider selection
count and reference-tier count. Counts measure provider lookups, not completed
kernels. Explicit counted fallbacks are included; cached calls through
`GetOpFallbackUncounted` are excluded. XPU entry points additionally emit
`tensor_arguments` with dtype, shape, stride and queue identity without reading
device contents. Casts share the Copy kernel and its argument record.

The validated B70 operator run recorded 167 native XPU selections and 167
argument records with zero CPU-reference selections. CPU oracle operations
appear separately as CPU selections. With tracing disabled no file is opened.
