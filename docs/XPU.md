# Native XPU development

The optional SYCL/Level Zero backend provides the resource and core-operator
foundation from PR00–PR02 of the
[B70 implementation plan](B70-SYCL-Qwen38-EXL3-Implementation-Plan.md).
It does not yet advertise a supported model architecture: EXL3, GDN, attention,
and the complete Qwen text path require PR03–PR06. The kernels prioritize
correctness; no model throughput or XMX performance claim is made.

## Build and focused checks

Use an Intel oneAPI compiler environment with `icpx`, SYCL headers/runtime, and
a Level Zero GPU driver. The local B70 checks used oneAPI 2026.1.1
(20260724), driver `1.17.39758+10`, device ID 57891, subgroups 16/32.
`test_xpu_backend` prints the device, compiler, driver/runtime and supported
matrix combinations as JSON. A missing GPU is an error, never a CPU substitute.

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
Auxiliary execution, graphs and compressed-state capabilities remain disabled.

`vt::xpu::GetMemoryInfo()` separates physical capacity, process budget, live device
allocations and pinned host allocations. `VT_XPU_MEMORY_BUDGET_BYTES` sets a
positive device-allocation ceiling no larger than capacity. Driver free memory
is reported only when available (`free_known`); it is never fabricated from the
process allocation count. The budget does not reserve memory against other
processes.

The 15 registered native operations are `Copy`, `CastBf16`, `CastF16`, `CastF32`,
`Add`, `SiluAndMul`, `MoeSiluMul`, `SigmoidGateBf16`, `IndexSelect`, `IndexCopy`,
`Embedding`, `Matmul`, `MatmulBT`, `RmsNorm`, and `GreedyArgmax`.
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
actual full-model reachability and a model-derived execution trace remain the
PR06 acceptance gate.

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
