# EXL3 projection fixtures and replay

These tools implement the active B70 plan's S0b projection check. They use the
explicit reference manifest and current checkpoint, with no legacy model pins,
weight download, dense full-model expansion, server or Torch-linked native code.

`extract_projection.py` preserves real trellis/scale/marker bits and adds exact
synthetic FP16 inputs at M1/M4. It accepts only whole 128-column Hadamard blocks,
checks metadata/header/revision identity and refuses overwrites. The current
fixture root is `/home/sebastian/LocalLLM/b70-exl3-fixtures/s0b`.

`capture_projection.py` runs in production image
`sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`.
It verifies the pinned EXL3 binary/wrapper and fixture hashes before GPU work.
Allocation retention redispatches the original XPU linear and retains its real
input-Hadamard and split-K buffers. The observed output must equal an ordinary
call bit-for-bit. A CPU F32 increasing-split sum is recorded as a derived stage.
The fixed-four-split raw helper and identical-input FP16 output-Hadamard probe
remain separately attributed diagnostics. Every finite stage has a dtype, shape
and SHA256 receipt; the native product does not import this oracle's Torch code.

Build only the native projection replay in the resolved builder, with repository
mounted read-only at `/work` and a writable fixture/build directory at `/fixtures`:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S /work/tools/exl3_reference -B /fixtures/native-build \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /fixtures/native-build --target exl3_projection_replay -j2
```

Builder: `local/b70-oneapi-2026.1.1-builder:vllm030`, immutable image
`sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca`.
The target builds actual native XPU leaves, queue/allocator, Copy and safetensors
reader dependencies with strict FP32 flags. It requires no Torch, oneDNN or
SYCL-TLA linkage. This is a standalone operator replay, not a full engine build
or qualification of its wrapper registration/model loader.

For each fixture `mlp_gate`, `head_first`, `head_last`, capture in the pinned
oracle container (read-only `/tools` mount of repository `tools/`), then run the
native builder container with `/dev/dri`. Use distinct output names on reruns:

```sh
python /tools/exl3_reference/capture_projection.py \
  --fixture /fixtures/head_first.safetensors \
  --output /fixtures/head_first_oracle_v2.safetensors \
  --image-identity sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918

VT_XPU_EXL3_TRACE=1 /fixtures/native-build/exl3_projection_replay \
  /fixtures/head_first.safetensors /fixtures/head_first_oracle_v2.safetensors \
  /fixtures/head_first_native_v2.safetensors
```

Leave the production service stopped. Run oracle/native GPU containers
sequentially, using `--rm --pull=never --network none`, `/dev/dri`, its render
group, an 8 GiB host-memory limit and four CPUs. Compilation needs no GPU.
No power/driver change or resource-controller lease is used.

Compare on the host (standard library only), adjusting the three capture paths:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/exl3_reference/compare_projection.py \
  --fixture /home/sebastian/LocalLLM/b70-exl3-fixtures/s0b/head_first.safetensors \
  --oracle /home/sebastian/LocalLLM/b70-exl3-fixtures/s0b/head_first_oracle_v2.safetensors \
  --native /home/sebastian/LocalLLM/b70-exl3-fixtures/s0b/head_first_native_v2.safetensors \
  --binary /home/sebastian/LocalLLM/b70-exl3-fixtures/s0b/native-build/exl3_projection_replay \
  --report /home/sebastian/LocalLLM/b70-exl3-fixtures/s0b/head_first_comparison_v2.json
```

Exit 0 means the local projection gate passed; exit 1 records a numerical failure
and exit 2 rejects an input/identity/error condition. Bit differences, finite
status, relative norm, max error and exact hashes remain in the report. Neither
a local pass nor an isolated head block qualifies the full engine/model/state.
The normal native packed output must equal the diagnostic replay output, and
invalid raw storage must be refused. An unsupported capture route can be tested
with `VT_XPU_EXL3_STRATEGY=fused` and `--expect-route-rejection`; it creates no
output capture and does not replace the requested kernel.

Focused host tests are `tests/scripts/test_exl3_projection_{extract,capture,compare}.py`.
Current numerical/build results and remaining prerequisites are recorded in
[the execution map](../../docs/b70-exl3/EXECUTION_MAP.md) and its reference manifest.

`capture_grouped.py` captures complete packed source groups through the same
immutable producer binary, without dense reconstruction. Supply multiple
`--fixture` arguments and a new `--output`; source checkpoint, full-width
geometry, marker, bytes and runtime identities are checked. It captures the
independently merged trellis/transforms/map and actual input Hadamard, split
partials and output at M1/4/12/16/128. Each allocation-observed invocation must
match its ordinary producer call bit-for-bit. The host guard test is
`PYTHONDONTWRITEBYTECODE=1 python3 tests/scripts/test_exl3_grouped_capture.py`.

For layer0 Gate/Up, keep the immutable S0b gate fixture, extract full-width
`model.language_model.layers.0.mlp.up_proj` with `extract_projection.py` and
name it `mlp_up.safetensors`. In the pinned oracle container, run:

```sh
python /tools/exl3_reference/capture_grouped.py \
  --fixture /s0b/mlp_gate.safetensors --fixture /fixtures/mlp_up.safetensors \
  --output /fixtures/mlp_gate_up_oracle.safetensors \
  --image-identity sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918
```

After that GPU job finishes, the regular native engine test
`test_xpu_exl3_smallm '--test-case=*real grouped*'` needs both
`VT_B70_EXL3_S0B_FIXTURES` and `VT_B70_EXL3_S1_FIXTURES`. It checks the actual
native loader/merge and both the typed operator and model-resident wrapper.
This is a grouped-linear arithmetic gate, not whole MLP/model parity.

For the complete target head, extract full-width `lm_head` into
`full_head.safetensors` and pass exactly one fixture with `--single-source`.
This explicit mode avoids an extra host packed concatenation; it retains the
same runtime, full-width and tensor identity checks. The native focused case
`test_xpu_exl3_smallm '--test-case=*full 6bpw head*'` needs only
`VT_B70_EXL3_S1_FIXTURES`. It checks all 953,548,800 packed GPU bytes in bounded
64 MiB chunks and the actual model linear seam at M1/4/12/16/128 without a
dense head copy. The observed full-head producer uses one split at every M.
This qualifies the complete projection, not complete model/state execution.

`capture_swiglu.py` consumes the immutable layer0 grouped capture and resolves
the pinned profile via `EngineArgs.create_engine_config`, without starting a
worker. Supply `--source`, new `--output`, `--reference-manifest`, `--model-dir`
and `--image-identity` in the pinned oracle image. It checks the source identity,
all source tensors and the active guard union, then captures the original eager
`SiluAndMul` method on those real Gate/Up outputs at M1/4/12/16/128. The resolved
profile uses `custom_ops=['none']`, `forward_native`: SiLU narrows through FP16
before multiplying up. A diagnostic without that narrowing differs. Compilation
of the native method is disabled for this isolated arithmetic gate; graph
fusion remains unqualified. Native filter `*real Gate/Up SwiGLU*` needs both
S0b/S1 fixture environments and compares direct SiLU plus the actual model
GateUp seam. Down projection and full MLP/block/state parity remain required.

`capture_block.py` starts one isolated pinned worker and observes actual target
layer0 at P128 followed by D1. Supply a new `--output`, `--reference-manifest`,
`--model-dir` and `--image-identity`; keep production stopped and run native GPU
work only after this worker finishes. The S1 overrides disable MTP, compilation
and graphs, preserving the profile's resolved `custom_ops=['none']`. Ordinary
and observed requests use the same128 token IDs and greedy two-token sampling;
prefix reset makes the observed request cold. Matching greedy IDs is observer
sanity, not bitwise whole-model parity or a performance measurement.

The bounded hooks copy original module input/output stages and only the active
Conv/FP32 recurrent cache slot. Ignored cold initial cache is explicitly left
unread. The current44 tensor receipts include matching P128-after/D1-before
Conv and SSM bytes. Native filter `*real block Gemma*` in
`test_xpu_qwen_exl3_fp16` needs `VT_B70_EXL3_MODEL` and the S1 fixture environment.
It replays real input/post-attention norms and checks the FP32-add-before-FP16-
residual-rounding boundary, including narrow/wide and output aliases. Only
normalization has been replayed so far; mixer, complete block and state numeric
qualification remain required. The actual producer IR norm tolerance is
rtol0.002/atol0.01; P128 still has small half-bit differences, explicitly recorded.


`capture_runtime_layout.py` runs only in the immutable pinned production image,
with the pinned checkpoint, runtime plugin mount and profile. Keep production
stopped and run it sequentially with native GPU work. Supply
`--reference-manifest`, `--model-dir`, `--image-identity` and a new `--output`
path; existing output is never overwritten. `--config-only` checks direct API
configuration without creating a worker.

The worker-extension RPC observes all65 bound layer caches and one synthetic
P128/O1 request. The temporary attention observer delegates the original guarded
function unchanged and describes actual target Q/K/V views. Only active small
metadata, scales and the first block-table column are copied; cache/state values
and unused capacity columns stay unread. The output includes source/profile/input
manifest hashes, actual physical strides, offsets, shared-owner pointers and
memory observations. It establishes a layout contract, not numerical parity,
peak memory or performance. The focused host check is
`PYTHONDONTWRITEBYTECODE=1 python3 tests/scripts/test_exl3_runtime_layout.py`.


`exl3_loader_check MODEL_DIR REFERENCE_MANIFEST NEW_REPORT` is a host-only
focused native-reader check. Build target `exl3_loader_check` from this CMake
directory in the pinned builder; no GPU device is required. It maps both real
shards, resolves all409 modules through current `LoadExl3`, verifies geometry,
codebook and exact borrowed pointer/span identity, and checks mapping ownership
after file destruction. `exl3_mul1_marker_test` exercises the actual shared
loader's exact multiplier and malformed-marker rejection. These checks do not
qualify complete Qwen loading, GPU upload, grouped transforms or model execution.


`dense_f16_loader_test` checks exact half-bit preservation/direct conversions,
merged BA row order and mapped ownership. `qwen_exl3_precision_test` checks the
explicit EXL3 FP16 policy and FP32 recurrent state. Both are host-only targets.
`qwen_exl3_parameter_check MODEL_DIR NEW_REPORT` loads all64 real target layers
through current `LoadQwen3_5DenseLayer` and checks their FP16 norm/Conv/BA source
values. Build target `qwen_exl3_parameter_check` in the same pinned builder; no
GPU device is needed. This does not load the complete embedding/vision/MTP or
perform a forward. The execution map records the remaining FP16 execution
wiring and full-model qualification work.

`capture_target.py --output-dir NEW_DIR --reference-manifest MANIFEST
--model-dir MODEL --image-identity PINNED_IMAGE` runs three cold eager P128/D1
repeats in the pinned original worker, with MTP and graphs disabled. It retains
layer0 block/state observations plus the actual gathered FP16 target-head input
and full 248,320-column logits. The observer delegates the original
`compute_logits` once without replacing arithmetic. Same-prefix IDs, tensor
hashes/layouts, TV/KL/top10 and unchanged strict state-band failures are retained.
The focused host check is `tests/scripts/test_exl3_target_capture.py`.

After the oracle exits, the native `test_xpu_qwen_exl3_fp16` case
`XPU EXL3 real target: eager P128 D1 full vocabulary comparison` loads the whole
target through the product loader and runs its own recurrent continuation with
FP16 activations, FP32 GDN state and FP8 KV. It gathers the last prompt row
before the full 6-bit head, saves bounded F32 logits and compares all three
repeats using the plan's existing TV/KL/top10 investigation thresholds. Set
`VT_B70_EXL3_MODEL` and `VT_B70_EXL3_S1_FIXTURES`; outputs are never overwritten.
This bounded target check does not waive failing block/state checks or qualify
D64, MTP, graphs, long context, serving or speed.

For the bounded D64 gate use `--decode-steps 64 --repeats 1` with a new output
directory `target-d64`. The observer records P128 plus64 decode heads; layer0
block/state capture remains limited to P128/D1. Ordinary and observed greedy
continuations may differ and are both preserved. The native test case
`XPU EXL3 real target: eager P128 D64 full vocabulary continuation` reads that
observed sequence and replays every prefix using its own persistent states.
It saves65 bounded raw logit rows, keeping all numerical failures.

To measure original variability on exactly that prefix, capture into another
new directory with `--decode-steps 64 --repeats 3 --trace-capture-json
/receipts/target-d64/repeat-0.json`. This explicitly enables the pinned engine's
existing trace replay, which overrides emitted IDs after sampling while
leaving the observed raw logits unchanged. These emitted IDs are forced trace
tokens, not evidence of identical greedy choices. `compare_target.py` compares
saved native rows against those matched-prefix original repeats using the same
fixed TV/KL/top10 thresholds and preserves every investigation trigger. No
new native GPU run or replacement of a failed test is implied.

The current observer also records actual embedding token IDs and all three
position axes for every head step. Each prefix is checked against the recorded
sequence before a capture is accepted. The host comparator requires these
direct witnesses, matching prefix IDs and valid capture hashes; equal emitted
trace IDs alone are insufficient evidence of identical model inputs.

`capture_attention.py --output NEW.safetensors --reference-manifest MANIFEST
--model-dir MODEL --image-identity PINNED_IMAGE` observes the actual first
full-attention layer3 at P128/D1. It retains input norm, merged QKV, Q/K norm,
post-RoPE Q/K, V, gate, attention and output projection boundaries. The original
projection method is delegated exactly once with the original arguments/result.
Actual block/slot/position metadata is checked before gathering only the written
128/129 FP8 rows; cold contents and unused page capacity are never read. Cache
continuity and unchanged existing rows at D1 are required. Host guard/delegation
checks live in `tests/scripts/test_exl3_attention_capture.py`.

After that oracle exits, the native case `XPU EXL3 real attention: identical
original operands FP8 bytes P128 D1` uses its captured post-RoPE Q/K/V and exact
active metadata. It checks byte-exact writes in the producer's interleaved
head layout, poisoned inactive capacity and native prefill/decode attention
against the unchanged rtol0.01/atol0.003 band. This qualifies that bounded
writer/core comparison only, not native QKV/norm/RoPE or the complete mixer.

The observer additionally retains actual normalized `rope_q_input`/
`rope_k_input` and selected FP16 `rope_cos_sin`. Use a new fixture path.
`replay_attention_rope.py` independently rounds FP16 products before
add/subtract on those captured operands; its focused host check is
`tests/scripts/test_exl3_attention_rope.py`. Native case
`XPU EXL3 attention RoPE: actual FP16 operands and coefficients` compares
these operands, the explicit fused primitive and generated coefficients.
The qualified bounded primitive passes38/38 assertions: captured-coefficient
rotation and all generated half coefficients at positions0-128 are exact.
The mode is now selected in scoped XPU FP16 EXL3 model execution. The actual
model-owned mixer passes655/657 assertions: numerical bands pass, but one
K-byte difference from normalization persists through P128/D1. The complete
mixer gate remains open; see the execution map for retained failed attempts.


For the focused fused-preamble diagnostic, set `VT_XPU_ATTN_NORM_PROBE` to a
new writable file prefix. The observer waits for that kernel and writes
`PREFIX.SEQUENCE.tTOKENS.f32` without overwriting files. Current output is
little-endian F32 `[T,Hq+Hkv,3]`: Q heads then K heads, with mean, mean plus
epsilon, and inverse root. Unset the variable for ordinary execution. The
current actual-kernel means and variances match the bounded Torch replay;
inverse-root differences and the exact mixer K-byte failure remain open.
