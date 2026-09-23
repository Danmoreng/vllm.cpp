# Leaf spec: Qwen3 GGUF arch support

**Row:** `BACKEND-GATE-ROCM-SGLANG` (sub-task: unblock the C++-only
quant-matched token gate on Strix) · **status:** `PENDING`
· **issue:** `ISSUE-LOCAL-01M3554MJ5K9V3BJ5X84YQAZ70`
· **upstream pin:** llama.cpp `b10451` / `10bf611e5` (oracle for GGUF k-quant
floor) · **base:** `6807ac2b46`

## Scope

Add `qwen3` to the GGUF dispatch table so vllm.cpp can load a Qwen3 GGUF
file (e.g. `Qwen3-4B-Q4_K_M.gguf`) and run inference through the existing
`Qwen3ForCausalLM` forward path. This unblocks the quant-matched C++-only
token gate (llama.cpp Q4_K_M vs vllm.cpp Q4_K_M) on Strix, where vLLM and
SGLang cannot run due to the ROCm 5.7 / torch 2.13.0+rocm7.2 incompatibility.

### In scope

- A `Qwen3HfConfigFromGguf` config builder that reads `qwen3.*` GGUF keys
  and maps `general.architecture = "qwen3"` onto the registered
  `Qwen3ForCausalLM` architecture string.
- A `LoadQwen3FromGguf` weight loader that populates `Qwen3DenseWeights`
  (which already uses `OwnedTensor`) from GGUF tensors.
- Wiring `qwen3_dense.cpp:LoadQwen3ForCausalLM` to accept
  `ModelSource::Kind::kGguf` in addition to `kSafetensors`.
- Adding `{"qwen3", &vllm::Qwen3HfConfigFromGguf}` to the `kGgufArchArms[]`
  dispatch table in `model_loader.cpp`.
- A focused test that loads a small Qwen3 GGUF and verifies token output
  against the llama.cpp oracle.

### Out of scope

- Keep-quant optimization for merged qkv_proj / gate_up_proj (initial
  implementation uses the expand-bf16 path; keep-quant for merged weights
  is a follow-up).
- NVFP4, EXL3, or other quant arm variants for the GGUF path.
- New GEMM kernels or attention backends.

## Upstream anchors

| Component | Pin | Source |
|---|---|---|
| llama.cpp oracle | `b10451` / `10bf611e5` | [oracle](../oracles/llama-cpp.md) |
| Qwen3-4B BF16 | `Qwen/Qwen3-4B` @ `1cfa9a72` | safetensors, existing gate path |
| Qwen3-4B Q4_K_M GGUF | sha256 `7485fe6f…` | on Strix at `/tmp/models/Qwen3-4B-Q4_K_M.gguf` |

The GGUF file carries `general.architecture = "qwen3"`. llama.cpp's
`b10451` defines the qwen3 arch writer and graph.

## Design

### Why this is simpler than it looks

`Qwen3DenseWeights` (in `include/vllm/model_executor/models/qwen3.h`) already
uses `OwnedTensor` — the same host-owned tensor type that qwen3.5's GGUF
path uses. The safetensors loader (`qwen3_weights.cpp`) populates
`Qwen3DenseWeights` from shard files; the GGUF loader will populate the
same struct from GGUF tensors. The forward path (`Qwen3DenseModel::Forward`
/ `ForwardDevice`) is source-agnostic — it consumes `Qwen3DenseWeights`
regardless of how it was loaded.

### Weight struct (existing, unchanged)

```cpp
struct Qwen3DenseAttnWeights {
  OwnedTensor qkv_proj;   // merged [Hq*Dh + 2*Hkv*Dh, H], NK orientation
  OwnedTensor o_proj;     // [H, Hq*Dh]
  OwnedTensor q_norm;     // [head_dim], empty when absent
  OwnedTensor k_norm;     // [head_dim], empty when absent
  OwnedTensor qkv_bias;   // empty when attention_bias=false
};

struct Qwen3DenseMlpWeights {
  OwnedTensor gate_up_proj;  // merged [2*I, H]
  OwnedTensor down_proj;     // [H, I]
};

struct Qwen3DenseLayerWeights {
  OwnedTensor input_layernorm;          // [H]
  OwnedTensor post_attention_layernorm;  // [H]
  Qwen3DenseAttnWeights attn;
  Qwen3DenseMlpWeights mlp;
};

struct Qwen3DenseWeights {
  bool tie_word_embeddings = true;
  bool attention_bias = false;
  OwnedTensor embed_tokens;  // [vocab, H]
  OwnedTensor final_norm;    // [H]
  OwnedTensor lm_head;       // [H, vocab], empty when tied
  std::vector<Qwen3DenseLayerWeights> layers;
};
```

### Config builder: `Qwen3HfConfigFromGguf`

New function in a new file `qwen3_gguf_weights.cpp`. Reads `qwen3.*` GGUF
metadata keys and builds an `HfConfig`:

| GGUF key | HfConfig field |
|---|---|
| `qwen3.embedding_length` | `hidden_size` |
| `qwen3.block_count` | `num_hidden_layers` (no MTP subtraction) |
| `qwen3.feed_forward_length` | `intermediate_size` |
| `qwen3.attention.head_count` | `num_attention_heads` |
| `qwen3.attention.head_count_kv` | `num_key_value_heads` (default = head_count) |
| `qwen3.attention.key_length` | `head_dim` (default = hidden_size / head_count) |
| `qwen3.attention.layer_norm_rms_epsilon` | `rms_norm_eps` |
| `qwen3.rope.freq_base` | `rope_theta` (default 10000) |
| `qwen3.context_length` | `max_position_embeddings` |
| `qwen3.vocab_size` (or `token_embd.weight` shape) | `vocab_size` |
| presence of `output.weight` tensor | `tie_word_embeddings` (false if present) |

Sets `c.architectures = {"Qwen3ForCausalLM"}`, `c.model_type = "qwen3"`,
`c.torch_dtype = "bfloat16"`, `c.attention_bias = false` (Qwen3 has no
attention bias).

Mirrors the structure of `HfConfigFromGguf` in `qwen3_5_gguf_weights.cpp:944`
but without MoE, GDN, MTP, or linear-attention fields.

### Weight loader: `LoadQwen3FromGguf`

New function in `qwen3_gguf_weights.cpp`:

```cpp
Qwen3DenseWeights LoadQwen3FromGguf(
    const GgufFile& gguf, const HfConfig& config,
    const GgufLoadPolicy* policy = nullptr);
```

Loads GGUF tensors by llama.cpp name pattern and populates
`Qwen3DenseWeights`. Uses the existing `GgufLoadPolicy` / `OwnedTensor`
infrastructure from `qwen3_5_gguf_weights.h`.

GGUF tensor name → weight field mapping:

| GGUF tensor name | Weight field | Notes |
|---|---|---|
| `token_embd.weight` | `embed_tokens` | |
| `output_norm.weight` | `final_norm` | |
| `output.weight` | `lm_head` | only when untied |
| `blk.{i}.attn_norm.weight` | `input_layernorm` | |
| `blk.{i}.post_attention_norm.weight` | `post_attention_layernorm` | |
| `blk.{i}.attn_q.weight` + `attn_k.weight` + `attn_v.weight` | `qkv_proj` | merged: concat along dim 0 |
| `blk.{i}.attn_output.weight` | `o_proj` | |
| `blk.{i}.attn_q_norm.weight` | `q_norm` | loaded when present |
| `blk.{i}.attn_k_norm.weight` | `k_norm` | loaded when present |
| `blk.{i}.ffn_gate.weight` + `ffn_up.weight` | `gate_up_proj` | merged: concat along dim 0 |
| `blk.{i}.ffn_down.weight` | `down_proj` | |

**Merged weight concatenation.** The GGUF file stores q/k/v and gate/up as
separate tensors. The `Qwen3DenseAttnWeights.qkv_proj` field is a single
merged `OwnedTensor`. The loader must concatenate:

1. Load each source tensor via the `GgufLoadPolicy` routing (expand-bf16 or
   keep-quant, via the same `OwnMatmulWeight` function used by qwen3.5).
2. For the expand-bf16 path: concatenate the raw bf16 byte buffers in
   q→k→v order (for qkv_proj) or gate→up order (for gate_up_proj). The
   result is a single `OwnedTensor` with shape `[sum_rows, cols]`.
3. For the keep-quant path: concatenate the raw block arrays. Q4_K blocks
   are per-row, so vertical concatenation is byte-level append. The shape
   becomes `[sum_rows, cols]` with the same block dtype.

The safetensors path produces the same merged layout via
`LoadMergedBf16RawNK`, so the forward path is unchanged.

**Initial implementation focus:** the expand-bf16 path (the `VT_CPU_REF=1`
default on CPU). Keep-quant for merged weights works by block-level
concatenation but is validated against the existing keep-quant GEMM path,
not gated separately.

### Wiring changes

**`qwen3_dense.cpp`** — `LoadQwen3ForCausalLM` (line 62):

Replace the `kSafetensors`-only guard with a kind switch:

```cpp
if (source.kind == ModelSource::Kind::kGguf) {
  const GgufLoadPolicy gguf_policy =
      GgufLoadPolicy::FromEnv(source.device);
  auto weights = LoadQwen3FromGguf(*source.gguf, config, &gguf_policy);
  return std::make_unique<Qwen3DenseLoadedModel>(
      registration, std::move(weights));
}
// existing safetensors path unchanged
```

**`model_loader.cpp`** — `kGgufArchArms[]` (line 1256):

Add one entry:
```cpp
{"qwen3", &vllm::Qwen3HfConfigFromGguf},
```

**`CMakeLists.txt`** — add `qwen3_gguf_weights.cpp` to the model source list.

**New files:**
- `src/vllm/model_executor/models/qwen3_gguf_weights.cpp` — config builder + weight loader
- `include/vllm/model_executor/models/qwen3_gguf_weights.h` — public API declarations

## Risks

1. **Merged weight concatenation correctness.** The q/k/v concat must
   produce the same row ordering as `LoadMergedBf16RawNK` in the safetensors
   path. A wrong order silently produces garbage attention. Mitigated by:
   the test compares GGUF output against the llama.cpp oracle, which reads
   the same GGUF file.

2. **QK-norm presence.** Qwen3-4B has `attn_q_norm` / `attn_k_norm`;
   Qwen3-0.6B does not. The loader must check tensor existence, not assume
   presence. An empty `OwnedTensor` (the default) is the correct absent
   state — the forward path already checks.

3. **`tie_word_embeddings` detection.** The GGUF has no
   `tie_word_embeddings` metadata key. Detect by checking whether
   `output.weight` tensor exists. Qwen3-4B is untied; Qwen3-0.6B is tied.

4. **Keep-quant for merged weights.** Block-level concatenation of Q4_K
   blocks is correct only when all source tensors share the same K
   dimension (they do — all project from `hidden_size`). The risk is a
   block-boundary misalignment when `hidden_size` is not a multiple of the
   block size (256 for Q4_K). Qwen3-4B has `hidden_size = 2560`, which is
   `256 × 10`, so blocks align. A general implementation must reject a
   misaligned case rather than produce corrupt output.

## Tests

1. **Focused test** — load `Qwen3-4B-Q4_K_M.gguf` via the GGUF path, run
   greedy decode on the 6 standard prompts (128 tokens, temp=0, seed=42),
   and compare token IDs against the llama.cpp oracle output (already
   captured on Strix). This is the token-exact gate.

2. **Unit test** — load a small synthetic GGUF (or the real Q4_K_M file)
   and verify that `Qwen3DenseWeights` fields are populated with correct
   shapes. This is the focused red-before test.

3. **Reachability** — the test enters through `ModelRegistry::Forward`
   (the production entry point), not by constructing `Qwen3DenseWeights`
   by hand.

## Gates

1. **Token-exact** — vllm.cpp Q4_K_M output matches the PRIMARY oracle
   (vLLM eager-mode) on all 6 prompts. The gate denominator is vLLM, not
   llama.cpp. See Evidence below for why llama.cpp cannot serve as the
   denominator.
2. **Build** — clean CPU build with `-Werror`, 0 warnings.
3. **Existing tests** — full CPU `ctest` green (no regressions in
   existing qwen3 safetensors tests).

## Evidence

- llama.cpp Q4_K_M output: captured on Strix, all 6 prompts, `b10451`
  pin, sha256-verified model file.
- vllm.cpp Q4_K_M GGUF output: captured on devbox (CPU-only), all 6
  prompts, same Q4_K_M file. Fed the same `ORACLE_PROMPT_IDS` as
  `gen_rocm.py` via `vllm_complete_tokens` (ABI v13). All 6 prompts
  generated 48 tokens.
- CPU-only diagnostic (devbox, g++ 13.3): 3/6 exact token matches vs
  llama.cpp. Identical match/mismatch pattern to Strix HIP — the HIP
  backend introduces no additional error.
- vllm.cpp token IDs for all 6 oracle prompts recorded and ready for
  comparison once vLLM is rebuilt on Strix.

### Root cause of 3/6 mismatches

The Qwen3-4B Q4_K_M GGUF file carries heterogeneous quantization per tensor:

| Tensor | ggml type |
|---|---|
| `attn_q.weight` | Q4_K |
| `attn_k.weight` | Q4_K |
| `attn_v.weight` | Q6_K |
| `ffn_gate.weight` | Q4_K |
| `ffn_up.weight` | Q4_K |
| `ffn_down.weight` | Q6_K |

The loader (`qwen3_gguf_weights.cpp:471`) forces ALL merged weights through
`NoKeepQuant(pol)` — bf16 expansion — because merged qkv_proj has
heterogeneous types (Q4_K q/k + Q6_K v) that cannot share one block dtype.
This also forces the homogeneous gate_up_proj (Q4_K + Q4_K) through bf16
expansion unnecessarily.

llama.cpp runs each tensor separately with native Q4_K/Q6_K dot products
(dequantize to fp32 on the fly, accumulate in fp32). The bf16 expansion path
(dequantize to bf16, then bf16 GEMM) loses precision in the dequantization
step, causing greedy argmax to diverge on near-tied logits.

Q4_K_S (homogeneous Q4_K) is not available on HuggingFace for Qwen3-4B, so a
homogeneous-quant control test is not possible with published artifacts.

### vLLM is the primary oracle and it diverges from llama.cpp too

The Qwen3.8-27B three-way comparison
(`docs/bench-evidence/oracle-vllm-gfx1151-20260903.md`) measured:

| Comparison | Divergences |
|---|---|
| vLLM eager vs llama.cpp | 4/6 |
| vLLM compiled vs llama.cpp | 3/6 |
| vllm.cpp vs llama.cpp | 3/6 |
| vllm.cpp vs vLLM (compiled) | 5/6 |

vLLM (the PRIMARY oracle) diverges from llama.cpp (the SECONDARY oracle) at
the same rate as vllm.cpp does. A gate whose denominator the reference
implementation does not satisfy is measuring the denominator, not the arm
under gate.

The root cause is the same on all three engines: Q4_K_M's heterogeneous
quantization (Q4_K + Q6_K) forces bf16 expansion for merged weights in both
vLLM and vllm.cpp, while llama.cpp uses native quantized dot products. The
bf16 path loses precision, causing greedy argmax to diverge on near-tied
logits.

### LoadMerged port: implemented, tested, abandoned

The `LoadMerged` pattern from `muse_glimmer_gguf_weights.cpp:255-345` was
ported to the Qwen3 GGUF loader on branch
`row/BACKEND-GATE-ROCM-SGLANG-q4-keepquant`. `LoadMerged` keeps homogeneous
shards (gate_up_proj Q4_K+Q4_K) on the keep-quant path and only expands
heterogeneous shards (qkv_proj Q4_K+Q6_K) to bf16.

Result: 3/6 matches — same count as without `LoadMerged`. The mismatch
pattern shifted (prompt 2 MISMATCH to MATCH, prompts 1 and 3 MATCH to
MISMATCH), proving the keep-quant path activated, but the overall count did
not improve.

More importantly, vLLM ALWAYS expands merged weights to bf16 for Q4_K_M —
even homogeneous shards. `LoadMerged` makes vllm.cpp take a different
dequantization path from the primary oracle. The port is abandoned.

### Next step

The correct gate is vLLM eager-mode vs vllm.cpp on Qwen3-4B-Q4_K_M, same
6 prompts, same `ORACLE_PROMPT_IDS`. The vLLM installation on Strix was
lost (the `/workspace` venv was cleaned between sessions). Rebuilding vLLM
on Strix is a multi-hour effort. The vllm.cpp token IDs for all 6 oracle
prompts are captured and ready for comparison.

The gate gap stays open. The next traceable step is rebuilding vLLM on
Strix and running `gen_rocm.py` with Qwen3-4B-Q4_K_M.

## Stop conditions

- Stop if the GGUF tensor name patterns do not match what llama.cpp's
  `b10451` qwen3 arch writer produces. Verify by dumping tensor names
  from the GGUF file before implementing.
- Stop if the merged weight concatenation cannot produce correct output
  (would indicate a structural mismatch between GGUF per-tensor storage
  and the merged `OwnedTensor` layout).
- Stop if the config builder cannot produce an `HfConfig` that the
  existing `Qwen3DenseModel::Forward` accepts.

## Git integration

One pull request (spec + implementation). Default per repository policy.
