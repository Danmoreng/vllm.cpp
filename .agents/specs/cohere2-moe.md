# Cohere2MoeForCausalLM (North): the MoE text model with interleaved sliding window

**Row:** `MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm` (`INVENTORIED` ->
`SPIKE` with this spec).
**Issue:** `ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/cohere2-moe` (developer direction for this campaign: one
branch per item, local commits only, no pull request). The spec commit precedes
the implementation commits on the same branch.

## Now

`SPIKE`. The spec is committed; the port starts from it.

## Scope

Register `Cohere2MoeForCausalLM` and make it forward on CPU through the shared
seams, for the released `CohereLabs/North-Mini-Code-1.0`
(@ `d11e61a842617a22dc328552fa5bb86231ee4f37`, bf16, 56.8 GiB, 49 layers,
hidden 2048, 32 q heads / 4 kv heads, head_dim 128, 128 experts top-8,
vocab 262144). The mechanisms, each from the pinned file:

1. Per-layer sliding window from `layer_types`: a `sliding_attention` layer has
   window `sliding_window + 1` (`cohere2_moe.py:205-211`); a full layer has none.
2. RoPE only on sliding layers, and on prefix-dense layers when
   `prefix_dense_sliding_window_pattern == 1` (`:213-222`, `:242-243`). GPT-J
   interleaved style (`is_neox_style=False`, `:202`). A full-attention MoE layer
   is NoPE.
3. `RMSNorm` (f32 statistics, weight applied in f32, `:75-94`) when
   `rms_norm_eps` is set, else the Cohere `LayerNorm` (`:97-103`).
4. The parallel block: one input norm feeds attention AND the MLP, and
   `hidden = residual + attn + mlp` (`:371-384`).
5. A dense prefix MLP from `mlp_layer_types`, normalized from
   `first_k_dense_replace` when absent (`:405-419`), sized
   `prefix_dense_intermediate_size` (`:351-360`).
6. The router: `sigmoid(gate(x))` then top-k, renormalized only when
   `norm_topk_prob` (`:56-72`, `:301-314`). North sets it False.
7. Shared experts when `num_shared_experts > 0`, combined `average` (the MoE sum
   halved, `:326-327`) or `sum` (`:287-303`). North ships none; the arm is
   ported and synthetic-gated.
8. Tied embeddings: `compute_logits` uses `embed_tokens` with `logit_scale`
   (`:499-502`, `:526-530`); `lm_head.*` is dropped at load (`:478`).
9. `use_qk_norm` is NOT in the pinned `cohere2_moe.py`. A checkpoint that sets
   it True is refused by name (the pin has no such mechanism to mirror).

Out of scope, each refused by name or owed: the GGUF arm (llama.cpp `b10451`
defines `cohere2moe`; the loader arm is owed), FP8 / W4A16 / NVFP4 sibling
checkpoints, the EAGLE drafter (`North-Mini-Code-1.0-eagle`), the gated
`North-Small-Translate-1.0` (406 GiB), GPU device arms beyond what the shared
ops provide, and speed.

## Upstream chain

vLLM `e126687a9a`: `vllm/model_executor/models/cohere2_moe.py` (534 lines),
`commandr.py::LayerNorm`, the `Attention` layer's `per_layer_sliding_window`,
`get_rope(..., is_neox_style=False)`, `FusedMoEFactory` with
`custom_routing_function`. The HF config class is transformers' `Cohere2MoeConfig`
(`model_type: cohere2_moe`).

## Our baseline

At `b45a94273`: `CohereForCausalLM` is registered (dense, LayerNorm, full-width
GPT-J RoPE, parallel block, tied, `logit_scale`) and refuses `use_qk_norm` and
`sliding_window` by name (`commandr_registry.cpp:129-137`). Laguna
(`laguna*.cpp`) carries the closest shared pieces: interleaved sliding window
over one paged cache, a dense layer 0 and a sigmoid MoE router. The
sliding-window switch is `ENG-ATTENTION-WINDOW`.

## Port map

| Ours (new files) | Upstream |
|---|---|
| `cohere2_moe.h`, `cohere2_moe.cpp` (forward) | `Cohere2MoeAttention`, `Cohere2Moe`, `Cohere2MoeDecoderLayer`, `Cohere2MoeModel` |
| `cohere2_moe_weights.cpp` (loader) | `load_weights` + `hf_to_vllm_mapper` |
| `cohere2_moe_registry.cpp` | the registration, config hook, KV spec |

## Dependencies

- Shared ops (`vt::MatmulBT`, `vt::RmsNorm`, `vt::RopeFromCache` non-neox, the
  paged attention with a window, the grouped MoE GEMM and SiLU-and-mul).
- `ENG-ATTENTION-WINDOW` for the per-layer window.
- The Command-R `LayerNorm` path for the non-RMS arm.

## Tests to port

vLLM `e126687a9a` ships no `cohere2_moe` model test (`tests/models/` has no
file naming it). The registry example-config coverage entry is ported by
extending `test_model_registry`.

## Gates

- Config: the released `config.json` committed as a fixture and parsed; each
  mechanism flag read from it; `use_qk_norm: true` refused.
- Forward: a tiny synthetic checkpoint against a torch transcription of the
  pinned `cohere2_moe.py` (f32 arm tight, bf16 envelope), with one mutation-
  proven case per mechanism 1-8.
- Real tensors (env-gated): the first layers of North-Mini-Code-1.0, fetched by
  HTTP range, per-layer against the transcription.
- Reachability: the model loads through `ModelRegistry` and decodes through the
  runner.

## Work breakdown

1. This spec and the records.
2. Config parse and registration, with the refusals.
3. Loader (bf16 safetensors), enumeration accounted against the released index.
4. Forward (host reference and the runner path through the shared seams).
5. Synthetic gates, the real-tensor layer gate, reachability.
6. Records: FEATURES, USAGE weights, matrix row, `## Outcome`.

## Risks/decisions

- The window is `sliding_window + 1`, not `sliding_window`. Off by one keeps
  every shape.
- NoPE on full MoE layers but RoPE on the full prefix-dense layer. Getting
  either wrong keeps every shape.
- The router does NOT renormalize for North. Renormalizing keeps every shape.
- `shared_expert_combination_strategy == "average"` halves the WHOLE MoE output
  (routed + shared), not only the shared part (`:326-327`).
- A token gate on the real checkpoint needs 56.8 GiB over a ~5 MB/s link and a
  GPU oracle; it is owed.

## Stop conditions

- A mechanism the shared seams cannot represent: extend the seam or record the
  exact tracked exception, never a parallel path.

## Owed

- The end-to-end token gate against pinned vLLM (GPU lease, 56.8 GiB), and the
  GGUF arm against llama.cpp `b10451`. Tracked by
  `ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY`.
