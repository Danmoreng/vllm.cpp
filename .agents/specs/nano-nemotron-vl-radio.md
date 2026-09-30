# Nemotron Nano VL / Omni: the RADIO image path

**Row:** `MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2` (`INVENTORIED` ->
`SPIKE` with this spec).
**Issue:** `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/nemotron-omni-vl` (developer direction for this campaign:
one branch per item, local commits only, no pull request). The spec commit
precedes the implementation commits on the same branch.

## Now

`SPIKE`. The spec is committed. The implementation lands the image half of the
architecture in slices, and the end-to-end token gate is owed (see `## Owed`).

## 1. Scope

In scope, image modality only:

1. The RADIO vision tower (`C-RADIOv4-H`, ViT-H/16, 32 blocks, width 1280,
   16 heads, MLP 5120, 4 CLS + 6 register tokens, CPE positional table
   128 x 128).
2. The projector `mlp1`: RMSNorm(5120, eps 1e-5) -> Linear(5120 -> 20480, no
   bias) -> ReLU squared -> Linear(20480 -> 2688, no bias), after the v2 pixel
   shuffle (downsample ratio 0.5).
3. The dynamic-resolution image processor (`DynamicResolutionImageTiler`):
   the per-image patch-grid budget, the antialiased bicubic resize, the
   `(x/255 - mean)/std` normalization, and the patchify.
4. The prompt expansion: each `<image>` becomes
   `<img>` + `<image>` x N + `</img>`, with the embedding rows on the `<image>`
   positions only.
5. The registration of `NemotronH_Nano_VL_V2` and
   `NemotronH_Nano_Omni_Reasoning_V3`, whose language tower is the existing
   `NemotronHForCausalLM` forward under the `language_model.` prefix.

Out of scope, and refused by name: audio (`sound_encoder`, `sound_projection`,
`<so_embedding>`), video (`video_embedder`, `<video>`, EVS pruning), and the
static-tile (non-dynamic) InternVL tiling arm, which the released Omni
checkpoints do not select (`min_num_patches` is in `vision_config.args`).

## 2. Upstream anchors (vLLM `e126687a9a`)

| Ours | Upstream |
|---|---|
| RADIO patch generator, CPE table, CLS/register tokens | `vllm/model_executor/models/radio.py:109-421` |
| `Im2Patches`, `ViTPatchLinear` | `radio.py:424-461` |
| RADIO block (`norm1 -> attn -> +`, `norm2 -> mlp -> +`) | `radio.py:470-520`, `intern_vit.py:145-350` |
| per-image mask and `_extract_final` | `radio.py:579-604`, `:745-774` |
| RADIO config (ViT-H dims, eps 1e-6, gelu, qkv bias) | `vllm/transformers_utils/configs/radio.py:12-110` |
| `pixel_shuffle` v2, `pixel_shuffle_dynamic_res`, `extract_feature_dynamic` | `nano_nemotron_vl.py:1012-1058` |
| `mlp1` | `nano_nemotron_vl.py:955-976` |
| `get_vit_model_from_radio_config` | `nano_nemotron_vl.py:1568-1598` |
| `DynamicResolutionImageTiler` | `vllm/transformers_utils/processors/nano_nemotron_vl.py:252-570` |
| `_bicubic_resize_and_normalize` (antialiased bicubic) | `processors/nano_nemotron_vl.py:61-83` |
| `get_image_repl` | `processors/nano_nemotron_vl.py:1086-1098` |
| weight-name mapping (`language_model.backbone` -> `language_model.model`) | `nano_nemotron_vl.py:904-908`, `:1499-1566` |

The resize is torch `F.interpolate(mode="bicubic", align_corners=False,
antialias=True)`. Its kernel is `aten/src/ATen/native/cpu/UpSampleKernel.cpp`
(the `_upsample_bicubic2d_aa` path, `a = -0.5`, support scaled by the
downscale factor, weights normalized per output pixel, f32 throughout). It is
NOT Pillow's resize (`pil_resize.h`), which rounds through uint8 between the
two passes.

## 3. Design

- `include/vllm/model_executor/models/radio.h` + `src/.../radio.cpp`: the tower,
  composed from `vt::MatmulBT`, `vt::Add`, `vt::LayerNorm`, `vt::GeluErf`,
  `vt::AttentionDenseFlash`, and the shared merged-QKV split, exactly as the
  Muse Glimmer and Qwen3-VL towers are. bf16 production dtype, f32 for the
  numeric gate.
- `include/vllm/model_executor/models/nano_nemotron_vl.h` + `.cpp`: the pixel
  shuffle and the projector.
- `include/vllm/multimodal/nano_nemotron_vl_processor.h` + `.cpp`: the tiler,
  the antialiased bicubic resize, normalization, patchify, and the placeholder
  expansion through `multimodal::MakeTokenTripleReplacement`.
- `src/.../nano_nemotron_vl_registry.cpp`: the registration, `encode_mm`,
  `embed_mm`, and the forward, which delegates to the NemotronH forward.

## 4. Risks (each is silent, so each is a gate)

1. The CPE table is interpolated to a SQUARE `max(h, w)` grid and then cropped
   (`radio.py:401-410`). Interpolating straight to `(h, w)` keeps every shape.
2. The CLS and register tokens are prepended per image and stripped per image
   (`num_skip` = 10). Stripping 4 or 0 keeps the row count wrong only by a
   constant that a length check can miss when it is paired with a wrong grid.
3. Pixel shuffle v2 permutes `(0, 1, 3, 2, 4, 5)`. The v1 order transposes the
   image and keeps the shape and the multiset of values.
4. The resize is antialiased. A four-tap bicubic without support scaling
   differs on every downscale.
5. The vision output is cast to bf16 before the pixel shuffle
   (`nano_nemotron_vl.py:1055`), and the projector runs in bf16.

## 5. Tests and gates

- Stage gates against a torch transcription of the pinned vLLM formulas
  (`scripts/mm/nano_nemotron_vl_ref.py`), on the REAL checkpoint tensors
  (`vision_model.*`, `mlp1.*` of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
  at `e5e9932441de940c9a62185c870ea5bcd4cd24e2`): processor pixels, tower
  output, projector output. f32 arm with a tight bound, bf16 arm with the bf16
  envelope. The committed fixture holds reduced inputs and a digest, not the
  weights.
- Synthetic unit gates that need no checkpoint: CPE interpolation, pixel
  shuffle order, tiler geometry, placeholder expansion, config parse of the
  real released `config.json` (committed fixture).
- A reachability gate that enters through `ModelRegistry` (encode_mm and
  embed_mm through the registered architecture name).
- Mutation: each risk in §4 is mutated by the fresh reviewer.

## 6. Stop conditions

- A language-tower format this tree cannot load (the NVFP4 Omni checkpoint's
  per-module scheme differs from Nemotron 3.5 Lightning's: FP8 `o_proj` and FP8
  shared experts, bf16 `lm_head`). If the loader cannot represent it, it is
  refused by name and recorded under `## Owed`; it is not approximated.
- The end-to-end token gate needs the pinned vLLM oracle and a GPU lease for a
  23 GB (NVFP4) or 62 GB (bf16) checkpoint. If that cannot run in this session,
  it stays owed.

## Owed

- The end-to-end image token gate against pinned vLLM (`dgx:gpu0` lease,
  oracle run of `NemotronH_Nano_Omni_Reasoning_V3`). Tracked by
  `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
- Audio (`sound_encoder`) and video (`video_embedder`, EVS) arms. Refused by
  name. Tracked by the same issue.
- The GGUF arm of the language tower (inherited from `NemotronHForCausalLM`,
  `.agents/specs/nemotron-h-model.md` §5b W7).
