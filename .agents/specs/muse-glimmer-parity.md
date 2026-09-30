# Muse Glimmer parity: a vision reference run and the GGUF text path against llama.cpp

**Row:** `MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`
(`SPIKE`, unchanged by this spec).
**Issue:** `ISSUE-LOCAL-01M3S0ZTRRVQWCZ35W33AJJNN8`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/muse-glimmer-parity` (developer direction for this campaign:
one branch per item, local commits only, no pull request).

## Now

`SPIKE`. This spec is committed before any gate code. The row does not change
state until a token gate against a registered oracle passes.

## Scope

1. **Record correction.** The row, `docs/models/muse-glimmer.md` and
   `docs/FEATURES.md` say the pinned vLLM oracle cannot load `muse_glimmer`.
   That was true at `555967922`. The parity pin is now `e126687a9a`, which
   registers `MuseGlimmerForConditionalGeneration` and ships
   `vllm/model_executor/models/muse_glimmer.py`. The text is corrected; the
   oracle's gateability on this fleet stays unmeasured.
2. **Vision: the first reference run.** The perception encoder, the adapter and
   `vision_projection` run on the RELEASED tensors
   (`meta-models/Muse-Glimmer-30B` @ `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`)
   against a torch transcription of the pinned vLLM formulas, stage by stage, in
   f32 and in the production bf16.
3. **Text: the GGUF k-quant against llama.cpp.** Greedy decode of
   `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
   (`meta-models/Muse-Glimmer-30B-GGUF` @ `70bf1b61ac09f91b24d39038091b41c582bc5d7a`)
   through our loader and forward, compared token by token with the registered
   `llama-cpp` oracle at its pin `b10451` on the same file, CPU on both sides.
   Every divergence is classified by the logit margin on both sides at the
   first differing position: a near-tie (both margins small and the two argmax
   tokens swapped) or a defect (a large margin on either side).

Out of scope: the bf16 token gate against pinned vLLM (a ~60 GB checkpoint and
a `dgx:gpu0` lease with an oracle build, owed), the image processor (no C++
port exists; the tower is fed the reference's pixels), video, and speed.

## Upstream chain

vLLM `e126687a9a`: `vllm/model_executor/models/muse_glimmer.py`. Its vision
and text semantics are unchanged from vllm#51655 head `075d645af`, which the
existing port and `scripts/mm/muse_glimmer_vision_ref.py` cite: the diff between
the two revisions is line wrapping, the multimodal processor's call signature,
`keep_on_cpu` metadata and `get_mm_mapping` (checked with
`git diff 075d645af e126687a9a -- vllm/model_executor/models/muse_glimmer.py`).
llama.cpp `b10451` (`10bf611`), `LLM_ARCH_MUSE_GLIMMER`, for the GGUF text path.

## Our baseline

At `b45a94273`: the vision tower is gated only on synthetic LCG weights
(`test_muse_glimmer_vision`); no real tensor has passed through it. The GGUF
text path generates `" Paris. The capital of France is Paris. ..."` and agrees
with llama.cpp on the first token only (`docs/models/muse-glimmer.md`).

## Port map

| Ours | Upstream (`muse_glimmer.py` @ `e126687a9a`) |
|---|---|
| `MuseGlimmerVisionForward` | `MuseGlimmerVisionEncoder.forward` |
| `MuseGlimmerVisionAdapterForward` | `MuseGlimmerVisionAdapter.forward` |
| `MuseGlimmerEncodePixelGroups` (projection) | `_encode_pixel_groups` |

## Dependencies

- The staged tensors: `model.vision_*` from both shards, cut by HTTP range
  onto the NAS (`/mnt/nas_share/rc/muse-glimmer-30b/`), not committed.
- The GGUF file on the same NAS directory, and a CPU build of llama.cpp `b10451`.
- Existing: `muse_glimmer_vision.cpp`, `muse_glimmer_gguf_weights.cpp`,
  `muse_glimmer.cpp`, the GGUF tokenizer.

## Tests to port

None new from upstream: vLLM's `tests/models/multimodal` has no Muse Glimmer
numeric test at the pin (only `tests/transformers_utils/test_muse_glimmer_config.py`
and the tool/reasoning parser tests, already ported by the row).

## Gates

- Vision (env-gated, real tensors): per-stage `patchify`, positional table,
  `ln_pre`, block 0, tower output, adapter, projection; f32 arm relative max
  error at the level the synthetic gate holds (1e-4), bf16 arm on the bf16
  envelope with a per-row cosine floor.
- Text (env-gated, real file): greedy token ids on the prompts the row already
  uses, ours against llama.cpp, with both margins at the first divergence.

## Work breakdown

1. This spec, the issue and the record correction.
2. The vision reference run: extend the reference script with a real-weights
   mode and add the env-gated real arm to `test_muse_glimmer_vision`.
3. The GGUF comparison: a llama.cpp greedy driver that prints ids and margins,
   our side through the production GGUF load, and the classification.
4. Records: the model page, FEATURES, the matrix row, `## Outcome`.

## Risks/decisions

- A transcription is not the runtime. The vision gate establishes agreement
  with the pinned formulas on real tensors, not image-to-text correctness.
- A Q4_K_M comparison against llama.cpp can only show quantization-matched
  agreement. A near-tie divergence is recorded as such and is not a pass; a
  defect found is fixed in this branch only if it is inside this item.
- Staging: the 16.76 GB GGUF and ~3.7 GB of vision tensors live on the NAS,
  because the development host has under 10 GB free.

## Stop conditions

- The vision tensors or the GGUF cannot be staged.
- A divergence needs a runtime feature this item cannot finish; it is then
  reported with its size and left owed.

## Owed

- The bf16 token gate against pinned vLLM `e126687a9a` on a GPU lease.
  Tracked by `ISSUE-LOCAL-01M3S0ZTRRVQWCZ35W33AJJNN8`.
