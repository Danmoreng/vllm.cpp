# Spec: the GDN region's replay defect — where the zero starts, and the fix that follows

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-27).
Issue: `ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS` (this row closes it).
Follow-up to: #3325's isolation chain (the port landed; the region's
replay is broken on both GDN-hybrid models) and the new issue's
recorded next step.
Git integration: one pull request (spec + attribution + fix + gates),
branch `row/TT-GDN-REGION-REPLAY`.

## Problem

The first served replay of a GDN-hybrid (dense) model produces
**248,320/248,320 exact-zero logits** on the 27B anchor (process-
permanent) and coherent-wrong tokens on the bf16 9B. The zero is INSIDE
the 64-layer region (the `dnorm` tap reads all-zero at replay vs sane
at capture-launch). The classic-dense lane (qwen3-0.6B) serves 176
byte-correct replays on the same pin — the pin's machinery works. The
embedding is exonerated (#3325: replayed embedding byte-correct). The
culprit set: the GDN layer path, the fused preamble, or the unified-KV
PA — the things both GDN-hybrid models share and the classic lane
lacks. Layer-count bisection is unusable (the conv-shadow gate refuses
truncated-model captures).

## Attribution plan (find WHERE the zero starts)

1. **Per-layer replay taps**: at replay, probe the region layer-by-layer
   (the redesign's `VT_TT_GDN_STATE_PROBE` pattern extended to the
   replay step): each layer's input/output activations, the GDN state
   slots (ssm/conv) before/after, the kq keepquant activations. The
   FIRST layer whose output reads zero names the layer; the tap inside
   that layer that reads zero first names the op class.
2. **The three culprits, each with a discriminating tap**:
   - **GDN state binding**: do the ssm/conv state slots at replay read
     the CURRENT state (the last step's committed values) or a
     capture-time/stale binding? The redesign's W2 made the eager path
     serve the commit — check the REPLAY path binds the same tensors
     (a captured region referencing warmup-time state buffers would
     read zeros or stale values exactly once and stay broken).
   - **Unified-KV PA**: the paged-attention decode in-region — the KV
     reads at replay vs capture-launch (the KV pages the capture
     referenced vs the current pages).
   - **Fused preamble**: the preamble ops' in-region tensors.
3. **The 9B bf16 as the cleaner instrument**: coherent-wrong (not
   zeros) means partial staleness — its per-layer taps show WHICH
   values are stale (the 27B's quant amplification to full zeros is
   harder to read; the 9B's bf16 shows the drift).

## Fix follows the attribution

- State-binding stale → the capture region must reference the
  persistent state slots (the W3 in-region discipline, extended to the
  GDN states), not the warmup-time transients.
- KV staleness → the PA's in-region reads must bind the current pages
  (the paged module's persistent tensors, not capture-time views).
- Preamble → per the tap.
- Then the two `expected_cur_pos` increments LAND on top (proven
  correct, held since #3323).

## Gates

1. The served-replay test (design recorded in the issue) GREEN: the
   anchor's served stream equals eager — byte-identical to main's
   anchor (sha256 `13c3f70b…`), the 9B's served stream equals its eager.
2. The increments land: replays serve (replays > captures in steady
   state), no boundary resets.
3. Suite 92/92 / 525,723; the served-replay test lands in the tree (the
   arm's first coverage).
4. **TPOT re-measured on the genuinely served arm** — the honest
   baseline, with its decomposition (the ~30 ms/op dispatch term is
   next).
5. Standard gates.

## Risks

- The GDN state slots may be structurally capture-incompatible (like
  the embedding was) — the fix is the same in-region discipline; if a
  slot cannot enter, the row stops and records (the arm stays masked).
- The 9B's coherent-wrong may be a SECOND defect layered on the state
  binding — fix the anchor's zero first, then re-adjudicate the 9B.

## Non-goals

- No eager-path change; no dispatch-cost work (next row); no tt-metal
  change (an in-ttnn impossibility stops and escalates to the
  tt-metal#57970 lane).

## Stop conditions

- A state/KV slot cannot enter the region (design-level impossibility)
  → stop, record, the arm stays masked.
- Post-fix, served and eager diverge coherently on the anchor → stop,
  the adjudication row (which side matches the vLLM oracle).
