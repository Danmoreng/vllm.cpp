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

## Now

2026-09-27: the attribution FOUND the zero and the fix followed the cause.
The instrument (`VT_TT_GDN_REPLAY_TAP`: the state-slot checksum between
steps + the logits probe) reproduced the defect at HEAD — cold and
capture-launch sane, the FIRST SERVED REPLAY all-zero (248,320/248,320),
process-permanent after — and the state probes show every slot's checksum
advancing cold→capture and CLOBBERED at the first replay. The cause is
structural: both GDN state commits installed a FRESH device tensor every
step, so the address the captured trace baked at capture time was freed by
the very commit that produced it (the allocator recycles the block), and
every replay re-read the freed block, then re-scattered the garbage over
the LIVE slot. The first consumer is layer 0, the FIRST GDN layer — the op
class is the GDN state binding (the PA and the preamble carry no per-step
shadow replacement). The fix (the W3 in-place discipline extended to the
ssm and conv shadows) landed; the anchor's served stream is byte-identical
to main's recorded anchor (sha256 `13c3f70b…`, file-for-file), the
served-replay test (the issue's recorded design, red-first on the pre-fix
tree) is green, and the TPOT is re-measured on the genuinely served arm.
The 9B's residual single-token divergence is an EXACT TIE in the
transformers oracle (top-2 {17,16} at identical log-prob, 0.00 mnats both)
— the tied-runner-up class, recorded in the test's 9B case; neither stop
condition fired (the anchor is byte-identical). The increments were already
in the tree via `44b19a453` (the prior record's "stay out" wording
described the branch intent, not the tree); this row's gates prove them.

## Outcome

- Measured (P150, pin `d20b8e27f29`, all legs under the file mutex): the
  pre-fix red — the anchor tap leg's step table (cold argmax=17 sane /
  capture argmax=220 sane / first served replay argmax=0 with
  248,320/248,320 zeros / every later step zero), the served stream
  `a79a66bc…`, the state-slot clobber columns; the post-fix greens — the
  anchor served stream sha256 `13c3f70b…` byte-identical to main's
  recorded anchor file-for-file, the served-replay test green on both cases
  (the anchor byte-identical, the 9B byte-identical up to the
  oracle-exact-tie cell), TPOT mean 34,165.25 ms with the served replay
  measured at 12.0 ms wall (`replay_ms` 0.3) against the eager step's
  35,177.9 ms, `boundary=0` at the served steps, the device suite at the
  standing bar.
- Rejected: fixing the aux-tap seam's TT D2D arm as the instrument's
  per-layer channel (`CopyDeviceDeviceIfCapture` re-shadows the aux slot
  with each tap's clone) — the aux seam's TT defect is real but dormant
  (no DFlash drafter runs on TT) and fixing a shared Copy lane is its own
  row; the attribution rests on the state-slot probes + logits + the
  structural code facts. Rejected: keeping the replacing state commits and
  re-warming the slots outside capture every step — the state is the
  recurrence's own memory; the trace must read and write the SAME buffer
  or the recurrence freezes/clobbers (the W2 lesson, at replay scale).
  Rejected: gating the 9B case on byte-identity alone — the oracle's own
  top-2 at the divergent cell are IDENTICAL (-0.706930 both), so
  byte-identity is not a well-posed bar there; the recorded tie treatment
  (the 0.8B gate's own words) applies.
- Defaults and their values: the in-place commit is unconditional on the
  geometry-matched path with a shape-guard fallback (no env knob — the
  persistent binding is a correctness property, not a configuration); the
  instrument stays env-gated (`VT_TT_GDN_REPLAY_TAP`, one getenv when
  unset); the served-replay test's 9B tie set is the recorded oracle
  adjudication, not a tolerance (any OTHER cell divergence fails).
- Owed: the dispatch-cost decomposition (the ~30 ms/op term) is the next
  row's work per the spec's non-goals; the aux-tap seam's TT D2D defect is
  recorded in this row's evidence for whoever first runs a DFlash drafter
  on TT.
