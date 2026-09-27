ID: ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS
Title: The dense qwen3.5 captured region's replay is broken on both of its models: the anchor (quantized) replays exact zeros, the bf16 9B replays coherently-wrong tokens — the first served replay of either never matches eager
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: 2026-09-27

## Problem

Fix row: [tenstorrent-gdn-region-replay](../../specs/tenstorrent-gdn-region-replay.md) (2026-09-27).

TT-DENSE-EMBED-IN-REGION isolation finding (2026-09-27, thalia P150, pin d20b8e27): with the embedding moved inside the captured region (byte-correct at replay: the replayed hidden equals the eager embedding of the same token) and the two expected_cur_pos increments landing (replays serve mechanically, red-first proven), the FIRST served replay of the dense 27B region writes 248,320/248,320 exact-zero logits; the final-norm hidden tap (dnorm, the lm_head input) reads tmin=tmax=0.0 at replay while the same trace's capture-launch reads sane values (-14.9..22.0). The zero originates INSIDE the 64-layer region — not the embedding (proven byte-correct), not the lm_head (the tap is the zero point). After the first zeroed replay the process is PERMANENTLY broken: every later generate — including eager cold steps — embeds correctly (hid0 nonzero) but writes zero logits; the layer-count bisection axis is unusable because the conv-shadow serveability gate refuses to capture any truncated model (captures=0 for every K<64 on both the 27B and the 9B; K=64 captures and desyncs), so the culprit cannot be narrowed by layers.

THE DISCRIMINATOR (measured the same session, CORRECTED by the eager adjudication): the bf16 Qwen3.5-9B (dense GDN-hybrid, 32 layers — the SAME driver, the SAME capture path, the SAME in-region port + increments) ALSO cannot serve correct replays: a fresh-process served-9B leg reads [220,16,220,220] against the eager-9B reference [220,16,220,16] — the first replay-served token diverges COHERENTLY (both sane, no zeros), the spec's adjudication class, and the served logits are a different distribution than the correct step's (probe: argmax 494, first5 [4.5 7.34 3.02 2.36 0.80]) — a stale-or-wrong replay input. An earlier same-session reading ('the 9B serves correctly') checked only coherence and was wrong. The qwen3-0.6B classic-dense captured gate is green on the same pin (125/125, 176 served replays at 0.068 ms, byte-correct), so the pin's trace-replay machinery works. The corrected culprit set: the machinery the two dense models share and the classic-dense lane lacks — the GDN layer path (kGdnDecode/kCausalConv1dUpdate/kGdnPostConv/kRmsNormGated/kSigmoidGateBf16), the fused preamble (kAttnQkNormRopeGate + the per-step cos|sin refill), and/or the unified-KV PA decode — with the quantized arms (the keep-quant decode path / the forced int8-dot GEMMs, tenstorrent_keepquant.cpp:965-973) AMPLIFYING the coherent divergence into the exact-zero desync on the anchor.

NEXT (the follow-up row): per-op isolation inside the quant path — the keep-quant word-decode ops vs the int8-dot GEMMs — then the fix; the two expected_cur_pos increments and the served-replay test (red-first on the masked tree: replays==captures) land with that fix per the TT-DENSE-EMBED-IN-REGION spec's stop-condition remedy. Evidence: docs/bench-evidence/tt-dense-embed-in-region-20260927.md; logs /tmp/embed-region/ (green-served-replay, green-dump2, green-dump3, sweep, sweep2, dense9b-replay, qwen3-capture-gate, red-served-replay3).

## Resolution

**2026-09-27 (the TT-GDN-REGION-REPLAY session): FIXED — the GDN state
binding was the defect; the state slots now bind persistently in-region and
the first served replay of both dense models serves correctly.**

The instrument (VT_TT_GDN_REPLAY_TAP, the VT_TT_GDN_STATE_PROBE pattern
extended to the replay step: a per-state-slot checksum between steps plus
the logits' argmax/zero-count) reproduced the defect at HEAD: cold (eager)
argmax=17 sane, capture-launch argmax=220 sane, FIRST SERVED REPLAY
argmax=0 with 248,320/248,320 exact-zero logits, then process-permanent
zeros at every later step (the served stream
`[[220,17,220,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]`, sha256
`a79a66bc…` — the exact recorded signature). The state probes: every slot's
checksum ADVANCES cold→capture and is CLOBBERED at the first replay
(`conv[47]` 30,078/40,960 zeros tmax 1.0 — index/mask-like residue; the
live slots receive values recomputed from a garbage read). The cause is
structural: both GDN state commits installed a FRESH device tensor every
step (`GdnDecodeKernel`'s `CommitDeviceLogical2D(state, std::move(newc), …)`
and `CausalConv1dUpdateKernel`'s `CommitConvTransposed(conv_state,
std::move(rolled), …)`), so the address a captured trace baked at capture
time was FREED by the very commit that produced it — tt-metal's allocator
recycles the block into the trace's own later allocations — and every
replay re-read the freed block, then re-scattered the garbage over the LIVE
slot (the process-permanent corruption). The unified-KV PA and the fused
preamble carry no per-step shadow replacement, so the state slots were the
only churn; the first consumer is layer 0, the FIRST GDN layer, whose input
is the byte-correct in-region embedding — the zero enters at the layer-0
state reads.

The fix (the spec's "state-binding stale" remedy — the W3 in-region
discipline extended to the GDN states, the embedding shadow's proven
in-place shape): both kernels copy the updated state back INTO the same
device shadow the trace reads (`ttnn::copy` into the served buffer — the
RAC lane's preallocated-destination in-region write) and publish THAT
buffer; a shape guard keeps the replacing commit on geometries the update
does not reproduce exactly. The identical op sequence runs in the eager
warmup and the captured pass (the W4 discipline), so the copy's program is
program-cache-warm before capture, and a pure copy keeps the eager values
byte-identical.

The gates: the served-replay test (tests/parity/test_qwen35_paged_engine,
the design recorded above) is RED on the pre-fix tree (the anchor case's
byte-identity CHECK fails: `row 0 token 3: served 0, eager reference 17`;
the mechanical replays>captures half passes — the two expected_cur_pos
increments, qwen3.cpp:971/:1118's mirrors, ARE in the tree at
`44b19a453`, contrary to the TT-DENSE-EMBED-IN-REGION record's "stay out"
wording: that record described the branch intent, the tree carries them)
and GREEN on the fixed tree for the anchor (served == eager, byte-identical
to main's recorded anchor `13c3f70b…`). The 9B serves its first three
tokens byte-identical to its eager leg; the FOURTH cell is an EXACT TIE in
the transformers oracle (teacher-forced on the shared prefix, top-2 {17,16}
at identical log-prob, gaps 0.00 mnats; pinned oracle env torch
2.7.1+cpu/transformers 5.8.1) — the eager arm took 16, the served arm takes
17, both the oracle's argmax, the recorded tied-runner-up treatment; the
pre-fix defect had diverged that cell to 220 (a different distribution) and
corrupted every later step, so the residual tie is the fix's success
boundary, not a second defect. The anchor's stream is byte-identical
everywhere (no adjudication needed). The increments land proven (replays >
captures at both cases), the device suite and the served-arm TPOT are
re-measured in the row's evidence
(docs/bench-evidence/tt-gdn-region-replay-20260927.md).
