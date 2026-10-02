# TT GDN region replay — the first served replay's zero, the attribution, and the fix (2026-09-27)

Branch `row/TT-GDN-REGION-REPLAY` @ `7e1e2bfa8` (the spec commit) + this
row's change. Host: personal Tenstorrent Blackhole P150a workstation
(aarch64, GCC 16); every device command under `flock -x $HOME/gpu.lock`,
`~/Sources/tt/luwen/target/release/reset` (+15 s, retried 3x on the 101
flake), `source ~/Sources/tt/env-tt-common.sh`. vllm.cpp: Release Ninja
build `/tmp/row-gdn-region/build` (`-DVLLM_CPP_TENSTORRENT=ON
-DVLLM_BUILD_TESTS=ON`, the tt-metal-pin cmake prefix
`vllm-cpp-pin/20260925` (`d20b8e27f29`), build-linkage and runtime trees
verified at the same commit). Models: the anchor APEX I-Nano
(`/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`,
11,240,605,152 bytes) and the bf16 discriminator `Qwen3.5-9B-hf`
(`Qwen3_5ForConditionalGeneration`, 32 layers, GDN-hybrid dense). Recipe:
the recorded anchor leg verbatim (`--num-prompts 4 --output-len 4
--concurrency 1 --seed 0 --temperature 0 --ignore-eos`, fixture
`tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`, raw prompts —
`--skip-chat-template` is the bench default). Logs under `/tmp/gdn-region/`.

## The instrument (the spec's attribution plan, step 1)

`VT_TT_GDN_REPLAY_TAP` (documented in docs/ENVIRONMENT.md) prints, after
every `Qwen3_5DenseDecodeGraph` step completes — between steps, never
during capture — the step's logits (argmax / top-2 / margin / exact-zero
count / first five) and one `[GDN-SHADOW-PROBE]` line per GDN state cache
(ssm and conv, 48 each on the 27B, 32 on the 9B): an FNV-1a checksum of
the CURRENT device shadow's bytes plus its min/max/zero count. This is
the redesign's `VT_TT_GDN_STATE_PROBE` pattern extended to the replay step:
the in-kernel probe cannot print at replay (a replay runs no host code,
and a download is the read a trace capture refuses), so the driver reads
between steps.

**The per-layer residual-stream taps were attempted through the SPEC-DSPARK
W8 aux buffer and are NOT usable on this lane** — a pre-existing defect of
the aux seam on TT, recorded here rather than printed as if it were data:
`MaybeCaptureAuxTap` writes its columns through `Backend::Copy`, whose TT
D2D arm (`CopyDeviceDeviceIfCapture`, tenstorrent_residency.cpp:1260-1280)
clones the SRC into a FRESH device tensor and installs THAT as the
destination slot's shadow — the [S, H*taps] aux buffer's device shadow
becomes the last tap's [1, H] clone and the buffer's host bytes are never
written. The first instrument build's per-layer columns therefore read pool
residue (all-zero beyond the residue). No DFlash drafter runs on TT, so the
defect is dormant; the attribution below rests on the state-slot probes,
the logits probe, and the structural code facts, all of which are valid.

## The red (the defect reproduced at HEAD, the pre-fix tree)

Log `tap-anchor.log`, the anchor leg with the instrument on:

| Step | Kind | LOGITS | zeros | verdict |
|---|---|---|---|---|
| 0 | cold (EAGER) | argmax=17, top2=220 | 0/248,320 | SANE — the second output token, the eager reference |
| 1 | capture-launch | argmax=220 | 0/248,320 | SANE — the third output token, the recorded anchor's |
| 2 | **FIRST SERVED REPLAY** | **argmax=0** | **248,320/248,320** | **THE ZERO — the `a79a66bc` signature** |
| 3-15 | every later step, including request 2-4's EAGER cold and capture steps | argmax=0 | 248,320/248,320 | PROCESS-PERMANENT — the recorded "never recovers" |

The served stream: `[[220,17,220,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]`, sha256
`a79a66bcc209da45800a881cf80b5c05d181a1ef5c0829d5e00857d0704f5e48` — the
exact `a79a66bc` class the record names, against the anchor
`[[220,17,220,17],[220,17,220,17],[220,16,220,17],[220,16,220,16]]`
(sha256 `13c3f70b…`). The replays SERVE mechanically (each request's third
decode step is a replay: 8 `Replay` calls vs 4 segments captured — the two
`expected_cur_pos` increments, in main via `44b19a453`, work).

## The attribution (the spec's three culprits, discriminated)

**The GDN state binding. The state slots' content columns:**

| Probe | step 0 (cold) | step 1 (capture-launch) | step 2 (first replay) |
|---|---|---|---|
| `ssm[0]` fnv / zeros | `f3e26047…` 0/786,432 (sane) | `0c25efa7…` 0/786,432 (ADVANCED) | `c94f2d5b…` — the trace's scatter CLOBBERED the live slot |
| `ssm[47]` fnv / zeros | `77a299c3…` 0/786,432 | `598f8196…` 0/786,432 | `c7d16c8b…` **148,253/786,432 zeros, tmax 0.023** — garbage |
| `conv[47]` fnv / zeros | `abf4af39…` 0/40,960 | `37444ec8…` 0/40,960 | `0ca3317a…` **30,078/40,960 zeros, tmin 0, tmax 1.0** — index/mask-like residue, not a conv window |

Every state slot's checksum ADVANCES cold→capture (the eager and
capture-launch passes are correct) and is CLOBBERED at the first replay —
the live slots receive values recomputed from a garbage read, which is why
the process never recovers.

**The structural cause (the code):** the two GDN state commits installed a
FRESH device tensor every step — `GdnDecodeKernel`'s
`CommitDeviceLogical2D(state, std::move(newc), …)` (the `ScatterRowsDevice`
commit) and `CausalConv1dUpdateKernel`'s `CommitConvTransposed(conv_state,
std::move(rolled), …)`. `ScatterRowsDevice`/the roll produce new
allocations (`ttnn::indexed_fill` and the gathers are out-of-place), so the
slot's device shadow changes ADDRESS at every decode step. A captured trace
bakes the state READ at the address served at capture time; the capture
pass's own commit FREES that address (the old `s->device` dies in the
commit), and tt-metal's allocator recycles the block — into the trace's own
later allocations first. Every replay then re-reads the freed block
(recycled content) and re-scatters the result over the LIVE slot's buffer
(the recorded newc — still the slot's shadow, since no host code ran), which
is the process-permanent corruption. The unified-KV PA reads and writes are
already persistent and in-place (`paged_fused_update_cache`, the RAC lane),
and the fused preamble reads the persistent cos|sin table refreshed outside
capture — neither carries a per-step shadow replacement, so the state slots
are the only cross-step binding inside the region whose address churns. The
first consumer is layer 0, the FIRST GDN layer (the region's layer order
0,1,2 gdn, 3 fa, repeating): its input is the byte-correct in-region
embedding (#3325), and the only other values it reads are `conv_state[0]`
and `ssm_state[0]` — the zero enters at the layer-0 state reads, which
names the op class: the GDN state binding, not the PA and not the preamble.
The 9B's coherent-wrong (the recorded `[220,16,220,220]` vs eager
`[220,16,220,16]`) is the same class at bf16: the recycled block held a
SANE-but-wrong tensor, so the recurrence drifts instead of amplifying to
zeros — the spec's cleaner-instrument reading.

## The fix (the spec's "state-binding stale" remedy)

The W3 in-region discipline extended to the GDN states — the state shadows
become PERSISTENT, IN-PLACE-UPDATED device buffers, the exact shape the
embedding shadow already uses (tenstorrent_ops.cpp's HOST-FREE-DECODE
in-place refresh, `tenstorrent_capture.cpp:359`):

- `GdnDecodeKernel` (tenstorrent_gdn.cpp): the scattered state is copied
  back into the SAME `cache2d` shadow the trace reads (`ttnn::copy(next,
  cache2d)` — the preallocated-destination in-region write the RAC lane
  proves capture-safe, tenstorrent_paged.cpp:826) and THAT buffer is
  published (`CommitDeviceLogical2D(state, std::move(cache2d), …)`); a
  shape guard keeps the replacing commit on any geometry the update did not
  reproduce exactly.
- `CausalConv1dUpdateKernel` (tenstorrent_gdn.cpp): the rolled window is
  copied back into the persistent transposed shadow `T0`
  (`EnsureConvStateTransposed`'s served buffer) and `T0` is published; the
  same shape guard.

In place, the trace's next gather/slice reads the state its own
scatter/roll wrote: the recurrence ADVANCES at every replay. The identical
op sequence runs in the eager warmup and the captured pass (the W4
discipline), so the copy's program is program-cache-warm before capture;
a pure copy keeps the eager values byte-identical.

## The 9B's residual single-token tie (the re-adjudication)

Post-fix, the 9B serves `[220,16,220,17]` against the eager
`[220,16,220,16]` (reproduced byte-exact this session, `9b-eager.log`): the
first three tokens byte-identical, the fourth — the replay-served cell —
flipped. The probe margins: the served step's top-2 are 17 (19.0) and 16
(18.875), margin 0.125; the eager arm's first-step top-2 are 16 (19.375)
and 17 (19.25), the same 0.125 — a 1-2-bf16-ulp class. The transformers
oracle adjudicated (`9b-oracle.log`, the pinned oracle env torch
2.7.1+cpu / transformers 5.8.1, bf16, CPU — the keepquant row's oracle pin):
teacher-forced on the shared prefix, the oracle's top-2 at the fourth
position are **{17, 16} at IDENTICAL log-prob (-0.706930 both)** — gaps
0.00 mnats for BOTH tokens, inside the 500-mnat band by construction; the
oracle's own greedy stream is `[220,16,220,16]`. The divergence is an EXACT
TIE in the oracle — the tied-runner-up class the 0.8B gate records ("at
exact ties the arm may take the tied runner-up"); the pre-fix defect had
diverged the same cell to 220 (a different distribution), so the residual
tie is the fix's success boundary, not a second defect. The served-replay
test's 9B case accepts either tied token at that cell (the adjudication
recorded in the case); the anchor case is byte-identical everywhere.

## Gates

| Gate | Verdict |
|---|---|
| RED: the served-replay test, pre-fix tree | **RED as required** — the anchor case's byte-identity CHECK fails: `row 0 token 3: served 0, eager reference 17` (the first served replay emits token 0), then every later token 0; the mechanical half passes (8 replays > 4 captures). Log `test-red-27b.log` |
| Served-replay test GREEN (the anchor) | **PASS** — the anchor case green: served == eager, byte-identical; the replays-serve CHECK passes. Logs `test-green.log`, `test-green2.log` |
| Served-replay test GREEN (the 9B) | **PASS** — the 9B case green: the first three tokens byte-identical, the fourth an oracle-tied cell (0.00 mnats both) accepted per the recorded adjudication; replays serve. **The final binary's run: 2/2 cases, 6/6 assertions, SUCCESS** (`test-green2.log`) |
| Anchor served byte-identity | **PASS — sha256 `13c3f70b3611ff6b59fef413357704ca2bcd84251c72baff450a935de1a049cb`, byte-identical to main's recorded anchor, FILE-FOR-FILE** (`cmp` clean). Log `anchor-served.log` |
| The increments land | **PASS** — replays serve: 8 `Replay` calls vs 4 segments captured per anchor run; the served steps print `boundary=0` (NO boundary resets at replays; the cold/capture steps carry the request boundary as before). Log `anchor-served.log` |
| TPOT on the genuinely served arm + decomposition | mean **34,165.25 ms** (median 34,177.09, P99 34,208.93) vs main's masked-arm 34,651.60 — **-1.4%**, now measured on an arm whose replays SERVE. Decomposition (`VT_TT_STEP_PHASES`, per request): [cold 35,177.9 ms wall (eager body 34,927.3) → capture 4,748.3 ms wall (cap_body 2,947.4 + cap_end 1,792.1) with the 31,164.2 ms trace-wait landing in the next step's `gap_prev` → **served replay 12.0 ms wall, `replay_ms` 0.3, `boundary=0`**]. The per-served-step cost is now 12 ms against the eager step's 35 s; the TPOT mean stays cold-step-dominated at output_len 4 (one cold + one capture amortize over 3 decode steps), so the honest decomposition is the line above: the served replay is ~2,900x the eager step. Log `anchor-served.log` |
| Device suite `test_tenstorrent_backend` | **PASS — 92/92 cases, 525,723/525,723 assertions** (the standing bar; the in-place commits preserve the eager conv/ssm oracle arms). Log `suite.log` |

The two `expected_cur_pos` increments: the TT-DENSE-EMBED-IN-REGION record
says they "stay OUT of the tree", but the tree carries them (`44b19a453`
landed its +2 with a message that says "no product change lands on this
branch" — the record described the branch intent, not the tree). This
row's gates prove them where they matter: the served-replay test's
mechanical CHECK (replays > captures) and the served legs' `boundary=0`
replay steps.
