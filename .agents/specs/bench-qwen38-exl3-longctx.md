# `BENCH-QWEN38-EXL3-LONGCTX` — the head-to-head at the context length the server actually serves

Row: `BENCH-QWEN38-EXL3-LONGCTX`.
Issue: `ISSUE-LOCAL-01M2TNH3A9DKGWCRADM4Q8J30C`.
Base SHA: `7fa861392`.
Predecessors: [`bench-qwen38-exl3-variadic.md`](bench-qwen38-exl3-variadic.md) (the
harness and the mixed load), [`quant-exl3-recon-scratch.md`](quant-exl3-recon-scratch.md)
(W7, which this run must carry to reach c = 32).
Comparator: `MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`
([oracle](../oracles/exllamav3.md)). vLLM implements no EXL3.

## Now

`ACTIVE`. Spec committed before the harness change.

2026-09-18: the harness part landed on `row/BENCH-QWEN38-EXL3-LONGCTX`.
`build_corpus.py` carries the `XXL` band at `XXL_TARGET_CHARS = 21000`, and
`--weights` replaces the four `--weight-*` flags.

The band is sized to FIT, not to hit the 7000 tokens §2 asks for. At the 3.4
characters per prompt token this corpus realised on the published `XL` band,
21000 characters is about 6200 prompt tokens. The band's own overshoot at
`k_mid = 47` is about 12.5% (4.3% k jitter plus 8.3% sampling spread, scaled
from XL's measured 22.7% at `k_mid = 20`), so the expected ceiling is about
6900 tokens against the 8192 - 192 - 50 = 7950 the served configuration leaves,
which is about 1000 tokens of headroom. The first draft targeted 7000 tokens at
23800 characters and left about 150. That margin was rejected: a band that
overruns the context is voided by `G-FITS` and costs the dgx lease it was
measured on, and 6200 tokens is still about 2.3 times the `XL` median and well
past the 3.3k every published band stops at. `G-FITS` still decides.

`XXL` carries no default weight, so the four-band default is byte-identical to
the predecessor: the pre-change and post-change generators produced the same
corpus sha256 on five `(count, seed)` pairs over the same fixture sources, and
`tests/scripts/test_variadic_harness.py` pins that sha256 as a golden. The run
itself, `job.sh`, and the publication are not in this change.

## 1. The question

Does our prefill lead over exllamav3 hold at the context this server actually
serves? Every published band stops at about 3.3k prompt tokens, which is 40% of
the `--max-model-len 8192` both engines run.

The prediction is that the lead widens, because #3150's reconstruct + cuBLAS path
amortizes its one-time dequantization over more rows. The counter-pressure is on
our side too: draft acceptance falls with context (0.49 at 324 tokens, 0.37 at
8159), so decode throughput and TTFT move in opposite directions with length.
Both effects are ours; neither has been measured against their engine.

## 2. Scope

- One new corpus band, `XXL`, targeting about 7000 prompt tokens, so that a
  prompt plus its 192 output tokens fits inside `--max-model-len 8192` with the
  chat template's own overhead.
- The band is built from the same HumanEval source as `XL`, by the same
  concatenation rule, so length is the only variable that changes.
- The band weights become a command-line knob, because a run that wants long
  prompts must be able to ask for them without editing the generator.
- The publication is a NEW benchmark id, `qwen38-27b-exl3-longctx-gb10`. The
  existing `qwen38-27b-exl3-variadic-gb10` page keeps its numbers and its
  disposition; it measured a different corpus on a tree that predates #3150, and
  nothing here supersedes it.

Out of scope: a matched KV configuration (theirs is `-cs 262144`, ours auto-fits
8192; that is `#2620`), and any context above 8192, which needs a different
served configuration.

## 3. Design

- `build_corpus.py` gains `XXL_TARGET_CHARS` and a `--weights` argument. The
  default weights keep the existing four-band distribution byte-for-byte, so an
  unchanged invocation still produces the predecessor's corpus.
- The run uses weights that put real mass in `XXL` while keeping the shorter
  bands for continuity.
- Both engines, c = 1 and c = 16, two rounds, arms and rung order reversed in
  round 2, exactly as the predecessor harness does.
- c = 32 only if W7 has landed and its G-MEM gate passed; otherwise the rung is
  refused by name rather than run, because the host OOM is what W7 fixes.

## 4. Gates

- `G-BYTES`: the three checkpoint shard sha256 values recomputed on the device
  match the pins.
- `G-FITS`: every realised `usage.prompt_tokens` plus `max_tokens` is below the
  served context on both engines, read back from each server's own usage. A
  truncated or refused prompt voids the band.
- `G-RESOLVED`: our server reports the configured `max_num_seqs` against the KV
  pool, so no leg runs below its rung.
- `G-SPREAD`: two rounds per cell, and the round-to-round spread is published
  beside every value.

## 5. Risks

- **The band does not fit.** 7000 tokens is a target in characters. G-FITS reads
  the realised counts back and voids the band rather than publishing a truncation.
- **Their engine refuses the length.** Their card runs `-cs 262144`, so it should
  not, but a refusal is recorded as their result, not worked around.
- **Acceptance collapse confounds decode.** Report acceptance per band from both
  engines' own counters, so a TPOT change is attributable.
- **dgx:gpu0 loses its rc stream under load.** Every leg is written to the share
  as it completes, and the job is resumable, as the predecessor's is.

## 6. Evidence

`docs/bench-evidence/qwen38-27b-exl3-longctx-<date>/`: the job as run, the corpus
manifest with its sha256, the per-leg JSON, the realised token histogram, both
engines' acceptance, and the lease ids and boot ids of every leg.

## 7. Stop conditions

- G-FITS fails: publish the refusal, not the numbers.
- W7 has not landed with G-MEM green: run c = 1 and c = 16 only, and record c = 32
  as refused with the issue that owns it.
