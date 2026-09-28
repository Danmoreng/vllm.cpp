# W4/W5: same-prefix sampled MTP4 verifier quality probe (2026-09-28)

The first production-sampler comparison with an **identical speculative
prefix** is now available. Both engines used the pinned local 27B GPTQ-G128
checkpoint, FP8 E4M3 KV, 1664-token effective attention page, eager C1,
separate INT4 draft head/five linears, MTP4, temperature 1, top-p .95,
top-k 20, seed 42 and no prefix cache. The 4096-token retrieval prompt had
token-ID FNV64 `4760835697920107937` in both engines. Python ran vLLM
`0.30.0+xpu` at source commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
with XPU kernels `0.1.15.4`; C++ ran the current native XPU build.

Python's `rejection_sampler._verify` was instrumented only for the first
five-row call with position at least 4095. The earlier Python startup warmup
calls at positions 6–10 were excluded. The active Python worker imports the
installed file under `site-packages`, even though an editable source copy also
exists under `/workspace/vllm`; the temporary probe was mounted at both
paths. The C++ runner was temporarily instrumented immediately before the
native one-hot rejection kernel. Both probes were removed from the live code
after capture. Raw arrays (about 20 MB total) remain outside Git under
`/tmp/b70_spec_probe_{cpp,python}`.

At the first real verification cycle Python saw positions 4096–4100 and
`draft_sampled=[271,90979,78,248046,198]`. C++ emitted the same first
token 271 and proposed the same four drafts in the same order. Python's
`draft_logits` was `None`, confirming the one-hot greedy-draft rejection
route. Therefore all five target rows score the same token prefixes.

| Row | Position | Raw TV | Raw KL(Python‖C++) | Raw top-10 overlap | Post-filter TV | Post-filter KL | Finite post-filter token IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4096 | 0.02445 | 0.002165 | 9 | 0.02475 | 0.002167 | 90979, 248068 |
| 1 | 4097 | 0.000002 | <0.000001 | 10 | 0 | 0 | 78 |
| 2 | 4098 | 0.000590 | 0.000036 | 10 | 0 | 0 | 248044 |
| 3 | 4099 | <0.000001 | <0.000001 | 10 | 0 | 0 | 198 |
| 4 | 4100 | 0.02329 | 0.001091 | 9 | 0.02329 | 0.001090 | 248044, 248045 |

All five raw top-1 IDs match. The finite post-filter support is identical
per row, and C++'s saved probability rows match a fresh softmax of its
post-filter logits within `1.3e-8`. Rows 0 and 4 exceed the frozen
`TV <= 0.02` target; row 0 also exceeds `KL(Python || C++) <= 0.002` in
`quality_thresholds.json`. These are **real failures for this newly selected
sampled-verification checkpoint**, not an average across unrelated text or
a claim that the previously frozen teacher-forced suite failed.

To test whether the recent W5 GDN workgroup change caused this difference,
the same C++ request was rerun with `VT_XPU_GDN_SPEC_WG=0`. The old and new
launches produced byte-identical five-row raw logits, post-filter logits
and probabilities; both emitted the same 32 token IDs. The Python-relative
gap therefore predates that performance change. The scoped workgroup gain
remains qualified as an internal C++ A/B, while Python quality parity at
these two rows remains open. The next correctness investigation should trace
the first divergent target layer/state or prefill activation on this fixed
prefix before tuning acceptance or changing the frozen threshold.

The dump SHA-256 values for reproducibility are:

| Array | SHA-256 |
| --- | --- |
| Python raw F32, 5×248320 | `f02d0ae6bf99ecd4057028b5dca64146fc67f3ef3adf518bdd3c0d9013fce13f` |
| Python processed F32 | `b7736ec3ca227976bce8d4efb9cabfbf4faad8ddf1da85b15ade9774c294eb36` |
| C++ raw F32 | `a706dd6021fd27a622877328854b33fd172279b28bdf2ea5de74e9241023951f` |
| C++ processed F32 | `681f6c31e0c2cf8a8e52a56e2aa2454821ace3a5515e6ce8878b0f6aeabafda1` |
