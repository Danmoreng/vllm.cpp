# W1: batch-eight case-5 first-divergence isolation (2026-09-28)

This is a diagnostic, not a scored performance run or graph promotion.
Production service `qwen38-vllm-production` was inactive. The real GPTQ-G128
model ran on the B70 with FP8 E4M3 KV, page 1600, native GDN and greedy
sampling. Full F32 logit dumps remain outside Git under
`/home/sebastian/LocalLLM/b70-gdn-real-layer-20260927/p7_graph/`.

## Test change

`test_xpu_qwen_checkpoint.cpp` now forces the eight-request expected-output
stage to eager when testing the GPTQ graph candidate. The mixed candidate
keeps its configured graph setting. This stops an anomalous serial graph
result from becoming the test's own expected output. Set
`VT_B70_BATCH_SERIAL_GRAPH_BASELINE=1` to reproduce the old graph-serial
history; the older `VT_B70_BATCH_GRAPH_AFTER_BASELINE=1` switch still works.
`VT_B70_BATCH_CASE5_ONLY=1` runs just the suspect prompt in a fresh engine and
prints its route, answer and graph capture/replay counts.

The focused target compiled in the pinned oneAPI builder with SYCL-TLA
`87f6850680a580654b9ea2c80dbc01aeb36ad231`, oneDNN 3.13 and no runtime
source change. The first attempted build used an incomplete archived TLA copy
and failed on `cutlass/util/sycl_event_manager.hpp`; the actual checkout fixed
the build input.

## Results

| Diagnostic | Result |
| --- | --- |
| C8 graph candidate, forced eager serial oracle | 80/80 assertions passed; case 5 matched `city 1`; graph capture/replay 1/1. |
| C1 case 5, fresh eager process | 27/27; `city 1`; 0 captures/replays. |
| C1 case 5, fresh graph process | 29/29; same 17 output IDs and `city 1`; 2 captures, 14 replays. |
| Historical serial-graph C8 history, three fresh processes | First two passed 78/78. Third failed 77/78 at case 5 with the **opposite direction** to the earlier retained failure: serial graph expected `city 1`, concurrent eager produced `city 5`. |

The third run's token-ID sidecars agree for case 5 up to row 1; at **row 2**
serial graph selects token 16 and concurrent eager selects token 20. The
callback's `first_chunk=2` also points there in this simple greedy test, but
the token IDs, not chunk numbering, establish the position. The concurrent
route at eight requests was `eager`, reason `batch_exceeds_graph_limit`.

At the first divergent logit row:

| | Token 16 | Token 20 | Selected |
| --- | ---: | ---: | ---: |
| Serial graph request `ours_5` | 17.375 | 17.375 | 16 (tie rule) |
| Concurrent eager request `ours_10` | 17.375 | 17.390625 | 20 |

Full-row max absolute logit difference was `0.072265625`, total variation
`0.00333436`, and KL(serial || concurrent) `0.00002920`. These are small
distribution differences with a zero or 0.015625 top-1 margin. In the fresh
C1 eager/graph pair, row 2 was an exact tie on both routes, and their output
IDs were identical; across its 17 rows, max TV was `0.00819896` and max KL
was `0.00020349`. Those numbers describe C++ route/batch variation, not a
Python-oracle quality result.

The earlier retained failure had the reverse string mismatch: the serial
graph expectation was `city 5` and concurrent eager was `city 1`. A fixed
stale-row explanation does not follow from these two directions. A near-tie
greedy flip is strongly supported at the *first* divergent row, but inputs,
metadata and state ownership have not yet been instrumented deeply enough to
exclude another cause of the bounded logit change. Keep the strict string
assertion, graph opt-in and the graph batch cap of four. Do not mark C8 or P7
fully qualified from this result.

The failed run's serial/mixed F32 files have SHA-256 respectively
`3dcf81edcb9989d5180b292085ec6e9bfd82a8406194a8f6a5b39e207a3828f7`
and `a29ebc77f1d38592c642a9d1100258e095dc2afda4da1ecfcff064e3b833a063`.
Compact raw logs are in `raw/w1_*.log.gz`; 17-row F32 arrays and the passing
full-batch dumps are excluded from Git.

## Next W1 gate

Replay serial histories 0–4 then case 5 under eager/eager, graph/graph and
graph/eager, recording request/row/slot mappings and the row-2 GDN/KV state
before concluding that the bounded difference is merely kernel arithmetic.
Use a stable external eager token/logit oracle. If no ownership violation
appears and only a near tie flips under valid route-dependent arithmetic,
document that narrow explanation and retain a numerical gate alongside the
strict-output diagnostic. W3 eager MTP1 work need not wait for graph promotion.
