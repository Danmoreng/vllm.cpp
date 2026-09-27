# P7: native decode graph at P4096, FP8 KV

The existing Qwen3.5 dense decode graph and XPU SYCL command-graph backend
were evaluated as a **separate configuration** after the eager path. Both
`VT_XPU_GRAPH=1` and `VT_GPTQ4_GRAPH=1` select the opt-in graph route;
without them the pinned GPTQ model stays eager. The route trace selected
eager for the 4096-query prefill and graph for single-token decode. The
C++ engine uses native SYCL and the pinned oneDNN GPU primitives throughout;
Python remains an external quality oracle only. No graph runtime code was
changed in this experiment.

Unprofiled, same-executable P4096/D64/O65 runs used batch 1, the real
GPTQ-G128 27B model, native GDN, FP8 E4M3 KV with page size 1600, and one
warmup request per process. There were five scored requests for each route
across four independent process starts, including three paired sessions
with Eager/Graph, Graph/Eager, Eager/Graph order:

| Route | Decode forwards/s, five scored runs | Median | Prefill median tokens/s |
| --- | --- | ---: | ---: |
| Eager | 25.5714, 25.5696, 25.5336, 25.5301, 25.5156 | 25.5336 | 1431.47 |
| Graph | 26.7576, 26.7442, 26.6512, 26.6312, 26.5976 | 26.6512 | 1429.91 |

The **decode median gain is 4.38%**. The decode ranges do not overlap;
the small prefill median difference is within this run's variation. All
64 measured decode forwards used graph replay (`graph_replays=64`,
`graph_captures=0` after warmup). Graph validation remained enabled. Peak
allocated GPU memory was 19,464,348,700 bytes with graph versus
19,460,738,408 bytes eager in the first A/B process; the graph backend
reported 8,192 graph-device bytes. This is a configuration gain, not an
identical-eager-kernel or Python-parity claim.

A separate P4096/D1024/O1025 graph run captured full-vocabulary F32 logits
at prefill and decode steps 1, 64, and 1024. All four files were bytewise
identical to the previous eager C++ capture. The frozen Python TV, KL,
top-10, and top-1 thresholds passed at all checkpoints; see
`p7_graph_quality_4096_1024.json`. The long run made 1,020 graph replays
of 1,024 decode forwards. Four fresh captures occurred in the measured
request as the retained graph changed from the warmup's four-column block
table back to three columns and later crossed to four columns again. The
first four steps and steps 705–708 showed the corresponding cold/warm/
capture cost. This proves the observed P4096 reset and page transition;
new requests, abort/reuse, unequal concurrent sequences, and other page
geometries remain unqualified.

An additional profiled P4096/D16 diagnostic recorded 16 graph compute and
16 graph-validation events. Validation GPU event duration summed to
0.325 ms; its host D2H wait summed to 0.714 ms. These are profiled sums,
not scored latency or removable exclusive savings. The graph-compute
event sum was 647.65 ms under profiling and must not be compared with
unprofiled per-forward time.

The compact raw A/B, independent-session, long-quality, and diagnostic
profile logs are under `raw/p7_*.log.gz`. F32 logits stay outside Git. The
native graph remains opt-in until concurrency 2/4/8, request lifecycle,
memory growth, and matched-scope Python comparisons are qualified.
