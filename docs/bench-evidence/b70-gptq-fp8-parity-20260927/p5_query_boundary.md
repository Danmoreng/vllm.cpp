# P5: FP8 attention at the 20-query boundary

The B70 FP8 E4M3 Split-K route previously stopped at 20 query tokens, while
the native Xe2 Q32 prefill route starts at 32. Queries 21–31 therefore used
the slow generic per-head reference kernel. The native C++/SYCL Split-K route
now covers 21–31 for the qualified Qwen3.8 shape only: 24 query heads, four
KV heads, D256, page 1600, causal, unit scales, no window or soft cap.
`VT_XPU_ATTN_SPLIT_EXTENDED=0` restores the old selector for A/B measurement.
Other shapes retain the old boundary.

The focused GPU test compared the selected route with the native reference at
Q20/21/31/32/33, checked the route and the opt-out, and passed 36 assertions.
Five timed operator calls after warmup gave:

| Query tokens / KV length | Old route / median | New route / median | Speedup |
| --- | ---: | ---: | ---: |
| 20 / 4116 | Split-K / 6.455 ms | Split-K / 6.483 ms | unchanged |
| 21 / 4117 | reference / 16.080 ms | Split-K / 3.773 ms | 4.26× |
| 31 / 4127 | reference / 23.884 ms | Split-K / 5.966 ms | 4.00× |
| 32 / 4128 | Q32 / 11.737 ms | Q32 / 11.737 ms | unchanged |
| 33 / 4129 | Q32 / 11.844 ms | Q32 / 11.845 ms | unchanged |

For P4117/O1, the optional benchmark chunk size 4096 made model calls of
4096 queries over an empty prefix and 21 queries over the 4096-token prefix.
With native GDN and page 1600, two timed rounds per route on the same binary
measured:

| Route at Q21 | Total prefill seconds | Prompt tokens/s | Q21 model call | Peak allocated bytes |
| --- | --- | --- | --- | ---: |
| Reference | 3.385, 3.395 | 1216.25, 1212.75 | 0.529, 0.531 s | 19,449,863,100 |
| Split-K | 3.175, 3.183 | 1296.55, 1293.35 | 0.310, 0.309 s | 19,466,640,316 |

Median throughput improved **6.6%** in this boundary case. The Split-K route
reserves another 16 MiB attention workspace. This is a C++ same-binary A/B,
not a Python versus C++ throughput result. Route tracing in the quality run
showed Q4096 on Xe2 and Q21 on Split-K; GDN at Q21 still uses its reference
path.

Pinned Python and C++ P4117/D64/O65 replays used identical prompt and
teacher-forced decode token hashes. The frozen full-vocabulary logit gates
passed at prefill, decode 1, and decode 64. Total variation was 0.00305,
0.01574, and 0.00321, respectively; KL, top-10, and top-1 gates also passed.
See `p5_query_boundary_quality_4117_64.json`. The threshold file was frozen
on P4096 cases and applied without modification to this P4117 extension.
Python was used only as an external oracle; the runtime route is C++/SYCL.

Raw operator/model logs and the two compact quality run logs are in
`raw/p5_query_boundary_*.log.gz`. Full F32 logits remain outside Git.
Remaining P5 work includes other ragged lengths and batches, Split-K occupancy
across contexts, and matched device-core timing against Python.
