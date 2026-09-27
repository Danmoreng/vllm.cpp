# P5: qualified native Xe2 FP8 prefill length range

The native C++/SYCL Xe2 FP8 E4M3 adapter now selects initial batch-1
prefill for 2048–8192 query tokens and continuation prefill for 2048–4096
queries behind an existing prefix of 1–4096 tokens, with KV length at most
8192. This replaces exact-length lists. The existing B70 compiler/driver,
Q24/KV4/D256, FP16 query/output, causal, unit-scale, page 64/1600, stride,
alignment and block-table gates remain. Other shapes use the previous path.
`VT_XPU_XE2_PREFILL=0` disables Xe2; `VT_XPU_XE2_CONTINUATION=0` disables
only its continuation selection for same-binary A/B runs.

The focused initial-prompt GPU test sampled 14 lengths spanning 2K, 3K,
4K, 6K and 8K, including one-token tile tails; at both page 64 and page
1600 it used reversed physical block IDs and a physical stride twice the
logical page size. All 196 assertions passed against Q64 and route checks.
The focused continuation test sampled ten query/prefix pairs, including
63-, 1600-, 2048- and 4096-token prefixes and ragged query tails, with both
page sizes. All 150 assertions passed. At Q3072/K7168/page1600, five
alternating operator calls measured Q64 median 305.694 ms versus Xe2
9.363 ms (32.7×). These samples cover the transition boundaries; arbitrary
values within the range were not individually replayed.

Same-binary eager C++ full-model A/B used one warmup and two timed rounds per
route, FP8 KV page 1600 and the pinned GPTQ model:

| Case | Q64 prefill tok/s | Xe2 prefill tok/s | Median gain | Peak GPU bytes |
| --- | --- | --- | ---: | ---: |
| P3072/O1, one model call | 834.44, 828.20 | 1238.65, 1197.36 | 46.5% | 19,026,167,140 |
| P7168/O1, 4096+3072 calls | 715.33, 714.02 | 1254.43, 1255.94 | 75.6% | 19,588,545,084 |

Peak allocated GPU bytes were identical between routes for each case. At
P7168, the second 3072-query model call fell from 7.140/7.146 seconds to
2.741/2.750 seconds. The first 4096-query call already used Xe2 in both
routes. Quality-run traces selected Xe2 at Q3072 (32 layer selections
across warmup and measured requests); P7168 additionally selected Xe2 at
Q4096. GDN remained chunked at Q3072 and native at Q4096.

Pinned Python and C++ used identical prompt and teacher-forced decode IDs,
effective KV page 1600 and full-vocabulary logits. Python's P7168 final
prefill call was also Q3072 after a 4096-token prefix. Both P3072/D64/O65
and P7168/D64/O65 passed the frozen TV, KL, top-10 and top-1 gates at
prefill, decode 1 and decode 64; see the two quality JSON files. P3072 TV
was 0.000850/0.016227/0.003897 and P7168 TV was
0.000059/0.018122/0.003530. P7168 decode 1 is close to the frozen TV
limit of 0.02. These thresholds were set on P4096 and were not relaxed.
Python remained an external quality oracle; the engine runs C++ and native
SYCL kernels only.

Small raw operator, A/B and quality logs are under `raw/p5_xe2_range_*.log.gz`.
Full F32 logits remain outside Git. This range does not qualify sequences
shorter than 2048, KV lengths above 8192, multiple requests, nonunit scales,
windowed attention or other layouts. Decode occupancy and matched Python
device-core timing remain P5 work.
