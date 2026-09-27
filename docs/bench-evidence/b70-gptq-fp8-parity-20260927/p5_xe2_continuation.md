# P5: native Xe2 attention for a 4K continuation over a 4K prefix

The native C++/SYCL Xe2 attention adapter now admits one further qualified
shape: 4096 query tokens with `max_seq_len=8192`, causal attention and an
existing 4096-token FP8 E4M3 KV prefix. It retains all B70/device/compiler,
scale, head, dtype, page and layout gates. The host block-table bound now
covers the full KV length. `VT_XPU_XE2_CONTINUATION=0` selects Q64 only for
this continuation, leaving the initial 4K Xe2 route active for A/B runs.
Other continuation lengths still use Q64.

The focused GPU test uses reversed physical block IDs and a physical page
stride twice the logical size. It passed 15 assertions across page sizes 64
and 1600, including the Xe2 route, Q64 opt-out, and output comparison. Xe2
versus Q64 relative RMS was 3.42e-5/3.51e-5; maximum absolute difference
was 3.05e-5. Five alternating timed operator calls after warmup gave:

| 4096 queries, 8192 KV tokens | Q64 median | Xe2 median | Ratio |
| --- | ---: | ---: | ---: |
| Page 64 | 443.62 ms | 13.39 ms | 33.1× |
| Page 1600 | 443.75 ms | 13.41 ms | 33.1× |

The benchmark harness has an optional `VT_B70_BENCH_PREFILL_CHUNK=4096`
mode. It makes two model calls for P8192, carries the same KV and GDN state,
marks the second GDN prefill as having an initial state, and reports each
model call's actual query/context length. The default remains one model call
and its existing timing scope. This optional mode matches the pinned Python
worker's two 4096-query prefill calls more closely.

With native GDN, FP8 KV page 1600 and one warmup per fresh process, two timed
P8192/O1 rounds per route measured:

| Second model call | Total prefill seconds | Total prompt tokens/s | Second call seconds | Peak allocated bytes |
| --- | --- | --- | --- | ---: |
| Q64 | 12.504, 12.503 | 655.15, 655.17 | 9.622, 9.622 | 19,601,128,000 |
| Xe2 default | 6.141, 6.148 | 1334.08, 1332.58 | 3.165, 3.158 | 19,601,128,000 |

The full two-chunk prefill speedup is **2.04×**. The remaining second-call
time includes all model layers and glue, not just attention. A route trace
selected Xe2 attention 64 times and native GDN 192 times across the warmup
and measured P8192/D64 run: two 4096-query calls, 16 attention and 48 GDN
layers each, repeated twice.

The P8192/D64/O65 two-chunk C++ replay used the same prompt and teacher-forced
decode hashes as pinned Python. It passed every frozen TV, KL, top-10 and
top-1 gate at prefill, decode 1 and decode 64; see
`p5_xe2_continuation_quality_8192_64.json`. Its C++ logits at all three
checkpoints are byte-identical to the earlier single-call P8192 C++ replay.
The original 4095/4096/4097/8192 prompt-route operator test also remains
green (56 assertions). The C++ runtime invokes native kernels only; Python
is an external oracle.

As a separate rejected P5 experiment, changing Split-K GQA reuse from two
query heads per K/V load to three or all six passed the focused numerical
test (80 assertions) but reduced full-model P4096/D64 decode from 23.89 to
22.97 and 22.07 forwards/s, respectively. The change was reverted. Register
and occupancy counters were not captured, so the exact cause of the regression
remains unproven. Small raw logs are under `raw/p5_gqa_reuse*_rejected.log.gz`.

Remaining P5 work includes other continuation lengths, ragged/multiple
sequences, a better occupancy-aware Split-K plan, the 20-query-token boundary
and a matched device-core latency comparison against Python.
