# P4 native GDN beyond 6656 tokens

The opt-in native Xe2 route now accepts one full P4096, P8192 or P16384
sequence. It divides longer calls into 4K macro segments, reuses the same
bounded A/W/U and Q/gate scratch, and carries the FP32 SSM state on VT's
in-order SYCL queue. The device query offsets for each macro are prepared
inside the native workspace. The public query offsets are checked against the
full logical length before any mutation. Unsupported shapes retain the old
path. The model benchmark now permits prompts through 16K subject to its
existing context and measured memory checks.

The isolated test duplicates one real projected P4096 activation set to form
P8192 and P16384. It compares one native long call with two or four existing
P4096 C++ calls that carry the same nonzero state. For P8192, native versus
old C++ relative RMS is 0.001468 for output and 0.002406 for final state;
for P16384 it is 0.001467 and 0.002463. The native workspace stays under
160 MiB. Five timed repeats after one warmup yield:

| Isolated GDN layer | Existing C++ macro calls | Native long call | Ratio |
| --- | ---: | ---: | ---: |
| P8192 | 22.82 ms | 3.47 ms | 6.59× |
| P16384 | 46.64 ms | 7.10 ms | 6.56× |

One complete C++ model P8192/O1 run without logit capture measured 65.49 s
with the old fallback and 14.75 s with native GDN, respectively 125 and
555 prompt tokens/s. This is **one measured round per route**, preceded by
one full warmup each, so it is directional full-model evidence. The route
trace from a separate diagnostic P8192/O1 run selected native GDN 96 times
(48 layers in warmup plus 48 in the measured request).

The pinned Python reference and C++ used identical P8192 prompt IDs, FP8
E4M3 KV with effective page 1600, and teacher-forced decode IDs in the
P8192/D64/O65 quality run. Python processed the prompt internally in 4K
model calls; C++ used one 8K model call with native GDN macro segments.
`p4_native_8192_quality_64.json` verifies the capture contexts and passes
the previously frozen TV, KL, top-10 and top-1 gates at prefill, decode 1,
and decode 64. Raw F32 logits remain outside Git at
`/home/sebastian/LocalLLM/b70-gdn-real-layer-20260927/`; the small compressed
run logs are under `raw/p4_native_long_*.log.gz`.

The Python P8192/O1 end-to-end request took 5.52 s in its separate quality
session. Its engine-step boundary differs from the C++ worker-forward timer;
these two durations are not a parity throughput comparison. The remaining
8K latency is a target for P5 attention and a matched-scope profile. P4
remains open for 16K full-model qualification, ragged/multiple sequences,
and wider default-route decisions.
