# P5: native Xe2 attention for 2K queries over a 4K prefix

The native C++/SYCL Xe2 FP8 E4M3 adapter now admits a second qualified
continuation shape: 2048 query tokens with 6144 KV tokens, including an
existing 4096-token prefix. It retains the device, compiler, driver, page,
layout, scale and causal gates of the earlier 4K-over-4K continuation.
`VT_XPU_XE2_CONTINUATION=0` restores Q64 only for continuation A/B runs.

The focused GPU test uses reversed physical page IDs and a physical page
stride twice the logical size. It compared Xe2 with Q64 at Q2048/K6144 and
the existing Q4096/K8192, for page sizes 64 and 1600. All 30 assertions
passed, including output comparison and the selected route. Five timed
alternating operator calls after warmup gave:

| Q2048/K6144 | Q64 median | Xe2 median | Ratio |
| --- | ---: | ---: | ---: |
| Page 64 | 186.093 ms | 5.833 ms | 31.9× |
| Page 1600 | 186.086 ms | 5.818 ms | 32.0× |

With native GDN enabled, FP8 KV page 1600 and optional benchmark chunk size
4096, P6144/O1 uses 4096-query and 2048-query model calls. One warmup and
two timed rounds per route on the same binary measured:

| Second-call attention | Total prefill seconds | Prompt tokens/s | Second call seconds | Peak allocated bytes |
| --- | --- | --- | --- | ---: |
| Q64 | 7.318, 7.336 | 839.63, 837.46 | 4.469, 4.474 | 19,610,059,448 |
| Xe2 | 4.772, 4.770 | 1287.38, 1288.03 | 1.825, 1.831 | 19,610,059,448 |

Median prompt throughput improved **53.6%**. This is a same-binary C++ A/B;
the two-round sample is short. The remaining second-call time includes GDN,
matmuls and model glue. The quality-run route trace selected Xe2 attention
at Q4096 and Q2048 (32 selections each across warmup and measured requests).
GDN used the native route for Q4096 and the existing chunked route for Q2048.

The pinned Python runner's final prefill invocation also had 2048 queries
after 4096 context tokens. Python and C++ P6144/D64/O65 replay used the
same prompt and teacher-forced decode hashes and effective KV page 1600.
Full-vocabulary logits passed the frozen TV, KL, top-10 and top-1 gates at
prefill, decode 1 and decode 64; TV was 0.000026, 0.013924 and 0.002572.
See `p5_xe2_2k_continuation_quality_6144_64.json`. Python was only the
external quality oracle; the runtime is native C++/SYCL.

Small raw operator/model/quality logs are under
`raw/p5_xe2_2k_continuation_*.log.gz`. Full F32 logits remain outside Git.
Other continuation lengths, ragged/multiple sequences, Split-K occupancy and
matched device-core Python timing remain open P5 work.
