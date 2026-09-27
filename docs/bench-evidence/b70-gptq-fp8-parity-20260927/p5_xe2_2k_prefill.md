# P5: native Xe2 FP8 attention at 2048 and 2049 queries

The qualified batch-1 Xe2 FP8 E4M3 prefill route now admits initial prompts
of exactly 2048 and 2049 query tokens, in addition to the previously
qualified lengths. The existing device, compiler, driver, shape, stride,
scale, causal and page gates remain in force. Other lengths still use Q64.
`VT_XPU_XE2_PREFILL=0` selects the Q64 baseline on the same binary.

The focused operator test compares Xe2 with Q64 at 2048/2049 and the older
4095/4096/4097/8192 lengths, using reversed physical page IDs, twice the
logical physical page stride and page sizes 64 and 1600. All 84 assertions
passed, including output and route checks. Relative RMS across the new 2K
cases was 1.25e-5 to 1.47e-5; maximum absolute difference was 0.0004883.

For the full Qwen3.8-27B GPTQ model with FP8 KV page 1600, a P2048/O1
same-binary A/B used one warmup and two timed rounds per route. The GPU peak
allocated bytes were 18,772,193,764 in both runs.

| Attention route | Prefill seconds | Prompt tokens/s | Median tokens/s |
| --- | --- | --- | ---: |
| Q64 | 2.218, 2.258 | 923.16, 907.11 | 915.13 |
| Xe2 | 1.659, 1.743 | 1234.17, 1174.94 | 1204.55 |

The median throughput gain is **31.6%** for this P2048 case. The nonoverlapping
rounds show a large gain, though this is a short two-round benchmark. The
quality-run route trace selected Xe2 attention 32 times (16 layers across
warmup plus one measured request). At 2048 tokens GDN selected its existing
chunked path 96 times; this experiment changes only attention selection.

Pinned Python and C++ P2048/D64/O65 used identical prompt and teacher-forced
decode hashes and effective KV page 1600. Prefill, decode 1 and decode 64
passed all frozen full-vocabulary TV, KL, top-10 and top-1 gates. TV was
0.000053, 0.010407 and 0.003235. See
`p5_xe2_2k_quality_2048_64.json`. The thresholds were originally frozen on
P4096 and applied unchanged to P2048. Python was an external oracle; the
engine runs C++ and native SYCL kernels only.

Small raw model and quality logs are in `raw/p5_xe2_2k_*.log.gz`; F32 logits
remain outside Git. The 2049 case has operator correctness and route evidence,
but no separate full-model Python replay. Further P5 work includes other
ragged lengths, continuation shapes, multiple sequences and matched
device-core Python timing.
