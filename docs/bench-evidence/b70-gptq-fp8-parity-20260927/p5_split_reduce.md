# P5: cooperative native Split-K decode reduction

The B70 E4M3 Split-K decode path now has a native SYCL reduction in which one
64-lane workgroup shares the split maxima and normalization weights across 64
output components. Four workgroups cover each 256-component query head.
The previous kernel recomputed the same maxima and denominator separately for
every component. The new path is the default only for B70 FP8 E4M3 with 24
query heads and dimension 256. Other shapes and formats retain the old scalar
reduction. `VT_XPU_ATTN_SPLIT_REDUCE=scalar` selects that baseline for A/B;
`cooperative` and `auto` select the new path on the qualified shape.

The focused GPU test `*split-KV: 32k*` passed 80 assertions with profiling
enabled, including the selected cooperative stage. It compares against the
native reference for E4M3 and float KV, M1–M5, batch 1 and 4, unequal
requests, windowed softcap, and empty split parts. Its relative RMS stayed
below 4.5e-6. The runtime contains no Python, PyTorch or Triton dependency.

Full-model A/B used the same binary, Qwen3.8-27B GPTQ G128, native GDN,
FP8 E4M3 KV page 1600, eager batch 1, one warmup then two measured
P4096/D64/O65 rounds per fresh process. Final scalar versus default runs:

| 4K route | Decode forwards/s | Prefill tokens/s | Peak allocated bytes |
| --- | --- | --- | ---: |
| Scalar | 23.602, 23.580 | 1392.97, 1397.12 | 19,460,738,408 |
| Cooperative default | 23.909, 23.873 | 1394.81, 1394.66 | 19,460,738,408 |

Median decode gain: **1.27%**. An earlier fresh-process A/B on this candidate
gave 23.612–23.624 versus 23.897–23.987 decode forwards/s, so the gain
exceeded the observed within-route variation. A P8192/D64/O65 A/B with the
same cooperative kernel opt-in measured 20.705/20.718 scalar versus
21.161/21.181 cooperative decode forwards/s: **2.22%** median gain, with
the same 20,661,169,524 peak bytes and unchanged prefill within noise.
The later default-route 8K quality replay produced byte-identical logits
to the opt-in replay. Raw run logs are under `raw/p5_split_reduce_*.log.gz`.

Pinned Python full-vocabulary quality passes every frozen TV, KL, top-10 and
top-1 gate at P4096 prefill and decode 1, 64, 256, 703, 704, 705, 1024;
see `p5_split_reduce_quality_4096_1024.json`. P8192 prefill and decode 1/64
also pass; see `p5_split_reduce_quality_8192_64.json`. The 8K decode-1 TV is
0.01913 against the frozen 0.02 limit, so future numerical changes need
special scrutiny there. Python is used only as an external quality oracle.

Before this change, a compact Split-K plan based on the host-known active
length was tested separately. With 4096 active tokens and 262400 reserved KV
slots, it reduced 256 parts to 128 and passed the focused numerical check,
but the full P4096/D64/O65 decode fell from 23.624/23.627 to 23.433/23.429
forwards/s. That candidate was reverted; its small raw logs remain under
`raw/p5_split_parts_*.log.gz`. Lower split counts need a different occupancy
or reduction plan before adoption.

Remaining P5 work includes further prefill lengths and continuation, mixed
requests and GQA reuse, split span/part configuration across context lengths,
and a matched-scope comparison with the pinned Python native attention route.
