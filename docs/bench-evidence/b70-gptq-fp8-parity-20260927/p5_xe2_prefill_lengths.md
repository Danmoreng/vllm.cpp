# P5: native Xe2 FP8 attention prefill beyond 4096 tokens

The C++ Xe2/SYCL-TLA adapter now admits exactly P4095, P4096, P4097 and
P8192 for the already qualified batch-1, causal, FP16-Q/output, unit-scale
E4M3 K/V contract. It keeps the existing checks for contiguous query/output,
1600- or 64-token pages, head-contiguous K/V, full-prompt metadata,
compiler/driver/device, and a contiguous block table. Other cases retain the
Q64 path. The runtime uses C++ and native SYCL kernels only; Python is an
external quality reference.

The focused operator test compares Xe2 with the existing Q64 GPU path for
each of those four lengths and both page sizes. It uses reversed physical page
IDs, a physical page stride twice the logical page size, and a causal tail.
All 56 assertions passed, including exactly one Xe2 event and no Q64 event
for each selected run. Relative RMS was 1.48e-5 to 1.80e-5; maximum absolute
difference was at most 0.0004883. Command: `VT_XPU_PROFILE=1
test_xpu_attention_fast --test-case="XPU Xe2 FP8 prefill: 4K boundaries and 8K with paged KV"`.

For the full Qwen3.8-27B GPTQ G128 model, FP8 E4M3 KV page 1600, one
P8192/O1 request, native GDN enabled, eager execution, the same binary was
measured with `VT_XPU_XE2_PREFILL=0` and with default Xe2 selection. Each
fresh process performed one warmup and two timed rounds; this was repeated
once per path. The synchronized C++ forward includes host metadata/input
preparation and upload and excludes model load/JIT/reset. Raw second-session
logs are under `raw/p5_xe2_8192_ab_*.log.gz`.

| Path | Timed prefill seconds (two sessions) | Median | Median throughput |
| --- | --- | ---: | ---: |
| Q64 | 14.686, 14.716; 14.693, 14.735 | 14.704 s | 557.1 tok/s |
| Xe2 | 6.031, 6.123; 6.032, 6.144 | 6.077 s | 1347.9 tok/s |

The full-prefill speedup is 2.42x. Both paths reported 20,644,268,596 peak
allocated bytes for this O1 case. A separate trace selected native GDN 96
times and Xe2 attention 32 times across warmup plus timed P8192 requests;
those are 48 GDN and 16 attention layers per forward.

Full-model P8192/D64/O65 logits were captured at prefill, decode 1 and
decode 64 with the same prompt/decode hashes as the pinned Python replay.
`p5_xe2_8192_quality_64.json` passes all frozen TV/KL/top-10/top-1 gates.
Prefill TV was 0.00240, decode-1 TV 0.01881, and decode-64 TV 0.00286.
Decode-1 TV is close to the frozen 0.02 limit and remains a watch item. A
second fresh-process P8192/D64/O65 capture produced byte-identical F32
logits at all three checkpoints.
One candidate D64 timing was 20.80 forwards/s versus 21.12 in the prior
Q64 quality run, within the observed short-run spread and not evidence of a
decode improvement. At P4096, the new run's prefill, decode-1 and decode-64
logit files are byte-identical to the previously Python-validated P4 run;
the new P4096/D64 run measured 1394.73 prefill and 23.62 decode forwards/s.

The Python P8192 prompt is internally split into two 4096-query model calls,
while this C++ case is one 8192-query forward. Python request timing is not
used as a matched-scope 8K performance claim. Other prompt lengths,
continuation-prefill, ragged/multiple sequences, nonunit scales, and other
layouts still need separate P5 qualification. This change adds no Python,
PyTorch or Triton dependency to the C++ runtime.
