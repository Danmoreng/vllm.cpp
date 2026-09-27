# P4 native Xe2 GDN, 4096-token prototype

`VT_XPU_GDN_NATIVE=1` selects a C++/SYCL-only GDN prefill route for one
P4096 sequence with F16 Q/K/V/output, 16 key heads, 48 value heads, D128,
and FP32 state. Other cases use the existing route. The PyTorch wrapper from
the version-matched XPU kernel source is excluded. The source branch is
`release/0.1.15.4` at `ddf336d86e3c8602888572a3502f951abd51df12`;
exact correspondence to the installed wheel remains unproven. The pinned
SYCL-TLA revision matches the donor branch. The native path uses VT's queue,
allocator, and bounded workspace; it has no Python/PyTorch/Triton dependency.

The donor expects head-major G/beta, cumulative G, and Q scaled by
`1/sqrt(128)`. VT already computes normalized Q/K and token-major G/beta.
The adapted preparation kernel transposes and scans G/beta, while one native
scaling kernel prepares Q. The five donor kernel stages then run with the
gathered FP32 state. This prototype also adopts the donor's F16 A/W/U scratch;
the quality gates below check that precision change. The original chunked
path and its 16/32 MiB workspace remain selectable.

## Focused correctness and route

- Real model layer 0, P4096: native output and SSM state pass the fixed
  Python-relative thresholds in `p4_native_real_layer_compare.json`. Relative
  RMS is 0.000762 for core output and 0.000949 for final SSM state; conv state
  remains exact. The same real activations replay with a nonempty state; native
  versus existing C++ gives relative RMS 0.001487 output and 0.002389 state.
- Full model, FP8 KV page 1600: every frozen periodic checkpoint through
  decode forward 1024 passes `quality_thresholds.json`, including 703–705
  across the context-4800 page transition. Top-1 agrees at all eight points.
  See `p4_native_quality_4096_1024.json`.
- A diagnostic P4096/O2 trace selected `native` 96 times: 48 GDN layers in
  warmup and 48 in the measured request. The raw trace is compressed under
  `raw/p4_native_native_route.log.gz`.
- The existing real-layer replay still passes 45/45 assertions when the
  selector is off. The native real-layer diagnostic passes with the selector
  enabled.

## Model performance

The same C++ binary family ran P4096/D64/O65, eager batch 1, fixed prompt and
decode IDs, FP8 E4M3 KV page 1600. Each of three independent process sessions
had one warmup and five measured rounds. Timings exclude load/JIT/reset and
quality capture. The medians below are medians of session medians; all round
values and per-step raw logs are in `p4_native_4k_benchmark.json` and the seven
small gzip files under `raw/p4_native_*.log.gz`.

| Route | Prefill token/s | Decode forward/s | Peak allocated GPU bytes |
| --- | ---: | ---: | ---: |
| Existing chunked | 1194.28 | 23.60 | 19,350,113,448 |
| Native Xe2 | 1395.47 | 23.57 | 19,460,738,344 |

Prefill improves by **16.85%** at an additional **105.50 MiB** peak GPU
allocation. Decode is unchanged within observed variation. Against the
earlier matched-scope Python worker reference (1593 prefill token/s), this C++
candidate reaches about 87.6% of Python prefill throughput; that Python number
was measured in a different session, not as a simultaneous A/B run.

This accepts a measured **4K opt-in improvement**. P4 remains open for 8K/16K,
bounded longer-context scratch, scheduler-chunked continuation, ragged and
multiple sequences, and default-route qualification. The final three candidate
sessions followed a preliminary alternating A/B series; they were not themselves
interleaved with the three baseline sessions after the last removal of
profile-only barriers. The layer microbenchmark was also rerun after that
removal.
