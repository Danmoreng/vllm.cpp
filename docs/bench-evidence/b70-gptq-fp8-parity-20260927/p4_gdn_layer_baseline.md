# P4 synthetic GDN layer baseline

The existing focused XPU GDN test ran with `VT_B70_GDN_BENCH=1`,
`VT_B70_GDN_BENCH_F16=1` and only the `XPU GDN chunk64: timing` case. It uses
Hk=16, Hv=48, D=128, FP16 q/k/v/output, FP32 state, one sequence and the
default two-chunk GDN path. The test first checks output and final state against
the same-binary sequential XPU path. The second local B70 run is recorded in
`raw/p4_gdn_layer_synthetic_baseline.log`.

| Tokens | Chunked median | Sequential median | Accuracy |
| ---: | ---: | ---: | --- |
| 128 | 0.36 ms | 17 ms | passed |
| 512 | 1.47 ms | 67.17 ms | passed |
| 2048 | 5.91 ms | 268.18 ms | passed |
| 4096 | 12.02 ms | 536.96 ms | passed |

The focused case passed 792/792 assertions. This test's synthetic normalized
inputs are useful for scheduling and numerical regression checks but do not
meet P4's real projected-activation and nonempty-state replay requirement. Its
timings are one-layer operator timings, not worker throughput; the Python
profile's 161 ms across 48 GDN layers has different boundaries. Capture of
selected real layer inputs, output, SSM state and conv state on both engines is
the next prerequisite before modifying the GDN schedule.
