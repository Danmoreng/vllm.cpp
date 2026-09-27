# P2 worker profiles: 4K FP8, eager batch 1

Separate profiler runs used identical periodic prompt/decode IDs at P4096/D64
and P4096/D256, with the effective 1600-token page size. The Python trace is
from the active XPU V2 worker. `b70_python_profile_summary.py` assigns kernels
to its prefill/decode annotations using the correlated CPU enqueue timestamp;
the dense logits head is separately identified after `execute_model`. The C++
trace uses SYCL event profiling. These runs are **diagnostic**: profiling adds
substantial overhead, and the Python/C++ kernel families are not identical.
The scored, unprofiled worker timings remain in
`worker_scope_4096_64_report.json`.

| Device event family | C++ P4096 prefill | Python P4096 prefill | C++ D256 per forward | Python D256 per forward |
| --- | ---: | ---: | ---: | ---: |
| GPTQ INT4 matmul | 1774.9 ms | 2061.5 ms | 23.02 ms | 22.65 ms |
| GDN core/state | 779.1 ms, 26,112 events | 161.4 ms, 336 events | 0.67 ms | 0.65 ms |
| GDN postconv | 112.4 ms | included in different native operators | 2.98 ms | included in native GDN/core/glue |
| Full attention | 135.1 ms | 162.7 ms | 4.64 ms | 1.93 ms |
| Dense FP16 and head | 9.4 ms | 9.9 ms | 4.48 ms | about 4.45 ms |

Each value is a **sum of device event durations**, not an exclusive share of
wall time. C++ GDN core remains around 779 ms in all three P4096 profiles;
Python is around 161 ms in both long profiles. C++ decode GPTQ, attention,
postconv and head times per forward are stable from D64 to D256. Python
decode GPTQ and head are also stable; attention is about 2.12 ms at D64 and
1.93 ms at D256. The exact native wheel source for 0.1.15.4 is still missing,
so the trace proves observed routes and costs, not an exact source-level port.

The C++ D64 host profile records 321 metadata readback waits during prefill
and 8,256 over 64 decode forwards (129 per forward). Their summed spans are
2549 ms and 1696 ms respectively. `staged_d2h_wait` is nested inside these
spans; workspace waits may also overlap. **Those numbers cannot be added to
the device times or read as exclusive savings.** An attempted D256 host trace
hit the existing 100,000-record limit during warmup (exit 139); the successful
D256 rerun profiles device events only. P3 first needs a top-level validation
span or focused microbenchmark to establish exclusive, removable host cost.

The Python D256 worker trace records 13,944 H2D bytes and 40 MiB D2D in
prefill, then 4,096 H2D bytes and 2.5 MiB D2D across decode. C++ copy byte
counts and peak scratch per family are not yet exposed by this profile. The
correlation-based Python summary leaves 55.9 ms of kernel durations without a
matching enqueue and 16.8 ms outside a worker annotation at D256; those are
reported explicitly in `profile_p2_diagnostics.json`.

**Priority from these profiles:** P3 measures and safely amortizes metadata
validation; P4 targets the large prefill GDN submission gap; P5 targets the
decode attention gap. The logit thresholds are already frozen in
`quality_thresholds.json`. Selected GDN layer/conv-state goldens are still
needed before a GDN implementation change. The raw traces and the failed
attempt are retained outside Git at
`/home/sebastian/LocalLLM/b70-gptq-fp8-p2-profiler-traces-20260927.tar.zst`
(91 MiB; SHA-256
`43fcd64e4714de84e9ef202d0328c90951278fc3190579f651bb1e8da4c24f78`).
