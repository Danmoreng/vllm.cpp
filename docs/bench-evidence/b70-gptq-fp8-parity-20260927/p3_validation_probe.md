# P3 isolated validation synchronization probe

`tests/vt/bench_xpu_validation_probe.cpp` compares repeated eager GPU checks of
one immutable, valid state slot against a single check followed by the same
number of ordered dummy device submissions. It keeps a device-to-host status
readback in both paths, alternates run order, warms up twice, checks the final
device output, and verifies that both paths reject an out-of-range slot.

Compiled with the pinned oneAPI 2026.1 builder using `icpx -fsycl -O2` and run
on the local B70 with the production service stopped. Raw per-run timings are
in `raw/p3_validation_probe.jsonl` (seven alternating pairs per shape). These
are wall-clock medians in one process, not independent model sessions.

| Repeated checks | Separate check + work | One check + same work | Difference |
| ---: | ---: | ---: | ---: |
| 96 (decode GDN/conv state-slot count) | 1.237 ms | 0.212 ms | 1.026 ms |
| 192 (prefill GDN/conv state-slot count) | 2.086 ms | 0.385 ms | 1.701 ms |

This isolates a potential synchronization cost; it is **not** a model
throughput improvement or an estimate of the full-model gain. The probe uses
one slot, a pinned result and trivial interleaved work. Production
`CheckDeviceMetadata` also validates uniqueness, allocates through the XPU
backend, stages readback for ordinary host pointers and runs amid real GDN
kernels. A production descriptor must bind the validated bytes and version to
the owning state allocation and step, preserve graph/error behavior, and be
tested for mutation, reuse, aliases and concurrent queues. The model profiles
show a substantially larger GDN core gap, so P4 is the next performance
candidate while P3 descriptor design remains open.
