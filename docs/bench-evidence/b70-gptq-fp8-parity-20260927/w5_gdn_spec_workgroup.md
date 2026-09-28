# W5: explicit workgroup for 27B speculative GDN (2026-09-28)

The sampled native C++ MTP4 P4096/O32 profile attributed 222.9 ms of GPU
event duration to 384 speculative GDN calls after the first output callback.
The kernel launched `requests * hv * dv` independent state-channel items with
a plain `sycl::range`. This experiment retains the same per-item arithmetic
and state writes but launches an explicit `nd_range` with a selectable
workgroup size. The optimized default is limited to one request with the
measured 27B geometry (`hk=16`, `hv=48`, `dk=dv=128`) and at most five
verification tokens. Other shapes keep the prior launch. The
`VT_XPU_GDN_SPEC_WG` override accepts 0 (old launch) or 32/64/128/256.

## Short profile and correctness

Same B70, pinned GPTQ-G128 model, FP8 E4M3 KV, automatic 1664-token hybrid
page, native C++ eager MTP4, T1/top-p .95/top-k 20, seed 42, retrieval prompt
P4096/O32. Each profile emitted the same 32 token IDs and made 33,129 GPU
submissions. The table uses the eight output callbacks after the first one.
GPU event durations are sums, not exclusive wall time.

| Workgroup | Post-first callback wall | GDN speculative GPU sum |
| ---: | ---: | ---: |
| Old range (0) | 748.23 ms | 222.91 ms |
| 32 | 637.55 ms | 102.78 ms |
| 64 | 636.99 ms | 102.52 ms |
| 128 | 639.14 ms | 105.34 ms |
| 256 | 642.61 ms | 107.63 ms |

The focused B70 test `XPU speculative GDN MTP4 restores every accepted 27B
snapshot` passed 102/102 assertions with workgroup 64 and again after making
64 the scoped default. All four candidate profiles passed the P4096/O32
full-model assertions. The short profiles rank the launch sizes, while the
long unprofiled runs below establish the request-level gain.

## Full-model A/B

Each row is an independent warmed process on the B70. Both routes use the
same 4096-token retrieval prompt and exactly 1024 emitted tokens, C1, FP8
KV, 1664-token effective page, MTP4, separate packed INT4 draft head and
linears, T1/top-p .95/top-k 20, seed 42, ignore EOS and no prefix cache.
The harness measures O1 TTFT in a separate request and derives decode rate
as `1023 / (O1024 wall - O1 TTFT)`. This is a client-scope estimate, not a
same-request first-to-last-token interval. The comparison alternated routes;
the first pair used candidate then baseline.

| Run | Route | O1 TTFT s | O1024 wall s | Derived decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Workgroup 64 | 3.92463 | 24.5994 | 49.4806 |
| 2 | Old range | 3.87898 | 28.1497 | 42.1496 |
| 3 | Old range | 3.86800 | 28.2657 | 41.9302 |
| 4 | Workgroup 64 | 3.89059 | 24.7321 | 49.0848 |
| 5 | Workgroup 64 | 3.88310 | 24.6905 | 49.1652 |
| 6 | Old range | 3.87979 | 28.3245 | 41.8496 |
| 7 | Old range | 3.88100 | 28.3739 | 41.7671 |
| 8 | Workgroup 64 | 3.88950 | 24.7223 | 49.1052 |
| 9 | Workgroup 64 | 3.88871 | 24.7223 | 49.1035 |
| 10 | Old range | 3.88466 | 28.3348 | 41.8402 |

Median **41.8496 -> 49.1052 emitted decode tok/s**, **+17.3%**. The
ranges did not overlap. All ten runs passed 50/50 assertions and reported
the same 4096-token prompt hash `4760835697920107937`, 1024-token output
hash `6972477010856893408`, 1012 proposed drafts, 770 accepted drafts and
253 speculative steps. Peak tracked GPU bytes were unchanged for the two
routes. Thus the measured gain is not due to a different sampled trajectory.

After rebuilding with the scoped workgroup-64 default, one full P4096/O1024
request with no override passed 50/50 assertions, retained the output hash
and counters, and measured 49.5478 derived decode tok/s. The same focused
GDN test passed 102/102 assertions. The previously measured matched-workload
Python eager result was 71.83 derived decode tok/s, making the new C++ median
about 68.4% of that reference. That Python run was earlier, not a simultaneous
paired measurement. This change does not improve prefill or establish
graph, C2/C4, long-context or full production parity.

Build: pinned oneAPI 2026.1.1 builder, oneDNN 3.13 and SYCL-TLA commit
`87f6850680a580654b9ea2c80dbc01aeb36ad231`. The pinned TLA checkout
was restored outside Git at `/tmp/b70-sycl-tla-87f685` because the current
container no longer had `/opt/tla`. The focused build target was
`test_xpu_gdn test_xpu_qwen_checkpoint`; both compiled and linked.

The next W5 work is an exact Python/C++ same-prefix draft/verification
probability comparison, then a different shared-KV or attention layout and
the remaining small-M/graph bottlenecks. The earlier shared-KV probe is
recorded separately in `w5_shared_kv_probe.md` and was not promoted.
