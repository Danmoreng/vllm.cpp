# P3 metadata validation baseline

`CheckDeviceMetadata` now records a top-level host span and a second span
named for its validation message when `VT_XPU_HOST_PROFILE=1`. The graph
capture path still uses its separate deferred validation. No check is skipped.
The focused benchmark target compiled, and FP8 model runs passed at P4096/D1
and P128/D1. Raw JSONL logs are retained in `raw/p3_metadata_*.log.gz`;
machine-readable counts and durations are in `p3_metadata_baseline.json`.

| Profiled phase | Checks | Top-level validation host span | Nested staged D2H wait | Remaining host time outside staged wait |
| --- | ---: | ---: | ---: | ---: |
| P4096 prefill | 321 | 2550.85 ms | 2549.39 ms | 1.47 ms |
| P4096 decode 1 | 129 | 33.30 ms | 32.76 ms | 0.53 ms |

The nested wait represents waiting for submitted device work and must not be
treated as 2.55 seconds of removable CPU code. The top-level spans do not
include the final scratch destructor; its separately measured `free_device`
span is under 0.05 ms for the 321 prefill checks. All durations are from
profiling-enabled runs and are unsuitable as scored throughput figures.

The P128 run identifies the repeated checks per model forward:

| Validation | Prefill | Decode |
| --- | ---: | ---: |
| GDN/conv state slot bounds and uniqueness | 192 | 96 |
| GDN/conv sequence offsets | 48 | 0 |
| Chunked GDN sequence offsets | 48 | 0 |
| KV slot bounds | 16 | 16 |
| Paged-attention offsets, lengths and block table | 16 | 16 |
| Index bounds | 1 | 1 |

This points to repeated state-slot validation across the 48 GDN layers, but
each check is tied to the bounds of a specific state allocation. A shared
descriptor must preserve that relationship and invalidate on mutation, buffer
reuse and new steps. The current measurement bounds non-wait host work to
roughly 1.5 ms in prefill and 0.5 ms in decode; any larger P3 gain would have
to come from removing synchronization-induced device gaps. Before changing
the execution path, measure that causal effect in a safe isolated validation
probe and retain invalid-metadata checks.
