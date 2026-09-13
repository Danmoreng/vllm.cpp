ID: ISSUE-LOCAL-01M2DQC3M3VHKDEZYPA8BSRCAV
Title: rocprofv3 1.1.0 writes ZERO dispatch records on the bare strix:gpu0 worker: ring_buffer.cpp:106 mmap fails with EINVAL at output generation, FATAL-aborts, then deadlocks in its own signal handler ignoring SIGTERM
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Measured 2026-09-13 on `strix:gpu0` (gfx1151, ROCm 7.2.4) across four leases, on vllm.cpp `cc0e827dd`. `rocprofv3 --kernel-trace` traces the workload correctly and at negligible cost -- profiled decode 4.804 tok/s against an unprofiled 4.823 tok/s on the same binary and artifact, a 0.4% difference, with rocprofv3 timing its own child at 47.86 s -- and then **writes nothing**. At output generation, after the traced process has exited: `E output_stream.cpp:111] Opened result file: ..._kernel_trace.csv` / `ring_buffer: munmap failed: Invalid argument` / `F ring_buffer.cpp:106] mmap failed with errno 22 :: Invalid argument`, logged at FATAL, which aborts. Its `rocprofv3_error_signal_handler` (`tool.cpp:3104`) then catches signal 6, re-enters, and DEADLOCKS: on one leg SIGTERM arrived 29 minutes later and was also survived, so only SIGKILL ends it. A leg budgeted at 3600 s consumed an entire hour without producing a byte. REPRODUCED on both artifacts (Qwen3.8-27B-Q4_K_M 17 GiB, rc=134 at 37-47 s; Qwen3.8-Flash-Next UD-IQ1_S 67 GiB, hung to SIGKILL) and both writers (`--output-format csv` gives a zero-row CSV; `--output-format rocpd` gives a 2.1-2.4 MB SQLite database carrying 639 `rocpd_info_kernel_symbol` rows and an EMPTY `rocpd_kernel_dispatch` table). Raising `vm.max_map_count` from 262144 to 2097152 did not move it. **This is not a version regression**: `rocprofv3 --version` reports 1.1.0 at git revision `97f5574fe2fdc7bef44fb01545347912ee9f1779`, byte-identical to the revision pinned by `docs/bench-evidence/strix-kernel-trace-3015-20260907/`, which wrote 85,737 dispatch rows from this same board. The difference is the ENVIRONMENT: that capture ran inside a purpose-built podman image `localhost/vllmcpp-strix-profile:3015-deps`, whereas the current `rc` worker is bare Ubuntu 24.04 with the same packages installed by apt and no `podman` at all. `ulimit -l` on the bare worker is 8192 (8 MiB). CONSEQUENCE: there is no per-kernel decode-time attribution on ROCm, so where Qwen3.8-Flash-Next spends its 5.0-5.3 tok/s on gfx1151 remains UNVERIFIED, and `scripts/rocm-rank-kernels.py` currently has no live trace to read on this fleet. NEXT: reproduce the 2026-09-07 image, or identify which of its properties the bare worker lacks (memlock limit and seccomp/`MAP_LOCKED` are the named suspects). Evidence: `docs/bench-evidence/rocm-kernel-attrib-gfx1151-20260913.md`.

## Resolution

-
