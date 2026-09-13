ID: ISSUE-LOCAL-01M2CZX87ZB0WHRW2YZYPW7VRZ
Title: The variadic harness never restores its cached vllm-server, because the CIFS share is mounted file_mode=0664 and the cache guard tests -x
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

benchmarks/variadic/job.sh:183 guards the binary cache with [ -x "$CACHED_BIN" ]. The rc share that holds the cache (/workspace on dgx:gpu0) is mounted cifs with file_mode=0664 and nounix, so no file on it is ever executable and the guard never holds. Measured 2026-09-13: /workspace/exl3-3150-ab/head/bin/39d3af455866bc47d76b0e0bd27fc3693e9e6ff5/vllm-server exists and reports -x FALSE, and the resubmitted #3150 A/B rebuilt that same pin from source (08:14:39 to 08:40:52 UTC, 26 minutes of a 35-minute setup) after a crash. dgx:gpu0 crashes under load about hourly, so the harness is resumable by design, and this guard throws away most of each boot's window: that boot crashed at 08:52, with 8 minutes of legs. The comment at line 226 names the cached binary as the resume guard, so the defect defeats the property it states. The restore branch already chmods the local copy (line 190), so an existence test is sufficient.

## Resolution

-

## Progress

2026-09-13: the guard in benchmarks/variadic/job.sh now tests `-f`, and the
restore branch keeps its chmod on the /tmp copy. The comment that names the
resume guard now names `-f`. tests/scripts/test_variadic_harness.py
`BinaryCacheRestores` executes the guard and restore branch cut from job.sh
against a 0664 cached binary. It failed with `REBUILD` before the fix, passes
after it, and fails again when `-x` is put back. The other `-x` tests in job.sh
(nvcc under /usr/local, the venv python under $SCRATCH=/tmp, `find -perm -u+x`
in the /tmp build tree) read local disk, so they do not have this defect.
