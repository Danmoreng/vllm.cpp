ID: ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q
Title: the decode step makes ~99 cudaFree calls at 634 us each, and DevicePool means it should be making almost none
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

MEASURED (host API side) on `dgx:gpu0` 2026-09-13, `nsys` over a 60 s window
inside a 3000-token decode at `ee0644eab`: `cudaFree` is **63.0% of all CUDA API
time** -- 52,130 calls at **634 us average**. Over the window's **527** steps
(see the step-count resolution in `ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142`) that
is **~99 calls and ~63 ms per step**. A 634 us free is itself anomalous and is
the reason to look: `cudaFree` synchronises the device, so each one drains the
stream before the next launch can be enqueued.

**THE STATIC FINDING, and it is the opposite of what the first revision of this
issue said.** The `vt::Free` seam itself does have no pooling: `vt::Free`
(`src/vt/backend.cpp:132-144`) forwards to the CUDA backend's `FreeOnDevice`
(`src/vt/cuda/cuda_dropin.cu:209-212`), a bare `Check(cudaFree(pointer), ...)`.
**But almost nothing in a decode step is supposed to reach it**, because
`DevicePool` (`include/vllm/model_executor/models/device_pool.h`) sits above that
seam and exists precisely to stop it. Its own `Get` says so
(`device_pool.h:111-125`): blocks "are never returned to the driver", and the
`VT_POOL_BYPASS=1` lane that turns every `Get`/`Put` back into a raw driver
`Alloc`/`Free` "reinstates the per-op `cudaMalloc`/`cudaFree` sync storm this pool
exists to remove, so it is never a timing configuration". A block only goes back
to the driver through `DevicePool::Drain` (`device_pool.h:322-338`), "one
`cudaFree` per retained block, once".

`qwen4_exp` routes its temporaries through that pool -- `qwen4_exp_forward.h:140`,
`qwen4_exp_qsa_block.h:200` and `qwen4_exp_ple_block.h:171` each describe "the
shared_ptr that returns its pool block to the `DevicePool` when the last reference
drops".

**SO THE MEASUREMENT IS MORE ANOMALOUS, NOT LESS.** A steady-state decode step
whose temporaries are pooled should free almost nothing, and a pool MISS costs an
`Alloc`, never a `Free`. ~99 `cudaFree` per step at 634 us is therefore not "the
allocator this tree uses"; it is ~99 frees that the pool's design says should not
be happening at all. **THE FIRST REVISION OF THIS ISSUE GOT THIS BACKWARDS** and
proposed `WorkspacePool` and the FA2 scratch as precedents for a caching allocator
to add. The cache already exists. Do not scope one.

**ONE CANDIDATE IS ALREADY REFUTED, STATICALLY.** A per-step `Drain` would
explain the count almost exactly -- "one `cudaFree` per retained block" against
~99 frees a step. It is not the cause: `Drain` has exactly two call sites in the
tree, `minimax_h3_pipeline.cpp:559` and `ltx2_video.cpp:5800`, both at a
diffusion/video phase boundary, and neither is reachable from a `qwen4_exp` text
decode. Whoever takes this should not spend a lease re-testing it.

Candidates that remain, none of them established here: allocations that call
`vt::Alloc`/`vt::Free` directly instead of taking a pool block; frees inside
cuBLAS or cuBLASLt that the profile attributes to our process; the `FreeGuard` at
`src/vt/cuda/cuda_qwen4_exp.cu:506`; and any path that constructs a device
`Tensor` without going through a `DBuf` at all.

**WHAT IS NOT ESTABLISHED, AND NO ROW MAY SCOPE A FIX UNTIL IT IS.** The spec's
`## Owed` already says this and it still holds:

1. **Which allocations, BY CALL SITE.** ~99 per step is a count, not an
   attribution, and the pool finding above means the interesting question is not
   "how do we cache these" but "why are these not in the cache". W6 already
   showed that guessing the population wrong sends a row at the wrong object.
2. **How much of the 63 ms is recoverable WALL time.** `cudaFree` is host API
   time and it can overlap device work. It also synchronises, so it can serialise
   work that would otherwise overlap -- which is the opposite sign. Only a
   measurement separates "63 ms of the step" from "63 ms of host time hidden
   behind device work".
3. **The workload.** These counts come from the same 3000-token profile whose
   mismatch is corrected in the spec's "### W9: the two numbers this row kept
   dividing into each other". Unlike the QSA kernel, `cudaFree` count per step
   should NOT scale with context -- which, if confirmed, is exactly what makes it
   the leading candidate at the 400-token reference workload while QSA is not.
   **CONFIRM THAT RATHER THAN ASSUMING IT**, by counting per step at two context
   lengths in the same run.

**Owed before a scope:** a per-step attribution of the `cudaFree` population by
call site, at the reference workload (400 tokens), with the per-step count taken
at two context lengths. `dgx:gpu0` has been unhealthy repeatedly; `thor:gpu0`
can carry the attribution, and the fleet-comparable numbers are owed on dgx.


## Resolution

-

**CONCURRENT WORK, DO NOT DUPLICATE.** On 2026-09-13 another session held
`dgx:gpu0` (`rc` job `072db212`) measuring whether `DevicePool`'s uncapped
retention exhausts GB10 unified memory at concurrency 32, arm `DEFAULT` against
arm `VT_POOL_BYPASS=1`. That is the same pool from the other side -- retention
cost rather than escape rate -- and its `BYPASS` arm is an upper bound on what
these ~99 frees per step would cost if every temporary escaped. Read its result
before taking a lease for this.

### CORRECTION 2026-09-13

Filed earlier the same day claiming "there is no pooling at that seam" and
proposing a caching allocator. `DevicePool` is that allocator and it has been in
the tree throughout, including for every measurement quoted here. The seam-level
reading was right and the inference drawn from it was wrong: reading one seam and
generalising to the path above it is the failure this record now carries.
