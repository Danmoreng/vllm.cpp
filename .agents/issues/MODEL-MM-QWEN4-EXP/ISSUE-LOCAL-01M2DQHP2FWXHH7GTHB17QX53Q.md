ID: ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q
Title: vt::Free calls cudaFree directly with no pooling, so every temporary device tensor in the decode step costs a synchronizing free; the profile counts ~99 per step at 634 us
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

**THE STATIC FINDING, which says where they come from.** `vt::Free`
(`src/vt/backend.cpp:132-144`) forwards to the backend, and the CUDA backend's
`FreeOnDevice` (`src/vt/cuda/cuda_dropin.cu:209-212`) is a bare
`Check(cudaFree(pointer), ...)`. **There is no pooling at that seam.** Every
temporary device tensor destroyed during a forward pass is therefore a real,
synchronizing `cudaFree`, and every one created is a real `cudaMalloc`. This is
not a per-model defect; it is the allocator every CUDA model in this tree uses.

**THE TREE ALREADY CONTAINS TWO WORKING PRECEDENTS for the shape a fix would
take**, which is what makes this tractable and is also the reason not to invent a
third:
- `WorkspacePool` in `src/vt/cuda/cuda_dropin.cu:87-198`, keyed and grown in
  place.
- The FA2 decode scratch (`src/vt/cuda/cuda_flash_attn_fa2.cu:45-48`), "keyed by
  device+stream+capture-stable shape and never moved until queue teardown: the
  cold eager graph step allocates it, capture/replay only reuses the same
  pointers". That sentence is also the CAPTURE-SAFETY argument a pooled allocator
  needs, already made and already shipped.

**WHAT IS NOT ESTABLISHED, AND NO ROW MAY SCOPE A FIX UNTIL IT IS.** The spec's
`## Owed` already says this and it still holds:

1. **Which allocations.** ~99 per step is a count, not an attribution. The
   candidates named so far are pooled `DBuf` lifetimes in the MoE and attention
   paths and any `ResidentWeight` whose `d_dev` does not memoise. Nothing here
   says which, and W6 already showed that guessing the population wrong sends a
   row at the wrong object.
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
