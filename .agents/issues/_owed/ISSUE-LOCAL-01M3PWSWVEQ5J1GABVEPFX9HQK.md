ID: ISSUE-LOCAL-01M3PWSWVEQ5J1GABVEPFX9HQK
Title: Close the Qwen3.8 Flash Next speed gap against TensorFold
Row: -
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

TensorFold v0.3.6.3 reports 62.4 tok/s single-request decode and 2.2-2.5k tok/s long-context prefill on one DGX Spark for Qwen3.8-Flash-Next. vllm.cpp has no reproducible TensorFold denominator and has not measured or ported TensorFold techniques such as persistent QSA state, native MTP, tiled long-context selection, and overlapped PLE staging.

## Resolution

Pin TensorFold and its deployment patch set as a secondary implementation/performance oracle; reproduce both engines on dgx:gpu0 under a declared protocol; profile prefill and decode; then land only correctness-gated changes that close measured gaps, beginning with persistent QSA state and native MTP when profiles justify them.
