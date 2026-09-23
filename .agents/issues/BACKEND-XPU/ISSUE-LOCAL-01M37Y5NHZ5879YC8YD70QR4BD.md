ID: ISSUE-LOCAL-01M37Y5NHZ5879YC8YD70QR4BD
Title: Implement native Intel B70 SYCL backend for Qwen3.8-27B EXL3
Row: BACKEND-XPU
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

vllm.cpp currently executes the pinned Qwen3.8-27B EXL3 checkpoint on B70 only through Vulkan, with CPU fallbacks and slow scalar EXL3 GEMM; a native Intel XPU/SYCL path needs staged implementation per the ChatGPT Pro plan.

## Resolution

-
