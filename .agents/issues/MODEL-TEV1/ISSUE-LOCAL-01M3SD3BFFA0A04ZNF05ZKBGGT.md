ID: ISSUE-LOCAL-01M3SD3BFFA0A04ZNF05ZKBGGT
Title: Serve Tev1 through /v1/systemone and vllm_decide
Row: MODEL-TEV1
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

Ollama 0.35 serves togethercomputer/Tev1-4B-experimental (tev1) and Tev1-0.8B-experimental (tev1:0.8b) on /v1/systemone, the same endpoint as nimble: it compiles typed choice/noul/score questions, scores the allowed answer letters from next-token logits, and returns probabilities with entropy confidence. vllm.cpp serves Tev1 only through /v1/chat/completions, which returns one sampled letter and no distribution, and vllm_decide refuses the architecture by name. LocalAI reaches vllm.cpp only through the C ABI, so it cannot get a Tev1 decision with probabilities. The request-level machinery that Nimble uses (question compilation, candidate token check, openjev answer) exists but is private to Nimble.

## Resolution

-
