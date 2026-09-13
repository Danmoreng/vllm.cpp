ID: ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG
Title: There is no declared token-exact gate for qwen4_exp on ROCm, so no cross-engine ratio may be taken on gfx1151
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: record
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

AGENTS.md Gates admits a performance result from an arm only after that arm's declared token-exact gate passes. The `qwen4_exp` ROCm arm has no such gate — not a failing one, an ABSENT one — and it cannot simply be written, because the architecture has no primary oracle: `Qwen/Qwen3.8-Flash-Next` declares `Qwen4ExpForConditionalGeneration` / `qwen4_exp`, which no vLLM revision implements (#1978). The pinned secondary oracle `llama-cpp-qwen4exp` (ggml-org/llama.cpp PR #27742 at 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3) now BUILDS with HIP for gfx1151 and DECODES the released UD-IQ1_S artifact on strix:gpu0, so a llama.cpp denominator exists on that board (docs/bench-evidence/qwen4exp-llamacpp-denominator-gfx1151-20260913.md, #2060). A denominator is a single-engine fact and needs no gate. A RATIO does, and this issue owns the missing precondition. Two further obstacles are specific rather than general. llama.cpp's own decode here is not token-gated against anything either, and the oracle record states its greedy decode is not deterministic across its own kernel paths. And the two engines are not serving the same surface: /props at the pin reports modalities vision=false video=false audio=false, so the llama.cpp arm is TEXT ONLY while MODEL-MM-QWEN4-EXP is a multimodal port. Until a gate is declared and ratified, every qwen4_exp gfx1151 number is reported per engine and never divided. #2497 already carries one retraction for taking a cross-engine number ahead of that gate.

## Resolution

-
