# B70 GPTQ/FP8 checkpoint for external review (2026-09-27)

Branch: `b70-gptq-int4`. Target: Qwen3.8-27B GPTQ INT4 G128, FP16
activations, FP8 E4M3 KV, 1600-token effective pages, FP32 GDN state on
Intel Arc Pro B70. The runtime goal is C++ plus native GPU kernels; Python is
an oracle, never a runtime dependency. The frozen Python reference is vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` with the archived local
patches, model revision `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.

## Progress against the ChatGPT Pro P0-P7 plan

| Phase | Current state | Remaining work |
| --- | --- | --- |
| P0 reference and provenance | Stack, selected V2 runner, primary routes and model manifest recorded. | Exact native source corresponding to installed XPU wheel 0.1.15.4 remains missing; no exact kernel-parity claim. |
| P1 timing and cases | P/D/O notation, step traces, token replay, 4K/64 worker-path diagnostic, long and boundary harness cases. | Current C++ and Python device-core, worker-step and end-to-end scopes need a new same-session comparison; wider matrix incomplete. |
| P2 profiling and quality | Critical paths profiled; frozen Python logits gates; 4K/1024 and selected 8K checks pass. | Full natural/code, ragged batch, longer-context and repeated-session coverage incomplete. |
| P3 metadata | Validation cost measured; candidate did not yield an accepted model-level gain. | Safe amortization remains possible if later profiles justify it. |
| P4 GDN | Native Xe2 chunked path and 4K/8K gains qualified. | Wider ragged/chunked cases and exact installed-reference comparison open. |
| P5 attention | Native FP8 Xe2 prefill length range, continuation, split reduction and active-page cap improved. | Python device-core comparison, ragged/multiple-request and broader contexts open. |
| P6 remaining costs | FP16 GDN post-conv subgroup change improved real model; oneDNN and state-copy audit found no justified further change. | Reprofile after concurrency and matched Python measurements. |
| P7 graphs/serving | Opt-in native graph qualified for single-request 4K/64 and 4K/1024 quality; 4-request functional check passes. | Eight-request intermittent output mismatch, lifecycle/concurrency matrix, serving TTFT/TPOT and final parity table open. |

The plan's definition of **done** is not met. In particular, the target is
no slower than Python in every agreed primary workload under identical timing
boundaries, with quality, memory and route evidence. Vision and MTP speculative
decoding were not part of this performance-parity plan.

## Current measured performance

Single request, P=4096, D=64, O=65, FP8 KV and page 1600:

| Engine/configuration | Prefill tokens/s | Decode forwards/s | Measurement |
| --- | ---: | ---: | --- |
| C++ eager, current | 1431.47 | 25.5336 | Five-run median, unprofiled forward benchmark |
| C++ graph, current | 1429.91 | 26.6512 | Five-run median; all 64 scored decode forwards replayed |
| Python V2, earlier | 1593 | 29.67 | Median of three session medians, worker-path diagnostic |

The graph improves C++ decode by **4.38%** versus current C++ eager. The
current graph figures are about 90% of the earlier Python rates, but those
runs do **not** have identical scope and date. This is a directional gap,
not a measured same-scope parity ratio. See `timing_contract.md` and
`p7_graph_decode_4096.md`. P4096/D1024 graph logits matched previous C++
eager captures byte-for-byte at four checkpoints and passed frozen Python
TV/KL/top-k gates. Graph remains opt-in (`VT_XPU_GRAPH=1`,
`VT_GPTQ4_GRAPH=1`).

## Concurrency diagnostic at the checkpoint

The opt-in real-model checkpoint harness now accepts `VT_B70_BLOCK_SIZE=1600`,
`VT_B70_BATCH_LONG_OUTPUTS=1` and `VT_B70_BATCH_EIGHT=1`. It compares serial
greedy responses and per-token streamed chunks with the same requests
submitted concurrently in a different order, then checks graph counters and
memory. `VT_B70_BATCH_GRAPH_AFTER_BASELINE=1` keeps serial baseline requests
eager and enables graph for the mixed batch. The focused test target compiled.

- Four concurrent requests with graph: 59/59 assertions passed; six captures
  and 63 replays. Matching eager run: 50/50 passed. This includes condensation,
  cancel and reuse.
- Eight unequal requests: eager repeated 3/3 passed (75/75 assertions each).
  Graph-enabled runs sometimes passed and sometimes produced a different
  response for case 5 at output chunk 2. A failing run had 77/78 assertions;
  another run with memory checks had 69/70. Three delayed-graph control runs
  passed (80/80 each). The mismatch is **unresolved**; eight-way graph
  concurrency is not qualified.
- At the first divergent step in a traced failing run, eight requests were
  active and the selected route was eager (`batch_exceeds_graph_limit`).
  An earlier graph request may be relevant, but this trace does not prove a
  cause. Do not attribute the failure to graph replay itself yet.

The route is capped at graph batch size four in `src/vllm/platforms/xpu.cpp`.
Representative passing, failing and control logs are in the review package.
Full local logs and large F32 dumps are deliberately excluded.

## Vision and MTP scope for a follow-up plan

`src/vllm/platforms/xpu.cpp` explicitly describes the B70 backend as a
native text path and says vision and speculative draft execution are not
enabled. There is generic Qwen3.5 MTP draft plumbing in
`src/vllm/model_executor/models/qwen3_5_dense.cpp`, but that is not a
qualification of MTP for this GPTQ/FP8/XPU configuration. Vision and MTP
need separate execution, quality, memory and performance plans. Both must
remain fully native C++/GPU implementations, even where the Python oracle
uses Triton or other Python-authored kernels.

## Questions for ChatGPT Pro

1. Given this source and trace evidence, what is the shortest isolation
   sequence for the intermittent batch-eight case-5 divergence? Which request,
   state, KV-slot and logit-margin probes distinguish numerical near ties,
   stale state, scheduler reuse and graph side effects without perturbing the
   scored timing path?
2. How should we finish the three identical-boundary Python/C++ timing scopes
   for 4K/64 and 4K/1024 and extend them to unequal concurrent requests?
   Which cases are the minimum credible final P7 matrix?
3. What native C++/SYCL vision path should be built first for this exact
   Qwen3.8-27B GPTQ checkpoint? Please map Python preprocessing, vision
   encoder, projector, token merge and multimodal cache behavior to required
   C++ components, quality gates and memory budget.
4. Which MTP draft weights are available for this checkpoint, and what is the
   smallest native C++/SYCL speculative-decoding implementation that preserves
   FP8 KV and GPTQ target correctness? Please specify token accounting,
   acceptance sampling, rollback, GDN/attention state handling and quality
   tests before speed claims.
5. In which order should the remaining P0/P1/P2/P4/P5/P7 gaps, Vision and
   MTP be tackled to reach production throughput and functionality without
   masking the current batch-eight issue?
