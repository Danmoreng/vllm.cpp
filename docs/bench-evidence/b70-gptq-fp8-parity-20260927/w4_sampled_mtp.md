# W4 sampled MTP, native XPU correctness increment (2026-09-28)

The installed Python vLLM 0.30.0 source was inspected in the pinned builder
image. With `draft_sample_method="greedy"`, `Speculator.draft_logits` is `None`.
The rejection sampler therefore treats the draft distribution as one-hot:
accept proposal `y` with probability `p_target(y)`; on rejection, draw from
`p_target` with `y` removed and renormalized; after all proposals are accepted,
draw a bonus token from the full target distribution. Python executes these
steps on the GPU. This is the production MTP profile's proposal rule, not a
generic probabilistic-draft implementation.

The active Python verification path calls
`vllm/v1/worker/gpu/sample/sampler.py::apply_sampling_params` for every
expanded target row. Its order is logit bias (including allowed/min-token
rules), penalties, bad-words mask, thinking-budget rule, temperature, min-p,
then top-k/top-p. Proposed token IDs and each row's local speculative position
are passed to the prefix-dependent processors. This differs from the older
generic `vllm/v1/sample/sampler.py` pipeline that our ordinary C++ sampler
mirrors. Implementing the remaining W4 processors therefore requires this
active GPU path's rowwise order and provisional-prefix semantics; simply
calling the ordinary sampler once per row would not establish Python parity.

The native XPU `SampleOneHotRejection` kernel now implements this rule on the
GPU. Its input is the per-expanded-row target probability tensor after
temperature, min-p, top-k and top-p processing. A separate seed belongs to
each request and logical output position, including each draft depth. The
accept/replacement walk remains device-side. The current implementation uses
one 128-lane workgroup per expanded row to draw from the full vocabulary; it
is a correctness rung, not an optimized small-top-k sampler.

The public engine routes sampled and mixed greedy/random XPU speculative
batches through this kernel. It expands temperature, top-k and top-p to every
verification row and uses existing native XPU transforms. Greedy rows use
deterministic argmax acceptance, correction and bonus inside the same native
accept walk. Bad words, custom processors, penalties and logprobs remain
unsupported until their rowwise semantics are implemented.
Greedy-only MTP and ordinary non-speculative sampling continue through their
existing routes. Python is not used at runtime.

## Focused evidence

| Case | Result |
| --- | --- |
| Tiny distribution `[.2,.3,.5]`, one-hot proposal `2`, 2048 seeded XPU requests | 8198/8198 assertions. Observed first-token frequencies and acceptance lie within ±0.035 of the target probabilities; rejection never emits proposal `2`, full acceptance emits a valid bonus. |
| Real GPTQ/FP8 model, short P64/O16, MTP4, T1/top-p .95/top-k 20 | 40/40 assertions, 16 tokens, 20 proposed/10 accepted, no reference-tier hits. Wall time includes load-independent request setup and prefill and is not a decode rate. |
| Real GPTQ/FP8 model, retrieval P4096/O32, MTP4, T1/top-p .95/top-k 20, seed 42 | 43/43 assertions, 32 tokens, 29 proposed/23 accepted, no reference-tier hits. The result is not expected to equal the greedy output string. |
| Mixed GPU rejection: greedy proposal rejected, sampled request, greedy proposal accepted | 8208/8208 assertions across the mixed and statistical tests. |
| Real GPTQ/FP8 model, two concurrent P4096/O8 MTP4 requests (one greedy, one T1/top-p .95/top-k 20) | 42/42 assertions, both completed, four output bursts each, zero reference-tier hits; automatic 1664-token hybrid page. This checks reachability and completion, not yet matched Python C2 quality or throughput. |

## First long C++ measurement

The focused `VT_B70_SAMPLED_MTP_BENCH=1` harness uses the same 4096-token
retrieval prompt, one request, FP8 E4M3 KV, 1600-token pages, eager execution,
MTP4, temperature 1, top-p .95, top-k 20, seed 42, and exactly 1024 emitted
tokens. It warms the route with O8, measures a separate P4096/O1 request for
TTFT, then measures the complete P4096/O1024 request. The arithmetic decode
rate subtracts that **separate-request TTFT** from the complete-request wall
time; it is not a same-request token timestamp. The second request's first
token matched the O1 control. No prefix cache or Python runtime was used.

| Metric | C++ result |
| --- | ---: |
| O1024 complete request | 28.7966 s, 35.56 emitted tokens/s over the entire request |
| Separate O1 TTFT | 4.18559 s, 978.6 input tokens/s client scope |
| Derived decode interval | 24.6110 s, **41.57 emitted tokens/s** |
| MTP proposals / accepted / steps | 1012 / 770 / 253 |
| XPU allocated / peak tracked | 27,881,959,058 / 27,891,892,054 bytes |
| Focused test | 50/50 assertions passed; zero reference-tier hits |

The first attempt with 16 reserved KV blocks was refused at load because
P4096+O1024 needs 0.83 GiB of KV while that pool supplied 0.78 GiB. The
reported run uses 20 reserved blocks.

## Python eager reference on the same prompt

A temporary vLLM 0.30.0+xpu server was started with the pinned production
GPTQ revision, native INT4 draft head/five linears, MTP4, FP8 KV, Eager
(`--enforce-eager`), C1, 4096 batched tokens, and prefix caching disabled.
The server received the same retrieval prompt string; the Hugging Face
tokenizer measured exactly 4096 prompt tokens. It used the same T1/top-p
.95/top-k 20/seed 42/ignore-EOS settings, warm O8, O1 TTFT control and
O1024 full request. The O1024 response reported exactly 1024 tokens.

| Matched workload metric | C++ | Python eager | C++ / Python |
| --- | ---: | ---: | ---: |
| Complete P4096/O1024 request | 28.7966 s | 16.8231 s | 0.584x emitted tokens/s |
| Separate O1 TTFT | 4.1856 s | 2.5815 s | 0.617x input tokens/s |
| Derived decode emitted tokens/s | 41.57 | 71.83 | **0.579x** |
| Full-request emitted tokens/s | 35.56 | 60.87 | 0.584x |

Python's server log reports effective 1664-token attention pages, while the
C++ run uses 1600-token pages; its FP8 KV pool and allocation policy also
differ. These results match the request workload and eager/sampling profile,
but not the page geometry or exact client boundary. The published production
67.6 decode tokens/s used a different prompt and graph-capable server recipe;
it remains a separate reference value.

The Python `/metrics` counters after warm O8, O1 and O1024 showed 1319 draft
tokens and 700 accepted tokens across those three requests. The C++ scored
O1024 request reported 1012/770. Because the Python counters include warmup
and the two engines use different request-local RNG streams, this is a
diagnostic difference, not a same-prefix acceptance comparison. Per-depth
draft/target probabilities at matched prefixes and the remaining rowwise
processors still need qualification. W2's exact top-20 path is recorded in
`w2_topk20_ab.md`; W5 still owes faster verification.

## After Python page parity and mixed-mode support

The separate page change `9a41281d8` made C++ derive the Python 1664-token
hybrid page for FP8/MTP4. The mixed-mode kernel change adds a per-row greedy
flag. On the same C1 P4096/O1024 retrieval request and settings above, a
focused rerun measured TTFT 4.01176 s, full wall 28.6342 s, derived decode
**41.5475 emitted tokens/s**, and 1012 proposed / 770 accepted drafts over
253 steps (50/50 assertions, zero reference-tier hits). This single rerun is
consistent with the preceding 1664-page 41.5738 decode tokens/s; it does not
establish a speed gain from mixed-mode support. The Python 1664-page eager
reference remains 71.8318 derived decode tokens/s under the matched client
recipe.

## Expanded logit bias and min-tokens (2026-09-28)

The active Python GPU sampler applies allowed-token restrictions, additive
logit bias and the min-token stop mask before penalties and temperature. Its
min-token predicate uses each expanded row's model position:
`pos + 1 < prompt_len + min_tokens`. For a completed prompt, the equivalent
C++ quantity is the accepted output count plus that row's draft depth. The
native C++ verification path now expands additive logit bias onto each row,
then masks stop tokens only while that provisional count is below the floor.
It uses the existing native XPU sparse bias and mask kernels. These processors
are absent from the production T1/p.95/k20 benchmark, so this is a correctness
increment, not a claimed speed gain.

The focused five-row XPU test covers two requests, rowwise bias, the
min-token boundary inside a draft span and the Python bias-before-mask order:
3/3 assertions, zero reference-tier hits. A real mixed C2 P4096/O8 FP8/MTP4
run, with one greedy request and one sampled request using `min_tokens=8`,
passed 42/42 assertions; both finished in four output bursts. The unchanged
production C1 P4096/O1024 route passed 50/50 assertions and retained the
same output-ID FNV64 `6972477010856893408`, 1012 drafts proposed, 770
accepted and 253 draft steps. Its derived decode rate was 42.1777 tok/s,
consistent with the W2 top-20 candidate series. The active Python sampler
source establishes the transformation order; a same-prefix Python/C++
probability comparison for these additional controls remains to be done.

## Expanded allowed-token mask (2026-09-28)

The C++ verification path now expands each request's allowed-token mask to
its speculative rows and applies the existing native XPU vocabulary mask
before logit bias and min-tokens, matching the active Python GPU sampler's
order. A focused five-row, two-request XPU test checks that one request's
restriction applies to every draft depth while the other request stays
unrestricted; it also checks bias and min-token masking on the same rows.
Result: 3/3 assertions, zero reference-tier hits. This feature is absent from
the production T1/p.95/k20 benchmark and from the C ABI's sampling-parameter
surface. A real 4096-token OpenAI request with allowed IDs, and a same-prefix
Python probability comparison, remain to be qualified before W4 closes.
