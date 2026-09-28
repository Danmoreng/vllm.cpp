# W3: first native XPU speculative recurrent operators (2026-09-28)

This is a correctness increment for **eager MTP1/2/4**. Full-model GPTQ
smokes now run on the B70, but W3 is not yet fully qualified and these short
probes are not a decode-performance result. The runtime operations are native
SYCL/C++; Python remains an external reference only. No graph or prefix-cache
behavior was changed.

## Implemented

`CausalConv1dSpecUpdate` and `GdnSpecDecode` now have XPU registrations and
native kernels. Both validate GPU-resident offsets, accepted counts and state
slots before mutating state; invalid metadata throws. FP16 activations and
outputs are accepted on XPU. The GPTQ conv cache uses FP16; GDN snapshots
remain FP32.

- The conv kernel reads the widened history at `accepted_count - 1`, emits
  each speculative row, then shifts the persistent window by one tap and
  appends this step's raw inputs. This follows the existing CPU reference's
  rollback contract without an inverse state update.
- The GDN kernel loads the previously accepted snapshot column, advances the
  recurrence in token order, and stores one FP32 state snapshot per verification
  row. A null initial slot emits zeros; a null snapshot column skips only its
  store. Distinct requests may not share live snapshot slots. The first
  correctness route uses private per-work-item state rows and supports
  `Dk <= 128`, covering this model's `Dk=128`; it is not yet optimized.
- The target's output precision and weights were not changed. Both operations
  remain unused by the default text-only non-speculative route.

## Focused tests on the B70

Build: `cmake --build /build/cmake --target test_xpu_gdn -j4` in the pinned
oneAPI builder with oneDNN 3.13 and SYCL-TLA
`87f6850680a580654b9ea2c80dbc01aeb36ad231`. The production service was
inactive. Each command selected only the named doctest case:

| Test case | Result |
| --- | --- |
| `XPU speculative conv preserves accepted-prefix windows and rejects bad metadata` | 172/172 assertions passed. F32/FP16, accepted counts 1–4, unequal request lengths, null slot, invalid count/duplicate slot and no state mutation on rejection. |
| `XPU speculative GDN snapshots match CPU for every accepted prefix` | 220/220 passed. F32/FP16, two successive steps with previous accepted counts 1–5, full output/state comparison, invalid count and no state mutation on rejection. |
| `XPU speculative GDN MTP1 uses 27B FP16 activations and FP32 snapshots` | 20/20 passed. Actual 27B dimensions Hk=16, Hv=48, Dk=Dv=128, two verification rows. |
| `XPU speculative GDN MTP4 restores every accepted 27B snapshot` | 102/102 passed. For every accepted count 1–5, the next GPU step and selected state slot match an independent serial CPU teacher-forced accepted-prefix computation. |

The focused CPU/XPU comparisons use the existing CPU operator as the
reference. They do not yet compare an actual loaded GPTQ model's internal
states or qualify k=4 throughput. The GDN kernel's
`Dk <= 128` restriction and private-row implementation must be included in
future route/memory profiling.

## Follow-up: target-to-draft dtype contract

`Qwen3_5MTPModel::ForwardPaged` now accepts the GPTQ target's FP16
post-final-normalization hidden tap and keeps the XPU draft activations FP16
through the head, decoder, carry and logits input. The temporary per-forward
FP16→BF16 bridge was removed after inspecting the installed production Python
image: `AutoRegressiveSpeculator` allocates hidden states in model dtype, which
is FP16 for this GPTQ model; the installed INT4 draft-linear helper expects
FP16 inputs and only casts if they are not FP16. The checkpoint's BF16 draft
weights are converted to INT4 once at load time by that helper. Native C++ now
performs the same symmetric G128 packing at load for the GPTQ/XPU draft:
FC, merged QKV, attention output, merged gate/up, down, and a separate draft
head copied from the target's FP16 checkpoint head. The target head remains
FP16. Generic CPU/BF16 behavior is retained.
This establishes the no-bridge activation contract, not full numerical parity.

The focused XPU FP16-output matmul test with BF16 weights passed 46/46
assertions against CPU F32 results rounded to FP16. The CPU test confirms
that an FP16 tap is rejected on the generic BF16 route (1/1 assertion).
The initial stage had no real GPTQ draft acceptance measurement; the first
short model-level observations appear below.

The focused real-checkpoint loader test opened only shard 5 from the pinned
local `Qwen3.8-27B-GPTQ-G128` model and called the C++ `LoadQwen3_5MTP`
loader with its actual config. All 6/6 assertions passed, covering the one
predictor layer, BF16 fc, head geometry, attention output gate and MLP shape.
This loads real weight payloads and executes the loader's full shape checks;
it does not by itself run draft inference.

A second focused GPU test now loads the same real MTP tensors, borrows the
checkpoint's FP16 shared embedding and output head, feeds a synthetic FP16
target tap directly into one native paged draft forward
with its own FP16 KV block, and computes all 248320 draft logits. It passed
29/29 assertions: FP16 output shape, finite nonzero hidden and logits,
two-step FP16-KV continuation matching a causal two-row forward exactly,
FP8 E4M3 draft KV write/read, packed INT4 draft/head execution, intact target
FP16 head, and zero reference-tier hits. The synthetic FP16 target tap changes
the draft top-1 from 14 (dense MTP) to 71262 (INT4 MTP); packing only the head
keeps top-1 at 14. Maximum hidden-state difference is 3.40479. This is a
synthetic-tap quality observation, not a production acceptance result. It
does not establish Python/C++ packed-output parity, actual target-tap quality,
or committed target state.

The first actual target-tap and eager MTP1 probes are recorded below. Graphs
and recurrent prefix snapshots remain disabled for the full-model W3 gate.

## First real GPTQ eager MTP1 model run

The real 4096-token target hidden tap now feeds dense and load-time INT4 draft
forwards with independent FP8 E4M3 KV. Both C++ draft variants selected the
same top-1 token on all 104 inspected rows. Their maximum logit difference was
3.46106, KL 0.0406666 and TV 0.0745358. This compares dense and packed C++
drafts, not the Python oracle; it is not a target-logit quality gate.

The first public-engine MTP1 attempt failed at the GPTQ speculative GDN guard.
The native spec-conv op now accepts the checkpoint's FP16 persistent conv
cache, while GDN snapshots remain FP32. The dedicated B70 CPU/XPU operator
case passed 41/41 assertions for previous accepted counts 1–3. Pure GPTQ spec
batches can enter the existing native GDN/conv route; mixed spec/prefill GPTQ
batches remain guarded until separately qualified.

The next model attempt found a missing XPU `GreedyRejectionSample` registration.
The native two-phase argmax and acceptance implementation now matches the CPU
reference on a ragged k=1/3/0/2 batch at the full 248320-token vocabulary,
including tie handling, rejection and invalid offsets (8/8 assertions).

With those operators, the public GPTQ engine completed an eager single-request
MTP1 run with FP8 E4M3 KV and no reference-tier hits. At a 64-token repeated
prompt it generated 8/8 requested tokens (34/34 assertions, 4/4 draft tokens
accepted). The same 8-token text was observed in a separate no-MTP run.
At a 4096-token repeated prompt with 1600-token pages and 16 reserved KV
blocks, the MTP1 and no-MTP runs each passed 35/35 assertions and emitted the
same 8-token text; MTP1 again accepted 4/4 drafts. The first 4K attempt used
the harness's default eight KV blocks and failed admission (0.36 GiB
available versus 0.52 GiB required); `VT_B70_NUM_BLOCKS=16` supplies enough
space for this focused run. These are short correctness probes, not scored
decode-throughput measurements. The repeated prompt also gives a weak greedy
equality witness; more diverse prompts and forced model-level rejection are
still required.

W3 remains **in progress** for EOS and cancellation. The independent
accepted-prefix recurrent-state gate now covers every MTP4 snapshot, while
model-level token-ID continuation, output cap and stop-string probes cover
rejection and stop behavior. Production sampling and full-cycle emitted-token
throughput belong to W4/W5. No MTP speed claim or performance promotion follows
from the short probes above.

## Token-ID and rejection probes

The public-engine test can call `vllm_complete_tokens` after the same tokenizer
encoding as `vllm_complete` (`VT_B70_TOKEN_IDS=1`), then prints the actual
generated IDs. This avoids treating equal detokenized text as token equality.
It also accepts `VT_B70_PROMPT_TEXT` for a targeted short prompt and
`VT_B70_RETRIEVAL_PROMPT=1` for a retrieval prompt padded to exactly the
requested token count. Each run used greedy sampling, FP8 E4M3 KV, one
request, eager target execution, an eight/sixteen-block KV allocation at
short/4K context, and no reference-tier hits.

| Prompt and output cap | No-MTP IDs versus MTP1 IDs | MTP1 proposed/accepted | Focused tests |
| --- | --- | ---: | --- |
| 4096 repeated `Hello` tokens, O=16 | Identical: token 21251 repeated 16 times | 8/8 | 37/37 each |
| `In Python, write a function that returns the sum of even integers in a list.`, O=32 | All 32 IDs identical, including continuation after two rejected proposals | 17/15 | 37/37 each |
| 4096-token retrieval prompt with bicycle records before filler, O=32 | All 32 IDs identical, including continuation after six rejected proposals | 18/12 | 40/40 each |
| Short Python-code prompt, O=32, MTP2 | All 32 IDs identical to no-MTP | 24/21 | 40/40 |
| Short Python-code prompt, O=32, MTP4 | All 32 IDs identical to no-MTP | 36/26 | 40/40 |
| 4096-token retrieval prompt, O=32, MTP2 | All 32 IDs identical to no-MTP | 28/16 | 43/43 |
| 4096-token retrieval prompt, O=32, MTP4 | All 32 IDs identical to no-MTP | 42/19 | 43/43 |
| 4096-token retrieval prompt, stop string `Bruno`, MTP1 | Same seven IDs and stop reason as no-MTP | 4/2 | 41/41 each |

For the retrieval case, the MTP1 output ID sequence was
`271 248068 271 248069 271 90979 78 248044 248045 271 248069 271
90979 78 248044 248045 271 248069 271 90979 78 248046 198 248044
248045 248046 198 248045 248046 198 248044 248045`; the separate no-MTP
run produced the same sequence. The traced MTP cycles include both immediate
rejection (`ns=1`) and acceptance (`ns=2`), followed by matching tokens. These
tests exercise target-state routing through actual rejection, but they do not
compare the internal recurrent state tensors directly. Output cap was reached
at 16/32 emitted tokens; EOS and cancellation remain separate gates. Short
completion wall times include prefill, scheduling and first-token work, so
they are not decode-throughput measurements.
