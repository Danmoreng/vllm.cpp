# B70 GPTQ/FP8 timing contract

This contract supersedes the historical comparison of Python engine-step time
with C++ synchronized `ModelRegistry::Forward` time. Those numbers are
directional only. The frozen first comparison uses the checkpoint and software
versions in `baseline_manifest.json`, one request, FP16 activations, FP8 E4M3
KV, effective 1600-token pages, FP32 persistent SSM state, eager execution,
no prefix cache, and no speculation.

## Counts and phase labels

`P` is the number of prompt tokens, `D` the number of additional autoregressive
decode forwards after prefill, and `O` the number of generated tokens. Without
speculation, `O = D + 1`. The historical anchor is `P=4096, D=64, O=65`;
`P=4096, D=63, O=64` is a distinct case.

For each *actual* model invocation record request ID, query-token count,
context length before and after, cache hit count, and whether it produced a
prompt or generated-token logit. A prefill may span more than one invocation.
Classify by scheduled tokens and context, never by engine-step ordinal. A
decode window lists each context length and per-step latency.

## Scopes

| Scope | Start | End | Included | Excluded |
| --- | --- | --- | --- | --- |
| Device model core | Identical device inputs and metadata are ready | Selected logits are complete on device | Embedding, model body, logit-row selection, head, required cast, internal glue | Input preparation/upload, scheduler, sampler, IPC, tokenizer |
| Worker step | Before per-step input/metadata preparation | Selected logits are complete on device | Input preparation/upload and model core | Scheduler, sampler, IPC, tokenizer |
| End to end | Request submitted | First token emitted / each later token emitted / request complete | Scheduler, worker, sampler, IPC | Tokenization only if separately measured on both sides |

The Python V2 runner returns hidden states from `execute_model()` and computes
logits later in `sample()`. A timer around `execute_model()` or `_model_forward`
alone is therefore incomplete. For scored runs, place only phase-boundary
synchronization; do not synchronize after every operator. GPU profiler runs are
separate from throughput runs.

The current C++ benchmark includes host-side construction/upload of inputs in
`ModelRegistry::Forward` and synchronizes after it. It is a *forward benchmark*,
not yet a verified match to either scope above. A C++ adapter must move the
device-core start boundary after preparation, and the Python adapter must stop
after `compute_logits()` completes but before sampling. End-to-end comparison
requires both serving paths and actual emission timestamps.

The C++ harness now emits `gptq4_benchmark_step` records after each measured
round. They contain the actual query-token count, context before and after,
input token for decode, and synchronized forward duration. Output is deferred
until the round ends so JSON serialization is outside the timed loop. The
current single-request unchunked harness records one prefill followed by `D`
one-token decode calls. `O=1` has `D=0` and null decode metrics. This does not
establish how many prefill calls the Python scheduler makes.

The harness accepts `1<=P<=6656` on the present unchunked native GDN path and
`O>=1`, subject to the model position limit, optional
`VT_B70_BENCH_MAX_CONTEXT`, and a measured GPU cache/state budget with a 1 GiB
workspace reserve. Cases above 6656 prompt tokens need a separately validated
chunked path; the 8192/16384/32768 cases in the plan remain unsupported here.
The focused functional checks in `harness_checks.json` cover pure prefill,
one context above 4096, and 65 decode forwards. Their single-run durations
must not be used for performance decisions.

Teacher-forced differential tests use identical prompt and decode token-ID
files with recorded hashes. Free-running generation is a separate end-to-end
case. FP8 scales, physical page mapping, effective page size, and state-slot
mapping must also be logged and checked before a parity ratio is published.

After warmup, use at least five scored runs per case across independent
sessions and alternate A/B order. Report all raw per-run and per-step values,
median, dispersion, active routes, quality status, and peak memory. Do not use
profiled durations as scored times or infer confidence from correlated token
steps as if they were independent sessions.
