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
For an end-to-end request, reserve a model-length limit of at least `P+O`:
the final emitted token is not fed to the C++ forward benchmark, but Python
serving still checks the full requested output length.

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

The pinned Python image supports `SamplingParams.trace_decode_token_ids` when
`enable_trace_replay=True`. `tools/bench/b70_gptq_fp8_python_replay.py` uses
that feature to feed the same first `D` output IDs into later Python forwards
as the C++ decode-ID file. The diagnostic scheduler hook in
`tools/bench/b70_scheduler_trace/` records actual query lengths and contexts;
it writes synchronously and must be disabled for scored runs. Its corrected
4096/2 trace contains exactly one 4096-token prefill call and two one-token
decode calls, with zero cache-restored tokens. The Python runner records
whole-engine step durations only. `LLM.generate` buffered all three outputs
until the final step in this short diagnostic, so its first nonempty API result
is **not** a streaming TTFT. A streaming request path remains necessary for
end-to-end TTFT and TPOT qualification.

At `P=4097, D=0, O=1`, the same Python settings schedule **two prefill model
calls** (`4096` then `1` query token), although the outer `LLMEngine.step`
wrapper sees only one call. The current C++ harness runs one unchunked
4097-token forward. Those are different schedules and must be compared only
after a corresponding C++ chunked replay exists or after a clearly labeled
operator-level decomposition. The raw diagnostic and exact call counts are in
`python_harness_checks.json`.

Teacher-forced differential tests use identical prompt and decode token-ID
files with recorded hashes. Free-running generation is a separate end-to-end
case. FP8 scales, physical page mapping, effective page size, and state-slot
mapping must also be logged and checked before a parity ratio is published.

After warmup, use at least five scored runs per case across independent
sessions and alternate A/B order. Report all raw per-run and per-step values,
median, dispersion, active routes, quality status, and peak memory. Do not use
profiled durations as scored times or infer confidence from correlated token
steps as if they were independent sessions.

## First 4096/64 worker-path diagnostic

`worker_scope_4096_64_report.json` contains three separate sessions, each
with one warmup and five measured requests per engine, in Python/C++, C++/Python,
Python/C++ order. The Python V2 hook starts at `execute_model` entry and stops
after `compute_logits` and one XPU synchronization, before sampling. It buffers
step records until worker shutdown. The C++ timer starts before prompt position
and attention/GDN metadata preparation and stops after synchronized logits.
Both exclude model loading, warmup, scheduler, and sampler; both include input
preparation/upload and the LM head. The joiner rejects any mismatch in P/D/O,
token hashes, effective page size, model-call count, phase, query length, or
context. The exact host preparation differs between the two workers, so this
is a worker-path diagnostic, not a device-core-only or end-to-end result.

At P=4096, D=64, O=65 with FP8 KV and 1600-token pages, median of the three
session medians was 2.571 s Python vs 3.418 s C++ for prefill (1593 vs 1198
tokens/s), and 33.71 ms Python vs 42.38 ms C++ per decode forward (29.67 vs
23.60 forwards/s). Decode windows 1-16 and 17-64 are in the report; every raw
step is in the compressed logs. These data establish a repeatable baseline under
the stated diagnostic boundary. Long-replay numerical quality, per-session
kernel-route evidence, device-core timing, and streaming TTFT/TPOT are still
open; this is not a performance-parity qualification.
