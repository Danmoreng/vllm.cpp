# Qwen3.8 TensorFold endpoint benchmark manifest

This directory defines the engine-neutral corpus used by
`tools/bench/qwen38_endpoint_bench.py`. The harness sends UTF-8 OpenAI-compatible
requests with seed 0, greedy decoding, `ignore_eos`, streaming enabled with
usage requested, and at least five closed-loop waves. Aggregate request and
token rates use the measured makespan of the complete workload, rather than an
individual request latency.

Run either endpoint with its runtime model identifier (model names are not
embedded in the harness):

```sh
python3 tools/bench/qwen38_endpoint_bench.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model "$MODEL" \
  --tokenizer-identity "$TOKENIZER_NAME_OR_DIGEST" \
  --adapter tensorfold \
  --corpus benchmarks/manifests/qwen38_tensorfold/corpus.json \
  --output raw.json --concurrency 1 --waves 5 --draft off
```

`--tokenizer-identity` must identify the common tokenizer by immutable revision
or digest; the harness fails closed when it is omitted, and comparisons refuse
different identities. Runtime model aliases and endpoint adapter extras are
excluded from canonical workload hashes, while semantic prompt and decoding
parameters remain covered.

`--adapter generic` adds no endpoint-specific semantic fields. With `--adapter
tensorfold`, draft mode is explicit on the wire (`draft:false` or `draft:true`)
and the harness sends `return_token_ids:true`. The terminal `tensorfold` SSE
block supplies token IDs/hash and cache, draft, prefill, and decode telemetry.
`--extra-json` may add other documented fields, but cannot contradict canonical
or adapter semantic fields. The effective draft policy is included in run
identity.

Keep every raw repetition. A cross-engine result may be called matched only
when neither run nor any sample was refused and canonical payload hashes,
tokenizer identities, prompt token counts, and generated-token fingerprints
are all present and equal. Missing streamed token IDs leave fingerprints null
but do not invalidate absolute timing or throughput measurements when usage
counts are present; only a matched-token comparison is refused. E2E ends at
the terminal SSE event. ITL is reported only when every timed delta is proven
to contain exactly one token; chunk spacing is never presented as token ITL.
The HTTP adapter requests streaming usage; if usage or reliable generated-token
accounting is absent, the sample and enclosing run are refused and token
throughput/TPOT are not reported as valid. Comparisons also require an equal
run identity covering draft mode, concurrency, waves, and canonical semantic
requests. Corpus changes alter the canonical payload hash and must start a new
comparison series.
