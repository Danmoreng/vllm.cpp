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
  --corpus benchmarks/manifests/qwen38_tensorfold/corpus.json \
  --output raw.json --concurrency 1 --waves 5 --draft off
```

`--tokenizer-identity` must identify the common tokenizer by immutable revision
or digest; the harness fails closed when it is omitted, and comparisons refuse
different identities. Runtime model aliases and endpoint adapter extras are
excluded from canonical workload hashes, while semantic prompt and decoding
parameters remain covered.

`--draft` is recorded as run metadata only and is never sent as a request
field. If an endpoint documents an endpoint-specific field, add it explicitly,
for example `--extra-json '{"draft_model":"MODEL"}'`. Extra JSON cannot
override canonical request fields. By default no draft-related wire field is
sent.

Keep every raw repetition. A cross-engine result may be called matched only
when neither run nor any sample was refused and canonical payload hashes,
tokenizer identities, prompt token counts, and generated-token fingerprints
are all present and equal. The HTTP adapter requests streaming usage; if usage
or reliable generated-token accounting is absent, the sample and enclosing run
are refused and token throughput/TPOT are not reported as valid. Endpoints that
do not expose token IDs are likewise refused rather than silently treated as
equivalent. Corpus changes alter the canonical payload hash and must start a
new comparison series.
