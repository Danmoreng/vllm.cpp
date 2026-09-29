# Qwen3.8 TensorFold endpoint benchmark manifest

This directory defines the engine-neutral corpus used by
`tools/bench/qwen38_endpoint_bench.py`. The harness sends UTF-8 OpenAI-compatible
requests with seed 0, greedy decoding, `ignore_eos`, streaming enabled, and at
least five closed-loop waves.

Run either endpoint with its runtime model identifier (model names are not
embedded in the harness):

```sh
python3 tools/bench/qwen38_endpoint_bench.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model "$MODEL" \
  --corpus benchmarks/manifests/qwen38_tensorfold/corpus.json \
  --output raw.json --concurrency 1 --waves 5 --draft off
```

Keep every raw repetition. A cross-engine result may be called matched only
when all generated-token fingerprints are present and equal. Endpoints that do
not expose token IDs are therefore recorded with a refusal reason rather than
silently treated as equivalent. Corpus changes alter the canonical payload
hash and must start a new comparison series.
