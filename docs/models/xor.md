# xor

xor is a 35B MoE System-1 decision model. Post-trained from
Qwen3.6-35B-A3B (35B total, ~3B activated per token), it extracts hidden
states via `ForwardMoeHidden` — the MoE variant of the `ForwardHidden`
pooling path — and reads candidate logits through the LM head. Each
question runs twice (forward and reverse option order) and the probabilities
are calibrated. It answers `choice`, `score`, and `noul` question types
through `/v1/systemone` and supports up to 8 images.

xor runs through the shared systemone lane, so [the usage guide](../USAGE.md)
covers the `/v1/systemone` request shape. This page carries the checkpoint,
the forward+reverse calibration, and what has not been measured.

## The checkpoint

| field | value |
|---|---|
| repo | [juspay/xor](https://huggingface.co/juspay/xor) |
| base | `Qwen/Qwen3.6-35B-A3B` (bf16, ~70 GB) |
| format | fully merged BF16 (no adapter, no LoRA) |
| architecture | `XorModel` |

The Qwen3.6-35B-A3B MoE backbone loads via the existing
`qwen3_5_weights.cpp` loader. The vision tower weights and decision head
weights are included in the merged checkpoint.

## Run it

```sh
build/examples/vllm-server --model /path/to/xor --port 8000
```

Then send a systemone request:

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"john works at google",
       "questions":{"pick":{"type":"choice","instructions":"entity type",
       "criteria":{"person":null,"organization":null}}}}'
```

Forward+reverse evaluation doubles inference cost per question. For a 35B
MoE this is significant but inherent to the method. Prefix caching of the
shared state prefix across the two passes is deferred. Confidence uses the
shared kev formulas (`ChoiceConfidence`, `ScoreConfidence`); probabilities
are rounded to 2 decimals.

## What has not been measured

No token gate result exists for this checkpoint yet. No byte has been
fetched or hashed. The golden tests (softmax, forward+reverse calibration,
confidence formulas, prompt construction) run on synthetic weights.

GPU (CUDA) build verification, GGUF k-quants, E2E parity vs the SGLang
oracle, and prefix caching are owed.

Spec: [`.agents/specs/xor.md`](../../.agents/specs/xor.md)
