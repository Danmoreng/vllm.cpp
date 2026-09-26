# GLiNER2.5-Decide

GLiNER2.5-Decide is a System-1 decision classifier. A DeBERTa-v3-large
encoder (already implemented for GLiNER2.5) extracts label embeddings, and
a classification head (`Linear(H, 2H) → ReLU → Linear(2H, 1)`) scores each
label in a single forward pass. It answers `choice`, `score`, and `noul`
question types through `/v1/systemone`.

GLiNER2.5-Decide runs through the shared paths, so [the usage guide](../USAGE.md)
covers the `/v1/systemone` request shape. This page carries the checkpoint
and what has not been measured.

## The checkpoint

| field | value |
|---|---|
| repo | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) |
| format | safetensors, F32, 486M parameters |
| architecture | `SpanExtractor` (same name as MODEL-GLINER25, with a classification head) |

The encoder config is read from `encoder_config/config.json` (DeBERTa v2).
The classifier weights load from the same safetensors checkpoint.

## Run it

```sh
build/examples/vllm-server --model /path/to/GLiNER2.5-Decide --port 8000
```

Then send a systemone request:

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"john works at google",
       "questions":{"pick":{"type":"choice","instructions":"entity type",
       "criteria":{"person":null,"organization":null}}}}'
```

The sequence layout is `[CLS] text [SEP] [P] task [L] label1 [L] label2 ...
[SEP]`. The encoder runs once; label embeddings are extracted at `[L]`
positions and scored by the classifier. Confidence uses the shared kev
formulas (`ChoiceConfidence`, `ScoreConfidence`).

## What has not been measured

No token gate result exists for this checkpoint yet. No byte has been
fetched or hashed. The golden tests (classification head, softmax/sigmoid)
run on synthetic weights.

GPU (CUDA) build verification, GGUF k-quants, E2E parity vs the GLiNER2
library oracle, and the constraint-feasibility decoding pipeline (`exact`
and `beam` decoders) are owed.

Spec: [`.agents/specs/gliner2.5-decide.md`](../../.agents/specs/gliner2.5-decide.md)
