# CLM

CLM (Contrastive-LM) is a bi-encoder System-1 decision model. A frozen
Qwen3-8B backbone extracts last-token hidden states via `ForwardHidden`,
and two MLP projection heads — a state head and an action head — project
state and candidate embeddings into a 512-d space. The score is
`exp(logit_scale) * cos(state_proj, action_proj)`, softmaxed over a
question's candidates. It answers `choice`, `score`, and `noul` question
types through `/v1/systemone`.

CLM runs through the shared paths, so [the usage guide](../USAGE.md) covers
the `/v1/systemone` request shape. This page carries the checkpoint setup,
the confidence formula, and what has not been measured.

## The checkpoint

| field | value |
|---|---|
| repo | [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) |
| base backbone | `Qwen/Qwen3-8B` (bf16, ~16 GB) |
| projection heads | `CLM_v0.1-8B.pt` (torch pickle, ~80 MB), converted to `head.safetensors` + `meta.json` via `scripts/convert-clm.py` |
| architecture | `ClmModel` |

The projection heads ship as a torch pickle. `scripts/convert-clm.py`
extracts the head config (`width=1536`, `depth=3`, `activation=gelu`,
`layernorm=true`, `residual=false`) from the checkpoint's `cfg` dict and
writes `head.safetensors` + `meta.json`. The base backbone loads from its
own safetensors via the existing Qwen3-8B loader.

## Run it

```sh
build/examples/vllm-server --model /path/to/CLM-v0.1-8B --port 8000
```

Then send a systemone request:

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"john works at google",
       "questions":{"pick":{"type":"choice","instructions":"entity type",
       "criteria":{"person":null,"organization":null}}}}'
```

CLM uses its own confidence formula: `max(0, min(1, p_max - mean(rest)))` —
a margin, not kev's `(max(p) - 1/K) / (1 - 1/K)` and not Laya's
`1 - H(p)/log(k)`. Probabilities are full-precision floats, not rounded to
2 decimals (unlike kev/Laya).

## What has not been measured

No token gate result exists for this checkpoint yet. No byte has been
fetched or hashed. The golden tests (13 cases: MLP head projection, L2
normalization, scaled cosine scoring, logit_scale clamp, bi-encoder
pipeline, confidence formula) run on synthetic weights.

GPU (CUDA) build verification, GGUF k-quants, E2E parity vs the CLM
Python reference, and action embedding caching are owed.

Spec: [`.agents/specs/clm.md`](../../.agents/specs/clm.md)
