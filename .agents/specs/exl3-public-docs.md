# EXL3 public documentation

## Now

Documentation correction for `QUANT-EXL3`, scoped to
`ISSUE-LOCAL-01M2CBGG169HDX0DWKHQWCBBGN`. The implementation row stays `ACTIVE`.
Base: `e030f1b90`. One pull request carries this spec and the documentation.

## Scope and inventory

| Surface | Current gap | Source and evidence | Change |
|---|---|---|---|
| README news | The older CUDA description omits mul1 and long-prefill dispatch | `src/vt/cuda/cuda_exl3.cu:2132`, `include/vllm/model_executor/models/dense_attn_block.h:315` | Replace the old EXL3 entry with a September capability update |
| `docs/FEATURES.md` EXL3 row | Attempt history obscures current support and contains a stale HumanEval obligation | `.agents/specs/quant-exl3-shared.md`, `.agents/specs/quant-exl3-perf.md`, `docs/benchmarks/qwen38-27b-exl3-gb10.md` | Concise support summary with evidence links |
| `docs/USAGE.md` EXL3 artifact entries | Current instructions mix with old attempts | Shared dense dispatch, EXL3 loader, and the quantization specs | Retain artifact pins, hashes, sizes, working paths, and refused arms. Remove obsolete chronology |

## Design and upstream anchors

This is an editorial correction, not a port. The existing `QUANT-EXL3` inventory
and committed implementation spec own the upstream surface and kernel gates.
The reference is exllamav3 at the revision in `.agents/oracles/exllamav3.md`.
`quant-exl3-shared.md` records the upstream dispatch and dependency chain.
Read the local loader, dispatch, CUDA instantiations, and CPU regression test
before editing their descriptions. Do not infer full model support from a kernel.

CUDA uses reconstruct plus cuBLASLt above 144 input rows when registered.
Other backends retain `Exl3Gemm`. Preserve the distinction between GEMM, GEMV,
and the DeepSeek-V4 fused MoE path. Do not generalize CUDA coverage to other
backends. Do not add new measured performance claims.

Preserve removed historical text in a dated file under `.agents/completed/`.
Keep the public feature row readable and link detailed constraints to the
existing specs. Change no benchmark values or benchmark dispositions.

## Tests and gates

No upstream tests to port: no executable behavior changes. No GPU, model
weights, external compute, or runtime benchmarks are required or authorized.
Use source inspection to verify every changed technical statement. Retain all
artifact filenames, sizes, revisions, and hashes byte-for-byte.

Focused CPU gate:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-quickstart-recipes.py
python3 scripts/check-benchmark-index.py
python3 scripts/check-agent-record.py
python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'
git diff --check
```

Run `scripts/agent-preflight.sh --quiet` and classify any failure against the
unchanged base. An independent reviewer checks the immutable implementation
commit and reruns the focused gate. Mutations apply only to executable test
claims. No new tests that merely copy the prose are needed for this edit.
The operator reruns the focused gate before the fork push.

## Work breakdown and authority

1. Operator commits this spec and the canonical local issue.
2. A fresh implementer edits only the three public documents above, the dated
   archive, and this spec's outcome. It records source anchors and gate output.
3. A fresh reviewer checks the immutable head without repairing findings.
4. Operator verifies and opens a pull request from the maintenance fork to
   `mudler/vllm.cpp:main`. Merge is not authorized by the user request.

## Risks and stop conditions

Do not promote a single completed benchmark round to an accepted headline.
HumanEval-style benchmark data is not a correctness gate. Preserve those limits.
Stop on an ambiguous source claim rather than guessing. Do not modify source,
tests, checkers, other rows, or unrelated documentation to repair baseline gates.
