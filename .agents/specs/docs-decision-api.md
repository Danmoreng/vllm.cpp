# Decision API documentation

## Scope

Describe the shipped ABI 29 decision API and announce the new decision models.
This is a documentation repair at upstream main `8205abd17`, with no runtime,
model lifecycle, or benchmark change. No GPU or model download is required.

| Item | Source | Public document | Verification |
|---|---|---|---|
| ABI and ownership | `include/vllm.h:372,1180`; `src/capi/vllm_c.cpp:1695,1903` | `docs/reference/c-api.md` | Header declarations, argument checks, allocation, and free implementation |
| Architecture dispatch | `src/capi/vllm_c.cpp:1713`; `src/vllm/model_executor/models/tev1_registry.cpp:8` | `docs/FEATURES.md`, C API reference | Exact allowlist and Tev1 generation path |
| New models | `src/vllm/model_executor/models/clm_registry.cpp`, `gliner25_decide_registry.cpp`, `xor_registry.cpp`, `tev1_registry.cpp` | README news | Registrations, server handlers, and existing model recipes |

## Design

Add a short README news item linking the decision model recipes. Distinguish
Tev1's chat completion path from the System-1 endpoint. Describe availability
without adding correctness or speed claims.

Correct the C API overview's stale version and document `vllm_decide`, its
request shapes, returned JSON ownership, and failure behavior. Link existing
HTTP examples instead of duplicating payloads. Limit any C snippet to declared
public functions and make its engine and error-handling assumptions explicit.
Clarify the `noul` decision versus NER behavior in `docs/USAGE.md`.
Consolidate the duplicate KevModel rows without dropping current evidence or
CUDA/GGUF limits. Correct the cua-s1 checkpoint repository from its pinned
weights record; distinguish the checkpoint from the `trycua/cua` code oracle.
Remove Tev1 from the capability row that attributes every decision model to
`vllm_decide`; retain Tev1 under the ordinary generation interface.

Keep prose short and preserve all existing performance caveats. Open PRs cover
GLiNER extraction at ABI 27, prompt log probabilities, EXL3, Qwen3.8, and
multimodal serving. Those topics are excluded. Do not change source or tests.

## Upstream chain and port map

This repair describes the checked-in C API adapter and server routing. It adds
no upstream port, kernel, or dispatch behavior. Existing model specs and recipe
pages retain their reference-engine evidence. No new oracle run or benchmark
applies to a prose-only change.

## Tests to port

None. Runtime tests and implementation remain unchanged. Verify every new
claim against the header, C adapter, server, and model registrations.

## Gates

Run README structure, supported-model, benchmark-index,
commit-style, and commit-trailer checks. Validate changed local links and
request examples. Run the full CPU-only preflight and distinguish unchanged
baseline failures from regressions. An independent reviewer inspects the
immutable implementation commit and checks broken-link or payload mutations
in scratch copies. The operator reruns the focused checks.

The initial baseline preflight reports unrelated missing issue records,
MiMoV2 checklist/support-list drift, and stale oracle-pin projections. Record
exact final dispositions rather than claiming a green full preflight.

## Dependencies and work breakdown

1. Commit this spec and its local issue before implementation.
2. A fresh implementer edits only README news, the C API reference, and the
   decision sections in `docs/FEATURES.md` and `docs/USAGE.md`.
3. A fresh reviewer checks the immutable change. The operator reruns gates
   and opens a pull request from the authorized fork.

Use an isolated worktree and local Python for document checks. No accelerator,
new weights, external compute, service management, or upstream merge is authorized.

## Risks and stop conditions

The header's explanatory model list can lag the adapter. Use the executable
allowlist for the documented model set. Do not imply that the HTTP NER fallback
or Tev1 is accepted by `vllm_decide`. Stop if a claim cannot be grounded in code.

## Owed

- ISSUE-LOCAL-01M3GD47P0MQRSHRCN8AP1NEXA: repair the public decision API documentation.
