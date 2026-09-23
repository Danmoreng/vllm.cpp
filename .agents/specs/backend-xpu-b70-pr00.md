# B70 XPU PR00: pinned inventory and operator trace

Row: `BACKEND-XPU`  
Issue: `ISSUE-LOCAL-01M37Y5NHZ5879YC8YD70QR4BD`  
Plan: [`docs/B70-SYCL-Qwen38-EXL3-Implementation-Plan.md`](../../docs/B70-SYCL-Qwen38-EXL3-Implementation-Plan.md), PR00

## Now

`SPIKE`. The pinned EXL3 checkpoint is present locally. PR00 must establish
its exact tensor inventory and an executable native versus fallback trace
before PR01 adds an XPU backend. This spec does not claim an XPU kernel.

## Scope

Add a read-only command that inventories the two Safetensors shards without
loading weights into a model or executing a runtime. Report tensor count,
stored bytes, dtype, shape, shard, and EXL3 component for each tensor. Report
totals for text, MTP, vision, embedding, head, and unclassified tensors.
Classify embedding and head before their broader text group. Print an error
for every unclassified tensor until its owner is established.

Read `model.safetensors.index.json`, both real shard headers, `config.json`,
and `quantization_config.json`. Validate the index and header name sets in
both directions, assigned shards, offsets, tensor byte sizes, duplicate
names, and the index's total byte count. Confirm the EXL3 bit widths and the
MTP tensor dtypes from the headers. Resolve the plan's 516 unspecified
tensors by listing their names and classes in the machine-readable result.
Generate the actual matrix descriptors from the checkpoint and compare their
count with the plan's 401. A count mismatch is a finding, not an adjusted
constant.

Add a machine-readable operator-provider trace to the existing provider
instrumentation. It must identify the selected native provider or CPU
reference fallback per operator and count real selections during execution.
The default runtime path must not write a trace. A separate microbenchmark
configuration consumes the inventory's matrix descriptors; it must not
register an XPU operator or claim performance before PR01.

PR01 owns SYCL, device allocation, queues, events, and the B70 matrix probe.
PR02 and later waves own model operators and end-to-end inference.

## Reference anchors

- The target local source revision is `9e63db5dd33b35e7cc57d0f0e80fe6c7d5ababa6`.
  The plan commit is `7190cd15f43bc4814669197debf2e892231565aa`.
  PR00 must record any difference that affects the inventory or trace.
- The model is `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` at
  `19441ac874c4018295da848e250f23511361cda4`. Verify the local
  provenance and both shard hashes before treating the headers as evidence.
- The current primary vLLM pin is in [`.agents/upstream-sync.md`](../upstream-sync.md).
  Read its XPU platform and external kernel dependency source at their
  recorded revisions. Name a missing source or revision as `PENDING`.
- Local header semantics come from
  `src/vllm/model_executor/model_loader/safetensors_reader.cpp::SafetensorsFile::Open`.
  Trace semantics come from `src/vt/op_provider.cpp::GetOp` and the provider
  statistics API in `include/vt/op_provider.h`.
- The plan cites a technical report by `[Lxx]` identifiers but does not
  include that report. PR00 must locate its source index or record the
  donor-source reconstruction as `PENDING`, without inventing revisions.

## Design

Read only the eight-byte header length and bounded JSON header of each shard.
Do not read tensor data for inventory generation. Reject a header length that
exceeds the file, out-of-range or overlapping spans, impossible dtype and
shape sizes, duplicate names, and index disagreement. Keep the result stable
across runs by sorting names and matrix descriptors. Include source pins and
checkpoint hashes in the output so another developer can reproduce the same
classification.

Reuse the provider registry and its existing counters. The trace must name
which operator and device each row describes, the selected provider, and
whether it is the CPU reference tier. Record the total reference-tier hits.
Keep the trace opt-in and preserve selection order and default dispatch.
Document exactly which counts are registrations, selections, or executions.
Do not label a registration count as a model call count.

## Tests and gates

1. Port or write the smallest fixture test first. Capture its intended red
   result, then green. Cover valid two-shard input, duplicate and missing
   names, wrong shard assignments, malformed lengths and offsets, dtype-size
   mismatches, and a total-size mismatch.
2. Run the inventory against the pinned real checkpoint. Assert the full
   index/header equality and record exact group totals, EXL3 bit-width
   distribution, MTP dtypes, and matrix descriptor count. Do not encode an
   unverified expected count into the parser.
3. Exercise the trace through the production provider lookup. Prove that a
   native selection and a CPU reference fallback are distinguished. Confirm
   no trace file appears without the opt-in setting.
4. Run the focused test suite, `scripts/agent-preflight.sh`, and applicable
   full CPU tests. An XPU hardware result remains `PENDING` until PR01.

A fresh reviewer must mutate one index/header consistency guard and the
fallback classification in a scratch copy. The focused tests must fail.
The reviewer must also remove the production trace call site and show that
its focused test fails. Restore the reviewed tree byte-for-byte.

## Risks and stop conditions

The 15 GB checkpoint can make an accidental data read expensive. The inventory
must use bounded header reads. The parent directory also contains unrelated
models, so the command accepts an explicit model directory and never selects a
checkpoint by a fuzzy name. A missing pinned source, shard, or revision is
`PENDING`; it is not replaced by a nearby revision. A count or dtype mismatch
stops claims about model support and remains visible in the output.

Do not download model assets, install a toolchain, run GPU work, or manage a
service as part of PR00. PR01 starts only after the PR00 inventory and trace
are reviewed and the operator reruns the gate.

## Evidence

The implementation report must give the immutable commit, exact commands,
exit codes, model shard hashes, inventory result path, focused red and green
outputs, full gate output, and review mutations. Distinguish observed header
facts from plan estimates. The operator records the PR00 outcome here after
the gate.

## Owed

PR01 in the plan owns the native XPU backend and B70 probe. This PR00
inventory and trace do not make XPU execution reachable. The `BACKEND-XPU`
row owns that wiring.
