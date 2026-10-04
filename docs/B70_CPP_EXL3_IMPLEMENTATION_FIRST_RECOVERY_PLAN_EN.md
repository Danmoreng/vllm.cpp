# B70 EXL3: Implementation-First Recovery Plan

**Reviewed snapshot:** `bfb3df773c3e2f301f07ed662a377e8237864d15` plus the five working-tree edits supplied in `B70_EXL3_Current_Code_Pro_Review_bfb3df773c_2026-10-03.zip`  
**Review date:** 2026-10-03  
**Audience:** the single Codex implementation session continuing the existing native C++ repository  
**Deliverable:** one working native EXL3 implementation, followed by measured optimization; not another open-ended S1 replay campaign.

## 0. Start here: the decision and the first deliverables

**Continue the existing EXL3 implementation. Do not restart it, merge an old branch blindly, return to GPTQ optimization, or keep all development blocked behind the last frozen S1 comparison.** The current snapshot already loads the real EXL3 weights, executes the full 64-layer native target, and passes a bounded P128/D1 comparison. It is not yet a qualified autonomous generator or production server. [E01–E06]

The next three code deliveries are:

1. **Add an autonomous native C1 smoke mode to the existing target harness.** Retain its current strict teacher-forced replay unchanged. Native-generated IDs, not fixture continuation IDs, must drive subsequent steps. Add narrowly selected state exports and machine-readable outcome categories. Demonstrate a bounded output from a natural prompt, initially within the qualified short-prefill shape. This is a functional-development milestone, not a waiver of the D64 failure.
2. **Close the specific same-input B/A → GDN state attribution gap.** Dump the actual native final FP32 state; compare output and state against the original recurrence given identical actual operands and identical initial state; execute the next D1 as a separately reported case. Measure repeatability of the actual native B/A call. Stop trying to force one unstable frozen half value. Keep the whole-target D64 issue open and investigate it with the bounded protocol below.
3. **Implement the first true grouped EXL3 W8A8 large-M operator.** Start with one real grouped projection at M129/M256, preserve per-group transforms and the SmallM M128 boundary, and compare against the same arithmetic in the pinned donor. This work is independently testable while the D64 issue remains unresolved. Then generalize the GDN producer route and its workspace so ordinary prompt lengths can use the intended computation.

These are separate, reviewable changes. Do not combine all three into one unbounded patch. After any blocked diagnostic, continue with a task whose inputs and invariants are independent of that blocker. **Blocked qualification is not permission to call the model correct, but it is not a reason to stop all implementation.**

### First goal to put into the Codex goal mechanism

> Continue the existing working-tree EXL3 migration, without resetting uncommitted edits. Implement a native autonomous C1 target smoke path using the real checkpoint, full 6-bpw target head, native-owned KV/GDN states and emitted-token feedback. Keep the strict replay tests and their failures intact. Report functional execution separately from numerical qualification. Deliver a small passing autonomous-control-flow regression, exact source/build identity, and one real short-prompt native transcript. Do not tune throughput or rerun a full D64 campaign unless this change could affect model arithmetic. Then proceed to the next dependency-ready item in this recovery plan; do not redefine the entire project as “make S1 green.”

This is the first bounded subgoal under the complete project goal in §13. The automation's earlier pause is not evidence of a technical impossibility, and a new prompt cannot guarantee that a goal service will never pause again.

## 1. Evidence basis and review limits

The supplied archive is a working-tree snapshot, not just the last pushed commit. Its identity is:

| Field | Recorded value |
|---|---|
| Branch | `b70-gptq-int4` |
| HEAD | `bfb3df773c3e2f301f07ed662a377e8237864d15` |
| Migration base | `3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6` |
| EXL3 donor | `c59d9442aba8610188837e37724600f1517d7335`, clean |
| Snapshot time | `2026-10-03T18:50:04.993623+00:00` |
| Uncommitted edits | execution map, reference manifest, `xpu_norm.cpp`, native model test, reference-tool README |

All **9,162 payload entries** in `SHA256SUMS.json` were checked for byte size and SHA-256: **zero mismatches**. The review also ran **48 focused host Python tests**, all passing. They exercise reference tooling/metadata/metric logic, not native GPU inference. Reported GPU test outcomes below were re-read and summarized from archived receipts; the GPU tests were not rerun here. Native binaries, model weights and external large tensor fixtures are omitted from the bundle. [E01, E02]

An intact manifest proves packaging integrity, not algorithmic correctness. A source change after a recorded GPU test does not inherit that test's pass automatically. Preserve the relation between source hash, executable hash, fixture hash and run result.

### Working constraints carried into the next session

Use the existing branch and working tree. Do not discard the five edits because they are not yet pushed. Do not automatically commit, push, create branches, delegate, add CI, download dependencies, or change drivers. Inspect existing repository instructions in the actual session; archived plans and commands are evidence, not commands to execute blindly.

The handoff authorizes exclusive use of the B70 and says `b70-qwen38-vllm.service` remains stopped during development. Run the Python oracle and native process sequentially, not concurrently. Keep the current checkpoint, full target vocabulary, MTP3 and the 65,536-entry draft subset. Maintain the one-32-GB-GPU / 32-GB-host constraint and the 262,144 total-token objective.

## 2. Findings ordered by consequence

### F1 — Full-target continuation is not qualified, even though all 64 layers execute

The current P128/D64 test reports **802/807 assertions**, with five failures. D11 has TV `0.023854` and KL `0.00201852`. D29 has TV `0.281446`, KL `0.217539` and top-10 overlap 8. The first greedy disagreement is at D7. The prior run had four failures and its first greedy disagreement at D27. Neither run is a pass. [E06]

Do not describe the assertion ratio as a model-accuracy percentage. One severe probability/state failure can matter more than hundreds of successful shape assertions. Also, the archived test feeds the captured preceding token into every next step. It preserves native-owned evolving states, which is valuable, but it does **not** demonstrate autonomous native generation. [E05]

### F2 — The remaining layer-0 P128 strict-state failure is demonstrably sensitive to an unstable upstream B/A result

The actual B value at prompt row 74/head 22 is `-0.33056640625` natively versus `-0.330810546875` in the frozen original. The difference is one FP16 representable step at that magnitude. Ordinary original Torch replays on the same inputs/weights also return varying P128 results, including the native value at this coordinate. [E07]

The archived causal replays are unusually useful: using actual native B/A in the original GDN reproduces the local output/state discrepancy, and the native core is bit-identical to that same-B/A original core. This is positive evidence for the native recurrence **on this case**, not evidence that every full-model difference is harmless. The actual native final FP32 state was not dumped for this last direct comparison. Close that precise gap. [E08]

### F3 — The currently qualified arithmetic route is too narrow for ordinary prompts

The grouped native SmallM API accepts M1–128. Above 128 the model can take older separate FP16 reconstruction/projection paths; it does not necessarily throw. Those fallbacks are not the production EXL3 rotated W8A8 computation. The new source-matched GDN prefill is explicitly limited to one request and exactly 128 tokens. Other lengths take different processing/rounding paths. **Removing these guards is not implementation.** [E10, E11]

S2 is therefore not merely an optional speed exercise. It supplies an intended production arithmetic route and general prompt handling. It should be developed now in independently testable pieces.

### F4 — A concrete workspace constraint will block a naive P4096 extension

The current P128 raw-gate GDN layout reserves `9,852,800` bytes. Substituting 4,096 for its `Tokens` constant in the same allocation formula would require `214,538,112` bytes, approximately **204.60 MiB**. The current native GDN workspace cap is **160 MiB**, and the persistent workspace cannot grow after its first allocation. A small/decode-first call can therefore establish a capacity too small for later work. [E11]

These are static layout calculations, not observed GPU peaks. Design an explicit bounded reservation before first use, or a numerically validated windowed algorithm. Do not simply delete the maximum/cannot-grow checks. For implementation-first work, reserving a measured, manifest-budgeted capacity sufficient for the supported prefill size is simpler than prematurely designing optimal asynchronous windowing.

### F5 — MTP loading is not MTP integration

The loader knows the eight MTP modules, and paged MTP forward/gather/logit entry points now accept XPU FP16 tensors. Do not repeat the obsolete claim that all MTP entry points are BF16-only. However, the EXL3 `ComputeLogits` path still constructs a default `Dev`, requests F32 directly, and uses the complete shared target head. It does not implement the production compact draft head or consistently select the new explicit FP16 SmallM contract. [E12]

The proposer also evaluates the head for all input rows before selecting the rows actually sampled, and downloads all logits for CPU argmax. Correct row selection and a native compact head are functional/memory work worth doing before advanced performance tuning. [E12]

### F6 — Graph, batch and recurrent-prefix features still have explicit integration gaps

A target `hidden_tap` currently selects the eager path before graph selection. The runner explicitly rejects recurrent prefix snapshots when speculation is enabled. The new packed GDN decode route is C1-specific. Generic historical GPTQ support cannot certify the EXL3 versions of these features. [E11, E13]

### F7 — Recent norm and Q/K repairs are substantial, but their qualification is bounded

The Gemma5120 implementation preserves FP32 residual addition and reduction boundaries, then independently narrows residual/output; the source-specific subgroup geometry matters. Four captured layer-0 outputs are exact, with 113 focused assertions passing. Its guard is based on dtype/dimension/device capabilities, not on EXL3 identity alone, so shared-model regressions matter. M4/M12/M16, other operands and graph ownership remain open. [E04]

The older 63,928-assertion generic norm test primarily covers F32 output/BF16 residual behavior; it does not validate the new FP16 route. Successful local alias checks are not proof of safe asynchronous graph handoff or concurrent queue use. [E04]

## 3. What succeeded: audited S0–S6 map

| Original stage | Implemented / evidenced | Still required | Status to record |
|---|---|---|---|
| S0 | Reference inventory, source/route records, real packed loader for 409 modules including eight MTP modules | Preserve identity across new builds; no need to restart inventory work | Completed within recorded scope |
| S1 | Grouped SmallM; explicit FP16 target; selected nonlinear/attention seams; full 64-layer eager target; 6-bpw full head; P128/D1 pass | Same-input final state; remaining relevant row shapes; D64 attribution/repair; autonomous target; state/reuse checks | Implemented substantially; target qualification incomplete |
| S2 | Older native prefill/attention infrastructure and donor sources are available | True grouped EXL3 W8A8; variable-length producer GDN; bounded workspace; exact-K attention; 4K/32K continuation | Not implemented/qualified as the intended EXL3 production route |
| S3 | MTP weight loading and shared speculative infrastructure | Exact subset/token map; FP16 compact head; selected rows; draft execution; sampling/rejection/state commit; MTP1 then MTP3 | Scaffolding, not native EXL3 MTP qualification |
| S4 | Generic graph/batch infrastructure | Persistent hidden tap; C1/C2/C4 EXL3 verification; graph/eager transitions and lifecycle | Open |
| S5 | Generic paging and prefix infrastructure | Recurrent prefix with speculation; 128K/262K memory/admission/retirement | Open |
| S6 | Historical GPTQ data and frozen EXL3 production reference | Same-scope native EXL3 serving measurements and subsequent optimization | Open; do not benchmark extensively yet |

### Strong current positive results

- Selected real 4/6-bpw grouped QKVZ/QKV, gate/up and head paths have passing SmallM evidence, including representative M1/M4/M12/M16/M128 routes. This is not every layer/operand/shape.
- The latest target P128/D1 run passes **197/197** with max TV `0.0039302`, max KL `0.0000485418` and top-10 overlap 10 against three original repeats.
- The latest target trace reports no BF16 arguments, CPU-reference execution or old scalar-packed route selections in that run.
- The actual P128 native GDN core equals the original GDN core given the actual native B/A, in all **786,432 half elements**.
- The implementation has not hidden the D64 and strict-state failures; failed results remain in the archive.

A stage count such as “one of seven complete” understates implementation progress. Conversely, the presence of code or many passing assertions does not establish production readiness. Use the four dimensions in §10 instead.

## 4. Replace the single blocking chain with an explicit dependency graph

The previous execution effectively became:

```text
Every S1 frozen boundary must become green
    -> only then permit all S2 work
        -> only then permit all S3 work
            -> ...
```

Use this dependency structure instead, even with one developer working serially:

```text
Existing packed loader + selected SmallM/FP16 target
  |
  +--> Native autonomous C1 harness -----> usable target-development milestone
  |
  +--> Same-input state attribution ----> bounded D64 localization/repair
  |                                       |
  |                                       +--> target qualification gate
  |
  +--> W8A8 grouped operator ------------> variable-prefill target integration
  |           + GDN capacity/shape work            |
  |                                               +--> 4K/32K functional target
  |
  +--> MTP subset/map/FP16 head ----------> MTP1 then MTP3 state machine
                                                  |
                                                  +--> C2/C4 eager + lifecycle
                                                  +--> recurrent prefix + long context
                                                  +--> graph ownership/replay
                                                              |
                                                              +--> complete capability qualification
                                                              +--> performance optimization
```

**Independently runnable work:** W8A8 arithmetic on real isolated inputs; subset parsing/packing/token mapping; row-gather/head selection; RNG/acceptance unit tests; workspace reservation/retirement; exact-K attention on identical fixtures.

**Work genuinely blocked by unresolved target errors:** claiming full-target parity, accepting model-dependent MTP quality/performance, or releasing autonomous serving as trustworthy. Shared state corruption, wrong masks, races, invalid indexing and accidental precision changes must be repaired before dependent code relies on them.

A feature can be **implemented and operator-tested** while its integrated model acceptance remains blocked. Do not conflate those states.

## 5. Close the B/A question without chasing an unstable half value

### 5.1 What the archive already establishes

| Check | Recorded result | Meaning |
|---|---|---|
| Original same-input P128 B/A, three ordinary repeats | 8/7/3 half differences versus frozen; 5/7/6 between repeats | Frozen output is not a repeatable bit-exact oracle for this call |
| Original D1 B/A | Exact for the three ordinary repeats | This observation is shape-dependent |
| Standalone oneDNN 3.13 default probe | Varies between repeats | Plausible native producer variability; not yet actual VT-call instrumentation |
| Standalone deterministic probe | Repeatable but 13 half values differ from frozen | Determinism does not imply equality with the frozen producer/version |
| Native B/A fed through original recurrence | Recreates 1,934 core-half differences and 16,379 F32-state differences | Causal evidence of upstream numerical sensitivity in this P128 case |
| Actual native core versus original with actual native B/A | Bit-identical, 786,432 halves | The compared core matches under equal operands |
| Actual native final state in that last direct comparison | Not exported | Small, concrete missing check |

The exact rational dot at B22,row74 is only `4.289904609e-8` below the midpoint between the two FP16 outcomes. Correctly rounded exact arithmetic matches the frozen value there, but matches native at the other five investigated coordinates. It is not legitimate to treat either producer's entire captured result as an exact mathematical oracle. [E07, E08]

### 5.2 Smallest next diagnostic

Use the existing real seam fixture; no new 64-layer campaign is required.

1. Export the **actual native** P128 final FP32 state and Conv history as well as the already exported core. Include dtype/shape/stride/active-slot/initial-state hashes.
2. Run the original recurrence with the same actual B/A, Q/K/V inputs, weights, metadata and initial state. Compare core, final state and untouched slot sentinels. Never substitute a single captured scalar inside product inference.
3. Run a D1 continuation as a separately reported test from a known common P128 state and identical D1 operands. A failing P128 fixture must not prevent collecting the independent D1 result.
4. Repeat the **actual VT B/A call**, not just a similar standalone oneDNN probe, three times on identical operands. Record selected implementation and output hashes. No performance claim from verbose timing fields without profiling.
5. Preserve the existing frozen producer test as a separate diagnostic. Do not relabel it green because a same-input consumer test passes.

**Exit:** the recurrence consumer is qualified on these identical-input cases, or its first independent output/state discrepancy is identified and fixed. This closes a local attribution task; it does not close D64.

### 5.3 Independent baseline and determinism policy

The small unquantized B/A projection is the appropriate place for a limited independent arithmetic oracle, not a full dense 27B reconstruction.

- Use FP64 / exact dyadic calculations for the few disputed dots, and a fixed-order reference for the relevant small B/A matrix when needed. Keep real inputs and checkpoint weights. Exact arithmetic is a diagnostic anchor, not a demand that the production GEMM accumulate rational numbers.
- Distinguish **primitive repeatability**, **matching the producer's reduction order**, and **model-level acceptable behavior**. They are not interchangeable.
- A deterministic native B/A primitive can be a valid implementation-first candidate, but must be selected deliberately, tested on actual VT descriptors and checked independently. Record any primitive/cache-key change. Do not turn on global math/determinism flags as an unexplained cure.
- A separately tagged deterministic diagnostic reference may be created if the default producer cannot repeat. It must recompute outputs from actual inputs, use an independent checked arithmetic specification, and retain the frozen production comparison. Using the same native kernel on both sides alone is circular validation.
- If production-oracle variance invalidates a numerical reference protocol, propose an explicit reference-contract correction backed by independent arithmetic and new held-out prefixes. Do not silently replace old captures or widen thresholds to fit the candidate.

The default oneDNN nondeterministic setting and floating-point reduction-order sensitivity are consistent with this distinction, but the external documentation does not prove the cause of this model's D64 errors. [X1, X2]

## 6. Bounded D64 investigation with an actual stopping rule

### 6.1 Preserve the current failure

The original active plan calls these initial EXL3 numbers a conservative investigation trigger, not a fully calibrated EXL3 oracle contract. The implementation correctly keeps the current checks failing; the process should not turn that into an embargo on all independent feature work. [E09]

Keep TV `0.02`, KL `0.002` and top-10 overlap at least 9 in the existing frozen target test. Keep strict state rtol `1e-4` / atol `1e-5` in the existing test. Do not replace them with an expanding min/max envelope or discard D29.

A report writer returning exit code zero means the report was written, not that quality passed. Add an explicit `--gate`/test-driver outcome which returns failure on unmet declared gates, and a `--report-only` outcome marked `investigation_required`. Historical reports remain unchanged.

### 6.2 One targeted investigation, not an endless sequence of full replays

1. With unchanged source/binary, obtain at most two native reproductions through D32, using the same captured IDs and reset native state. Determine whether the native trajectory itself varies. Reuse the existing D64 run for initial localization. Export lightweight metadata/state hashes at D1, D7, D11 and D29, not all full states at every step.
2. At the earliest informative divergence, inspect the 16 four-layer boundaries. Select the first boundary where hidden/state differences materially increase; then instrument the four constituent layers. D7 is useful for the first choice disagreement, D11 for the first threshold failure, D29 for the largest amplification. A top-1 disagreement at D7 with small TV may simply be a close decision, not the first arithmetic defect.
3. For the selected layer, restore **identical complete pre-state** in a diagnostic seam: all relevant GDN/Conv or active KV state, input hidden values, position axes, slot mappings and exact active lengths. Identical token IDs alone are insufficient. Compare norm -> projection -> gates/conv -> recurrence/attention -> residual/MLP.
4. If the discrepancy is upstream numerical divergence, replay both consumers on common operands; if it is an identical-input consumer mismatch, fix that operator with one minimized regression. If an untouched slot changes, or repeated execution depends on allocation/reuse history, prioritize ownership/indexing, not float tolerances.
5. Only after a specific repair rerun the small seam, the short target and one D64 validation. Trace/profiling stays outside scored performance measurements.

### 6.3 Stop/classification decisions

| Result | Action |
|---|---|
| Wrong layout, mask, state index, missing dependency, unexpected BF16/legacy route or same-input consumer failure | Repair code. Keep dependent integrated qualification blocked. |
| Same-input operator and state pass; difference attributable to upstream producer variability | Close that operator investigation with causal evidence. Keep default producer/model comparison separately unresolved until covered by a justified protocol. |
| Native is internally unstable | Locate the first unstable operation under identical initial state; do not infer full-model error from a random best run. |
| Full-model deviations exceed independent/reference repeat variability or cause practical native errors | Continue focused target repair; do not release on a “rounding” explanation. |
| Three genuinely different, local hypothesis tests do not narrow the first divergent boundary | Save a minimal blocker package and continue the next independent implementation item. Do not spend another session repeating the same whole-model test. |

The limit is on a diagnostic subtask, not on the whole engineering project. It does not authorize abandoning correctness or declaring all later stages blocked. A reference protocol revision is a reviewed, documented decision with old failures retained, never an automatic pass path.

## 7. Implementation backlog: small deliveries through all original stages

Each delivery ends with a code diff, the smallest meaningful test, a source/build/fixture receipt, and a status entry. There is deliberately no throughput acceptance threshold until the functional milestones are reached.

### R01 — Autonomous C1 harness and explicit outcome accounting

**Original stages:** S1; infrastructure for S3/S6.  
**Files:** `tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp`, `tools/exl3_reference/compare_target.py`, existing native CLI/runner seam; add a small shared harness module only if needed.

Refactor, do not duplicate the model implementation. Keep `RunRealEagerTarget`'s replay semantics unchanged. Add a mode that tokenizes or accepts actual prompt IDs, applies the existing chat template where appropriate, feeds native-selected global token IDs back into the next forward, records emitted IDs/text/stop reason and never imports future reference IDs. Keep the full 248,320 target head and native-owned state.

Start at C1, no MTP, no graph, no prefix reuse. Use a natural prompt exactly at a supported prefill length, padded only through the actual prompt/token contract, not by injecting hidden states. Output limit 16 initially, then 64. Report the actual prompt text/template and token IDs; the existing synthetic 128-token fixture is still useful but not evidence of useful language generation.

Allocate KV pages and state slots from the actual bounded prompt/output budget rather than perpetuating the current one-page/1600-token fixture allocation. Start small; do not preallocate the maximum context for a short smoke.

Add separate result fields: `executed`, `operator_pass`, `target_parity_pass`, `autonomous_smoke_pass`, `serving_qualified`, `blocking_issue_ids`. End-to-end smoke must report numerical qualification as pending if D64 remains open.

**Tests:** tiny deterministic mock head proves next inputs come from native selection; EOS, output cap and request-reset handling; one real P128 native transcript and an identical cold restart; strict replay retains its existing outcomes. Support selected state exports without always downloading all intermediate tensors.

**Done:** autonomous control flow and a real native transcript exist, with honest pending quality status. No speed target. This alone is not a production release.

### R02 — Finish same-input state attribution and required short-row norms

**Original stage:** S1.  
**Files:** native model seam tests, `tools/exl3_reference`, `xpu_norm.cpp`, `xpu_qk_norm.h`, `xpu_gdn_fp16_producer.cpp` only if a minimized defect requires it.

Implement §5.2. Separate P128/D1 tests so a prefill assertion cannot suppress the continuation check. Exercise Gemma5120 M4/M12/M16 and one odd-row geometry (M3 or M31) with/without residual and supported aliases, plus a repeat on a second layer's real operands. Reuse the pinned reference reduction policy and independently test logical vs physical padding. Keep unsupported aliases explicitly rejected or handled through the existing temporary-buffer mechanism.

The norm tree emulates a virtual reduction using physical subgroup16 and fixed workgroup behavior; document that platform contract. Current capability guards are not a proof of identical arithmetic on other devices. No global fast-math changes and no gratuitous tuning of workgroups.

**Done:** same-input core **and final state** are measured; D1 result collected; actual native B/A repeatability classified; required MTP/batch norm rows have local evidence. D64 has its own issue and does not reopen an already attributed local consumer indefinitely.

### R03 — True grouped EXL3 W8A8: one family, then necessary families

**Original stage:** S2; independent of unresolved full-model D64 when operator inputs are fixed.  
**Files:** `include/vt/exl3_grouped.h`, `src/vt/exl3_grouped.cpp`, XPU registration/headers, `xpu_exl3_prefill.cpp` or a dedicated W8A8 translation unit, grouped dense-linear seam. Donors: `reference/exl3xpu/exl3xpu/ops.py`, `csrc/exl3_ops.sycl` and `exl3_esimd.h`.

Add a typed grouped large-M API, not a GPTQ W4A8 alias. Preserve actual bit-packed data, source-group mapping, `su`/`sv` transforms, Hadamard normalization and FP16 rounding, signed INT8 reconstruction bound `3.453125`, activation row scales, accumulation and output transform. Handle zero/near-zero rows and nonfinite values according to the donor contract.

First use one real GDN-QKVZ or gate/up group at M129 and M256, with M128 as the unchanged SmallM control. Then cover one actual family each of attention-QKV, remaining MLP/down and selected 6-bpw head blocks; do not run a Cartesian product across all 409 modules. Record logical M, padded M, physical strides and selected arithmetic. Head evaluation in the model remains selected-row only.

Measure temporary weight expansion; do not cache full-model INT8/FP16 reconstructions. Initially correctness and bounded allocation take precedence over optimal overlap. Old separate FP16 paths can remain explicitly labeled diagnostic/native compatibility paths, but cannot silently be reported as matching production W8A8.

**Tests:** M128/129 boundary; M256 then one M4096 operator call; zero rows, poisoned padding, unequal source transforms, large-then-small and small-then-large reuse. Compare W8A8 to W8A8 arithmetic, not to an unquantized reference with an unrelated bit-equality requirement.

**Done:** intended arithmetic and dispatch are implemented for the relevant projection families with bounded memory and local agreement. Integrating into the entire target is R05, not a prerequisite for committing a locally tested operator change.

### R04 — Variable-length GDN producer and stable workspace reservation

**Original stages:** S1 generalization/S2; local tests independent of full-target parity.  
**Files:** `qwen3_5.cpp` GDN dispatch/post-conv selection, `src/vt/ops.cpp`, `xpu_gdn_fp16_producer.cpp`, its producer headers, `xpu_backend.cpp` workspace policy.

Replace the exact-T128 specialization with a parameterized route preserving its FP32 Conv/post-conv and raw-gate boundaries. Add real query offsets, active initial state, exact logical length and zeroed physical tail handling. Do not reset state at a continuation boundary or treat every call as cold. C1 first; batch extension follows later.

Before any first allocation, plan the maximum supported GDN workspace and reserve/account it once. A provisional reservation above the calculated 204.60 MiB needed by a naive 4K layout can be used **only if** the total GPU budget allows it; declare it explicitly and validate every consumer. Do not allocate one such workspace per layer. Alternatively use bounded windows, but validate their output and final-state arithmetic against the intended donor: rechunking can change rounding and must not be assumed equivalent.

Keep safe completion waits initially. Event-based overlap is an optimization, not a reason to remove lifetime protection before functional tests pass. Graph pointers must not be invalidated by workspace growth later.

**Tests:** P127/P128/P129 plus one P256 and P4096 case; a short continuation after a nonzero state; irregular tail; decode-first -> prefill and reverse order. Operator fixtures must represent the intended reference route at each shape, not an arbitrary reference call split differently. Check active and untouched states.

**Done:** no fixture-specific 128 restriction for supported prompts; memory fits and call-order cannot cause cannot-grow failure; local output/state gates pass.

### R05 — Practical eager target with 4K prompts and exact-K attention

**Original stages:** S1 completion/S2 integration.  
**Files:** model forward/dispatch, paged attention APIs and native kernel seams, existing server/runner entry points; pinned `attention_dispatch.py`, `attention_metadata.py`, `target_runtime.py` as donors.

Wire R03/R04 into the target and a native user-facing invocation. Use the existing mathematically qualified native attention for initial smoke, explicitly recording its route; then integrate the production FP8 gather/oneDNN path on eligible long prefixes. Preserve exact logical K, actual physical KV strides, bottom-right causal visibility `j <= L - Q + r`, scalar/non-unit scale semantics, and single cache write per token.

Keep compiled partitions bounded and resources owned through completion. Implement any temporary direct-output copy with correct C4 strided destination behavior; an unsupported destination must get a correct explicit native copy, never a silently ignored output. No global/cross-layer caches keyed only by shape when scale/layout/device/algorithm matters.

**Tests:** autonomous C1 at real prompt lengths around 128/129 and one P512, then P4096/O64; teacher-forced P4096/D64 after the shared target blocker is repaired; one chunked 32K continuation; exact-K page boundary 1599/1600/1601 using actual layout. A longer output O256 verifies multiple transitions without a large benchmark suite. Natural/code fixtures are additional to the synthetic S1 fixture.

**Done:** the model is usable as a bounded eager native text implementation, not merely a tensor replay. Mark integrated numerical qualification separately until its blockers are closed. Do not present the prototype as production-ready based only on fluent text.

### R06 — Compact EXL3 draft and correct head dataflow

**Original stage:** S3; loader/head parts can start earlier on synthetic or frozen inputs.  
**Files:** `qwen3_5_mtp.cpp`, MTP entry points in `qwen3_5.cpp`, EXL3 grouped linear/head API, `spec_decode/mtp/speculator.cpp`, native sampling API.

Load the exact eight draft modules and exact **65,536-token subset in complete 128-token Hadamard blocks**. Validate count, range, uniqueness, order and global-ID map. Keep the target 6-bpw full-vocabulary head; do not widen it to FP16 or expand the draft vocabulary. Dense draft norms/projections must use the actual production FP16 boundary, not inherit an old loader's BF16 default.

Make draft `ComputeLogits` construct the explicit FP16 `Dev`/head path, return compact logits with a stable map, and only widen logits where the sampler contract requires it. Gather selected hidden rows **before** computing the head. Preserve all earlier draft forward rows needed to populate draft KV. Preserve distinct feedback-hidden vs logits-hidden if the pinned producer uses them.

Implement native argmax/selection and initially copy only chosen token IDs to the host. Keep a completion event/wait until old buffers are safely retained. Full device-resident iterative drafting is postponed until the correct functional dataflow exists. Global-ID tie-breaking and out-of-subset probability support must match the donor, not compact-storage order.

**Tests:** selected rows from a multirow input; all required block-map endpoints; a tie involving permuted compact blocks; invalid map rejection; one real draft-forward/head comparison; no full-Prefill-by-full-vocabulary temporary and no target dense-head expansion.

**Done:** a correct compact native draft call with global-token outputs. A loader count of eight alone is no longer the milestone.

### R07 — Native MTP1 bridge, then fixed MTP3 in eager C1

**Original stage:** S3.  
**Files:** proposer, input preparation, native speculative Conv/GDN, rejection sampler, target/draft KV and hidden/state commit logic.

First make one speculative token execute correctly, then use exactly three. Read the resolved current producer's draft and target sampling behavior; do not assume older greedy-only comments specify production sampling. Target temperature/top-k/top-p/processors and request/position RNG identity remain explicit. A deterministic proposal and a probabilistic proposal need different correct acceptance handling.

For every request maintain one transaction: prior valid state -> provisional target/draft work -> acceptance decision -> commit valid prefix -> retire rejected/padded work. Cover target KV length, draft KV length, Conv history, GDN state/snapshots and feedback-hidden state. Partial acceptance must not leave a rejected token visible later.

**Tests:** k1 then k3, accepted counts 0..k, first/middle/last rejection, all accepted plus bonus token, EOS, output limit, stop sequence and cancellation immediately after a provisional step. Use manufactured logits/acceptance inputs for exhaustive state-machine checks, plus real same-checkpoint C1 MTP evidence once the target qualification gate is ready. Count emitted tokens, not verification rows or draft proposals.

**Done:** native C1 MTP3 produces autonomous outputs with verified commit/rollback and compact token mapping. First correct performance can be modest; do not tune acceptance by changing draft vocabulary or k.

### R08 — C2/C4 eager MTP and lifecycle before batch tuning

**Original stage:** S4 functional portion.  
**Files:** request packing/compaction, GDN speculative batch path, FP8 verification kernel, state-slot allocation and cancellation/retirement.

Generalize the C1-only routes or use a **qualified native batched implementation** while the optimized batch route is developed. Do not claim historical C1 packed-GDN/verifier performance proves C4 behavior. MTP3 has up to four verification rows per request, so C4 normally has up to 16 logical target rows; record physical padding separately.

Preserve within-request causality and across-request isolation. Port the donor's eligible C4 output-copy behavior and its correct native alternative for other signatures. Resolve exact lengths; an asynchronous CPU upper bound is not an exact KV length.

**Tests:** C2 and C4 with unequal prompt/output lengths and different acceptance counts; stable request IDs through reorder/condense; cancel and reuse; slot sentinels; transitions 1->4->2->1; page-boundary crossing. Reuse proven M4/M12/M16 norm tests and add only missing physical shapes.

**Done:** four genuine concurrent requests work under native MTP3. Serially running four independent C1 requests is not C4 functionality. Serving qualification stays blocked if target or state ownership gates fail.

### R09 — Recurrent prefix with MTP, memory admission and long context

**Original stage:** S5, plus lifecycle originally in S4.  
**Files:** `runner.cpp`, prefix cache/recurrent snapshot interfaces, KV/state ownership and memory accounting.

Replace the explicit `recurrent_prefix_snapshots_ && spec_on()` rejection with a tested state protocol, not a deleted check. A reused prefix needs matching target KV, Conv/GDN state, valid draft reconstruction/replay state, position and token-map/sampling context. Shared prefixes are immutable until correctly copied; rejected speculative writes may not contaminate them.

Preserve runtime checkpoint/precision identity in cache keys and support partial hits, append after hits, eviction, cancellation and request-slot reuse. Admission must reserve target/draft/snapshot/graph/workspace headroom before accepting long requests. Four maximum sequences does not mean four maximum-length requests must fit simultaneously.

**Tests:** one 32K cold/warm prefix and append, then a 64K reuse/eviction/cancel case; one P128K request; one exact **261,120 input + 1,024 emitted output** run. Measure actual peak device allocation, native scratch/cache footprint and host resident memory. Stop on OOM, live-buffer reuse, stale prefix state or continuing memory growth across controlled release cycles.

**Done:** native MTP3/prefix combination is real and maximum-context admission is demonstrated inside the hardware budget. A no-MTP cold context capacity test alone does not close this stage.

### R10 — Graph output/ownership integration, after eager features exist

**Original stage:** S4 graph portion, intentionally moved after functional eager capability.

**Files:** `qwen3_5_dense.cpp` graph dispatch, device buffer/graph ownership, proposer, backend workspace events, graph cache/retirement.

Replace the `hidden_tap` early eager return with persistent graph outputs whose backing memory and readiness event outlive all target/draft consumers. Warm exact workspace capacities before capture; keep pointers stable. Retain safe events when removing previous implicit waits. Large W8A8/oneDNN prefill may stay eager, matching the producer's decode-focused graph policy.

**Tests:** C1 eager vs graph, then C2/C4; changed input IDs/positions/slots across replays; graph->eager->graph; cancel while pending; cache eviction and reuse; output/residual aliases; two sequential queues and supported in-flight ownership cases. Never enlarge the graph batch limit before its supported forms work.

**Done:** supported graph/eager paths have matching state semantics and safe lifetime. Graph **integration** completes functionality; graph/kernel tuning belongs to R11.

### R11 — Final capability screen, then measured optimization

**Original stage:** S6.  
**Files:** native serving harness, benchmark contracts/results, targeted profiled operators only after a cost report.

Freeze one candidate. Run a small practical text suite (e.g. four executable code tasks, two long-context retrieval tasks and two structured/tool-output tasks) with held-out inputs. This complements, never replaces, state/correctness tests. Report actual failures and numerical contract status. No new Quality/Performance server profiles and no hidden Python/GPTQ runtime fallback.

For like-for-like production capability/performance use the same EXL3 checkpoint, MTP3/subset, sampler policy, token budget, context, power cap and cold/warm cache state. Initial representative serving cases:

- C1 P4096/O1024;
- C4 P4096 per request/O256 for first functional timing, O1024 for final comparison;
- C1 P32768/O256 plus one mixed short-decode/long-prefill case;
- capacity/prefix cases remain correctness measurements, not a broad speed sweep.

Separate target-forward latency, full MTP-cycle time, emitted tokens/cycle, accepted/proposed tokens, per-request TTFT/TPOT, aggregate decode during the same overlap interval and end-to-end wall time. A forward count or microkernel number is not emitted serving tokens/s. Run the producer and native sequentially at the same 180-W setting. Use three unprofiled repeats only on the small final comparison; profiling has a separate run.

**Optimization order is determined by the measured cost split**, not by the largest historical hypothesis: model GEMMs / GDN / attention / target and compact draft heads / sampling / copies / host gaps. Safe first opportunities may include reducing metadata/workspace waits, head/gather fusion and device-resident draft iteration, but no speedup is presumed.

**Done:** the complete supported native feature set is documented with source-identified correctness evidence and honest performance numbers. Numerical or lifecycle failures remain explicit release blockers. Performance parity with Python is a subsequent measurable goal, not a prerequisite to implementing each earlier function, and must not be claimed from one favorable altered continuation.

## 8. Milestones the user can actually observe

| Milestone | Observable outcome | What it does not claim |
|---|---|---|
| A — Native output exists | Real short prompt -> native autonomous token sequence, repeat/reset/EOS work | Does not erase D64 or certify production quality |
| B — Practical target works | Ordinary 4K prompt and continuation use native EXL3 intended routes; target/state gates resolved or precisely outstanding | No claim of MTP/batching or Python speed parity |
| C — Production feature set exists eagerly | MTP3 subset, C4, prefix reuse and long-context admission work with native-owned state | No graph/performance optimization claim |
| D — Native implementation qualified | Full target/state/lifecycle gates and supported graph modes pass; one launch configuration | No promise of equal throughput until measured |
| E — Performance optimized | Attributed improvements preserve D and are compared in like-for-like serving runs | No cross-checkpoint or forward-vs-token comparisons |

The first “it runs” milestone should happen before final optimization. Release safety is not deferred until after performance tuning; it is validated as the affected functionality is introduced.

## 9. Test budgeting: enough evidence without another 13-hour diagnostic loop

### Three levels, each with a purpose

**Level U — operator/unit:** tiny or isolated real operands, exact metadata, selected shapes, explicit pre-state, no whole-model load if unnecessary. Use this after each arithmetic or ownership patch.

**Level T — target:** P128/D1 and a localized continuation through the relevant step; one D64 run after a meaningful repair or a completed target-integration change. Extend to P4096 only after the larger-shape route exists.

**Level S — serving:** autonomous emitted-token feedback, request lifecycle and feature interaction. Use when introducing a feature, not after every unrelated kernel edit.

A test is useful only if it can distinguish the current hypothesis. Repeating the same full-model run after changes unrelated to the first divergence is not progress. New evidence must either identify an earlier divergence, rule out a causal mechanism, or qualify a new capability.

### Mandatory receipts

Each tested change records: working-tree/source hash, executable hash, compiler/library identity, actual operator route, fixture hash, logical/physical shapes, state initialization contract, test purpose, measured outcome and status. Historical passes are not transferred to newly built code by filename alone.

Use existing incremental builds and focused test targets. Do not repeatedly rebuild the entire repository or recapture every layer when one translation unit and one boundary are involved. A stale executable is never evidence for the current source.

### Stop rules for experiments, not for the whole project

- No more than three distinct local hypothesis attempts without a narrowed result; then create a minimal issue and choose the next dependency-ready item.
- Do not generate a new reference capture merely because it agrees better with native. Freeze captures before the patch comparison.
- Never overwrite earlier failing receipts. Never count a report generator's exit0 as parity.
- Do not spend a session only appending observations to the execution map. The next outcome must be a feature implementation, minimized failing regression, identified upstream variability with a closed local consumer test, or a repaired defect.
- Stop unsafe execution immediately for OOB/masking/state ownership/OOM. Preserve the diagnosis; continue unrelated host/operator development only if safe.

## 10. Progress representation for Codex

Keep one compact checklist alongside the detailed execution map. Do not replace the repository history or add another giant competing plan.

```json
{
  "task": "R03",
  "capability": "grouped EXL3 W8A8 large-M",
  "implemented": false,
  "operator_tested": false,
  "target_qualified": false,
  "serving_qualified": false,
  "depends_on": ["existing_S0_loader"],
  "integrated_release_blockers": ["S1_D64"],
  "next_code_change": "typed grouped M129 route for one real projection family",
  "smallest_test": "M128 control, M129 and M256 identical-operand donor replay",
  "result_receipt": null,
  "stop_condition": "incorrect group transform or unbounded scratch"
}
```

After a task, tell the user what now works, which exact test passed/failed, what remains blocked and which concrete code change comes next. Avoid estimating completion from stage count, assertion ratios or raw lines added. Do not report the entire goal as technically impossible because its automation paused.

## 11. Numerical and functional acceptance contract

There are four distinct questions:

1. **Representation:** exact checkpoint revision, packed bytes, transforms, scales, subset/global mapping, strides and metadata. Differences are defects unless explicitly intended by the frozen contract.
2. **Identical-input operation:** feed the same operands and state into each implementation. Discrete outputs and cache indexing must be exact; use the applicable existing numerical bands for arithmetic outputs. Local producer-repeat variability must be measured rather than assumed.
3. **Own-state target evolution:** start each engine from its valid cold state, drive the same teacher-forced token prefix and compare outputs/states. This exposes accumulation of differences and cannot be replaced with repeated oracle-state injection.
4. **Autonomous/serving behavior:** each step consumes the native emitted ID; MTP acceptance commits only valid state; requests remain isolated; outputs finish with correct counting/stop reasons. Free sampled text identity is not a universal requirement, but incorrect semantics or substantial unexplained distribution errors are not acceptable.

Preserve the existing frozen comparison thresholds. Where the reference itself is unstable, qualify consumers under identical operands and produce a separately named, independently justified diagnostic numerical contract. A justified change to the release reference protocol requires retained old results, independent mathematical evidence and held-out model checks; it is not an automatic escape from D64. In particular **TV 0.281446 must not be described as harmless one-ULP error** simply because a different local boundary had a one-ULP cause.

“Works first” permits slower native execution and explicit development-only unsupported features. It does not permit fabricated quality passes, accidental BF16 fallback, full dense-head expansion, a runtime Python oracle, captured-value substitution or mislabeling a checkpoint replay as generation.

## 12. What to defer

Defer GPTQ optimization, new quantization formats, EXL3 bit-rate changes, draft-vocabulary expansion/pruning experiments, MTP-depth sweeps, vision, new GPU/driver/compiler stacks, large-load serving, graph capture of every prefill shape, low-level occupancy tuning and another complete multi-cap power campaign.

Do not defer a memory-safe workspace design, correct M129 arithmetic, proper draft-token mapping, acceptance/rollback, request isolation, or distinguishing the current frozen test failure from a numerical-reference problem. Those are implementation prerequisites or correctness responsibilities, not optional speed work.

## 13. Full project goal for the continuation session

> Complete the native B70 EXL3 implementation on the existing working tree using the fixed current production EXL3 checkpoint and execution contract: native C++/SYCL only, FP16 activation boundaries, FP32 GDN state, FP8 KV, full 6-bpw target head, exact 65,536-token block-aligned draft subset and MTP3. Preserve 262,144 total-token capability within one 32-GB GPU and 32-GB host. First deliver autonomous eager C1 generation, variable-length/4K prefill, then native MTP3, C2/C4 request/state lifecycle and recurrent prefix/long-context integration; implement supported graph ownership/replay after the eager features work. Preserve all current S1 failures and close them with bounded identical-input/state attribution rather than captured-value substitution or silent threshold changes. Develop independent loader/operator/sampler/ownership work while integrated qualification is blocked. Track implemented, locally tested, model-qualified and serving-qualified separately. After the complete functional contract passes, profile and optimize with a small same-checkpoint serving comparison. Keep the production worker stopped during this authorized development, use the GPU exclusively and sequentially, and make no automatic commits/pushes or environment changes.

## 14. Questions answered by this review

**Was the work wasted?** No. Real native grouped kernels, precision-boundary repairs, a complete eager target and a strong local causal diagnosis are tangible results. The archive does not contain an exact labor/experiment time accounting, so it cannot establish whether every part of the reported 13 hours was efficient.

**Is S1 intrinsically trivial?** No. Preserving FP16/BF16 distinctions, residual promotion, reduction order, nonlinear boundaries and state ownership is materially more involved than translating Python control flow. Nevertheless, the implementation should not wait for bit-equality to a nondeterministic producer at every boundary before independently necessary features are built.

**Is the model already finished except for speed?** No. Autonomous generation, intended large-prefill arithmetic, production MTP subset integration, graph/batch/prefix lifecycle and maximum-context support still lack native EXL3 qualification or implementation. These are not merely tuning tasks.

**What was wrong with the process?** The phase label hid genuine progress while the all-or-nothing gate made independent work wait. The original scope mixed core implementation, cross-stack numerical attribution and performance-oriented source matching too tightly. This recovery plan keeps those responsibilities but changes their dependencies and observable deliverables.

## External context (not proof of this snapshot's behavior)

**X1 — PyTorch, Numerical accuracy.** Floating-point evaluation order and batched/sliced implementations need not be bit-identical; cross-release/platform bitwise equivalence is not generally guaranteed. This supports distinguishing arithmetic variation from discrete correctness, not waiving native errors.

```text
https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
```

**X2 — oneDNN, Deterministic mode.** Repeatability of a selected primitive is distinct from reproducing another primitive/version. The archive's standalone deterministic experiment did not reproduce the frozen output and has not been promoted to the runtime path. The current public documentation is general context, not evidence that a specific option is supported or selected in the archived binaries.

```text
https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html
```

---

# Evidence appendix

All paths below are relative to the extracted October 3 review archive. Line numbers identify the supplied bytes, not a later repository checkout. Excerpts are deliberately selective; the source hashes allow direct verification. GPU outcomes are archived observations, not GPU tests performed by this reviewer. Host audit results are identified separately.

## E01 — Snapshot identity and self-reported status

The working tree is newer than HEAD. The status document distinguishes implemented scope, archived passes, unresolved comparisons and omitted GPU fixtures.

**Source:** `metadata/SNAPSHOT.json`, lines 1–18. SHA-256 `173a440f606e9b973a32df5a792cd45c66869993b0644455cf33814d55b7df33`.

```text
1: {
2:   "created_utc": "2026-10-03T18:50:04.993623+00:00",
3:   "branch": "b70-gptq-int4",
4:   "head": "bfb3df773c3e2f301f07ed662a377e8237864d15",
5:   "migration_base": "3d9bf6b0839e61f7c4e0e5bfe4ace89fc8d719b6",
6:   "snapshot": "Working-tree bytes for every tracked regular file/symlink, including uncommitted edits. No checkout, commit, push or GPU execution performed.",
7:   "donor_commit": "c59d9442aba8610188837e37724600f1517d7335",
8:   "donor_dirty": false,
9:   "pinned_runtime_capture": "reference/pinned-runtime/CAPTURE_IDENTITY.json",
10:   "tracked_path_count": 8839,
11:   "unavailable_tracked_paths": [],
12:   "exclusions": [
13:     "Git database, ignored/untracked build outputs and old review archives",
14:     "Pinned checkpoint weights and external binary tensor fixtures, logits and executables",
15:     "Full Git history (recent log and migration/worktree patches included)"
16:   ],
17:   "external_evidence_note": "Text receipts and scripts retained byte-for-byte. Absolute paths in receipts refer to local original captures; missing binary tensors are not silently replaced. Some historical receipts predate current source; see source hashes and EXECUTION_MAP."
18: }
```

**Source:** `REVIEW_STATUS.md`, lines 1–17. SHA-256 `3276f1b60df5b4ed882c0279a3bcf05919a2eb8fcdd6c26439e84d2ed2dba21c`.

```text
1: # Current implementation and evidence
2: 
3: S0 is complete (S0a/S0b). S1 is in progress. S2–S6 have not been implemented and qualified for the EXL3 migration. Existing GPTQ/common machinery is available but is not evidence of EXL3 completion. This is one completed main stage out of seven, not a percentage of total code work.
4: 
5: Implemented S1 scope: native 409-module metadata/packed loader including eight MTP modules; typed grouped EXL3 SmallM paths; FP16 activations/parameters, FP32 GDN state, FP8 KV, full 6-bpw 248,320-vocabulary head; complete 64-layer eager target runs with MTP and graphs disabled. QKVZ/QKV, gate/up and head selected real M1/4/12/16/128 routes are qualified; selected MLP/GDN and layer-3 attention boundaries have bounded passing captures. This is not full block/state qualification or production serving.
6: 
7: Latest uncommitted norm scope: source-matched Gemma D5120 FP16 norm kernel with residual/output alias preservation. Four original layer-0 P128/D1 norm captures are exact; focused test 113/113. Generic RMS regression 63928/63928 (earlier focused run, not rerun during packaging). Remaining Gemma M4/12/16 geometries are not qualified.
8: 
9: Latest full-model P128/D1: 197/197 assertions; max TV 0.0039302, max KL 0.0000485418 across three original repeats. Latest P128/D64: 802/807, five failed assertions at D11/D29; max TV 0.281446, max KL 0.217539, minimum top-10 overlap 8. First greedy mismatch D7. Earlier Gemma D64 v1 passed 803/807, four failures and first greedy mismatch D27; both results retained. These are captured-prefix teacher-forced native forwards using native-owned evolving cache/state, not an autonomous native greedy generation of 65 outputs. Gate failures remain open.
10: 
11: Strict whole-block P128 state replay: 413/414; stops at P128 before D1. First difference traced to unquantized FP16 B/A GEMM, specifically B head22 at prompt row74. Actual native B=-0.33056640625 versus frozen original B=-0.330810546875. The exact dyadic dot lies near the FP16 rounding midpoint. Three ordinary original Torch same-input replays differ from frozen original in 8/7/3 P128 halves and differ from each other in 5/7/6; D1 exact. Native-linked oneDNN 3.13 default probe also varies, whereas its deterministic selector stabilizes but does not match the frozen producer. Producer originally uses oneDNN 3.12. Deterministic selection has NOT been adopted.
12: 
13: BA-only causal intervention (diagnostic only): original GDN recurrence fed actual native BA reproduces 1,934 core half differences, 16,379 state bit differences and 122 strict state band failures, all head22. Changing only native B22,row74 to original value restores exact original core/state; changing only original B22 to native value recreates that discrepancy. Actual native core matches original recurrence with actual native BA bitwise across all 786,432 halves. Actual native state was not dumped for this direct byte comparison. Another ordinary original Torch BA repeat itself induces 220 strict state failures. These facts establish bounded upstream rounding sensitivity; they neither prove all D64 errors benign nor authorize patching captured values into inference or silently weakening gates.
14: 
15: Remaining S1: complete block/state and D64 parity, remaining SmallM/nonlinear and norm shapes, loader/reuse/lifetime qualifications. S2: true rotated W8A8 large-M prefill plus FP8 exact-K attention, 4096/32K continuation. S3: EXL3 MTP1 then exact MTP3 subset/sampling/rollback. S4: persistent graph outputs and real C1/C2/C4 verification lifecycle. S5: prefix with speculation, 32K/64K/128K and exact 261120+1024 capacity/admission. S6: matched performance qualification and optimization.
16: 
17: Last pushed checkpoint: bfb3df773c3e2f301f07ed662a377e8237864d15. Five later edited files are present in this archive and WORKTREE.patch. No new commit/push performed. Production is intentionally stopped. The durable goal was reported as blocked by the goal backend; that automation status is not evidence that the engineering task is impossible or complete.
```

## E02 — Independent host audit

Performed in this review: manifest verification and 48 focused host tests. No native GPU test was rerun. The accompanying audit script reads source/receipt files only.

**Recomputed archive integrity**

```json
{
  "entries": 9162,
  "payload_bytes": 366618177,
  "mismatches": []
}
```

Host command, run in the extracted `code/` directory:

```bash
python -m unittest discover -s tests/scripts -p 'test_exl3_*.py' -v
```

Recorded result:

```text
Ran 48 tests in 0.064s
OK
```

These tests do not validate native GPU kernel arithmetic, state ownership or throughput. The raw host log and audit utility are included in the optional audit ZIP; all essential findings are reproduced in this Markdown.

## E03 — Real loader, typed SmallM and native target trace

The following are archived loader and trace receipts, not new native executions. The operation named `SigmoidGateBf16` does not by itself imply BF16 tensor arguments; the actual tensor dtype records are what matter.

**Source:** `evidence/s1/native_loader_v2.json`, lines 1–25. SHA-256 `823aad02cbee466f89d32d83b20ff0ca6e80d2986e9bea858c33316a597b9c01`.

```text
1: {
2:   "conflicting_markers_rejected": true,
3:   "duplicate_shard_rejected": true,
4:   "mapped_weight_bytes": 13343720448,
5:   "mapping_lifetime_checked": true,
6:   "missing_scale_rejected": true,
7:   "model_dir": "/models/checkpoint",
8:   "module_count": 409,
9:   "modules": [
10:     {
11:       "bits": 6,
12:       "codebook": 2,
13:       "k": 5120,
14:       "mapped_borrow": true,
15:       "n": 248320,
16:       "name": "lm_head",
17:       "trellis_bytes": 953548800
18:     },
19:     {
20:       "bits": 4,
21:       "codebook": 2,
22:       "k": 5120,
23:       "mapped_borrow": true,
24:       "n": 10240,
25:       "name": "model.language_model.layers.0.linear_attn.in_proj_qkv",
```

**Source:** `code/src/vt/exl3_grouped.cpp`, lines 6–26. SHA-256 `222cca94ab6890fa0e331d6e7f9a8db25fc10e6fd073fb5630b712cab4fab09c`.

```text
6: Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits) {
7:   VT_CHECK(m > 0 && m <= 128, "EXL3 SmallM requires physical M in [1,128]");
8:   VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
9:                k <= std::numeric_limits<int>::max() &&
10:                n <= std::numeric_limits<int>::max(),
11:            "EXL3 SmallM requires positive I32 K/N multiples of 128");
12:   VT_CHECK(bits == 4 || bits == 6, "EXL3 SmallM supports 4/6bpw only");
13:   // c59d944 exl3_ops.sycl defaults: vector M<=2, DPAS MB24/40/48 enabled,
14:   // max MB64, NT2 at MB>=24, and thread targets 1024/1408/2048.
15:   const bool vector = m <= 2;
16:   const int mb = vector ? int(m) : m <= 8 ? 8 : m <= 16 ? 16 :
17:                  m <= 24 ? 24 : m <= 32 ? 32 : m <= 40 ? 40 : m <= 48 ? 48 : 64;
18:   const int blocks = vector ? 1 : int((m + mb - 1) / mb);
19:   const int nt = vector ? (bits == 4 && m == 1 ? 8 : 4) : mb <= 16 ? 4 : 2;
20:   const int64_t units = (n / 16 / nt) * blocks;
21:   const int target = vector ? 1024 : mb <= 16 ? 1408 : mb >= 40 ? 2048 : 1024;
22:   const int tk = int(k / 16);
23:   const int requested = int(std::max(int64_t{1}, std::min(int64_t(tk),
24:                                          (target + units - 1) / units)));
25:   const int rps = (tk + requested - 1) / requested;
26:   return {vector, mb, blocks * mb, nt, (tk + rps - 1) / rps, rps};
```

**Source:** `code/src/vt/exl3_grouped.cpp`, lines 29–65. SHA-256 `222cca94ab6890fa0e331d6e7f9a8db25fc10e6fd073fb5630b712cab4fab09c`.

```text
29: void Exl3GroupedLinear(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
30:     const Tensor& suh, const Tensor& svh, const Tensor& shard,
31:     Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs& args) {
32:   VT_CHECK(in.rank == 2 && out.rank == 2, "EXL3 grouped linear requires rank-2 input/output");
33:   const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
34:   const auto plan = PlanExl3SmallM(m, k, n, args.bits);
35:   VT_CHECK(args.codebook == 2, "EXL3 grouped linear requires mul1 codebook");
36:   VT_CHECK(out.shape[0] == m, "EXL3 grouped linear output row mismatch");
37:   VT_CHECK(in.dtype == DType::kF16 && out.dtype == DType::kF16,
38:            "EXL3 grouped linear requires F16 model input/output");
39:   VT_CHECK(trellis.dtype == DType::kI8 && trellis.rank == 3 &&
40:                trellis.shape[0] == k / 16 && trellis.shape[1] == n / 16 &&
41:                trellis.shape[2] == 32 * args.bits,
42:            "EXL3 grouped linear requires opaque packed trellis [K/16,N/16,32*bits]");
43:   VT_CHECK(suh.dtype == DType::kF16 && suh.rank == 2 && suh.shape[0] > 0 &&
44:                suh.shape[0] <= std::numeric_limits<int>::max() && suh.shape[1] == k,
45:            "EXL3 grouped linear requires F16 suh[S,K]");
46:   VT_CHECK(suh.shape[0] * m * (k / 128) <= std::numeric_limits<int>::max(),
47:            "EXL3 grouped linear input launch exceeds I32 indexing");
48:   VT_CHECK(svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
49:            "EXL3 grouped linear requires F16 svh[N]");
50:   VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
51:            "EXL3 grouped linear requires I32 shard_of_nb[N/128]");
52:   VT_CHECK(in_had.dtype == DType::kF16 && in_had.rank == 4 &&
53:                in_had.shape[0] == suh.shape[0] && in_had.shape[1] == k / 16 &&
54:                in_had.shape[2] == plan.padded_rows && in_had.shape[3] == 16,
55:            "EXL3 grouped linear input scratch layout mismatch");
56:   VT_CHECK(partials.dtype == DType::kF32 && partials.rank == 3 &&
57:                partials.shape[0] == plan.splits && partials.shape[1] == m &&
58:                partials.shape[2] == n, "EXL3 grouped linear partial scratch layout mismatch");
59:   for (const Tensor* t : std::initializer_list<const Tensor*>{
60:            &out, &in, &trellis, &suh, &svh, &shard, &in_had, &partials}) {
61:     VT_CHECK(t->IsContiguous(), "EXL3 grouped linear requires contiguous tensors");
62:     VT_CHECK(t->device == q.device, "EXL3 grouped linear device mismatch");
63:   }
64:   reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedLinear, q.device.type))(
65:       q, out, in, trellis, suh, svh, shard, in_had, partials, args);
```

**Recounted archived full D64 trace**

```json
{
  "events": 89530,
  "event_counts": {
    "selection": 46325,
    "tensor_arguments": 43205
  },
  "selection_counts": {
    "Embedding": 65,
    "RopeCosSinCache": 65,
    "RmsNorm": 8385,
    "Exl3GroupedLinear": 16705,
    "MatmulDenseF16": 3120,
    "GdnStateGather": 96,
    "CausalConv1dFwd": 48,
    "GdnStateScatter": 96,
    "GdnPostConv": 48,
    "GdnPrefillRawGate": 48,
    "RmsNormGated": 3120,
    "SiluAndMul": 4160,
    "AttnQkNormRopeGate": 1040,
    "ReshapeAndCacheFp8": 1040,
    "PagedAttention": 1040,
    "SigmoidGateBf16": 1040,
    "CastF32": 65,
    "CausalConv1dUpdate": 3072,
    "GdnPackedDecode": 3072
  },
  "BF16_tensor_arguments": 0,
  "cpu_reference_selections": 0,
  "head_F16_M1_tensor_calls": 65
}
```

## E04 — Gemma5120 arithmetic, selection and limited qualification

The new implementation deliberately reproduces residual promotion and reduction geometry. Existing generic RMS tests do not automatically cover this new branch.

**Source:** `code/src/vt/xpu/xpu_norm.cpp`, lines 6–23. SHA-256 `312fb8dd871617131db145deea5ee0af7ec5fc24fb77cd4dcf72f1a05976a8f2`.

```text
6: template <bool Residual>
7: float Gemma5120Value(View src, View res, int64_t row, int col) {
8:   const float value = Load(src, row * src.stride[0] + col);
9:   if constexpr (Residual)
10:     return sycl::ext::intel::math::fadd_rn(value, Load(res, row * res.stride[0] + col));
11:   return value;
12: }
13: 
14: template <bool Residual>
15: void Gemma5120Kernel(Queue& q, View dst, View src, View w, View res,
16:                      int64_t rows, float eps) {
17:   // Pinned Torch ReduceConfig: max WG1024, logical SG32, contiguous vec4.
18:   // Output count determines group_height; group_x_reduce first halves the
19:   // virtual lanes down to32, then uses ascending subgroup offsets.
20:   int height = 1;
21:   while (height < 32 && height * 2 <= rows) height *= 2;
22:   const int width = 1024 / height;
23:   constexpr int lanes = 16, workgroup = 128;
```

**Source:** `code/src/vt/xpu/xpu_norm.cpp`, lines 31–70. SHA-256 `312fb8dd871617131db145deea5ee0af7ec5fc24fb77cd4dcf72f1a05976a8f2`.

```text
31:     float first[32], second[32];
32:     const int chunks = width / 32;
33:     for (int chunk = 0; chunk < chunks; ++chunk) {
34:       float partial[2];
35:       for (int half = 0; half < 2; ++half) {
36:         float regs[4] = {};
37:         for (int col = 4 * (32 * chunk + lane + half * lanes); col < 5120;
38:              col += 4 * width) {
39:           for (int j = 0; j < 4; ++j) {
40:             const float value = Gemma5120Value<Residual>(src, res, row, col + j);
41:             regs[j] += sycl::ext::intel::math::fmul_rn(value, value);
42:           }
43:         }
44:         partial[half] = ((regs[0] + regs[1]) + regs[2]) + regs[3];
45:       }
46:       first[chunk] = partial[0]; second[chunk] = partial[1];
47:     }
48:     for (int offset = chunks / 2; offset > 0; offset /= 2)
49:       for (int chunk = 0; chunk < offset; ++chunk) {
50:         first[chunk] += first[chunk + offset];
51:         second[chunk] += second[chunk + offset];
52:       }
53:     auto group = item.get_sub_group();
54:     float a = first[0], b = second[0];
55:     for (int offset = 1; offset < lanes; offset *= 2) {
56:       a += sycl::shift_group_left(group, a, offset);
57:       b += sycl::shift_group_left(group, b, offset);
58:     }
59:     // MeanOps projects with an F32 reciprocal factor, not division by D.
60:     const float mean = sycl::ext::intel::math::fmul_rn(
61:         sycl::group_broadcast(group, a + b, 0), 1.0f / 5120.0f);
62:     const float inverse = sycl::rsqrt(mean + eps);
63:     for (int col = lane; col < 5120; col += lanes) {
64:       const float value = Gemma5120Value<Residual>(src, res, row, col);
65:       if constexpr (Residual) Store(res, row * res.stride[0] + col, value);
66:       Store(dst, row * dst.stride[0] + col,
67:             ProducerQkNormValue(value, inverse, Load(w, col), true));
68:     }
69:   });
70:   RecordProfileEvent(q, "rms_norm_gemma5120_fp16", event);
```

**Source:** `code/src/vt/xpu/xpu_norm.cpp`, lines 121–139. SHA-256 `312fb8dd871617131db145deea5ee0af7ec5fc24fb77cd4dcf72f1a05976a8f2`.

```text
121:     if (width == 5120 && gemma && src.dtype == DType::kF16 &&
122:         dst.dtype == DType::kF16 && w.dtype == DType::kF16 &&
123:         (!has_res || res.dtype == DType::kF16)) {
124:       const auto device = NativeQueue(q).get_device();
125:       const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
126:       if (device.get_info<sycl::info::device::max_work_group_size>() == 1024 &&
127:           !sizes.empty() && *std::min_element(sizes.begin(), sizes.end()) == 16 &&
128:           *std::max_element(sizes.begin(), sizes.end()) == 32) {
129:         if (has_res) Gemma5120Kernel<true>(q, dst, src, w, res, x.shape[0], eps);
130:         else Gemma5120Kernel<false>(q, dst, src, w, res, x.shape[0], eps);
131:         return;
132:       }
133:     }
134:     // Pinned EXL3 FP16 GemmaRMSNorm uses the native producer IR:
135:     // normalize x.float()+res.float(), but return that sum narrowed as res.
136:     // Do not normalize a reread of the FP16 store. Other dtype/weight modes
137:     // retain their existing residual-rounding contract.
138:     const bool fp32_sum = has_res && gemma && src.dtype == DType::kF16 &&
139:                           dst.dtype == DType::kF16 && res.dtype == DType::kF16;
```

**Source:** `evidence/s1/gemma5120-native-test-v3.log`, lines 1–10. SHA-256 `3b219646cd593ab117aabe5198aa949b9229373262237ded2821ab6b0988ec24`.

```text
1: [doctest] doctest version is "2.5.2"
2: [doctest] run with "--help" for options
3: REAL_BLOCK_NORM phase=p128 post=0 relative=0 max_error=0 different_half_values=0
4: REAL_BLOCK_NORM phase=p128 post=1 relative=0 max_error=0 different_half_values=0
5: REAL_BLOCK_NORM phase=d1 post=0 relative=0 max_error=0 different_half_values=0
6: REAL_BLOCK_NORM phase=d1 post=1 relative=0 max_error=0 different_half_values=0
7: ===============================================================================
8: [doctest] test cases:   2 |   2 passed | 0 failed | 13 skipped
9: [doctest] assertions: 113 | 113 passed | 0 failed |
10: [doctest] Status: SUCCESS!
```

**Source:** `code/docs/b70-exl3/EXECUTION_MAP.md`, lines 1521–1538. SHA-256 `f52d865e6e2468f404eee057848cb37ff354027fe0f30f822435823480cbe3ef`.

```text
1521: The new separate producer kernel emulates this tree with physical SG16,
1522: uses rsqrt and explicit F32 residual/square/Gemma multiplication boundaries,
1523: and returns the independently narrowed residual. Selection is scoped to
1524: Gemma D5120 F16 input/output/weight and optional F16 residual, maxWG1024,
1525: minSG16/maxSG32. Other geometries keep generic RMS; this step does not qualify
1526: those fallbacks. No public RMS argument or model policy changed.
1527: 
1528: Focused v1/v2/v3 builds exit0. Initial captured norm run passes40/40;
1529: final2 Gemma cases pass **113/113 assertions,exit0**, including separate output,
1530: output/input aliases and output/residual aliases. All4 captured norm outputs
1531: have zero differing half words; residuals remain exact where still visible.
1532: The existing generic RMS regression passes **63928/63928,exit0**; its F32
1533: outputs/BF16 residuals do not exercise the new path. Internal native D5120
1534: means/inverses are not separately observed by these output checks. M4/12/16,
1535: other layer operands, performance and graph/multi-queue behavior remain open.
1536: Initial v1 used an unsupported trace environment variable and produced no
1537: trace; finalv3 uses VT_OP_PROVIDER_TRACE. An initial source command exits1
1538: because its second grep finds no MeanOps in ReduceOps.h; the actual
```

## E05 — P128/D1 success and the current target test semantics

The full target runs native-owned state, but the continuation input is selected from captured reference IDs. This is why autonomous generation is a separate deliverable.

**Source:** `evidence/s1/target-native-gemma5120-test-v2.log`, lines 1–15. SHA-256 `6f1972af0ef767b7234fe78f35546cb9d1cc6a4649f09349b5af43f89026fd60`.

```text
1: [doctest] doctest version is "2.5.2"
2: [doctest] run with "--help" for options
3: REAL_TARGET_LOAD layers=64 head_bits=6 vocab=248320
4: REAL_TARGET_FORWARD_START p128
5: REAL_TARGET_COMPARE p128 repeat=0 TV=0.00206823 KL=1.39684e-05 top10=10 greedy=13 reference_greedy=13
6: REAL_TARGET_COMPARE p128 repeat=1 TV=0.00220001 KL=1.53485e-05 top10=10 greedy=13 reference_greedy=13
7: REAL_TARGET_COMPARE p128 repeat=2 TV=0.0022555 KL=1.63734e-05 top10=10 greedy=13 reference_greedy=13
8: REAL_TARGET_FORWARD_START d1
9: REAL_TARGET_COMPARE d1 repeat=0 TV=0.0039302 KL=4.85418e-05 top10=10 greedy=198 reference_greedy=198
10: REAL_TARGET_COMPARE d1 repeat=1 TV=0.00355419 KL=4.05824e-05 top10=10 greedy=198 reference_greedy=198
11: REAL_TARGET_COMPARE d1 repeat=2 TV=0.00340776 KL=3.89044e-05 top10=10 greedy=198 reference_greedy=198
12: ===============================================================================
13: [doctest] test cases:   1 |   1 passed | 0 failed | 14 skipped
14: [doctest] assertions: 197 | 197 passed | 0 failed |
15: [doctest] Status: SUCCESS!
```

**Source:** `code/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp`, lines 936–962. SHA-256 `bcdf9b0f98c00494ae882648b2a602f9c51145f7bfaba42d2c7b6bd12f02a024`.

```text
936:   const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
937:   if (!model || !fixtures) std::exit(77);
938:   const std::filesystem::path model_dir(model), receipts(fixtures);
939:   REQUIRE((decode_steps == 1 || decode_steps == 64));
940:   const auto capture_dir = receipts / (decode_steps == 1 ? "target-repeats" : "target-d64");
941:   const int repeats = decode_steps == 1 ? 3 : 1;
942:   std::ifstream record(capture_dir / "repeat-0.json");
943:   const auto captured_ids = nlohmann::json::parse(record).at("output_ids").at(0)
944:                                 .get<std::vector<int32_t>>();
945:   REQUIRE(captured_ids.size() == static_cast<size_t>(decode_steps + 1));
946:   std::vector<vllm::SafetensorsFile> oracles;
947:   for (int repeat = 0; repeat < repeats; ++repeat)
948:     oracles.push_back(vllm::SafetensorsFile::Open(
949:         (capture_dir / ("repeat-" + std::to_string(repeat) + ".safetensors")).string()));
950:   const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
951:   REQUIRE(config.num_hidden_layers == 64);
952:   REQUIRE(config.vocab_size == 248320);
953:   std::vector<vllm::SafetensorsFile> shards;
954:   for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
955:     shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
956:   xpu_test::Queue gpu(vt::DeviceType::kXPU);
957:   auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
958:   REQUIRE(weights.exl3_checkpoint);
959:   REQUIRE(weights.precision.activation == DType::kF16);
960:   REQUIRE(weights.layers.size() == 64);
961:   REQUIRE(weights.lm_head_exl3.Bits() == 6);
962:   REQUIRE(weights.lm_head_exl3.OutFeatures() == 248320);
```

**Source:** `code/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp`, lines 977–1017. SHA-256 `bcdf9b0f98c00494ae882648b2a602f9c51145f7bfaba42d2c7b6bd12f02a024`.

```text
977:       vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
978:       states.push_back(state);
979:       owners.push_back(std::move(conv)); owners.push_back(std::move(ssm));
980:     } else {
981:       auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
982:           std::initializer_list<int64_t>{2 * page * 4 * 256});
983:       std::vector<uint8_t> poison(2 * page * 4 * 256, 0x7f);
984:       kv->upload(poison.data());
985:       vllm::PagedKvCache cache;
986:       cache.data = kv->tensor.data; cache.dtype = DType::kI8;
987:       cache.num_blocks = 1; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
988:       cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
989:       caches.push_back(cache); owners.push_back(std::move(kv));
990:     }
991:   }
992:   REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
993:   for (int step = 0; step <= decode_steps; ++step) {
994:     const bool prefill = step == 0;
995:     const int rows = prefill ? 128 : 1;
996:     const std::string phase = prefill ? "p128" : "d" + std::to_string(step);
997:     std::vector<int32_t> ids(rows), positions(rows);
998:     for (int i = 0; i < rows; ++i) {
999:       ids[i] = prefill ? 1000 + (i * 37) % 4096 : captured_ids[step - 1];
1000:       positions[i] = prefill ? i : 127 + step;
1001:     }
1002:     vllm::v1::CommonAttentionMetadata am;
1003:     am.num_reqs = 1; am.num_actual_tokens = rows;
1004:     am.query_start_loc = am.query_start_loc_cpu = {0, rows};
1005:     am.seq_lens = am.seq_lens_cpu = {128 + step};
1006:     am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
1007:     am.block_table_num_cols = 1; am.block_table_tensor = {0};
1008:     am.slot_mapping.assign(positions.begin(), positions.end());
1009:     am.causal = true;
1010:     vllm::v1::GDNAttentionMetadata gm;
1011:     gm.num_actual_tokens = rows;
1012:     gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
1013:     gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
1014:     if (prefill) {
1015:       gm.num_prefills = 1; gm.num_prefill_tokens = rows;
1016:       gm.has_initial_state = std::vector<uint8_t>{0};
1017:       gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
```

**Source:** `code/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp`, lines 1065–1082. SHA-256 `bcdf9b0f98c00494ae882648b2a602f9c51145f7bfaba42d2c7b6bd12f02a024`.

```text
1065:       int overlap = 0;
1066:       for (const auto id : actual_top)
1067:         overlap += std::find(expected_top.begin(), expected_top.end(), id) != expected_top.end();
1068:       std::cout << "REAL_TARGET_COMPARE " << phase << " repeat=" << repeat << " TV=" << tv
1069:                 << " KL=" << kl << " top10=" << overlap << " greedy=" << actual_top[0]
1070:                 << " reference_greedy=" << captured_ids[step] << '\n';
1071:       CHECK(tv <= 0.02); CHECK(kl <= 0.002); CHECK(overlap >= 9);
1072:     }
1073:   }
1074:   CHECK(vt::GetReferenceTierHits() == 0);
1075: }
1076: 
1077: TEST_CASE("XPU EXL3 real target: eager P128 D1 full vocabulary comparison") {
1078:   RunRealEagerTarget(1);
1079: }
1080: 
1081: TEST_CASE("XPU EXL3 real target: eager P128 D64 full vocabulary continuation") {
1082:   RunRealEagerTarget(64);
```

## E06 — D64 failures, not an assertion-based accuracy percentage

Recomputed log summaries preserve both runs. The later run is not replaced by the earlier, more favorable one.

**Archived target log summaries**

```json
[
  {
    "path": "target-native-gemma5120-test-v2.log",
    "sha256": "6f1972af0ef767b7234fe78f35546cb9d1cc6a4649f09349b5af43f89026fd60",
    "recorded_assertions": {
      "total": 197,
      "passed": 197,
      "failed": 0
    },
    "comparisons": 6,
    "max_TV": 0.0039302,
    "max_KL": 4.85418e-05,
    "min_top10": 10,
    "first_greedy_mismatch": null,
    "failed_rows": []
  },
  {
    "path": "target-native-D64-gemma5120-test-v1.log",
    "sha256": "4189f7524b1eba9eeb8f6240f82403b934684475f315e606f134af4ce6a3d042",
    "recorded_assertions": {
      "total": 807,
      "passed": 803,
      "failed": 4
    },
    "comparisons": 65,
    "max_TV": 0.247458,
    "max_KL": 0.180265,
    "min_top10": 8,
    "first_greedy_mismatch": "d27",
    "failed_rows": [
      {
        "step": "d27",
        "repeat": 0,
        "TV": 0.0294933,
        "KL": 0.00196467,
        "top10": 10,
        "greedy": 74455,
        "reference_greedy": 248045
      },
      {
        "step": "d29",
        "repeat": 0,
        "TV": 0.247458,
        "KL": 0.180265,
        "top10": 8,
        "greedy": 248068,
        "reference_greedy": 248068
      }
    ]
  },
  {
    "path": "target-native-D64-gemma5120-test-v2.log",
    "sha256": "9ad4dcb7da17d4bf23d7ea623040c527aac5a2b1d5be94ceaeb152ac5c34d399",
    "recorded_assertions": {
      "total": 807,
      "passed": 802,
      "failed": 5
    },
    "comparisons": 65,
    "max_TV": 0.281446,
    "max_KL": 0.217539,
    "min_top10": 8,
    "first_greedy_mismatch": "d7",
    "failed_rows": [
      {
        "step": "d11",
        "repeat": 0,
        "TV": 0.023854,
        "KL": 0.00201852,
        "top10": 10,
        "greedy": 248046,
        "reference_greedy": 248046
      },
      {
        "step": "d29",
        "repeat": 0,
        "TV": 0.281446,
        "KL": 0.217539,
        "top10": 8,
        "greedy": 248068,
        "reference_greedy": 248068
      }
    ]
  }
]
```

**Source:** `evidence/s1/target-native-D64-gemma5120-test-v2.log`, lines 18–35. SHA-256 `9ad4dcb7da17d4bf23d7ea623040c527aac5a2b1d5be94ceaeb152ac5c34d399`.

```text
18: REAL_TARGET_FORWARD_START d7
19: REAL_TARGET_COMPARE d7 repeat=0 TV=0.00962862 KL=0.000282665 top10=10 greedy=248046 reference_greedy=248044
20: REAL_TARGET_FORWARD_START d8
21: REAL_TARGET_COMPARE d8 repeat=0 TV=0.0127604 KL=0.000611437 top10=10 greedy=248046 reference_greedy=248046
22: REAL_TARGET_FORWARD_START d9
23: REAL_TARGET_COMPARE d9 repeat=0 TV=6.917e-07 KL=2.37704e-08 top10=10 greedy=198 reference_greedy=198
24: REAL_TARGET_FORWARD_START d10
25: REAL_TARGET_COMPARE d10 repeat=0 TV=0.00348788 KL=2.72328e-05 top10=10 greedy=248045 reference_greedy=248045
26: REAL_TARGET_FORWARD_START d11
27: REAL_TARGET_COMPARE d11 repeat=0 TV=0.023854 KL=0.00201852 top10=10 greedy=248046 reference_greedy=248046
28: ===============================================================================
29: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1081:
30: TEST CASE:  XPU EXL3 real target: eager P128 D64 full vocabulary continuation
31: 
32: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1071: ERROR: CHECK( tv <= 0.02 ) is NOT correct!
33:   values: CHECK( 0.023854 <= 0.02 )
34: 
35: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1071: ERROR: CHECK( kl <= 0.002 ) is NOT correct!
```

**Source:** `evidence/s1/target-native-D64-gemma5120-test-v2.log`, lines 71–85. SHA-256 `9ad4dcb7da17d4bf23d7ea623040c527aac5a2b1d5be94ceaeb152ac5c34d399`.

```text
71: REAL_TARGET_COMPARE d28 repeat=0 TV=0.00535261 KL=0.000227577 top10=10 greedy=198 reference_greedy=198
72: REAL_TARGET_FORWARD_START d29
73: REAL_TARGET_COMPARE d29 repeat=0 TV=0.281446 KL=0.217539 top10=8 greedy=248068 reference_greedy=248068
74: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1071: ERROR: CHECK( tv <= 0.02 ) is NOT correct!
75:   values: CHECK( 0.281446 <= 0.02 )
76: 
77: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1071: ERROR: CHECK( kl <= 0.002 ) is NOT correct!
78:   values: CHECK( 0.217539 <= 0.002 )
79: 
80: /work/tests/vllm/models/test_xpu_qwen_exl3_fp16.cpp:1071: ERROR: CHECK( overlap >= 9 ) is NOT correct!
81:   values: CHECK( 8 >= 9 )
82: 
83: REAL_TARGET_FORWARD_START d30
84: REAL_TARGET_COMPARE d30 repeat=0 TV=0.00142799 KL=0.000149849 top10=10 greedy=198 reference_greedy=198
85: REAL_TARGET_FORWARD_START d31
```

**Source:** `code/docs/b70-exl3/EXECUTION_MAP.md`, lines 1543–1558. SHA-256 `f52d865e6e2468f404eee057848cb37ff354027fe0f30f822435823480cbe3ef`.

```text
1543: They are not autonomous native-greedy continuation. P128/D1 passes197/197
1544: in both runs. Finalv2 maxTV0.0039302/maxKL0.0000485418, top10all10, greedy13/198.
1545: Earlier v1 maxTV0.00311897/maxKL0.0000323448 is retained, not substituted.
1546: 
1547: D64 remains failed in both runs. Initialv1: **803/807,exit1**; D27 TV0.0294933
1548: fails while KL0.00196467 passes; D29 TV0.247458/KL0.180265/top10overlap8 fails.
1549: Greedy differs atD27 (native74455/reference248045). Finalv2 against the final
1550: v3 binary: **802/807,exit1**; D11 TV0.023854/KL0.00201852 fails, D29
1551: TV0.281446/KL0.217539/top10overlap8 fails. Greedy differs atD7. The source/device
1552: eligibility guard does not establish a cause for this variation. All65 steps
1553: execute in each run; neither failed result is waived by the exact norm.
1554: Final traces have65 full-head M1 calls and zero BF16 arguments, CPU-reference
1555: or old scalar-packed selections. Independent same-prefix/3-position-axis
1556: witness comparison retains11/195 triggers forv1 and10/195 forfinalv2. Final
1557: maxTV0.2633897193/maxKL0.2041601301 remains failing; report generation's exit0
1558: is not a parity pass. Frozen thresholds and all earlier captures are unchanged.
```

## E07 — B/A repeatability and exact-dot diagnostics

Same-input original replays vary. The native-linked standalone deterministic probe is not actual VT instrumentation and is not a promoted production change.

**Source:** `code/docs/b70-exl3/EXECUTION_MAP.md`, lines 1581–1606. SHA-256 `f52d865e6e2468f404eee057848cb37ff354027fe0f30f822435823480cbe3ef`.

```text
1581: 
1582: Literal pinned Torch F.linear with unchanged frozen P128/D1 norm inputs and
1583: byte-identical actual worker/checkpoint BA weights (B then A, SHA256
1584: e5dbd32b6b1adb82dc5803a7239b9ae005ed621af21a89984512a111022760e9) completes,
1585: exit0. Verbose identifies original oneDNN3.12; native uses the pinned3.13.
1586: Original P128 repeats differ8/7/3 half values from the frozen capture and
1587: 5/7/6 from one another. All3 D1 repeats are exact. Split/F32 diagnostics
1588: retain10/14 P128 differences; neither replaces original F16 production math.
1589: All3 original repeats give the captured native B22,row74 value rather than
1590: the frozen original value. This strengthens the earlier live-operand variance
1591: observation; it does not qualify the failing strict state gate.
1592: 
1593: A standalone native oneDNN3.13 probe reproduces VT's F16 descriptors, strict
1594: accumulation and user-scratchpad recipe. Build/run exit0. Default P128 repeats
1595: differ6/5/12 halves from original and9/9/10 from one another. Deterministic
1596: selection differs13 halves in each repeat, with zero pair differences, but
1597: still misses B22. D1 is exact in both selections. The source selector rejects
1598: kParallel catalogue candidates when deterministic is set; the physical selected
1599: subkernel is not separately observed here. This is not actual VT instrumentation.
1600: Its SYCL queue has no profiling, so verbose time0 fields are not measurements.
1601: Deterministic selection is not adopted as a product repair.
1602: 
1603: Exact rational dot products of the6 differing FP16 operand pairs are a
1604: mathematical diagnostic. Correct rounding matches native at5 points and frozen
1605: original atB22,row74. That exact sum is-0.3306885194615461, only4.2899046e-8
1606: below the F16 midpoint-0.3306884765625. This shows rounding sensitivity; it
```

**Source:** `evidence/s1/ba-exact-dyadic-points-v1.json`, lines 34–51. SHA-256 `77b734ef44ff2cb5e51d27d7cd8a2dc2158d30affd5030139b2415b01d0797c4`.

```text
34:     },
35:     {
36:       "row": 74,
37:       "head": 22,
38:       "native": -0.33056640625,
39:       "original": -0.330810546875,
40:       "part": "b",
41:       "exact_numerator": -5681185505,
42:       "exact_denominator": 17179869184,
43:       "exact_as_F64": -0.3306885194615461,
44:       "native_original_midpoint": -0.3306884765625,
45:       "exact_minus_midpoint": -4.289904609322548e-08,
46:       "correctly_rounded_F16": -0.330810546875,
47:       "rounded_matches_native": false,
48:       "rounded_matches_original": true
49:     },
50:     {
51:       "row": 1,
```

**Checked arithmetic of printed exact fractions (not recomputed missing input tensors)**

```json
[
  {
    "row": 46,
    "head": 34,
    "exact_fraction_value": -0.0004235941651131725,
    "reported_exact_value_agrees": true,
    "midpoint": -0.0004235506057739258,
    "difference": -4.3559339246712625e-08
  },
  {
    "row": 70,
    "head": 39,
    "exact_fraction_value": 0.000108610594224956,
    "reported_exact_value_agrees": true,
    "midpoint": 0.00010862946510314941,
    "difference": -1.8870878193411045e-08
  },
  {
    "row": 74,
    "head": 22,
    "exact_fraction_value": -0.3306885194615461,
    "reported_exact_value_agrees": true,
    "midpoint": -0.3306884765625,
    "difference": -4.289904609322548e-08
  },
  {
    "row": 1,
    "head": 10,
    "exact_fraction_value": -6.545568248839118e-05,
    "reported_exact_value_agrees": true,
    "midpoint": -6.541609764099121e-05,
    "difference": -3.958484739996493e-08
  },
  {
    "row": 39,
    "head": 17,
    "exact_fraction_value": 0.0006796755515097175,
    "reported_exact_value_agrees": true,
    "midpoint": 0.0006797313690185547,
    "difference": -5.581750883720815e-08
  },
  {
    "row": 97,
    "head": 45,
    "exact_fraction_value": 0.16998287496971898,
    "reported_exact_value_agrees": true,
    "midpoint": 0.16998291015625,
    "difference": -3.518653102219105e-08
  }
]
```

## E08 — Causal local state attribution and the missing actual-state export

These interventions were diagnostic experiments. They are not instructions to patch B22 or inject oracle values into generation.

**Source:** `evidence/s1/ba-state-causality-original-v1.json`, lines 1–75. SHA-256 `de1b61429f76a9f35a522fe066a2eb2a727963951c7a846ab596bb772f1b7c8e`.

```text
1: {
2:   "scope": "Original fused GDN on unchanged original QKVZ and weights; only BA varied. Causal same-input diagnostic, not a modified product route or a waived state gate.",
3:   "capture_sha256": "195d9ee81287406b4e0d6e05dc1611627bd7b3ad3fbd10377c58f793f03d79fc",
4:   "cases": {
5:     "original": {
6:       "BA_half_differences": 0,
7:       "core_half_differences": 0,
8:       "state_F32_differences": 0,
9:       "state_relative": 0.0,
10:       "state_max_error": 0.0,
11:       "strict_state_band_failures": 0,
12:       "failing_heads": [],
13:       "first_failing_index": null,
14:       "state_value_at360512": -0.011011045426130295,
15:       "original_state_value_at360512": -0.011011045426130295
16:     },
17:     "captured_native": {
18:       "BA_half_differences": 6,
19:       "core_half_differences": 1934,
20:       "state_F32_differences": 16379,
21:       "state_relative": 1.746155685513628e-05,
22:       "state_max_error": 0.0026378631591796875,
23:       "strict_state_band_failures": 122,
24:       "failing_heads": [
25:         22
26:       ],
27:       "first_failing_index": 360512,
28:       "state_value_at360512": -0.01099583599716425,
29:       "original_state_value_at360512": -0.011011045426130295
30:     },
31:     "native_with_original_B22": {
32:       "BA_half_differences": 5,
33:       "core_half_differences": 0,
34:       "state_F32_differences": 0,
35:       "state_relative": 0.0,
36:       "state_max_error": 0.0,
37:       "strict_state_band_failures": 0,
38:       "failing_heads": [],
39:       "first_failing_index": null,
40:       "state_value_at360512": -0.011011045426130295,
41:       "original_state_value_at360512": -0.011011045426130295
42:     },
43:     "original_with_native_B22": {
44:       "BA_half_differences": 1,
45:       "core_half_differences": 1934,
46:       "state_F32_differences": 16379,
47:       "state_relative": 1.746155685513628e-05,
48:       "state_max_error": 0.0026378631591796875,
49:       "strict_state_band_failures": 122,
50:       "failing_heads": [
51:         22
52:       ],
53:       "first_failing_index": 360512,
54:       "state_value_at360512": -0.01099583599716425,
55:       "original_state_value_at360512": -0.011011045426130295
56:     },
57:     "Torch_repeat0": {
58:       "BA_half_differences": 8,
59:       "core_half_differences": 2661,
60:       "state_F32_differences": 54443,
61:       "state_relative": 2.2940637332751293e-05,
62:       "state_max_error": 0.0026378631591796875,
63:       "strict_state_band_failures": 220,
64:       "failing_heads": [
65:         2,
66:         5,
67:         22
68:       ],
69:       "first_failing_index": 34008,
70:       "state_value_at360512": -0.01099583599716425,
71:       "original_state_value_at360512": -0.011011045426130295
72:     }
73:   },
74:   "replay_sha256": "7599db9315561d38f71e982630578a67c489f2e1d25689e5bd251d459f1e53b6"
75: }
```

**Source:** `evidence/s1/ba-state-causality-core-native-v1.json`, lines 1–9. SHA-256 `7410a315d96628f58ad337da5c365ef51e7c1c5d11e772046c67b0b3e252e19d`.

```text
1: {
2:   "scope": "Actual native own-state GDN core versus original fused GDN using the same actual native BA and unchanged original QKVZ. Bounded P128 attribution only, not whole-target/state parity.",
3:   "native_core_sha256": "7ebe5cf7cc828fe3a0f80e01ca7b201e5d820df7f30c329419101d1657af40eb",
4:   "original_same_BA_core_sha256": "7ebe5cf7cc828fe3a0f80e01ca7b201e5d820df7f30c329419101d1657af40eb",
5:   "bit_exact": true,
6:   "bit_mismatches": 0,
7:   "first_bit_mismatch": null,
8:   "elements": 786432
9: }
```

**Source:** `code/docs/b70-exl3/EXECUTION_MAP.md`, lines 1618–1636. SHA-256 `f52d865e6e2468f404eee057848cb37ff354027fe0f30f822435823480cbe3ef`.

```text
1618: 
1619: The actual native P128 core dump and original fused GDN core from the same
1620: actual native BA are byte-identical in all786432 halves. Both SHA256 values
1621: are7ebe5cf7cc828fe3a0f80e01ca7b201e5d820df7f30c329419101d1657af40eb.
1622: Thus the current bounded recurrence/core discrepancy is causally attributed
1623: to the BA operand, not unexplained downstream core arithmetic. Actual native
1624: state is not separately dumped for byte comparison in this step. A same-input
1625: original Torch BA repeat also produces220 strict state failures in3 heads
1626: when replayed by the original GDN body. Original variability is retained,
1627: not a replacement capture, relaxed threshold or whole-target parity claim.
1628: 
1629: No BA selector/arithmetic or product B22 substitution is implemented. New
1630: root-owned safetensors initially had mode600; only those receipt read permissions
1631: were corrected to644 so host hashing/comparison could complete. Bytes remain
1632: unchanged. All identities/results are recorded in
1633: `reference_B.S1_BA_same_input_reproducibility_and_state_causality`.
1634: **Next:** qualify remaining Gemma M4/12/16 geometries and continue same-input
1635: block/D64 localization; S1 and S2-S6 remain open. Production remains inactive,
1636: and all jobs from this step are terminal. No commit/push is made for this step.
```

## E09 — Original numerical contract and report-only semantics

The original plan did not establish the initial EXL3 target thresholds as a universal property of the reference. Keep explicit existing failures and distinguish an investigation report from a passing test.

**Source:** `code/docs/B70-EXL3-Migration-and-Parity-Plan.md`, lines 311–320. SHA-256 `f82186533265b69e204019ca3b2d2693090d635e7fd0e602c41f51b03db5eb87`.

```text
311: 
312: ### Parity and measurement rules
313: 
314: 1. **Exact invariants:** checkpoint bytes, packed layout, token IDs/positions, subset map, request identity, causal visibility, accepted counts, logical lengths and ownership. FP8 writer bytes must agree when given identical source values and scales.
315: 2. **Operator arithmetic:** producer's existing real-linear relative-norm threshold <2e-3; attention rtol .01/atol .003 on matched inputs. SmallM and W8A8 use their respective independent references. Record max error and finite status, not only one aggregate metric. Exact data rearrangements use bit-pattern equality, including direct-copy tests.
316: 3. **Whole-target comparison:** begin with the frozen GPTQ diagnostic envelope (TV .02, KL .002, top-10 overlap ≥9) as a conservative **EXL3 investigation trigger**, not evidence already validated for EXL3. Calibrate an EXL3 same-checkpoint/route baseline before accepting a different envelope. Near ties may explain a token mismatch, never an ownership/mask failure. Do not weaken a gate after observing a failure just to green the port.
317: 4. **Quantization quality versus engine correctness:** current EXL3-vs-BF16 differences are part of the chosen checkpoint/arithmetic. Reimplementing that recipe should not introduce an unexamined new quality tradeoff. Use held-out application tests to supplement, not replace, matched-state checks.
318: 5. **Three timing scopes:** synchronized operator replay; complete worker step including the target head and required sampling/state work; end-to-end emitted tokens/TTFT/request wall. Never compare one scope's number with another.
319: 6. **MTP accounting:** count proposed, accepted and emitted tokens, target cycles and draft steps separately. MTP3 normally verifies up to four rows per request, not five. O counts total emitted tokens; D counts extra target forwards in non-speculative tests. First output arises from prefill. Do not infer work from an output token repeated 1,024 times.
320: 7. **Performance attribution:** for device-core comparisons restore identical full states, tokens, proposals, accepted lengths and physical batch forms. For serving, keep prompts and sampling controls fixed but report different continuations and acceptance; the same seed need not produce the same text across arithmetic implementations.
```

**Source:** `code/tools/exl3_reference/compare_target.py`, lines 35–65. SHA-256 `51a921a3d373cac8213f2aaaa45ad776eb78ab2e53a365c169639b6729a6c889`.

```text
35:         headers.require(record["output_ids"] == native_ids and
36:                         digest(path.read_bytes()) == identity["sha256"] == record["capture_sha256"],
37:                         "reference prefix or capture identity mismatch")
38:         references.append((path, headers.read_shard_header(path)))
39:     headers.require(bool(references), "reference repeats missing")
40:     rows, native_identities = [], []
41:     for phase in target_phases(64):
42:         path = Path(str(args.native_prefix) + "-" + phase + "-logits.f32")
43:         native = path.read_bytes()
44:         native_identities.append({"path": str(path), "sha256": digest(native)})
45:         for repeat, (reference, header) in enumerate(references):
46:             entry, expected = blob(reference, header, phase + "_logits")
47:             headers.require(entry["shape"] == [1,248320], "requires full target vocabulary")
48:             result = compare_row(native, expected, entry["dtype"], 248320)
49:             rows.append({"phase": phase, "reference_repeat": repeat, **result})
50:     result = {"schema": 1, "image": IMAGE, "native_trace_json_sha256": digest(args.native_trace_json.read_bytes()),
51:               "reference_comparison_sha256": digest((args.reference_dir / "comparison.json").read_bytes()),
52:               "native_logits": native_identities, "comparisons": rows,
53:               "investigation_triggers": [r for r in rows if r["investigation_trigger"]],
54:               "scope": "Same-prefix whole-target logits only. Does not replace failed native tests or qualify state/performance."}
55:     with args.report.open("x") as stream:
56:         json.dump(result, stream, indent=2); stream.write("\n")
57:     print("TARGET_COMPARISONS", len(rows), "INVESTIGATION_TRIGGERS", len(result["investigation_triggers"]))
58: 
59: 
60: if __name__ == "__main__":
61:     parser = argparse.ArgumentParser(description=__doc__)
62:     parser.add_argument("--native-prefix", type=Path, required=True)
63:     parser.add_argument("--native-trace-json", type=Path, required=True)
64:     parser.add_argument("--reference-dir", type=Path, required=True)
65:     parser.add_argument("--report", type=Path, required=True)
```

## E10 — M128 boundary and legacy routes

SmallM supports physical M<=128. Model call sites can fall through to older reconstruction paths at larger M; this is not a claim that the full model always rejects larger inputs.

**Source:** `code/include/vllm/model_executor/models/dense_attn_block.h`, lines 348–374. SHA-256 `dc75f2c1fe373ed76e4adcd57695ba3cdc8acaaa53f3b245e2fbdeae69a17baa`.

```text
348: inline DBuf Exl3GroupedMatmulD(Dev d, const vt::Tensor& x,
349:                                const Exl3GroupedWeight& w) {
350:   const int64_t M = x.shape[0], K = w.suh.shape[1], N = w.svh.shape[0];
351:   VT_CHECK(d.q.device.type == vt::DeviceType::kXPU &&
352:                d.activation_dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 &&
353:                x.rank == 2 && x.shape[1] == K && w.codebook == 2,
354:            "exl3 grouped model: requires scoped XPU FP16 and matching K/mul1");
355:   const int bits = static_cast<int>(w.trellis.shape[2] / 32);
356:   const auto plan = vt::PlanExl3SmallM(M, K, N, bits);
357:   const bool upload = !w.trellis.d_dev || !w.suh.d_dev || !w.svh.d_dev || !w.source_map.d_dev;
358:   auto trellis = ResidentWeight(d, w.trellis);
359:   auto suh = ResidentWeight(d, w.suh);
360:   auto svh = ResidentWeight(d, w.svh);
361:   auto map = ResidentWeight(d, w.source_map);
362:   if (upload) {
363:     d.b.Synchronize(d.q);
364:     w.trellis.ReleaseHost();
365:     w.suh.ReleaseHost();
366:     w.svh.ReleaseHost();
367:     w.source_map.ReleaseHost();
368:   }
369:   DBuf had(d, vt::DType::kF16, {w.suh.shape[0], K / 16, plan.padded_rows, 16});
370:   DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
371:   DBuf out(d, vt::DType::kF16, {M, N});
372:   vt::Exl3GroupedLinear(d.q, out.t(), x, trellis, suh, svh, map,
373:                        had.t(), parts.t(), {bits, w.codebook, w.name.c_str()});
374:   return out;
```

**Source:** `code/include/vllm/model_executor/models/dense_attn_block.h`, lines 377–420. SHA-256 `dc75f2c1fe373ed76e4adcd57695ba3cdc8acaaa53f3b245e2fbdeae69a17baa`.

```text
377: inline DBuf Exl3MatmulD(Dev d, const vt::Tensor& x, const Exl3Weight& w,
378:                         vt::DType out_dtype) {
379:   const int64_t M = x.shape[0];
380:   const int64_t K = w.InFeatures();
381:   const int64_t N = w.OutFeatures();
382:   VT_CHECK(x.rank == 2 && x.shape[1] == K,
383:            "exl3 linear: activation is [" + std::to_string(x.shape[0]) + "," +
384:                std::to_string(x.rank == 2 ? x.shape[1] : -1) +
385:                "] but the weight needs K=" + std::to_string(K));
386:   VT_CHECK(out_dtype == vt::DType::kF32 || out_dtype == vt::DType::kBF16 ||
387:                out_dtype == vt::DType::kF16,
388:            "exl3 linear: out_dtype must be f32, bf16 or f16");
389: 
390:   // Scoped XPU FP16 model projections use the pinned producer's arithmetic.
391:   // A single checkpoint projection is one source group. Grouped QKV/QKVZ/
392:   // gate-up storage will use the same VT seam; legacy callers remain below.
393:   if (d.q.device.type == vt::DeviceType::kXPU &&
394:       d.activation_dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 &&
395:       out_dtype == vt::DType::kF16 && M <= 128 && w.codebook == 2 &&
396:       (w.Bits() == 4 || w.Bits() == 6)) {
397:     const auto plan = vt::PlanExl3SmallM(M, K, N, w.Bits());
398:     auto trellis = ResidentWeight(d, w.trellis);
399:     auto suh = Reshape(ResidentWeight(d, w.suh), {1, K});
400:     auto svh = ResidentWeight(d, w.svh);
401:     DBuf shard(d, vt::DType::kI32, {N / 128});
402:     shard.Zero(d);
403:     DBuf had(d, vt::DType::kF16, {1, K / 16, plan.padded_rows, 16});
404:     DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
405:     DBuf out(d, vt::DType::kF16, {M, N});
406:     vt::Exl3GroupedLinear(d.q, out.t(), x, trellis, suh, svh, shard.t(),
407:                          had.t(), parts.t(), {w.Bits(), w.codebook, w.name.c_str()});
408:     return out;
409:   }
410: 
411:   static const bool xpu_cast_fusion = [] {
412:     const char* setting = std::getenv("VT_XPU_EXL3_CAST_FUSION");
413:     return setting == nullptr || std::string(setting) != "0";
414:   }();
415:   const bool fuse_casts = d.q.device.type == vt::DeviceType::kXPU && xpu_cast_fusion;
416:   DBuf a_owned;
417:   vt::Tensor a = x;
418:   if (x.dtype != vt::DType::kF16 && !fuse_casts) {
419:     a_owned = DBuf(d, vt::DType::kF16, {M, K});
420:     vt::CastF16(d.q, a_owned.t(), x);
```

**Source:** `code/include/vllm/model_executor/layers/quantization/exl3.h`, lines 87–118. SHA-256 `44d5c4e2f74403e72361437011a2c133470a4521d596855ba1b66940f21851db`.

```text
87:                       Exl3GroupedWeight* grouped = nullptr)
88:       : gate_(gate), up_(up), grouped_(grouped) {}
89: 
90:   DBuf Apply(Dev d, const vt::Tensor& x) const override {
91:     const int64_t M = x.shape[0];
92:     const int64_t I = gate_->OutFeatures();
93:     VT_CHECK(up_->OutFeatures() == I,
94:              "exl3 gate_up: gate is [.., " + std::to_string(I) + "] but up is [.., " +
95:                  std::to_string(up_->OutFeatures()) + "]");
96:     // `vt::MoeSiluMul` rather than `vt::SiluAndMul`, and it is the op written
97:     // for this shape: SiluAndMul consumes ONE [M, 2I] operand with gate rows
98:     // first, which is what the MERGED arms hand it, while this one "takes the
99:     // two separately-produced projections so no concat/copy is needed"
100:     // (`ops.h`). The operator rounds SiLU through the gate dtype before
101:     // multiplying by up. Preserve the model's explicit activation precision
102:     // at both projections and the output; callers without a policy retain
103:     // the legacy BF16 behavior.
104:     const vt::DType dtype = d.activation_dtype.value_or(vt::DType::kBF16);
105:     if (grouped_ && d.q.device.type == vt::DeviceType::kXPU &&
106:         dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 && M <= 128) {
107:       if (grouped_->Empty())
108:         *grouped_ = MergeExl3Weights({gate_, up_}, gate_->name + "+" + up_->name);
109:       DBuf gu = dense_attn::Exl3GroupedMatmulD(d, x, *grouped_);
110:       DBuf act(d, dtype, {M, I});
111:       vt::SiluAndMul(d.q, act.t(), gu.t());
112:       return act;
113:     }
114:     DBuf g = dense_attn::Exl3MatmulD(d, x, *gate_, dtype);
115:     DBuf u = dense_attn::Exl3MatmulD(d, x, *up_, dtype);
116:     DBuf act(d, dtype, {M, I});
117:     vt::MoeSiluMul(d.q, act.t(), g.t(), u.t());
118:     return act;
```

## E11 — Exact-P128 GDN contract and workspace constraint

The source-matched P128 path also changes the Conv/post-conv dtype boundary, so simply lifting the shape guard changes more than a launch size. Static scratch figures below are not measured GPU peaks.

**Source:** `code/src/vllm/model_executor/models/qwen3_5.cpp`, lines 5666–5677. SHA-256 `e8d9a6e6ec91a03caf3ef590f55d924e6c00d0bfae33aa5cbd54e8341c93857d`.

```text
5666:   // The qualified EXL3 P128 producer retains the unrounded FP32 Conv result
5667:   // through normalization, rounds scaled Q to FP16, and prepares gate prefixes
5668:   // directly from raw A. Its state remains FP32 across the following decode.
5669:   const bool fp16_producer_prefill = d.q.device.type == vt::DeviceType::kXPU &&
5670:       d.activation_dtype == DType::kF16 && mixed.dtype == DType::kF16 && gptq == nullptr &&
5671:       !w.in_proj_qkv_exl3.Empty() && !spec && np == 1 && nd == 0 && T == 128 &&
5672:       Hk == 16 && Hv == 48 && Dk == 128 && Dv == 128 && Kw == 4 &&
5673:       vt::OpRegistered(vt::OpId::kGdnPrefillRawGate, d.q.device.type);
5674:   const DType convdt = fp16_producer_prefill ? DType::kF32 : mixed.dtype;
5675:   Tensor dcw = mixed.dtype == DType::kBF16 || mixed.dtype == DType::kF16
5676:                    ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
5677:                    : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
```

**Source:** `code/src/vt/xpu/xpu_gdn_fp16_producer.cpp`, lines 10–40. SHA-256 `9a1c8035a2b167d6bd20b896213cd3a2871f14a34b89d1a68038e806cb658214`.

```text
10: constexpr int Tokens = 128, Capacity = Tokens + 63, Hk = 16, Hv = 48, D = 128;
11: using T = cutlass::half_t;
12: constexpr size_t QBytes = size_t(Capacity) * Hk * D * sizeof(T);
13: constexpr size_t VBytes = size_t(Capacity) * Hv * D * sizeof(T);
14: constexpr size_t ABytes = size_t(Hv) * Capacity * 64 * sizeof(T);
15: constexpr size_t WBytes = size_t(Hv) * Capacity * D * sizeof(T);
16: constexpr size_t GateBytes = size_t(Hv) * Capacity * sizeof(float);
17: constexpr size_t Bytes = 2 * QBytes + VBytes + ABytes + 2 * WBytes +
18:     2 * GateBytes + 128 + 128;
19: }  // namespace
20: void GdnPrefillRawGateKernel(Queue& queue, Tensor& out, const Tensor& qi,
21:     const Tensor& ki, const Tensor& vi, const Tensor& raw_a, const Tensor& beta,
22:     const Tensor& a_log, const Tensor& dt_bias, Tensor& state, const Tensor& qsl,
23:     const GdnArgs&) {
24:   auto& native = NativeQueue(queue);
25:   const auto device = native.get_device();
26:   VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
27:                device.get_info<sycl::ext::intel::info::device::device_id>() == 57891 &&
28:                device.has(sycl::aspect::ext_intel_matrix),
29:            "gdn_prefill_raw_gate: requires the B70 matrix device");
30:   for (const Tensor* input : {&qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &qsl}) {
31:     VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
32:              "gdn_prefill_raw_gate: output/state overlaps an input");
33:   }
34:   VT_CHECK(!Overlap(out, state), "gdn_prefill_raw_gate: output overlaps state");
35:   TraceXpuOp(OpId::kGdnPrefillRawGate, queue,
36:              {&out, &qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &state, &qsl});
37:   RecordGraphWrite(queue, state.data, Span(state));
38:   const auto* offsets = static_cast<const int32_t*>(qsl.data);
39:   CheckDeviceMetadata(queue, [=] { return offsets[0] == 0 && offsets[1] == Tokens; },
40:                       "gdn_prefill_raw_gate: requires offsets [0,128]", {&qsl});
```

**Source:** `code/src/vt/xpu/xpu_gdn_fp16_producer.cpp`, lines 60–79. SHA-256 `9a1c8035a2b167d6bd20b896213cd3a2871f14a34b89d1a68038e806cb658214`.

```text
60:     // The producer reads its zero-padded physical capacity at tile tails.
61:     native.memset(storage, 0, Bytes);
62:     native.memcpy(q, qi.data, size_t(Tokens) * Hk * D * sizeof(T));
63:     native.memcpy(k, ki.data, size_t(Tokens) * Hk * D * sizeof(T));
64:     native.memcpy(v, vi.data, size_t(Tokens) * Hv * D * sizeof(T));
65:     const View av(raw_a), bv(beta), dv(dt_bias);
66:     native.parallel_for(sycl::range<1>(size_t(Hv) * Capacity), [=](sycl::id<1> id) {
67:       const int h = id[0] / Capacity, t = id[0] % Capacity;
68:       if (t < Tokens) {
69:         a[id] = Load(av, t * av.stride[0] + h);
70:         b[id] = Load(bv, t * bv.stride[0] + h);
71:       }
72:       if (t == 0) bias[h] = sycl::half(Load(dv, h));
73:     });
74:     native.single_task([=] { index[0] = 0; initial[0] = true; });
75:     gdn::fp16_producer::kernel_launcher<T, float>(native,
76:         static_cast<T*>(out.data), q, k, v, A, w, u, b, a,
77:         static_cast<const float*>(a_log.data), reinterpret_cast<const T*>(bias),
78:         static_cast<float*>(state.data), Hv * D * D, offsets, index, initial,
79:         nullptr, 1, Capacity, Hk, D, Hv, D);
```

**Source:** `code/src/vt/xpu/xpu_gdn_fp16_producer.cpp`, lines 88–115. SHA-256 `9a1c8035a2b167d6bd20b896213cd3a2871f14a34b89d1a68038e806cb658214`.

```text
88:   const auto device = native.get_device();
89:   VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
90:                device.get_info<sycl::ext::intel::info::device::device_id>() == 57891,
91:            "gdn_packed_decode: FP16 producer requires the B70");
92:   VT_CHECK(mixed.dtype == DType::kF16 && mixed.shape[0] == 1 &&
93:                mixed.shape[1] == (2 * Hk + Hv) * D &&
94:                raw_a.dtype == DType::kF16 && raw_b.dtype == DType::kF16 &&
95:                out.dtype == DType::kF16 && state.dtype == DType::kF32 &&
96:                a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32 &&
97:                state.shape[1] == Hv && state.shape[2] == D && state.shape[3] == D &&
98:                state.shape[0] > 0 &&
99:                std::abs(args.scale - 1.0f / std::sqrt(float(D))) < 1e-6f,
100:            "gdn_packed_decode: qualified FP16 producer geometry is C1/Hk16/Hv48/D128/F32 state");
101:   for (const Tensor* input : {&mixed, &raw_a, &raw_b, &a_log, &dt_bias, &indices}) {
102:     VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
103:              "gdn_packed_decode: output/state overlaps an input");
104:   }
105:   VT_CHECK(!Overlap(out, state), "gdn_packed_decode: output overlaps state");
106:   TraceXpuOp(OpId::kGdnPackedDecode, queue,
107:              {&out, &mixed, &raw_a, &raw_b, &a_log, &dt_bias, &state, &indices});
108:   RecordGraphWrite(queue, state.data, Span(state));
109:   const auto* index = static_cast<const int32_t*>(indices.data);
110:   const int64_t slots = state.shape[0];
111:   CheckDeviceMetadata(queue, [=] { return index[0] >= 0 && index[0] < slots; },
112:                       "gdn_packed_decode: invalid active state slot", {&indices});
113:   // Reserve the same completion-owned capacity as P128, including when
114:   // decode runs first. This workspace cannot grow after first allocation.
115:   const bool submitted = WithGdnNativeWorkspace(queue, Bytes, [&](void* storage) {
```

**Source:** `code/src/vt/xpu/xpu_backend.cpp`, lines 733–779. SHA-256 `7c35948d3c286e5704d75d52b710951ca9bc8c9ba2658105e76431701847376e`.

```text
733: bool WithWorkspace(Queue& q, Workspace& workspace, unsigned mask,
734:                    const char* wait_stage, size_t bytes,
735:                    const std::function<void(void*)>& launch) {
736:   auto& native = NativeQueue(q);
737:   auto& c = GetContext(q.device.index);
738:   std::lock_guard<std::mutex> execution(workspace.mutex);
739:   bool capturing = false;
740:   {
741:     std::lock_guard lock(c.mutex);
742:     if (auto it = c.recordings.find(&native); it != c.recordings.end()) {
743:       VT_CHECK(workspace.data != nullptr, "XPU graph workspace must be warmed before capture");
744:       it->second->workspace_mask |= mask;
745:       capturing = true;
746:     }
747:   }
748:   if (!workspace.data) {
749:     std::lock_guard<std::mutex> lock(c.mutex);
750:     // Reserve and account under the same lock as ordinary allocations: another
751:     // queue must not consume this budget between the check and allocation.
752:     if (bytes > c.budget - c.allocated - c.graph_bytes) return false;
753:     void* storage = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
754:     VT_CHECK(storage != nullptr, "XPU persistent workspace allocation failed");
755:     try { c.allocations.emplace(storage, bytes); }
756:     catch (...) { sycl::free(storage, c.context); throw; }
757:     c.allocated += bytes;
758:     c.peak_allocated = std::max(c.peak_allocated, c.allocated);
759:     workspace.data = storage;
760:     workspace.bytes = bytes;
761:   }
762:   VT_CHECK(bytes <= workspace.bytes, "XPU persistent workspace cannot grow");
763:   if (!capturing && workspace.last) native.ext_oneapi_submit_barrier({*workspace.last});
764:   try {
765:     launch(workspace.data);
766:     if (!capturing) {
767:       const auto wait_start = HostProfileEnabled() ? SteadyNs() : 0;
768:       native.wait_and_throw();
769:       if (wait_start)
770:         RecordHostProfileSpan(q, wait_stage, wait_start, SteadyNs());
771:       workspace.last.reset();
772:     }
773:   } catch (...) {
774:     // Even when host-side submission throws, complete earlier kernels before
775:     // releasing this workspace to another queue.
776:     if (!capturing) { try { native.wait_and_throw(); } catch (...) {} }
777:     throw;
778:   }
779:   return true;
```

**Source:** `code/src/vt/xpu/xpu_backend.cpp`, lines 787–797. SHA-256 `7c35948d3c286e5704d75d52b710951ca9bc8c9ba2658105e76431701847376e`.

```text
787: bool WithGdnWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
788:   VT_CHECK(bytes > 0 && bytes <= 32 * 1024 * 1024, "XPU GDN workspace exceeds 32 MiB budget");
789:   return WithWorkspace(q, GetContext(q.device.index).gdn, 2,
790:                        "workspace_wait_gdn", bytes, launch);
791: }
792: bool WithGdnNativeWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
793:   VT_CHECK(bytes > 0 && bytes <= 160 * 1024 * 1024,
794:            "XPU native GDN workspace exceeds 160 MiB budget");
795:   return WithWorkspace(q, GetContext(q.device.index).native_gdn, 16,
796:                        "workspace_wait_gdn_native", bytes, launch);
797: }
```

**Direct substitution into the current layout formula**

```json
{
  "128": {
    "bytes": 9852800,
    "MiB": 9.3963623046875
  },
  "2048": {
    "bytes": 108894080,
    "MiB": 103.8494873046875
  },
  "4096": {
    "bytes": 214538112,
    "MiB": 204.5994873046875
  }
}
```

## E12 — MTP: FP16 is partly accepted, but compact head/dataflow are missing

The paged input/gather acceptance is newer than older BF16-only descriptions. The current head path still constructs a default Dev and requests F32 from the full EXL3 head, unlike the new target SmallM route. The proposer chooses rows after evaluating the whole head.

**Source:** `code/src/vllm/model_executor/models/qwen3_5.cpp`, lines 10135–10162. SHA-256 `e8d9a6e6ec91a03caf3ef590f55d924e6c00d0bfae33aa5cbd54e8341c93857d`.

```text
10135: Qwen3_5MTPHiddenStates Qwen3_5MTPModel::ForwardPaged(
10136:     const std::vector<int32_t>& input_ids,
10137:     const std::vector<int32_t>& positions,
10138:     const vt::Tensor& target_hidden_states,
10139:     const v1::CommonAttentionMetadata& attn_meta, PagedKvCache& draft_kv,
10140:     vt::Queue& queue, int64_t spec_step_idx) const {
10141:   const int64_t tokens = static_cast<int64_t>(input_ids.size());
10142:   const int64_t hidden_size = config_->hidden_size;
10143:   const int64_t num_layers = weights_->NumLayers();
10144:   VT_CHECK(tokens > 0, "qwen3_5 MTP paged forward: empty input_ids");
10145:   VT_CHECK(static_cast<int64_t>(positions.size()) == tokens,
10146:            "qwen3_5 MTP paged forward: positions length must equal token count");
10147:   VT_CHECK(spec_step_idx >= 0 && num_layers > 0,
10148:            "qwen3_5 MTP paged forward: invalid spec step/layer count");
10149:   VT_CHECK(target_hidden_states.rank == 2 &&
10150:                target_hidden_states.shape[0] == tokens &&
10151:                target_hidden_states.shape[1] == hidden_size &&
10152:                (target_hidden_states.dtype == DType::kBF16 ||
10153:                 (queue.device.type == vt::DeviceType::kXPU &&
10154:                  target_hidden_states.dtype == DType::kF16)) &&
10155:                target_hidden_states.IsContiguous() &&
10156:                target_hidden_states.device == queue.device,
10157:            "qwen3_5 MTP paged forward: target hidden states must be contiguous "
10158:            "bf16 [T,H] or xpu f16 [T,H] on the queue device");
10159:   // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): ONE precondition, two containers.
10160:   // The trellis stores [K=2H, N=H] where the torch Linear stores [N=H, K=2H],
10161:   // so the same projection is asserted through the orientation its own owner
10162:   // uses. `IsExl3()` is the loader's answer, not a second derivation.
```

**Source:** `code/src/vllm/model_executor/models/qwen3_5.cpp`, lines 10261–10300. SHA-256 `e8d9a6e6ec91a03caf3ef590f55d924e6c00d0bfae33aa5cbd54e8341c93857d`.

```text
10261: ForwardLogits Qwen3_5MTPModel::ComputeLogits(
10262:     const vt::Tensor& hidden_states, vt::Queue& queue) const {
10263:   const int64_t hidden_size = config_->hidden_size;
10264:   VT_CHECK(hidden_states.rank == 2 &&
10265:                hidden_states.shape[1] == hidden_size &&
10266:                (hidden_states.dtype == DType::kBF16 ||
10267:                 (queue.device.type == vt::DeviceType::kXPU &&
10268:                  hidden_states.dtype == DType::kF16)) &&
10269:                hidden_states.IsContiguous() &&
10270:                hidden_states.device == queue.device,
10271:            "qwen3_5 MTP logits: hidden states must be contiguous bf16 or xpu f16 [T,H] "
10272:            "on the queue device");
10273:   Dev device{vt::GetBackend(queue.device.type), queue};
10274:   if (weights_->IsGptq4Draft()) {
10275:     VT_CHECK(weights_->draft_lm_head_gptq4.k == hidden_size &&
10276:                  weights_->draft_lm_head_gptq4.n == config_->vocab_size,
10277:              "qwen3_5 MTP: packed draft head geometry mismatch");
10278:     DBuf half = dense_gptq4::Packed(
10279:         device, hidden_states, weights_->draft_lm_head_gptq4,
10280:         dense_gptq4::Projection::kMtpHead);
10281:     DBuf logits(device, DType::kF32,
10282:                 {hidden_states.shape[0], config_->vocab_size});
10283:     vt::CastF32(device.q, logits.t(), half.t());
10284:     return WrapDeviceLogits(device, std::move(logits), config_->vocab_size);
10285:   }
10286:   // MODEL-QWEN35-EXL3-HEAD (#2495 item 5). The arms are ordered
10287:   // `exl3 -> fp4 -> bf16`, which is `DenseLogitsF32D`'s order and
10288:   // `DflashLogitsF32D`'s, so the three readers of ONE target head cannot
10289:   // disagree about precedence. The trellis head is COMPUTED WITH, never widened:
10290:   // a dequantized copy of the real 248320x5120 head is 2.543 GB, and upstream
10291:   // reaches the packed path without a branch because `_apply_head` calls
10292:   // `lm_head.quant_method.apply` (logits_processor.py:132-142).
10293:   DBuf logits =
10294:       (lm_head_exl3_ != nullptr && !lm_head_exl3_->Empty())
10295:           ? dense_exl3::Linear(device, hidden_states, *lm_head_, *lm_head_exl3_,
10296:                                DType::kF32)
10297:           : ((lm_head_fp4_ != nullptr && !lm_head_fp4_->Empty())
10298:                  ? MatmulNvfp4F32D(device, hidden_states, *lm_head_fp4_)
10299:                  : MatmulF32D(device, hidden_states, *lm_head_));
10300:   return WrapDeviceLogits(device, std::move(logits), config_->vocab_size);
```

**Source:** `code/src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, lines 19–55. SHA-256 `37d5290657b4edadb9b146913d164004831bfcd0b6d09d149f3f8ac2900808ff`.

```text
19: // `_greedy_sample_draft` (spec_decode/speculator.py:276-280 @ 555967922, on
20: // `DraftModelSpeculator` (:69), which AutoRegressiveSpeculator inherits it from,
21: // and not on autoregressive/speculator.py): argmax over each named row of
22: // a device [rows, vocab] logits buffer, downloaded once. Lowest-index tie-break,
23: // matching our sampler's argmax. `rows` names which logits row each request
24: // samples from: the prefill's last_token_indices, and the identity on a decode
25: // step.
26: std::vector<int32_t> GreedySampleDraft(const vllm::ForwardLogits& logits,
27:                                        const std::vector<int64_t>& rows,
28:                                        vt::Queue& queue) {
29:   const int64_t vocab = logits.vocab;
30:   const int64_t num_rows = logits.rows;
31:   std::vector<float> host(static_cast<size_t>(num_rows) *
32:                           static_cast<size_t>(vocab));
33:   vt::Backend& backend = vt::GetBackend(queue.device.type);
34:   backend.Copy(queue, host.data(), logits.device_tensor.data,
35:                host.size() * sizeof(float));
36:   backend.Synchronize(queue);
37: 
38:   std::vector<int32_t> drafted(rows.size(), 0);
39:   for (size_t r = 0; r < rows.size(); ++r) {
40:     const int64_t row = rows[r];
41:     VT_CHECK(row >= 0 && row < num_rows,
42:              "MtpPropose: draft logits row out of range");
43:     const float* logit_row =
44:         host.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab);
45:     int32_t best_idx = 0;
46:     float best_val = logit_row[0];
47:     for (int64_t v = 1; v < vocab; ++v) {
48:       if (logit_row[static_cast<size_t>(v)] > best_val) {
49:         best_val = logit_row[static_cast<size_t>(v)];
50:         best_idx = static_cast<int32_t>(v);
51:       }
52:     }
53:     drafted[r] = best_idx;
54:   }
55:   return drafted;
```

**Source:** `code/src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, lines 106–129. SHA-256 `37d5290657b4edadb9b146913d164004831bfcd0b6d09d149f3f8ac2900808ff`.

```text
106:   // ── The one paged draft forward (I5c) + shared lm_head. ──────────────────────
107:   vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
108:       spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue);
109:   vllm::ForwardLogits logits = draft.ComputeLogits(hidden.tensor, queue);
110:   VT_CHECK(logits.on_device() && logits.rows == T,
111:            "MtpProposePrefill: unexpected draft logits shape");
112: 
113:   PrefillOutcome out;
114:   // ── Greedy draft pick over each request's last (sampled) row
115:   // (spec_decode/speculator.py:276-280). ──────────────────────────────────────
116:   out.sampled_rows.assign(
117:       spi.last_token_indices.begin(),
118:       spi.last_token_indices.begin() + static_cast<size_t>(num_reqs));
119:   out.draft_tokens = GreedySampleDraft(logits, out.sampled_rows, queue);
120: 
121:   // :346 — the positions of those same rows. The decode half advances from them.
122:   // The k=1 caller drops them, and dropping a host vector costs nothing.
123:   out.positions.resize(static_cast<size_t>(num_reqs));
124:   for (int64_t r = 0; r < num_reqs; ++r) {
125:     out.positions[static_cast<size_t>(r)] =
126:         positions32[static_cast<size_t>(out.sampled_rows[
127:             static_cast<size_t>(r)])];
128:   }
129:   out.hidden = std::move(hidden);
```

**Source:** `reference/exl3xpu/exl3xpu/vllm_plugin.py`, lines 334–374. SHA-256 `4c89fe610ce9d5aeabec0740eed1ab9a63136b7d885d54e273ed60ba1f52c0d6`.

```text
334: def _build_draft_head(layer, path):
335:     with open(path) as f:
336:         spec = json.load(f)
337:     n_total_blocks = layer.svh.shape[0] // 128
338:     ids=spec['blocks']
339:     if not ids or any(type(i) is not int or i<0 or i>=n_total_blocks for i in ids) or len(set(ids))!=len(ids):
340:         raise ValueError(f'exl3: draft vocabulary contains empty/invalid/duplicate blocks: {path}')
341:     blocks = torch.tensor(ids, dtype=torch.long)
342:     dev = layer.trellis.device
343:     tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
344:     trellis = layer.trellis.data.index_select(1, tiles).contiguous()
345:     svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
346:     idx = (blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
347:     layer.exl3_draft = dict(trellis=trellis, svh=svh, idx=idx,
348:                             shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
349:                             bounds=[0, len(blocks) * 128])
350:     logger.info("exl3: MTP draft head uses %d of %d vocab blocks (%.1f%% of lm_head)",
351:                 len(blocks), n_total_blocks, 100.0 * len(blocks) / n_total_blocks)
352: 
353: 
354: def _patch_mtp_draft_logits():
355:     try:
356:         from vllm.model_executor.models import qwen3_5_mtp
357:     except Exception:
358:         return
359:     cls = qwen3_5_mtp.Qwen3_5MTP
360:     if getattr(cls, "_exl3_patched", False):
361:         return
362:     orig = cls.compute_logits
363: 
364:     def compute_logits(self, hidden_states, spec_step_idx: int = 0):
365:         lm = self.lm_head
366:         d = getattr(lm, "exl3_draft", None)
367:         if d is None:
368:             return orig(self, hidden_states, spec_step_idx)
369:         from . import ops
370:         sub = torch.ops.exl3xpu_C.linear(hidden_states, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
371:                                          lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
372:         logits = hidden_states.new_full((hidden_states.shape[0], lm.svh.shape[0]), float("-inf"))
373:         logits.index_copy_(1, d["idx"], sub)
374:         return logits[:, : self.config.vocab_size]
```

## E13 — Explicit graph and recurrent-prefix integration gates

Existing generic infrastructure is useful but these guards must be replaced by tested contracts, not deleted. Graph pointer and state lifetime qualification belongs after a functional eager flow.

**Source:** `code/src/vllm/model_executor/models/qwen3_5_dense.cpp`, lines 204–219. SHA-256 `d014379e0cf63494bfd0254f1564c5244b15431083917a375f335f824bb9c348`.

```text
204:   // SPEC-MTP I5d-pre hidden-state tap. When the spec verify forward requests the
205:   // drafter's [T,H] post-final-norm hidden (I5d), route to the EXISTING
206:   // ForwardDeviceTap: byte-identical logits to ForwardDevice, plus the hidden
207:   // moved into *input.hidden_tap. Null (every spec-off run) falls through to the
208:   // unchanged path below, so the forward is byte-identical when spec is off.
209:   if (input.hidden_tap != nullptr) {
210:     if (weights.gptq4_checkpoint && Gptq4RouteTraceEnabled())
211:       std::fprintf(stderr,
212:                    "{\"event\":\"gptq4_route\",\"selected\":\"eager\","
213:                    "\"reason\":\"hidden_tap\",\"tokens\":%zu,"
214:                    "\"requests\":%d}\n",
215:                    input.token_ids.size(), input.num_reqs);
216:     return Qwen3_5DenseModel::ForwardDeviceTap(
217:         input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
218:         input.attn_kv, input.gdn_state, weights, input.config, input.queue,
219:         input.hidden_tap, input.logits_indices);
```

**Source:** `code/src/vllm/v1/worker/gpu/runner.cpp`, lines 756–767. SHA-256 `55a0887d3529e9360fb98a496be13e45555446d5a0d4c11e8734b6193cc9a8a5`.

```text
756:   // selects among, spec §3). Size the compact pool max_num_reqs*(num_spec+1) and
757:   // let remap_gdn_state_slots hand each sequence a base of num_spec+1 slots.
758:   const int spec_cols = spec_on() ? num_spec() + 1 : 1;
759:   const int64_t base_slots = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
760:   prefix_snapshot_base_ = base_slots * spec_cols;
761:   recurrent_prefix_snapshots_ = kv_cache_config.recurrent_prefix_snapshots;
762:   VT_CHECK(!recurrent_prefix_snapshots_ || !spec_on(), "recurrent prefix snapshots require non-speculative execution");
763:   gdn_state_slots_ = prefix_snapshot_base_ + (recurrent_prefix_snapshots_ ? recurrent_prefix_snapshots_->capacity() : 0);
764:   gdn_slot_of_req_.clear();
765:   gdn_free_slots_.clear();
766:   gdn_free_slots_.reserve(static_cast<size_t>(base_slots));
767:   // The free list holds each sequence's BASE state slot. Under speculation a
```

## E14 — Current safety mechanisms versus later overhead reduction

Output alias handling and metadata/workspace waits are deliberately conservative. Performance work must retain the safety they currently provide.

**Source:** `code/src/vt/xpu/xpu_common.h`, lines 108–159. SHA-256 `7116bdaa6a423e32138f4c39bedf03718b877a1cc1be76181e5a8cccf8ad3aa1`.

```text
108: void CheckDeviceMetadata(Queue& q, Check check, const char* message,
109:                           std::initializer_list<const Tensor*> inputs) {
110:   const auto check_start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
111:   if (CaptureMetadataCheck(q, [check](sycl::handler& h, int* result) {
112:         h.single_task([=] { *result = check() ? 1 : 0; });
113:       }, message, inputs)) {
114:     if (check_start) {
115:       const auto check_end = HostProfileClockNs();
116:       RecordHostProfileSpan(q, "metadata_validation_host", check_start,
117:                             check_end);
118:       RecordHostProfileSpan(q, message, check_start, check_end);
119:     }
120:     return;
121:   }
122:   Scratch scratch(q.device, sizeof(int));
123:   auto* result = static_cast<int*>(scratch.data);
124:   NativeQueue(q).single_task([=] { *result = check() ? 1 : 0; });
125:   int valid = 0;
126:   auto& backend = GetBackend(q.device);
127:   const auto wait_start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
128:   backend.Copy(q, &valid, result, sizeof(valid));
129:   backend.Synchronize(q);
130:   if (wait_start)
131:     RecordHostProfileSpan(q, "metadata_d2h_wait", wait_start,
132:                           HostProfileClockNs());
133:   if (check_start) {
134:     const auto check_end = HostProfileClockNs();
135:     RecordHostProfileSpan(q, "metadata_validation_host", check_start,
136:                           check_end);
137:     RecordHostProfileSpan(q, message, check_start, check_end);
138:   }
139:   VT_CHECK(valid, message);
140: }
141: // The correctness path snapshots only when the output could clobber an input.
142: // Temporary releases drain their final kernel/copy use through Backend::Free.
143: template<class Launch>
144: void WithOutput(Queue& q, Tensor& out, std::initializer_list<const Tensor*> inputs, Launch launch,
145:                 bool preserve = false) {
146:   bool alias = false;
147:   for (const auto* in : inputs) if (in && Overlap(out, *in)) alias = true;
148:   if (!alias) { launch(out); return; }
149:   VT_CHECK(static_cast<uint64_t>(out.Numel()) <= SIZE_MAX / SizeOf(out.dtype), "XPU scratch size overflow");
150:   Scratch scratch(q.device, static_cast<size_t>(out.Numel()) * SizeOf(out.dtype));
151:   Tensor temp = out;
152:   temp.data = scratch.data;
153:   int64_t stride = 1;
154:   for (int d = temp.rank - 1; d >= 0; --d) { temp.stride[d] = stride; stride *= temp.shape[d]; }
155:   if (preserve) vt::Copy(q, temp, out);
156:   launch(temp);
157:   vt::Copy(q, out, temp);
158:   // Surface asynchronous failures before the nonthrowing cleanup.
159:   GetBackend(q.device).Synchronize(q);
```

## E15 — Large-M donor is rotated W8A8, not legacy FP16 reconstruction

The donor first selects SmallM, then W8A8 for eligible larger input. With EXL3_DNNL, rows are padded to 256 and INT8 weights are reconstructed for each complete source group. The separate FP16 slice setting does not bound this weight allocation.

**Source:** `reference/exl3xpu/exl3xpu/ops.py`, lines 10–18. SHA-256 `9c05ea0aefca817fd37cee27f6e6216b6cee2d2a16c6480cafff52dc5ff67c13`.

```text
10: 
11: # Rows at or below this use the fused decode-in-GEMM kernel; above it we reconstruct fp16
12: # weight slices and use the oneDNN GEMM (compute bound regime).
13: SMALL_M_MAX = int(os.environ.get("EXL3_SMALL_M_MAX", "128"))
14: # Prefill (M > SMALL_M_MAX) GEMMs in int8 XMX (2x fp16 rate): per-token activation scale (Hadamard-rotated
15: # activations have no outliers), one static weight scale (the mul1 codebook bound). Opt-in. The C++ op fuses the
16: # quantization into had_in / reconstruct / had_out; the Python fallback below is the unfused prototype.
17: INT8_PREFILL = os.environ.get("EXL3_INT8_PREFILL", "0") == "1"
18: RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))
```

**Source:** `reference/exl3xpu/csrc/exl3_ops.sycl`, lines 576–590. SHA-256 `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`.

```text
576: // W8A8 prefill (opt-in): mul1 codebook values are bounded by 3.453125, one static int8 scale for every tensor
577: static bool g_int8 = false;
578: void exl3_set_int8(int64_t on) { g_int8 = on != 0; }
579: constexpr float kQ8WMax = 3.453125f;
580: 
581: template <int K>
582: static void launch_reconstruct_q8(const at::Tensor& tr, at::Tensor& w, int n0, sycl::queue& q) {
583:     int tk = tr.size(0), tiles_n = tr.size(1), n_out = w.size(1);
584:     constexpr int NT = 8;
585:     int n_strips = n_out / (16 * NT);
586:     ReconstructKernel<K, 2, NT, int8_t> k{reinterpret_cast<const uint32_t*>(tr.data_ptr()),
587:                                           reinterpret_cast<int8_t*>(w.data_ptr()), tiles_n, n0 / 16, n_out, n_strips, tk,
588:                                           127.0f / kQ8WMax};
589:     int n = n_strips * tk, local = 8;
590:     q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, local), local), k); });
```

**Source:** `reference/exl3xpu/csrc/exl3_ops.sycl`, lines 701–771. SHA-256 `b70880d1e92032de697c1d432c6899e34e17662e2086a7fbba80fa17b28e41cf`.

```text
701: at::Tensor exl3_linear(at::Tensor x, at::Tensor trellis, at::Tensor suh, at::Tensor svh, at::Tensor shard_of_nb,
702:                        std::vector<int64_t> group_bounds, int64_t K, int64_t cb, int64_t small_m_max,
703:                        int64_t slice_n) {
704:     auto shape = x.sizes().vec();
705:     int64_t k = shape.back();
706:     int64_t n = svh.size(0);
707:     auto x2 = x.reshape({-1, k});
708:     if (!x2.is_contiguous()) x2 = x2.contiguous();
709:     int64_t M = x2.size(0);
710:     shape.back() = n;
711:     auto out = at::empty({M, n}, x.options());
712:     if (M == 0) return out.view(shape);
713:     if (M <= small_m_max && exl3_supported(K, cb)) {
714:         exl3_gemm_small(x2, trellis, suh, svh, shard_of_nb, out, K, cb);
715:         return out.view(shape);
716:     }
717:     int64_t S = suh.size(0);
718:     if (g_int8 && cb == 2 && (K == 4 || K == 6) && M > 16) {
719:         auto& q = queue_of(x2);
720: #ifdef EXL3_DNNL
721:         // GEMM rows padded to 256: bounds the number of oneDNN primitives (one per (Ms, K, N)) on real traffic
722:         const int64_t Ms = (M + 255) / 256 * 256;
723: #else
724:         const int64_t Ms = M;
725: #endif
726:         auto xq = at::empty({S, Ms, k}, x.options().dtype(at::kChar));
727:         auto sx = at::empty({S, Ms}, x.options().dtype(at::kFloat));
728:         int n_in = S * M, local = 8;
729:         auto run_in = [&](auto tag) {
730:             using TIn = decltype(tag);
731:             int kb_n = (int)(k / 128);
732:             auto wg = [&](auto kern, int tpr) {
733:                 q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)n_in * tpr, tpr), kern); });
734:             };
735:             if (kb_n <= 8 * 6) {
736:                 wg(HadInQ8WgKernel<TIn, 8, 6>{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
737:                                               reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
738:                                               (int)x2.stride(0), (int)Ms}, 8);
739:                 return;
740:             }
741:             if (kb_n <= 16 * 9) {
742:                 wg(HadInQ8WgKernel<TIn, 16, 9>{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
743:                                                reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
744:                                                (int)x2.stride(0), (int)Ms}, 16);
745:                 return;
746:             }
747:             HadInQ8Kernel<TIn> kin{reinterpret_cast<const TIn*>(x2.data_ptr()), reinterpret_cast<const fp16*>(suh.data_ptr()),
748:                                    reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), (int)M, (int)k, (int)S,
749:                                    (int)x2.stride(0), (int)Ms};
750:             q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n_in, local), local), kin); });
751:         };
752:         if (x2.scalar_type() == at::kHalf) run_in(fp16{});
753:         else if (x2.scalar_type() == at::kBFloat16) run_in(sycl::ext::oneapi::bfloat16{});
754:         else TORCH_CHECK(false, "x dtype");
755:         const float sw = kQ8WMax / 127.0f;
756: #ifdef EXL3_DNNL
757:         {
758:             static std::vector<at::Tensor> swd(16);
759:             auto& swt = swd.at(x.device().index());
760:             if (!swt.defined()) swt = at::full({1}, sw, x.options().dtype(at::kFloat));
761:             auto y = at::empty({Ms, n}, x.options().dtype(at::kHalf));
762:             for (size_t g = 0; g + 1 < group_bounds.size(); ++g) {
763:                 int64_t g0 = group_bounds[g], g1 = group_bounds[g + 1], ng = g1 - g0;
764:                 auto w = at::empty({k, ng}, x.options().dtype(at::kChar));
765:                 if (K == 4) launch_reconstruct_q8<4>(trellis, w, g0, q); else launch_reconstruct_q8<6>(trellis, w, g0, q);
766:                 q8dnnl::gemm(q, x, reinterpret_cast<const int8_t*>(xq.data_ptr()) + g * Ms * k,
767:                              reinterpret_cast<const int8_t*>(w.data_ptr()), swt.data_ptr<float>(),
768:                              sx.data_ptr<float>() + g * Ms, reinterpret_cast<fp16*>(y.data_ptr()) + g0, Ms, k, ng, n);
769:             }
770:             exl3_had_out_h(y.narrow(0, 0, M), svh, out);
771:             return out.view(shape);
```

---

End of plan. The first next action is R01; the first unresolved numerical issue has its own bounded R02/§6 investigation. Do not restart the prior monolithic S1 goal.
