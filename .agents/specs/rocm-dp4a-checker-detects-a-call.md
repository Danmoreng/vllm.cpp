# The dp4a gate has to see a call, and it only sees a string

Issue: `ISSUE-LOCAL-01M2TQYXN85A4FK48B8N2WFQ8G`
Row: `BACKEND-ROCM`

## Why

`scripts/check-rocm-dp4a-intrinsic.py` exists because a CPU-only `ctest` run
cannot compile `src/vt/rocm/rocm_grouped_gemm.hip`, so nothing else in the tree
notices if the hardware dot product leaves `Dp4a`. The checker reads the source
and is the whole guarantee that `v_dot4_i32_i8` is still emitted.

It no longer is one. `tests/scripts/test_check_rocm_dp4a_intrinsic.py:126`
mutates the real source and the checker returns an empty error list.
`scripts/agent-preflight.sh` reports `1 gate(s) failed` on `main` for this.

The mutation replaces the call line, and the mutated `Dp4a` body reads:

```
#if defined(__has_builtin)
#if __has_builtin(__ockl_sdot4)
  int sum = 0;
  ...                        <- the scalar expansion, no intrinsic anywhere
#endif
#endif
  // Fallback: scalar expansion
  ...
```

Zero call sites. Both of the checker's disjuncts pass anyway, each on its own,
because the token `__ockl_sdot4` survives on the `__has_builtin` **probe** line
and both tests are substring tests:

- `INTRINSIC in body` matches the probe.
- `"__has_builtin(__ockl_sdot4)" in body` matches the probe by construction.

So the second disjunct can never fail while the first passes, and the first
passes on a body that calls nothing. Measured in-tree before any change:
`body.count("__ockl_sdot4") == 1`, real call sites `== 0`, both flags `True`.

## Which half is stale

The checker. `git log --follow` on the three files settles it:

| File | Last touched |
|---|---|
| `tests/scripts/test_check_rocm_dp4a_intrinsic.py` | `6fd5443fa` |
| `scripts/check-rocm-dp4a-intrinsic.py` | `2bde17f6c` |
| `src/vt/rocm/rocm_grouped_gemm.hip` | `2bde17f6c` (for this hunk) |

`2bde17f6c` ("fix(ROCM): add conditional `__ockl_sdot4` intrinsic with scalar
fallback") changed the source and the checker in one commit and did not touch
the test. Its own message says *"Update checker to accept conditional
fallback pattern"*, and the diff adds the `has_conditional_fallback` disjunct.
That is the shape AGENTS.md names: *"Never make a red gate green by deleting an
assertion or widening its scope."* The test is the half that did not move, so
the test is not the stale one.

The **product** intent of `2bde17f6c` is legitimate and is kept: `__ockl_sdot4`
is absent from some ROCm installs, so the call is guarded and a scalar fallback
follows it. The defect is that the checker was widened to accept the guard
*instead of* the call rather than *around* it.

## Scope

- `scripts/check-rocm-dp4a-intrinsic.py`: detect a genuine call site.
- `tests/scripts/test_check_rocm_dp4a_intrinsic.py`: pin the new shape, in both
  directions, and repair one case that does not test what it is named for.

Out of scope: `src/vt/rocm/rocm_grouped_gemm.hip`. The live source is correct
under the restored rule and is not edited. No kernel, build or ROCm behaviour
changes; this is a checker-only change and needs no GPU.

## Design

The rule, restated so the checker and this file say the same thing: **the
`Dp4a` body must contain at least one genuine call to `__ockl_sdot4`.** A
`#if __has_builtin(__ockl_sdot4)` guard around that call is allowed, and a
scalar fallback after it is allowed. Neither is a substitute for the call.

Detection, in order:

1. Strip `//` and `/* */` comments from the extracted body, so an intrinsic
   named only in prose cannot satisfy the gate.
2. Delete every `__has_builtin( __ockl_sdot4 )` probe, whitespace-tolerant.
   The probe asks whether the compiler knows the name; it is not a use of it.
3. Require `__ockl_sdot4\s*\(` to match what is left.

The `has_conditional_fallback` disjunct is removed. It is not an assertion
being deleted: it is the widening, and after step 2 it cannot be satisfied by
anything the first test does not already cover.

`_SCALAR_MARKERS` is removed in the same change. It is dead — defined, never
read — and it describes a rule (`a * b` means regression) the checker has never
enforced, so a reader of the checker is misled about what the gate does.

## Risks

- **Over-tight regex refuses the live tree.** Mitigated by
  `test_live_tree_passes` and by a new case that builds the exact live shape
  (guarded call plus scalar fallback) in a scratch tree.
- **A future legitimate spelling is refused**, for example a call through a
  macro or a function pointer. Accepted: the checker is a source-shape gate by
  construction, and a spelling change to `Dp4a` should re-read this gate.
- **Vacuous green elsewhere in the file.** Addressed below rather than assumed.

## Tests

Red before: `test_live_scalar_mutation_fails` fails with `0 != 1 : []`.

Green after, plus these new cases:

| Case | Asserts |
|---|---|
| `test_has_builtin_probe_alone_fails` | the probe without a call is red — pins the removed disjunct dead |
| `test_intrinsic_in_comment_only_fails` | prose is not a call |
| `test_guarded_call_with_scalar_fallback_passes` | `2bde17f6c`'s real shape is still accepted |
| `test_missing_dp4a_function_fails` | repaired: it now removes the signature, so it reaches the `Dp4a function not found` branch it is named for |
| `test_empty_dp4a_body_fails` | the case the old `test_missing_dp4a_function_fails` actually ran, kept under an honest name |

Audit of the five pre-existing cases for vacuity is recorded in `## Outcome`.

## Gates

`python3 tests/scripts/test_check_rocm_dp4a_intrinsic.py`,
`python3 scripts/check-rocm-dp4a-intrinsic.py`, and
`scripts/agent-preflight.sh`. No GPU lease is required and none is taken.

## Evidence

Both mutation directions, run by hand and recorded in the pull request body:

1. Mutate the **source**: replace the guarded call with the scalar expansion in
   the real `.hip` and confirm the repaired checker reports one error.
2. Mutate the **checker**: defeat the call detection and confirm the test file
   goes red, then restore and verify with `sha256sum -c`.

A restored `.py` can still run mutant `__pycache__` bytecode, so
`__pycache__` is purged between arms and every restore is hash-verified.

## Stop conditions

- Stop and return `NEEDS_DECISION` if the repaired checker refuses the live
  tree, because that would mean the source, not the checker, is the stale half.
- Stop if repairing detection would require editing
  `src/vt/rocm/rocm_grouped_gemm.hip`; that is a different change with a
  different reviewer.

## Now

`BACKEND-ROCM` does not change lifecycle state. This is a gate repair inside
the row, not a capability move.
