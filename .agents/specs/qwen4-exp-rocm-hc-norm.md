# ROCm `HcGroupedNormKernel`: one block per group, fp32 accumulate

Row: `MODEL-MM-QWEN4-EXP`
Issue: `ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F` (row-owned, filed by the CUDA
twin's landing `ee0644eab` on 2026-09-13, `OPEN`)
Donor: `src/vt/cuda/cuda_qwen4_exp.cu` at `ee0644eab`

## 1. Scope

ONE kernel and its launch site:

- `src/vt/rocm/rocm_qwen4_exp.hip:220` -- `HcGroupedNormKernel`
- `src/vt/rocm/rocm_qwen4_exp.hip:361` -- `<<<GridFor(T * hc), kBlock, ...>>>`

and the gate that is missing for it. Nothing else in the file moves: the
write-back, the SiLU scale, the mix and the inject kernels are already
element-parallel and are not touched.

OUT OF SCOPE. The per-call `hipMalloc` scratch (recorded as a SPEED item in the
flash-next spec's `## Owed`), the three `vt::MatmulBT` projections, the CUDA arm
(already landed), and the ROCm arms of any other `qwen4_exp` op.

## 2. The defect

The ROCm arm launches **one thread per (token, hc stream) group** and each
thread walks its whole group serially, twice, accumulating the sum of squares in
`double` with `__dadd_rn`/`__dmul_rn`.

The released `Qwen3.8-Flash-Next` has `hc_count = 4` (`qwen4_exp.h`) and decode
runs at `T = 1`, so **the kernel does its work with FOUR ACTIVE THREADS**, each
walking `H = 2560` elements twice, on a 40-CU gfx1151.

MEASURED on `strix:gpu0`, 2026-09-13, `rocprofv3 --kernel-trace` ranked by
`scripts/rocm-rank-kernels.py` over 660,273 dispatch rows with the window
validated at one dispatch count per step:

| | |
|---|---:|
| share of decode kernel time | **35.10%** |
| dispatches per step | 97 |
| mean duration | 626.9 us (613.6 -- 652.7) |
| step wall median | 197.50 ms |
| step kernel-busy | 173.24 ms |
| implied per step | ~60.8 ms |

It is the top of the kernel budget on this board, and it was the top of the
budget on GB10 too (40.7%, `nsys`, 626.94 ms over 1,439 instances) until
`ee0644eab` fixed the CUDA arm and left this one behind.

## 3. Is the summation ORDER load-bearing? -- the question answered BEFORE the change

A parallel reduction re-associates the sum, and floating-point addition is not
associative. Two separate questions, answered separately.

### 3a. Is bit-exactness with the CPU arm REQUIRED? NO.

Decided at `src/vllm/model_executor/models/qwen4_exp_hc.h:99-104`, which is the
CPU reference's own header and predates both device arms:

> fp32 interior throughout, with the per-group sum of squares accumulated in
> double -- the `deepseek_v4_mhc.cpp` house convention for a host reference.
> Upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise
> (`x = x.float()`), so nothing here is a widening of the model path: this is a
> CPU reference and **the device arm is the thing that must be fp32-accumulate
> and gated against these numbers.**

So the `double` in the device arm is an INHERITANCE, not a requirement, and the
contract says fp32. The ROCm file's own header (`:49-55`) argues the opposite
-- that the width is "inherited, not chosen" and must not be unified with
`vt::RmsNormGroup`'s f32 -- and that paragraph is REPLACED rather than amended,
exactly as `ee0644eab` replaced the CUDA one.

Independently, the arm cannot be bit-identical to the CPU sibling anyway and
never was: its three projections go through `vt::MatmulBT`, a device GEMM that
re-associates the K reduction. `rocm_qwen4_exp.hip:199-201` already states this
and names the tolerance as the gate.

### 3b. Does anything GATE the current order? -- MEASURED, not assumed

The only committed test that reaches the ROCm arm of this op is
`tests/vt/test_backend_cross_device.cpp:5580`, "qwen4_exp gated-residual MIXER
matches the CPU oracle and is NATIVE on ROCm", at `kH = 7`, `kHc = 3`, `T = 4`,
bar `kNmseTol`.

A **seven-element group** cannot express a reduction shape. One thread does
seven adds either way.

MUTATION M0, run against the UNCHANGED kernel to answer this empirically rather
than by argument: reverse the per-group walk (`for h = H-1 down to 0`) in
`HcGroupedNormKernel`'s sum-of-squares loop, leaving everything else alone.

> `[RESULT M0 -- filled from the measurement, see §7]`

This row does not ship a reassociation into a path whose numerics nothing
measures. §5 adds the fixture that can see the shape first.


### 3c. There IS a fixture that convicts an f32 accumulator, and it runs CPU-ONLY

`tests/vllm/models/test_qwen4_exp_hc_device.cpp:504`, "the grouped norm needs a
WIDER-THAN-f32 accumulator": `H = 2560`, `hc = 4`, every element 1.0f except one
dominant element per group at `{4096, 2048, 8192, 1024}`. Its recorded band at
`:606` is `kAccumBound = 1e-5`, and the header records what it measured -- a
`double` accumulator lands `1.173e-06` and a `float ss` lands `6.702e-04`, a
**571x** separation.

That case never runs a device arm. It is the CPU reference's own gate, and the
reference is the thing that must be `double`. So moving the ROCm arm to f32 does
NOT red it -- and that is precisely why it has to be said out loud rather than
discovered later: **on magnitude-separated data an f32 device arm is a worse
approximation of the reference, deliberately, because fp32 is what upstream and
vLLM compute.**

The separation is not 571x here, because that number is for a SERIAL f32 walk.
A block tree over 2560 elements has depth ~10 instead of 2560, so its error
grows with the tree depth and not with `H`. HOW MUCH BETTER IS A MEASUREMENT AND
NOT AN ARGUMENT: case 5 of §5 replays this exact fixture against the ROCm arm and
holds it to the same `1e-5`. If the tree does not make that bar, the number is
recorded and the bar is restated with its derivation -- it is never widened to
let the kernel pass.

## 4. Design

Mirror the donor. One block per `(token, hc)` group; a strided per-thread
partial; a cross-lane tree; a cross-wave fold through shared memory; the
normalize-and-write loop strided across the block. `GridForGroups` caps the grid
at 4096 blocks exactly as `GridFor` does, so the group loop stays a grid stride.

ONE DIFFERENCE FROM THE DONOR, AND IT IS THE ONLY ONE. CUDA's tree is written
against a fixed 32-lane warp (`__shfl_down_sync(0xffffffffu, part, off)`, `off`
from 16). AMD wavefronts are 32 on RDNA and 64 on CDNA, so every lane constant
here is derived from `warpSize`:

```
for (int off = warpSize / 2; off > 0; off >>= 1)
  part = __fadd_rn(part, __shfl_down(part, off));
```

`warpSize` is `__builtin_amdgcn_wavefrontsize()` and is constant-folded per
target, so this costs nothing and the file stays gfx-family agnostic. The shared
partial array is sized `kBlock / 32` -- the MAXIMUM wave count, reached on
wave32; a wave64 target uses the first four slots and the `lane < nwave` guard
keeps it off the rest.

The file's header paragraph "NO MFMA, NO DP4A, NO WAVEFRONT ASSUMPTION" becomes
false in its second clause and is rewritten: the file now uses a cross-lane
primitive, and the statement that survives is that it makes no assumption about
the wavefront SIZE.

WIDTH. `double` -> `float`, for §3a's reason. The `1 +` gamma fold stays f32 and
stays where it is (#2218); eps stays inside the rsqrt, added to the mean square;
the stream stays read-only.

BARRIERS. Two, both inside the loop body, neither trailing -- the donor's shape
and the donor's argument, which rests only on the trip count being
block-uniform. It is: `g` is seeded from `blockIdx.x` and stepped by
`gridDim.x`. HIP's `__syncthreads()` is the same barrier. The donor's racecheck
measurement is CUDA's and is NOT inherited as evidence; see `## Owed`.

## 5. Tests

`tests/vllm/models/test_qwen4_exp_rocm_reductions.cpp`, the ROCm twin of the
donor's `CUDA W7:` battery, registered in `tests/CMakeLists.txt`.

The four W7 helpers -- `W7NormRel`, `SynthHc`, `MakeSynthHc`, `RunSynthMixer`
-- move to `tests/vllm/models/qwen4_exp_hc_synth.h` and the CUDA file includes
them instead of defining them. They are NOT copied. A second copy of
`W7NormRel`'s derivation is a duplicated fact, and repairing one copy of a
duplicated fact creates the next contradiction. `RunSynthMixer`'s device branch
takes its `DeviceType` from the argument instead of hard-coding `kCUDA`, which
is the one behavioural change in that move and is why the CUDA cases still pass
unchanged.

Cases, each named with the literal `ROCM W7` so `-tc='*ROCM W7*'` selects the
battery alone (the donor's rename trap: a filter that matches nothing prints
`Status: SUCCESS!` and exits 0 -- run every filtered arm through
`scripts/run-doctest-selected.sh`, and report case AND assertion counts):

1. **MODEL WIDTH** -- `hidden 2560, hc 4, lowrank 320, T 3`. Ten elements per
   thread at a 256-thread block, every wave live. The goldens' `hidden 6` and
   the cross-device case's `hidden 7` reach neither.
2. **PAST THE GRID CAP** -- 4800 groups, so the group loop takes a SECOND trip
   and both shared slots are re-read after a block has used them.
3. **RAGGED** -- `hidden 100` (under the block, most waves contribute an exact
   zero, and the cross-wave stage must not read a slot no wave wrote) and
   `hidden 777` (a multiple of neither 32, 64 nor 256, so the last strided trip
   is ragged on either wavefront size). `hc = 3` keeps the stream count off a
   power of two.
4. **SEPARATION PROBE** -- replays case 1 with every group of token 0 forced to
   group 0's data and PRINTS the margin, so a passing gate says how far it
   passed rather than only that it passed.

5. **MAGNITUDE-SEPARATED** -- `test_qwen4_exp_hc_device.cpp:504`'s fixture
   (`H 2560, hc 4`, all ones but one dominant element per group at
   `{4096, 2048, 8192, 1024}`) against the CPU `double` reference, held to that
   case's own `1e-5`. This is the ONE case where the width change, not the
   ordering change, is what is under test. See §3c.

Bound: `W7NormRel(H)` -- the donor's derived reduction-width bound, unchanged
and not refitted. `CheckBitwise(hyper_after, hyper)` on every case: the stream
is read-only, and a normalize-in-place slip would double-normalize at the second
site of every layer.

RED-FIRST. Expected UNAVAILABLE on this fixture for the same reason the donor
recorded: before the change both arms walk the group ascending in `double`, so
they agree exactly and the new cases pass on the unchanged kernel. That is a
measurement to take, not a prediction to assume -- §7 records what it read. The
discriminating evidence is therefore the §6 mutation battery, and the cases
above are justified by WHICH mutations they convict, not by an initial red.

## 6. Mutation battery (IMP-MUTATE)

Each mutation is applied alone to the FIXED kernel, the binary is proved changed
(`sha256` both arms -- a surviving mutation whose binary did not change measured
nothing), the battery is run through `run-doctest-selected.sh`, and the tree is
restored byte-for-byte (`git diff --exit-code`).

| | mutation | must convict |
|---|---|---|
| M1 | delete the cross-wave fold (use warp 0's partial as the whole sum) | yes |
| M2 | off-by-one the strided walk (`h += blockDim.x + 1`) | yes |
| M3 | drop the final partial (`h + blockDim.x < H` bound) | yes |
| M4 | drop the group grid stride (`g += gridDim.x` -> `break`) | yes, case 2 only |
| M5 | broadcast `r` from group 0 | yes, the probe |
| M6 | accumulate in `double` again (the pre-change width) | case 5 must MOVE; if it does not, case 5 is not measuring the width |

## 7. Evidence

> filled by the measurement. Archived under
> `docs/bench-evidence/qwen4exp-rocm-hcnorm-gfx1151-20260913/` and to
> `/workspace/`, because `rc` logs age out within a day.

## 8. Prediction, stated so it can be falsified

On `strix:gpu0` the kernel is 35.10% of 173.24 ms of kernel-busy, ~60.8 ms of a
197.50 ms step. The donor removed most of its own share on thor. If this change
removes the same FRACTION here, the step falls toward ~140 ms and decode moves
from 5.0-5.3 tok/s toward ~7 tok/s.

THE PREDICTION IS NOT A RESULT, and three things can make the measured number
smaller. The step is only 88% kernel-busy, so a kernel saving does not convert
one-for-one. The remaining 64.9% of the budget does not shrink. And gfx1151 is
an APU whose 40 CUs are fed by host memory, so a kernel that becomes
bandwidth-bound rather than latency-bound stops improving. The direct evidence
is therefore the kernel's own new SHARE from `scripts/rocm-rank-kernels.py`, not
tok/s.

## 9. Risks

- **Wavefront size.** gfx1151 is wave32. A wave64 target (gfx90a, gfx942) takes
  the same code path with four slots instead of eight, and the `lane < nwave`
  guard is what holds it. NOT MEASURED on a wave64 board -- see `## Owed`.
- **Shared-memory hazards.** The donor's ordering argument is inherited but its
  `compute-sanitizer racecheck` run is not; ROCm's equivalent is
  `rocgdb`/`--tool` coverage this row does not have. Case 2 is the only case
  that takes a second trip and is the one to drive any future check at.
- **gfx1151 illegal-memory-access race.** `ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`
  -- about 2 of 5 identical greedy runs die on this board for reasons that
  predate this row. Every crash is checked against that signature before it is
  attributed here.
- **No cross-engine ratio is admissible.** `qwen4_exp` on ROCm has no token-exact
  gate (`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`): llama.cpp aborts on this
  architecture and no vLLM revision registers it. Single-engine facts and
  within-engine A/Bs only.

## 10. Stop conditions

- The reduction cannot hold `W7NormRel(2560)` against the CPU arm -> STOP and
  report. A fast wrong norm is worthless.
- A mutation in §6 survives with a proven-changed binary -> the fixture does not
  gate what this change claims; fix the fixture before landing the kernel.
- The full ROCm cross-device suite moves off its measured baseline at the base
  commit -> STOP. The baseline is READ at `60990ee78`, never assumed.

## Now

`ACTIVE`. Spec committed; implementation and the `strix:gpu0` measurement
follow in the same pull request.

## Owed

- **A wave64 measurement.** Only gfx1151 (wave32) is in this fleet, so the
  `warpSize`-derived tree is REASONED correct on wave64 and MEASURED on wave32
  alone.
- **A ROCm race check on the shared-memory shape.** The donor gates its barriers
  with `compute-sanitizer --tool racecheck`; no equivalent was run here, and the
  ordering argument is carried as an argument.
