# W2: exact native top-20 candidate, P4096/O1024 (2026-09-28)

The production sampler requests temperature 1, top-p .95 and top-k 20 on
each of up to five expanded MTP4 verification rows. The previous XPU path
stable-sorted all 248,320 vocabulary entries. The new native candidate finds
the best 20 IDs per 256-element tile, merges the tile winners, checks the
whole row for non-finite values and ties at rank 20, then applies the top-p
cutoff. It returns without mutating logits when the p boundary is numerically
close, a tie/non-finite value occurs or p is invalid. The caller then uses
the existing stable sort. `VT_B70_FAST_TOPK20=0` selects the old route for A/B;
the candidate is the default only when every row has k=20 and p is supplied.
No Python code runs in the C++ inference path.

## Correctness and isolated cost

The focused XPU test compares the resulting full-vocabulary F32 masked logits
bit-for-bit with the old stable-sort route for 257, 1025 and 248320 entries,
five rows and p values .05/.37/.95/1/.999. Tied logits and a near-uniform
top-p boundary leave logits unchanged and take the fallback. Result: 24/24
assertions, zero reference-tier hits. In a separate five-row, peaky-logit
microbenchmark, median host-observed operation time over five repetitions was
5.07414 ms for full sort and 0.885119 ms for top-20. The microbenchmark is
diagnostic; the full-model result below is the promotion evidence.

## Matched full-model A/B

Pinned local GPTQ-G128 checkpoint, B70, C1, eager, FP8 E4M3 KV, automatic
1664-token hybrid attention page, native MTP4, separate INT4 draft head and
five packed draft linears, no prefix cache, 4096-token retrieval prompt,
exactly 1024 emitted tokens, seed 42, ignore EOS, T1/p.95/k20. The benchmark
warms O8, measures a separate O1 TTFT, then measures O1024. The reported
decode rate is 1023 / (O1024 wall - separate O1 TTFT); it is not an
instrumented first-to-last-token interval. Each row below is a separate
warmed process. The output ID-sequence FNV hash was recorded on every row.
The prompt-ID hash was added for the final post-default run. The temporary
`VT_XPU_GDN_NATIVE=1` A/B probe was
excluded because it selects a different prefill route and changes the MTP
trajectory.

| Order | Route | O1 TTFT s | O1024 wall s | Derived decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | candidate | 3.98717 | 28.1682 | 42.3058 |
| 2 | full sort | 3.98208 | 28.7168 | 41.3589 |
| 3 | full sort | 3.92692 | 28.6899 | 41.3116 |
| 4 | candidate | 3.95201 | 28.3802 | 41.8778 |
| 5 | candidate | 3.94180 | 28.3745 | 41.8700 |
| 6 | full sort | 3.93131 | 28.8113 | 41.1174 |
| 7 | candidate | 3.94059 | 28.3811 | 41.8568 |
| 8 | full sort | 3.96205 | 28.8749 | 41.0631 |
| 9 | full sort | 4.01193 | 28.8907 | 41.1194 |
| 10 | candidate | 3.94984 | 28.3999 | 41.8404 |

Median full sort **41.1194**, candidate **41.8700 emitted decode tok/s**,
**+1.83%**. Ranges were 41.0631–41.3589 and 41.8404–42.3058 respectively;
there was no overlap. Mean gain was 1.84%. Every run emitted 1024 tokens,
proposed 1012 drafts, accepted 770, and used 253 draft steps. All runs with
the output-hash field recorded the same FNV64 ID-sequence hash
`6972477010856893408`. The final post-default run, with no selector
environment variable, passed 50/50 assertions, recorded prompt-ID FNV64
`4760835697920107937`, the same output hash and acceptance counters, and
measured 42.219 derived decode tok/s. Candidate tracked peak GPU allocation
was 27,936,844,838 bytes versus 27,936,456,539 for full sort in the A/B.

The earlier matched Python eager reference measured 71.83 derived decode
tok/s under the same request profile. The candidate's five-run median is
about 58.3% of that rate. This is still a single prompt/seed and separate-O1
timing scope, not production graph/C4 parity or a matched-prefix Python
distribution test. W4 rowwise prefix-dependent processors and W5 faster
verification remain open.

## Reproduction

Builder: `local/b70-oneapi-2026.1.1-builder:vllm030`, oneDNN 3.13 at
`/home/sebastian/LocalLLM/b70-onednn-3.13-install`, build tree
`build-b70-xpu/cmake`. Focused targets:

```sh
cmake --build /build/cmake --target test_xpu_sampling test_xpu_qwen_checkpoint -j4
```

The full-model container mounts `build-b70-xpu` at `/build`, the checkpoint
at `/models`, and oneDNN at `/opt/dnnl`; it passes `/dev/dri`, video/render
groups and `--ipc=host`. Its environment is:

```text
VT_B70_MODEL_DIR=/models VT_B70_KV_DTYPE=fp8
VT_B70_BLOCK_SIZE=1600 VT_B70_NUM_BLOCKS=20
VT_B70_BATCH_TOKENS=4096 VT_B70_MTP_K=4
VT_B70_SAMPLED_MTP_BENCH=1 VT_B70_PROMPT_TOKENS=4096
VT_B70_MAX_TOKENS=1024
```

Run `/build/cmake/tests/test_xpu_qwen_checkpoint --no-colors` with
`LD_LIBRARY_PATH=/opt/dnnl/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH`.
Set `VT_B70_FAST_TOPK20=0` for the full-sort control; leave it unset for the
default candidate. The five-row diagnostic uses
`VT_B70_SAMPLING_BENCH=1` and
`--test-case="XPU sampling: five-row production top-20 timing"` on
`/build/cmake/tests/test_xpu_sampling`.
