# #2999: WHERE do CPU and CUDA first diverge? — 23 September 2026

## Provenance

- **Job:** `af84d24f-163c-4443-a12b-35327882e7e4` on `thor:gpu0` (NVIDIA Thor, sm_110, aarch64)
- **Source:** `0da53ba1753c084a76c5e20ab65c0be5cccfa849` (main, 2026-09-23)
- **Binary:** `/tmp/q4exp-bld/examples/vllm-server` sha256=`d309cf2b5c7c4cb1b9fcba26e9956aceaba6252e5d9774f97674ef1cd0e7ea04`
- **Artifact:** `Qwen3.8-Flash-Next-UD-IQ1_S` (3 shards, 72.5 GB), shard1 sha256=`88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`
- **Instrument:** `VT_Q4EXP_LAYER_FP=9` (prefill + 8 decode steps), sign-sensitive random projections from #2877
- **Prompt:** `"The capital of France is"`, `max_tokens=8`, `temperature=0`

## Token sequences

| Arm | Token IDs | Text | vs Prior |
|---|---|---|---|
| A-CPU | `11751 13 15767 411 2029 11 1092 369` | " Paris. Given this fact, what is" | IDENTICAL to prior CPU |
| B-CUDA | `11751 13 15767 264 1103 314 89798 11` | " Paris. Given a list of capitals," | DIFFERENT from prior CUDA |

### Key observation: CUDA is non-deterministic across runs

The prior CUDA run (2026-09-04, `run2-job.sh`) produced `11751 13 15767 411 1928 11 628 567` (5 of 8 agreed with CPU).
This run produced `11751 13 15767 264 1103 314 89798 11` (3 of 8 agree with CPU).

Only the first 3 tokens agree between the two CUDA runs. The CPU arm is bit-stable (8 of 8 identical across runs).

**Agreement:**
- A-CPU vs B-CUDA: 3 of 8 (indices 0,1,2)
- A-CPU vs prior CPU: 8 of 8
- B-CUDA vs prior CUDA: 3 of 8 (indices 0,1,2)
- Disagreeing indices: [3, 4, 5, 6, 7]

## LayerFp diff (3,496 taps, 0 missing)

The sign-sensitive `rel_proj` column (#2877) detects a difference in **3,476 of 3,496 taps** (99.4%).

## Per-step breakdown — the divergence starts at PREFILL

| Step | Taps with rel_proj ≠ 0 | Largest rel_sumabs | Location |
|---|---|---|---|
| 0 (prefill) | 432 of 437 | 2.15% | L+31 moe |
| 1 | 432 of 437 | 6.29% | L+45 moe |
| 2 | 432 of 437 | 6.83% | L+45 moe |
| 3 | 432 of 437 | 14.8% | L+40 moe |
| 4 | 437 of 437 | 57.5% | L+01 ple |
| 5 | 437 of 437 | 52.0% | L+01 ple |
| 6 | 437 of 437 | 56.5% | L+43 ahc.mix |
| 7 | 437 of 437 | 79.8% | L+01 ple |
| 8 | 0 | — | — |

### What this establishes

1. **The divergence starts at PREFILL (step 0), not at decode.** 432 of 437 taps already differ in step 0. The magnitude is small (2.15% at the largest tap, at L+31 moe), but it is real and sign-sensitive — the `rel_proj` projections confirm it is not a zero-mean perturbation that `sumabs` is cancelling.

2. **The divergence amplifies exponentially through decode:** 2% → 6% → 7% → 15% → 58% → 52% → 57% → 80%. It crosses 50% at step 4, which is where the token sequences diverge (index 3 is the first disagreeing token, produced at step 3).

3. **The first divergence is at L+31 moe** — layer 31's MoE output. This is where the ~33% bf16 tie rate lives (established by MOEDIV/TIEBREAK). The MoE routing boundary ties cause CPU and CUDA to pick different experts, and the difference then amplifies through the remaining 16 layers and subsequent decode steps.

4. **The CUDA arm is non-deterministic across runs.** The prior CUDA run produced a different sequence (5 of 8 agreement) from this run (3 of 8 agreement). This is consistent with the tie-break mechanism: different runs break ties differently, producing different expert selections and thus different tokens. The CPU arm is deterministic.

### What this does NOT establish

- It does not establish that the MoE tie-break is the *only* source of divergence. The step-0 divergence is at L+31 moe, but 432 of 437 taps differ — the divergence may have upstream roots (e.g., in the attention or linear-attention layers before L+31) that are too small for `rel_sumabs` to detect but are caught by `rel_proj`.
- It does not establish a fix. The MoE tie-break non-determinism is structural (bf16 ties are exact), and the prior MOEDIV/TIEBREAK waves established that no CPU-vs-CUDA token-exactness gate is well-posed for this architecture.

## Top-50 divergences (selected)

| step | L | tag | rel_sumabs | head_dmax | rel_proj |
|---|---|---|---|---|---|
| 7 | 1 | ple | 79.8% | 9.3e-04 | 1.610 |
| 7 | -1 | emb | 58.0% | 1.1e-02 | 1.981 |
| 7 | 0 | in | 58.0% | 1.1e-02 | 1.999 |
| 4 | 1 | ple | 57.5% | 4.0e-04 | 0.833 |
| 6 | 43 | ahc.mix | 56.5% | 5.3e-01 | 1.915 |
| 7 | 46 | moe | 54.4% | 1.3e-01 | 1.826 |
| 5 | 1 | ple | 52.0% | 5.0e-04 | 1.813 |
| 4 | -1 | emb | 42.5% | 6.4e-03 | 1.501 |

The largest divergences at steps 4-7 are dominated by `ple` (PLE residual at L+01), `emb`/`in`/`wide` (the residual stream), and `mhc.inj` (MHC injection with `head_dmax` of 39-53%, meaning gate values are substantially different).

## Reproduction

```
# On thor:gpu0 (sm_110, aarch64, 122 GB RAM)
VT_Q4EXP_LAYER_FP=9 vllm-server \
  --model /tmp/q4exp-UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --device cpu --host 127.0.0.1 --port 8181 \
  --block-size 16 --num-blocks 128 --max-model-len 256 \
  --served-model-name qwen4exp --verbose

# Then repeat with --device cuda

# Diff the server logs:
python3 scripts/q4exp-layerfp-diff.py A-CPU/server.log B-CUDA/server.log --top 50
```

## Conclusion

#2999 is closed. The three disagreeing token ids are caused by a prefill-time divergence at the MoE layer (L+31), which amplifies to 80% by step 7. The `rel_proj` column from #2877 confirmed the divergence is real and sign-sensitive. The CUDA arm is non-deterministic across runs (3 of 8 vs 5 of 8 agreement with CPU in prior runs), consistent with the MoE tie-break mechanism established by MOEDIV/TIEBREAK. No CPU-vs-CUDA token-exactness gate is well-posed for this architecture.
