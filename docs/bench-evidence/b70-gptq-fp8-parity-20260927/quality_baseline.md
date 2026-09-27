# P2 FP8 replay quality baseline (2026-09-27)

The pinned Python V2 runner and the C++ model benchmark consumed the same
prompt and teacher-forced decode IDs, with FP8 E4M3 KV, eager batch 1 and
effective 1600-token attention pages. Full-vocabulary F32 logits were captured
after prefill and selected decode forwards. Capture runs are **not** performance
measurements: both engines copy logits to the host, and the C++ benchmark also
executes a full warmup replay.

The comparator validates P/D/O, token hashes, page size, Python capture
contexts and file hashes before computing float64-normalized TV,
KL(Python || C++), top-10 overlap, top-1 and logit margins. The committed
`quality_thresholds.json` was fixed from these baselines before P3/P4 tuning.

| Replay | Decode checkpoints | Maximum TV | Maximum KL(Python || C++) | Minimum top-10 overlap | Top-1 |
| --- | --- | ---: | ---: | ---: | --- |
| Periodic, P128 | 1, 2 | 0.012761 | 0.000549 | 10 | 3/3 agree |
| Periodic, P4096 | 1, 64, 256, 703–705, 1024 | 0.018244 | 0.001123 | 9 | 8/8 agree |
| German/English text, P4096 | 1, 64, 256, 703–705, 1024 | 0.012998 | 0.000544 | 10 | 8/8 agree |
| C++ code, P4096 | 1, 64, 256, 703–705, 1024 | 0.015406 | 0.000911 | 10 | 8/8 agree |
| Periodic, P4096 | 1, 1024, 2048 | 0.018244 | 0.001123 | 9 | 4/4 agree |

The 703–705 checkpoints surround the transition to context 4800. The 2048
checkpoint ends at context 6144. In both engines, periodic prefill, decode 1
and decode 1024 logits were byte-identical between the D1024 and D2048 runs.
Every listed replay passed the fixed thresholds. Individual metrics, source
logit SHA-256 digests and verified contexts are in `quality_*.json`.

The complete raw F32 captures and run logs are stored outside Git in
`/home/sebastian/LocalLLM/b70-gptq-fp8-quality-goldens-20260927.tar.zst`
(28 MiB; SHA-256
`f221522fdf971293578a6ec303dc1651b4f1712e56af849c63db60e7b40f1eae`).
The 4K text and code token arrays and their tokenizer/source hashes are in
`fixtures/`. The periodic array follows the benchmark's documented default.

This establishes selected logit baselines. Selected per-layer GDN/conv state
comparisons and the P2 host/device critical-path profile are still open.
