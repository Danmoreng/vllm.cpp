#!/usr/bin/env python3
"""Capture and compare full-vocabulary B70 GPTQ FP8-KV distributions.

The native arm is ``bench_gptq4_model`` with VT_B70_BENCH_QUALITY_DIR set to
the same output directory and VT_B70_BENCH_DECODE_FIRST_TOKEN set to the first
ID in each prompt's metadata JSON. Use matching 64-token KV pages.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def reference(args):
    from vllm import LLM, SamplingParams, TokensPrompt

    args.out.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model, tokenizer=args.model, revision=args.revision,
        quantization="gptq", dtype="float16", kv_cache_dtype="fp8",
        max_model_len=8192, max_num_seqs=1,
        max_num_batched_tokens=max(args.lengths), max_logprobs=-1,
        gpu_memory_utilization=0.85, enable_prefix_caching=False,
        enforce_eager=True, mamba_cache_mode="align",
    )
    config = llm.llm_engine.vllm_config
    hf_config = config.model_config.hf_config
    vocab = getattr(hf_config, "vocab_size", None)
    if vocab is None:
        vocab = hf_config.text_config.vocab_size
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=2, min_tokens=2,
        ignore_eos=True, logprobs=-1,
    )
    for length in args.lengths:
        prompt = TokensPrompt(prompt_token_ids=[100 + i % 11 for i in range(length)])
        output = llm.generate([prompt], sampling)[0].outputs[0]
        if len(output.token_ids) != 2 or output.logprobs is None or len(output.logprobs) != 2:
            raise RuntimeError(f"unexpected output at {length} tokens")
        for phase, token_logprobs in zip(("prefill", "decode"), output.logprobs):
            values = np.full(vocab, -np.inf, dtype=np.float32)
            for token_id, record in token_logprobs.items():
                values[int(token_id)] = record.logprob
            if len(token_logprobs) != vocab or not np.isfinite(values).all():
                raise RuntimeError(f"incomplete logprobs at {length} {phase}")
            values.tofile(args.out / f"p{length}_{phase}_logprob.f32")
        metadata = {
            "prompt_tokens": length,
            "vocab_size": vocab,
            "generated_token_ids": list(output.token_ids),
            "logprobs_per_step": [len(x) for x in output.logprobs],
            "kv_dtype": str(config.cache_config.cache_dtype),
            "model_dtype": str(config.model_config.dtype),
            "mamba_cache_dtype": str(config.cache_config.mamba_ssm_cache_dtype),
            "prefix_caching": config.cache_config.enable_prefix_caching,
            "speculative": config.speculative_config is not None,
            "revision": args.revision,
        }
        (args.out / f"p{length}_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


def normalize_logprobs(values):
    values = values.astype(np.float64)
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def compare(args):
    reports = []
    for length in args.lengths:
        metadata = json.loads((args.out / f"p{length}_metadata.json").read_text())
        vocab = metadata["vocab_size"]
        for phase in ("prefill", "decode"):
            native = np.fromfile(args.out / f"p{length}_{phase}.f32", dtype=np.float32)
            reference = np.fromfile(
                args.out / f"p{length}_{phase}_logprob.f32", dtype=np.float32)
            if native.size != vocab or reference.size != vocab:
                raise RuntimeError(f"{length} {phase}: wrong vocab size")
            if not np.isfinite(native).all() or not np.isfinite(reference).all():
                raise RuntimeError(f"{length} {phase}: nonfinite values")
            native_logp = normalize_logprobs(native)
            reference_logp = normalize_logprobs(reference)
            p, q = np.exp(native_logp), np.exp(reference_logp)
            reference_top = int(np.argmax(q))
            native_top = int(np.argmax(p))
            report = {
                "prompt_tokens": length,
                "phase": phase,
                "reference_top1": reference_top,
                "native_top1": native_top,
                "top10_overlap": len(set(np.argpartition(p, -10)[-10:]) &
                                     set(np.argpartition(q, -10)[-10:])),
                "kl_reference_to_native": float(np.sum(q * (reference_logp - native_logp))),
                "total_variation": float(0.5 * np.abs(q - p).sum()),
                "reference_top1_probability": float(q[reference_top]),
                "native_probability_for_reference_top1": float(p[reference_top]),
            }
            report["passed"] = (native_top == reference_top and
                                report["kl_reference_to_native"] <= args.max_kl and
                                report["total_variation"] <= args.max_tv)
            reports.append(report)
            print(json.dumps(report), flush=True)
    (args.out / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    if not all(report["passed"] for report in reports):
        raise SystemExit("FP8 quality gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reference", "compare"))
    parser.add_argument("--model", help="local pinned Hugging Face snapshot")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(512, 4096))
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--max-tv", type=float, default=0.02)
    args = parser.parse_args()
    if args.action == "reference":
        if not args.model:
            parser.error("reference requires --model")
        reference(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
