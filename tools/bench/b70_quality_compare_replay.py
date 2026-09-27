#!/usr/bin/env python3
"""Compare selected C++ and pinned Python full-vocabulary replay logits.

This reports baseline diagnostics. Acceptance thresholds are a separate,
versioned input so a candidate cannot silently choose its own limits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_logits(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % 4:
        raise ValueError(f"invalid F32 logits size: {path}")
    values = np.frombuffer(data, dtype="<f4").astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite logits: {path}")
    return values, hashlib.sha256(data).hexdigest()


def log_probs(values: np.ndarray) -> np.ndarray:
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"vocabulary mismatch: {reference.shape} vs {candidate.shape}")
    ref_logp, cpp_logp = log_probs(reference), log_probs(candidate)
    ref_p, cpp_p = np.exp(ref_logp), np.exp(cpp_logp)
    ref_order = np.argsort(reference)[-2:][::-1]
    cpp_order = np.argsort(candidate)[-2:][::-1]
    ref_top, cpp_top = int(ref_order[0]), int(cpp_order[0])
    return {
        "vocab": int(reference.size),
        "total_variation": float(0.5 * np.abs(ref_p - cpp_p).sum()),
        "kl_python_to_cpp": float(np.sum(ref_p * (ref_logp - cpp_logp))),
        "top10_overlap": int(len(set(np.argsort(reference)[-10:]) &
                                 set(np.argsort(candidate)[-10:]))),
        "python_top1": ref_top,
        "cpp_top1": cpp_top,
        "python_top1_probability": float(ref_p[ref_top]),
        "cpp_probability_for_python_top1": float(cpp_p[ref_top]),
        "python_top1_logit_margin": float(reference[ref_order[0]] - reference[ref_order[1]]),
        "cpp_top1_logit_margin": float(candidate[cpp_order[0]] - candidate[cpp_order[1]]),
        "top1_agrees": ref_top == cpp_top,
    }


def validate_replay(args: argparse.Namespace) -> dict:
    cpp_config = next(
        json.loads(line) for line in args.cpp_log.read_text().splitlines()
        if line.startswith("{") and '"event":"gptq4_benchmark_config"' in line
    )
    python_config = next(
        json.loads(line.removeprefix("B70_CONFIG "))
        for line in args.python_log.read_text().splitlines()
        if line.startswith("B70_CONFIG ")
    )
    for cpp_key, python_key in (
        ("prompt_tokens", "P"), ("decode_forward_steps", "D"),
        ("output_tokens", "O"), ("kv_block_size", "actual_kv_block_size"),
        ("prompt_ids_fnv1a64", "prompt_ids_fnv1a64"),
        ("decode_ids_fnv1a64", "decode_ids_fnv1a64"),
    ):
        if cpp_config[cpp_key] != python_config[python_key]:
            raise ValueError(f"replay config mismatch: {cpp_key}/{python_key}")
    if (cpp_config["prompt_tokens"] != args.prompt_tokens or
            cpp_config["kv_dtype"] != "fp8_e4m3" or
            not python_config["trace_replay"] or
            cpp_config["quality_decode_steps"] != sorted(args.decode_steps)):
        raise ValueError("quality capture config does not match requested replay")
    captures = [json.loads(line) for line in
                (args.python / "captures.jsonl").read_text().splitlines()]
    expected = [("prefill", 0), *(('decode', step) for step in args.decode_steps)]
    if [(record["phase"], record["decode_forward"]) for record in captures] != expected:
        raise ValueError("Python captures do not match requested checkpoints")
    for record in captures:
        step = record["decode_forward"]
        before = (record["context_before"] if step == 0
                  else args.prompt_tokens + step - 1)
        after = args.prompt_tokens if step == 0 else before + 1
        if (step == 0 and not 0 <= before < args.prompt_tokens or
                record["context_before"] != before or
                record["context_after"] != after or
                record["query_tokens"] != after - before or
                record["sha256"] != sha256_file(args.python / record["path"])):
            raise ValueError(f"invalid Python capture: {record['path']}")
    return {
        "prompt_ids_fnv1a64": cpp_config["prompt_ids_fnv1a64"],
        "decode_ids_fnv1a64": cpp_config["decode_ids_fnv1a64"],
        "attention_page_tokens": cpp_config["kv_block_size"],
        "generated_tokens": cpp_config["output_tokens"],
        "capture_contexts": [record["context_after"] for record in captures],
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_thresholds(record: dict, thresholds: dict) -> None:
    top1_required = (record["python_top1_logit_margin"] >
                     thresholds["top1_tie_margin_max"])
    checks = {
        "total_variation": record["total_variation"] <= thresholds["max_total_variation"],
        "kl_python_to_cpp": record["kl_python_to_cpp"] <= thresholds["max_kl_python_to_cpp"],
        "top10_overlap": record["top10_overlap"] >= thresholds["min_top10_overlap"],
        "top1_or_near_tie": record["top1_agrees"] or not top1_required,
    }
    record["checks"] = checks
    record["near_tie_exemption"] = not top1_required
    record["passed"] = all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--cpp-log", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, nargs="*", default=[])
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or any(step < 1 for step in args.decode_steps):
        parser.error("prompt tokens and decode steps must be positive")
    if len(set(args.decode_steps)) != len(args.decode_steps):
        parser.error("decode steps must be unique")
    provenance = validate_replay(args)
    records = []
    for label in ["prefill", *(f"decode_{step}" for step in args.decode_steps)]:
        name = f"p{args.prompt_tokens}_{label}.f32"
        reference, reference_hash = read_logits(args.python / name)
        candidate, cpp_hash = read_logits(args.cpp / name)
        records.append({"checkpoint": label,
                        "python_sha256": reference_hash,
                        "cpp_sha256": cpp_hash,
                        **compare(reference, candidate)})
    thresholds = None
    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        for record in records:
            apply_thresholds(record, thresholds)
    result = {"prompt_tokens": args.prompt_tokens,
              "replay": provenance,
              "kl_direction": "D_KL(Python || C++)",
              "normalization": "float64 log-sum-exp of finite F32 logits",
              "quality_pass": (all(record["passed"] for record in records)
                               if thresholds is not None else None),
              "thresholds": thresholds,
              "records": records}
    output = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    if result["quality_pass"] is False:
        raise SystemExit("B70 quality threshold failed")


if __name__ == "__main__":
    main()
