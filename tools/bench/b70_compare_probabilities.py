#!/usr/bin/env python3
"""Compare VT_B70_QUALITY runs of the same checkpoint at identical prefixes.

Usage: b70_compare_probabilities.py REFERENCE_DIRECTORY CANDIDATE_DIRECTORY
Each directory comes from VT_DUMP_LOGITS + test_xpu_qwen_checkpoint. No third
party dependencies. Thresholds are fixed before qualification, not fitted to
observed activation/logit differences. This small regression corpus does not
establish general benchmark quality or parity with a differently quantized model.
"""
import argparse
import array
import json
import math
from pathlib import Path
import sys

VOCAB = 248320
MAX_KL = 0.01  # nats, reference || candidate
MAX_TV = 0.02


def read_dump(directory, case):
    path = directory / case["logits"]
    values = array.array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) and len(values) % VOCAB == 0, path
    assert all(math.isfinite(x) for x in values), path
    ids = [tuple(map(int, line.split())) for line in path.with_suffix(".ids.txt").read_text().splitlines()]
    assert len(ids) == len(values) // VOCAB, path
    rows = []
    for step, (recorded_step, best) in enumerate(ids):
        row = values[step * VOCAB:(step + 1) * VOCAB]
        assert recorded_step == step and max(range(VOCAB), key=row.__getitem__) == best, path
        rows.append((row, best))
    return rows


def distances(reference, candidate):
    def normalized(row):
        peak = max(row)
        weights = [math.exp(x - peak) for x in row]
        total = math.fsum(weights)
        return [w / total for w in weights], peak + math.log(total)

    p, log_zp = normalized(reference)
    q, log_zq = normalized(candidate)
    kl = math.fsum(prob * ((x - log_zp) - (y - log_zq))
                   for prob, x, y in zip(p, reference, candidate))
    tv = 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q))
    return max(0.0, kl), tv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--long-context", action="store_true", help="compare the separate one-case long retrieval corpus")
    args = parser.parse_args()
    reference = json.loads((args.reference / "results.json").read_text())
    candidate = json.loads((args.candidate / "results.json").read_text())
    assert len(reference) == len(candidate) == (1 if args.long_context else 8)
    if args.long_context:
        assert reference[0]["name"] == candidate[0]["name"] == "long_retrieval"
        assert reference[0]["response"]["usage"]["prompt_tokens"] >= 4096
    failures = 0
    compared = 0
    for ref, got in zip(reference, candidate):
        assert ref["name"] == got["name"] and ref["request"] == got["request"]
        assert ref["response"]["usage"]["prompt_tokens"] == got["response"]["usage"]["prompt_tokens"]
        left, right = read_dump(args.reference, ref), read_dump(args.candidate, got)
        count, worst_kl, worst_tv = 0, 0.0, 0.0
        for (p, p_best), (q, q_best) in zip(left, right):
            kl, tv = distances(p, q)
            worst_kl, worst_tv = max(worst_kl, kl), max(worst_tv, tv)
            failures += int(kl > MAX_KL or tv > MAX_TV)
            count += 1
            # The current distribution has the same conditioning prefix. Once
            # greedy choices diverge, subsequent rows no longer do; don't claim
            # that comparing them measures numerical drift at identical input.
            if p_best != q_best:
                break
        assert count > 0
        compared += count
        print(json.dumps({"case": ref["name"], "comparable_steps": count,
                          "reference_steps": len(left), "candidate_steps": len(right),
                          "max_kl": worst_kl, "max_tv": worst_tv,
                          "same_answer": ref["answer"] == got["answer"]}), flush=True)
    print(json.dumps({"cases": len(reference), "compared_distributions": compared,
                      "failed_distributions": failures, "max_allowed_kl": MAX_KL,
                      "max_allowed_tv": MAX_TV}), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
