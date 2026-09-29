#!/usr/bin/env python3
"""Canonical, engine-neutral OpenAI endpoint benchmark for Qwen3.8 campaigns."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a UTF-8 JSON corpus (a list or ``{"prompts": [...]}``)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("prompts") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("corpus must contain a non-empty prompt list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or (("text" in row) == ("messages" in row)):
            raise ValueError(f"corpus item {index} must have exactly one of text or messages")
        if "text" in row and not isinstance(row["text"], str):
            raise ValueError(f"corpus item {index} text must be UTF-8 text")
        if "messages" in row and not isinstance(row["messages"], list):
            raise ValueError(f"corpus item {index} messages must be a list")
        if int(row.get("max_tokens", 1)) < 1:
            raise ValueError(f"corpus item {index} max_tokens must be positive")
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _payload(model: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "max_tokens": int(row.get("max_tokens", 128)),
        "seed": 0,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "messages" in row:
        payload["messages"] = row["messages"]
    else:
        payload["prompt"] = row["text"]
    return payload


def _semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return engine-neutral request fields used for cross-endpoint identity."""
    return {key: value for key, value in payload.items() if key != "model"}


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    valid = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "count": len(valid),
        "mean": sum(valid) / len(valid) if valid else None,
        "p50": _percentile(valid, .50),
        "p90": _percentile(valid, .90),
        "p95": _percentile(valid, .95),
        "p99": _percentile(valid, .99),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
    }


def summarize(samples: list[dict[str, Any]], workload_makespan_s: float | None = None) -> dict[str, Any]:
    """Return percentile aggregates without discarding any raw repetition."""
    successful = [s for s in samples if not s.get("refusal_reason")]
    elapsed = float(workload_makespan_s) if workload_makespan_s is not None else None
    generated_values = [s.get("generated_tokens") for s in successful]
    generated = sum(int(x) for x in generated_values if x is not None)
    accounting_valid = len(successful) == len(samples) and all(x is not None for x in generated_values)
    itls = [x for s in successful for x in s.get("itl_s", [])]
    return {
        "request_count": len(samples),
        "successful_requests": len(successful),
        "workload_makespan_s": elapsed,
        "metrics": {
            "ttft_s": _stats(s.get("ttft_s") for s in successful),
            "tpot_s": _stats(s.get("tpot_s") for s in successful),
            "itl_s": _stats(itls),
            "e2e_s": _stats(s.get("e2e_s") for s in successful),
        },
        "request_rate": len(successful) / elapsed if accounting_valid and elapsed else None,
        "token_throughput": generated / elapsed if accounting_valid and elapsed else None,
        "raw_repetitions": samples,
    }


def _sample(response: dict[str, Any], started: float) -> dict[str, Any]:
    events = response.get("events", [])
    offsets = [float(event["offset_s"]) for event in events]
    token_ids = [token for event in events for token in event.get("token_ids", [])]
    texts = [str(event.get("text", "")) for event in events]
    generated_value = response.get("generated_tokens")
    generated = int(generated_value) if generated_value is not None else None
    e2e = float(response.get("e2e_s", offsets[-1] if offsets else time.perf_counter() - started))
    ttft = offsets[0] if offsets else None
    itl = [b - a for a, b in zip(offsets, offsets[1:])]
    tpot = ((e2e - ttft) / (generated - 1)) if ttft is not None and generated is not None and generated > 1 else None
    fingerprint = _hash(token_ids) if token_ids else None
    refusal = response.get("refusal_reason")
    if generated is None and not refusal:
        refusal = "endpoint did not expose reliable generated-token usage"
    if not fingerprint and not refusal:
        refusal = "endpoint did not expose generated token IDs; token fingerprint unavailable"
    prompt_value = response.get("prompt_tokens")
    if prompt_value is None and not refusal:
        refusal = "endpoint did not expose prompt-token usage"
    return {
        "prompt_tokens": int(prompt_value) if prompt_value is not None else None,
        "generated_tokens": generated,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itl_s": itl,
        "e2e_s": e2e,
        "token_ids": token_ids,
        "token_fingerprint": fingerprint,
        "generated_text": "".join(texts),
        "cache": response.get("cache"),
        "draft": response.get("draft"),
        "memory_samples": response.get("memory_samples", []),
        "refusal_reason": refusal,
    }


def run_workload(config: dict[str, Any], transport: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Run fixed-size closed-loop waves using the supplied endpoint transport."""
    waves = int(config.get("waves", 5))
    concurrency = int(config.get("concurrency", 1))
    if waves < 5:
        raise ValueError("serving benchmarks require at least five closed-loop waves")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    corpus = config["corpus"]
    payloads = [_payload(config["model"], row) for row in corpus]
    samples: list[dict[str, Any]] = []
    workload_started = time.perf_counter()

    def one(wave: int, slot: int) -> dict[str, Any]:
        index = (wave * concurrency + slot) % len(payloads)
        payload = payloads[index]
        started = time.perf_counter()
        try:
            sample = _sample(transport(payload), started)
        except Exception as error:  # preserve failed repetitions too
            sample = {"prompt_tokens": None, "generated_tokens": None, "ttft_s": None,
                      "tpot_s": None, "itl_s": [], "e2e_s": time.perf_counter() - started,
                      "token_ids": [], "token_fingerprint": None, "generated_text": "",
                      "cache": None, "draft": None, "memory_samples": [],
                      "refusal_reason": f"transport error: {error}"}
        sample.update({"wave": wave, "slot": slot, "corpus_index": index,
                       "payload_hash": _hash(_semantic_payload(payload))})
        return sample

    for wave in range(waves):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, wave, slot) for slot in range(concurrency)]
            samples.extend(f.result() for f in futures)

    workload_makespan_s = time.perf_counter() - workload_started
    aggregate = summarize(samples, workload_makespan_s)
    tokenizer_identity = getattr(transport, "tokenizer_identity", None)
    refused = any(s.get("refusal_reason") for s in samples)
    refusal_reason = "one or more samples were refused" if refused else None
    if not tokenizer_identity:
        refusal_reason = "tokenizer identity was not provided"
        aggregate["request_rate"] = None
        aggregate["token_throughput"] = None
    aggregate.update({
        "schema_version": 1,
        "refusal_reason": refusal_reason,
        "endpoint": config["endpoint"],
        "model": config["model"],
        "draft_mode": config.get("draft", "off"),
        "concurrency": concurrency,
        "waves": waves,
        "canonical_payload_hash": _hash([_semantic_payload(payload) for payload in payloads]),
        "tokenizer_identity": tokenizer_identity,
        "token_fingerprints": [s["token_fingerprint"] for s in samples],
        "samples": samples,
    })
    return aggregate


def comparison_verdict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless both runs contain equal canonical correctness evidence."""
    reason = None
    left_samples, right_samples = left.get("samples"), right.get("samples")
    if left.get("refusal_reason") or right.get("refusal_reason"):
        reason = "an input run was refused"
    elif not isinstance(left_samples, list) or not isinstance(right_samples, list) or not left_samples:
        reason = "per-sample evidence is missing"
    elif len(left_samples) != len(right_samples):
        reason = "sample counts differ"
    elif any(s.get("refusal_reason") for s in left_samples + right_samples):
        reason = "an input sample was refused"
    elif not left.get("canonical_payload_hash") or left.get("canonical_payload_hash") != right.get("canonical_payload_hash"):
        reason = "canonical payload hashes are missing or differ"
    elif not left.get("tokenizer_identity") or left.get("tokenizer_identity") != right.get("tokenizer_identity"):
        reason = "tokenizer identities are missing or differ"
    else:
        for left_sample, right_sample in zip(left_samples, right_samples):
            if (not left_sample.get("payload_hash") or
                    left_sample.get("payload_hash") != right_sample.get("payload_hash")):
                reason = "per-sample payload hashes are missing or differ"
                break
            if (left_sample.get("prompt_tokens") is None or
                    left_sample.get("prompt_tokens") != right_sample.get("prompt_tokens")):
                reason = "prompt token counts are missing or differ"
                break
            if (not left_sample.get("token_fingerprint") or
                    left_sample.get("token_fingerprint") != right_sample.get("token_fingerprint")):
                reason = "consumed token fingerprints are missing or differ"
                break

    left_fp, right_fp = left.get("token_fingerprints"), right.get("token_fingerprints")
    if reason is None and (not left_fp or left_fp != right_fp or any(not x for x in left_fp + right_fp)):
        reason = "token fingerprint evidence is missing or differs"
    return {"verdict": "refused" if reason else "matched", "refusal_reason": reason}


class HTTPTransport:
    """Minimal UTF-8 OpenAI-compatible streaming transport."""

    def __init__(self, endpoint: str, tokenizer_identity: Any,
                 extra_json: dict[str, Any] | None = None):
        self.endpoint = endpoint
        self.tokenizer_identity = tokenizer_identity
        self.extra_json = dict(extra_json or {})

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        overlap = self.extra_json.keys() & payload.keys()
        if overlap:
            raise ValueError(f"endpoint extra JSON cannot override canonical fields: {sorted(overlap)}")
        return {**payload, **self.extra_json}

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(self.endpoint, data=_canonical(self.prepare_payload(payload)),
                                         headers={"Content-Type": "application/json; charset=utf-8"})
        started = time.perf_counter()
        events, usage = [], {}
        with urllib.request.urlopen(request) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:].strip())
                usage.update(chunk.get("usage") or {})
                choice = (chunk.get("choices") or [{}])[0]
                text = choice.get("text", choice.get("delta", {}).get("content", ""))
                ids = chunk.get("token_ids", choice.get("token_ids", []))
                if text or ids:
                    events.append({"offset_s": time.perf_counter() - started,
                                   "text": text or "", "token_ids": ids})
        return {"events": events, "prompt_tokens": usage.get("prompt_tokens"),
                "generated_tokens": usage.get("completion_tokens")}


def _extra_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("extra JSON must be an object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-identity", required=True,
                        help="shared tokenizer name/revision or immutable digest")
    parser.add_argument("--draft", choices=("on", "off"), default="off",
                        help="run metadata only; does not alter the endpoint request")
    parser.add_argument("--extra-json", type=_extra_json, default={}, metavar="OBJECT",
                        help="documented endpoint-specific JSON fields to add to each request")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--waves", type=int, default=5)
    args = parser.parse_args()
    config = vars(args)
    config["corpus"] = load_corpus(args.corpus)
    result = run_workload(
        config, HTTPTransport(args.endpoint, args.tokenizer_identity, args.extra_json)
    )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
