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


def _payload(model: str, row: dict[str, Any], draft: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "max_tokens": int(row.get("max_tokens", 128)),
        "seed": 0,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "stream": True,
        "draft": draft == "on",
    }
    if "messages" in row:
        payload["messages"] = row["messages"]
    else:
        payload["prompt"] = row["text"]
    return payload


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


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return percentile aggregates without discarding any raw repetition."""
    successful = [s for s in samples if not s.get("refusal_reason")]
    elapsed = max((float(s.get("e2e_s", 0)) for s in successful), default=0.0)
    generated = sum(int(s.get("generated_tokens", 0)) for s in successful)
    itls = [x for s in successful for x in s.get("itl_s", [])]
    return {
        "request_count": len(samples),
        "successful_requests": len(successful),
        "metrics": {
            "ttft_s": _stats(s.get("ttft_s") for s in successful),
            "tpot_s": _stats(s.get("tpot_s") for s in successful),
            "itl_s": _stats(itls),
            "e2e_s": _stats(s.get("e2e_s") for s in successful),
        },
        "request_rate": len(successful) / elapsed if elapsed else None,
        "token_throughput": generated / elapsed if elapsed else None,
        "raw_repetitions": samples,
    }


def _sample(response: dict[str, Any], started: float) -> dict[str, Any]:
    events = response.get("events", [])
    offsets = [float(event["offset_s"]) for event in events]
    token_ids = [token for event in events for token in event.get("token_ids", [])]
    texts = [str(event.get("text", "")) for event in events]
    generated = int(response.get("generated_tokens", len(token_ids) or len(events)))
    e2e = float(response.get("e2e_s", offsets[-1] if offsets else time.perf_counter() - started))
    ttft = offsets[0] if offsets else None
    itl = [b - a for a, b in zip(offsets, offsets[1:])]
    tpot = ((e2e - ttft) / (generated - 1)) if ttft is not None and generated > 1 else None
    fingerprint = _hash(token_ids) if token_ids else None
    refusal = response.get("refusal_reason")
    if not fingerprint and not refusal:
        refusal = "endpoint did not expose generated token IDs; token fingerprint unavailable"
    return {
        "prompt_tokens": int(response.get("prompt_tokens", 0)),
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
    payloads = [_payload(config["model"], row, config.get("draft", "off")) for row in corpus]
    samples: list[dict[str, Any]] = []

    def one(wave: int, slot: int) -> dict[str, Any]:
        index = (wave * concurrency + slot) % len(payloads)
        payload = payloads[index]
        started = time.perf_counter()
        try:
            sample = _sample(transport(payload), started)
        except Exception as error:  # preserve failed repetitions too
            sample = {"prompt_tokens": 0, "generated_tokens": 0, "ttft_s": None,
                      "tpot_s": None, "itl_s": [], "e2e_s": time.perf_counter() - started,
                      "token_ids": [], "token_fingerprint": None, "generated_text": "",
                      "cache": None, "draft": None, "memory_samples": [],
                      "refusal_reason": f"transport error: {error}"}
        sample.update({"wave": wave, "slot": slot, "corpus_index": index,
                       "payload_hash": _hash(payload)})
        return sample

    for wave in range(waves):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, wave, slot) for slot in range(concurrency)]
            samples.extend(f.result() for f in futures)

    aggregate = summarize(samples)
    aggregate.update({
        "schema_version": 1,
        "endpoint": config["endpoint"],
        "model": config["model"],
        "draft_mode": config.get("draft", "off"),
        "concurrency": concurrency,
        "waves": waves,
        "canonical_payload_hash": _hash(payloads),
        "tokenizer_identity": getattr(transport, "tokenizer_identity", None),
        "token_fingerprints": [s["token_fingerprint"] for s in samples],
        "samples": samples,
    })
    return aggregate


def comparison_verdict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Match only complete runs having identical, non-empty token fingerprints."""
    left_fp, right_fp = left.get("token_fingerprints"), right.get("token_fingerprints")
    reason = None
    if left.get("refusal_reason") or right.get("refusal_reason"):
        reason = "an input run was refused"
    elif not left_fp or not right_fp or any(not x for x in left_fp + right_fp):
        reason = "token fingerprint evidence is missing"
    elif left_fp != right_fp:
        reason = "token fingerprints differ"
    return {"verdict": "refused" if reason else "matched", "refusal_reason": reason}


class HTTPTransport:
    """Minimal UTF-8 OpenAI-compatible streaming transport."""
    tokenizer_identity = None

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(self.endpoint, data=_canonical(payload),
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
        return {"events": events, "prompt_tokens": usage.get("prompt_tokens", 0),
                "generated_tokens": usage.get("completion_tokens", len(events))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft", choices=("on", "off"), default="off")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--waves", type=int, default=5)
    args = parser.parse_args()
    config = vars(args)
    config["corpus"] = load_corpus(args.corpus)
    result = run_workload(config, HTTPTransport(args.endpoint))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
