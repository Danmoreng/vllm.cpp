#!/usr/bin/env python3
"""Time one bounded GPTQ text request against the pinned local vLLM server."""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()
    prompt = [100 + i % 11 for i in range(args.prompt_tokens)]
    payload = {
        "model": "B70-GPTQ-INT4-Reference",
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "min_tokens": args.output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    emitted: list[float] = []
    usage = None
    started = time.monotonic()
    with urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            event = json.loads(body)
            if event.get("usage") is not None:
                usage = event["usage"]
            if any(choice.get("text") or choice.get("finish_reason") is None
                   for choice in event.get("choices", [])):
                emitted.append(time.monotonic())
    finished = time.monotonic()
    if len(emitted) < 2:
        raise SystemExit(f"expected at least two streamed tokens, got {len(emitted)}")
    report = {
        "prompt_tokens": args.prompt_tokens,
        "requested_output_tokens": args.output_tokens,
        "streamed_events": len(emitted),
        "usage": usage,
        "time_to_first_token_s": emitted[0] - started,
        "prompt_tokens_per_ttft_s": args.prompt_tokens / (emitted[0] - started),
        "decode_interval_s_per_token": (emitted[-1] - emitted[0]) / (len(emitted) - 1),
        "decode_tokens_per_s": (len(emitted) - 1) / (emitted[-1] - emitted[0]),
        "total_request_s": finished - started,
        "metric_scope": "API wall time including local HTTP, scheduling and sampling",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if usage and usage.get("completion_tokens") == args.output_tokens else 1


if __name__ == "__main__":
    raise SystemExit(main())
