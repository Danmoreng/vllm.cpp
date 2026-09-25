#!/usr/bin/env python3
"""Send bounded teacher-forced prompts to the local GPTQ capture server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


TOKEN_PATTERN = [33, 22, 15, 469, 2737, 48, 8948, 19, 5604, 15728, 13]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 16, 256])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    context_file = output_dir / "current_prompt.json"
    base_url = f"http://{args.host}:{args.port}"
    for length in args.lengths:
        if length not in {1, 16, 256}:
            raise SystemExit(f"unsupported capture length {length}; choose 1, 16, or 256")
        tokens = [TOKEN_PATTERN[i % len(TOKEN_PATTERN)] for i in range(length)]
        request_id = f"teacher_forced_m{length}_{int(time.time())}"
        context = {
            "request_id": request_id,
            "prompt_kind": "fixed_token_pattern",
            "token_ids": tokens,
            "position_ids": list(range(length)),
            "position_ids_source": "sequential positions for a single unpadded text-only request",
            "prompt_length": length,
        }
        context_file.write_text(json.dumps(context, indent=2) + "\n")
        payload = {
            "model": "B70-GPTQ-INT4-Reference",
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 0,
            "stream": False,
        }
        req = Request(
            f"{base_url}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=180) as response:
            body = json.loads(response.read())
        (output_dir / f"{request_id}_response.json").write_text(
            json.dumps(body, indent=2) + "\n"
        )
        print(json.dumps({"request_id": request_id, "prompt_tokens": length,
                          "completion_tokens": body.get("usage", {}).get("completion_tokens")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
