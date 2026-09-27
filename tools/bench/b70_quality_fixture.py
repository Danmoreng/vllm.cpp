#!/usr/bin/env python3
"""Freeze an exact-length quality prompt with the pinned local tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.length < 1:
        parser.error("length must be positive")
    source_bytes = [path.read_bytes() for path in args.source]
    source_text = [data.decode("utf-8") for data in source_bytes]
    text = "\n\n".join([args.prefix, *source_text])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < args.length:
        raise ValueError(f"sources contain only {len(tokens)} tokens")
    ids = tokens[:args.length]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(ids, separators=(",", ":")) + "\n"
    args.out.write_text(serialized, encoding="utf-8")
    metadata = {
        "tokens": len(ids),
        "available_source_tokens": len(tokens),
        "prompt_ids_sha256": sha256(serialized.encode()),
        "tokenizer_json_sha256": sha256((args.model / "tokenizer.json").read_bytes()),
        "prefix": args.prefix,
        "sources": [
            {"path": str(path), "sha256": sha256(data)}
            for path, data in zip(args.source, source_bytes)
        ],
    }
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
