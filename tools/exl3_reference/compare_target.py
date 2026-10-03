#!/usr/bin/env python3
"""Compare bounded native full-target logits to matched-prefix original repeats."""
import argparse
import json
from pathlib import Path
import struct

from capture_projection import IMAGE
from capture_target import probability_metrics, target_phases, validate_prefix_witnesses
from compare_projection import blob
from extract_projection import digest, headers


def compare_row(native, expected, dtype, columns):
    width, fmt = {"F32": (4, "<f"), "F16": (2, "<e")}[dtype]
    headers.require(len(native) == columns * 4 and len(expected) == columns * width,
                    "full-target logit byte count mismatch")
    a = [x for (x,) in struct.iter_unpack("<f", native)]
    e = [x for (x,) in struct.iter_unpack(fmt, expected)]
    return probability_metrics(a, e)


def compare(args):
    headers.require(not args.report.exists(), "refusing to overwrite target comparison")
    metadata = headers.read_json(args.reference_dir / "comparison.json")
    headers.require(metadata["image"] == IMAGE and metadata["decode_steps"] == 64,
                    "requires pinned original D64 reference")
    native_ids = headers.read_json(args.native_trace_json)["output_ids"]
    references = []
    for i, identity in enumerate(metadata["repeats"]):
        path = args.reference_dir / f"repeat-{i}.safetensors"
        record = headers.read_json(path.with_suffix(".json"))
        headers.require("actual_input_witnesses" in record, "actual reference inputs were not observed")
        validate_prefix_witnesses(record["actual_input_witnesses"], metadata["prompt_token_ids"], native_ids[0])
        headers.require(record["output_ids"] == native_ids and
                        digest(path.read_bytes()) == identity["sha256"] == record["capture_sha256"],
                        "reference prefix or capture identity mismatch")
        references.append((path, headers.read_shard_header(path)))
    headers.require(bool(references), "reference repeats missing")
    rows, native_identities = [], []
    for phase in target_phases(64):
        path = Path(str(args.native_prefix) + "-" + phase + "-logits.f32")
        native = path.read_bytes()
        native_identities.append({"path": str(path), "sha256": digest(native)})
        for repeat, (reference, header) in enumerate(references):
            entry, expected = blob(reference, header, phase + "_logits")
            headers.require(entry["shape"] == [1,248320], "requires full target vocabulary")
            result = compare_row(native, expected, entry["dtype"], 248320)
            rows.append({"phase": phase, "reference_repeat": repeat, **result})
    result = {"schema": 1, "image": IMAGE, "native_trace_json_sha256": digest(args.native_trace_json.read_bytes()),
              "reference_comparison_sha256": digest((args.reference_dir / "comparison.json").read_bytes()),
              "native_logits": native_identities, "comparisons": rows,
              "investigation_triggers": [r for r in rows if r["investigation_trigger"]],
              "scope": "Same-prefix whole-target logits only. Does not replace failed native tests or qualify state/performance."}
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("TARGET_COMPARISONS", len(rows), "INVESTIGATION_TRIGGERS", len(result["investigation_triggers"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--native-trace-json", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    compare(parser.parse_args())
