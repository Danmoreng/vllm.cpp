#!/usr/bin/env python3
"""Fail-closed validator for Qwen3.8 TensorFold baseline evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

BENCHMARK_ID = "bench-qwen38-tensorfold-gap"
BLOCKED = "BLOCKED_MISSING_ARTIFACTS"
MEASURED = "MEASURED"
MTP_VERDICTS = {"LOADABLE", "BLOCKED_NO_MTP_WEIGHTS", "BLOCKED_CONVERSION_REQUIRED"}
MTP_BLOCKED = "BLOCKED_NO_MTP_WEIGHTS"
WORKLOADS = {"serial_decode", "drafted_decode", "prefill_ladder", "serving_ladder"}
ENGINES = {"tensorfold", "vllm-cpp"}


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def load_object(path: pathlib.Path) -> dict:
    require(path.is_file(), f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def inspect_manifest(path: pathlib.Path) -> dict:
    require(path.is_file(), f"committed GGUF manifest is missing: {path}")
    text = path.read_text(encoding="utf-8")
    declared = re.search(r"kQwen4ExpGgufTensorCount\s*=\s*(\d+)", text)
    require(declared is not None, "GGUF manifest lacks its declared tensor count")
    names = re.findall(r'^\s*\{"([^"]+)",', text, flags=re.MULTILINE)
    mtp = [name for name in names if re.search(r"(?:^|[._])(mtp|nextn|draft|eh_proj|enorm|hnorm)(?:[._]|$)", name, re.I)]
    blocks = [int(value) for value in re.findall(r'^\s*\{"blk\.(\d+)\.', text, flags=re.MULTILINE)]
    return {
        "declared_tensor_count": int(declared.group(1)),
        "parsed_tensor_count": len(names),
        "mtp_name_matches": mtp,
        "max_block": max(blocks) if blocks else None,
    }


def validate(directory: pathlib.Path) -> None:
    summary = load_object(directory / "summary.json")
    require(summary.get("schema_version") == 1, "schema_version must be 1")
    require(summary.get("benchmark_id") == BENCHMARK_ID, "wrong benchmark_id")
    status = summary.get("status")
    require(status in {BLOCKED, MEASURED}, f"status must be {BLOCKED} or {MEASURED}")
    require(re.fullmatch(r"[0-9a-f]{40}", str(summary.get("source_sha", ""))) is not None,
            "source_sha must be full 40-hex")

    if status == MEASURED:
        require(summary.get("numbers_published") is True, "measured evidence must publish numbers")
        require(summary.get("cross_engine_ratio") is None,
                "unlike-artifact measured evidence must not contain a cross-engine ratio")
        artifacts = summary.get("artifacts")
        require(isinstance(artifacts, dict) and set(artifacts) == ENGINES,
                "measured evidence requires both engine artifacts")
        for engine in ENGINES:
            item = artifacts[engine]
            require(item.get("present") is True and item.get("hash_manifest_present") is True,
                    f"measured {engine} evidence requires present, hashed artifacts")
        require(summary.get("tensorfold_source_present") is True,
                "measured evidence requires pinned TensorFold source")
        workloads = summary.get("workloads")
        require(isinstance(workloads, dict) and set(workloads) == WORKLOADS,
                "all four measured workload families are required")
        require(all(value == "CAPTURED" for value in workloads.values()),
                "measured workloads must all be CAPTURED")
        profiles = summary.get("profiles")
        require(isinstance(profiles, dict) and set(profiles) == ENGINES and
                all(value == "CAPTURED" for value in profiles.values()),
                "same-tool profiles must be CAPTURED for both engines")
        mtp = summary.get("mtp")
        require(isinstance(mtp, dict) and mtp.get("verdict") in MTP_VERDICTS,
                "measured evidence requires an allowed MTP verdict")
        require((directory / "comparison.json").is_file(), "measured evidence lacks comparison.json")
        for engine in ENGINES:
            for name in ("result.json", "clocks-summary.json", "provenance.json", "profile.json"):
                require((directory / engine / name).is_file(),
                        f"measured evidence missing {engine}/{name}")
        return

    require(summary.get("numbers_published") is False, "blocked evidence must publish no benchmark numbers")
    require(summary.get("cross_engine_ratio") is None, "blocked evidence must not contain a ratio")

    leases = summary.get("lease_checks")
    require(isinstance(leases, list) and leases, "at least one lease check is required")
    for item in leases:
        require(isinstance(item, dict), "lease check must be an object")
        require(item.get("device") == "dgx:gpu0", "GPU discovery must use dgx:gpu0")
        require(item.get("state") == "succeeded", "lease discovery job did not succeed")
        require("job_id" not in item, "raw lease job IDs must not be committed")
        require(re.fullmatch(r"[0-9a-f]{64}", str(item.get("job_fingerprint_sha256", ""))) is not None,
                "lease check lacks a SHA-256 job fingerprint")

    artifacts = summary.get("artifacts")
    require(isinstance(artifacts, dict), "artifacts must be an object")
    require(set(artifacts) == ENGINES, "both engine artifacts must be recorded")
    for engine in ENGINES:
        item = artifacts[engine]
        require(item.get("required") is True, f"{engine} artifact must be required")
        require(item.get("present") is False, f"{engine} blocker must record artifact absent")
        require(item.get("hash_manifest_present") is False,
                f"{engine} must not claim a hash manifest for absent bytes")

    require(summary.get("tensorfold_source_present") is False,
            "TensorFold source absence must be explicit")
    denominator = summary.get("production_vllm_denominator")
    require(isinstance(denominator, dict) and denominator.get("status") == "BLOCKED",
            "production vLLM must remain a named blocked denominator")

    workloads = summary.get("workloads")
    require(isinstance(workloads, dict) and set(workloads) == WORKLOADS,
            "all four workload families must be accounted for")
    require(all(value == "NOT_RUN_PREREQUISITE" for value in workloads.values()),
            "blocked evidence cannot contain a timed workload")
    profiles = summary.get("profiles")
    require(isinstance(profiles, dict) and set(profiles) == ENGINES,
            "same-tool profile disposition is required for both engines")
    require(all(value == "NOT_RUN_PREREQUISITE" for value in profiles.values()),
            "blocked evidence cannot claim a profile")

    mtp = summary.get("mtp")
    require(isinstance(mtp, dict) and mtp.get("verdict") == MTP_BLOCKED,
            f"MTP verdict must be {MTP_BLOCKED}")
    manifest_rel = mtp.get("manifest")
    require(isinstance(manifest_rel, str) and manifest_rel and not pathlib.Path(manifest_rel).is_absolute(),
            "MTP manifest must be a repository-relative path")
    repo = pathlib.Path(__file__).resolve().parents[2]
    observed = inspect_manifest(repo / manifest_rel)
    require(observed["declared_tensor_count"] == mtp.get("tensor_count") == 1224,
            "MTP evidence tensor count disagrees with committed manifest")
    require(observed["parsed_tensor_count"] == 1224, "manifest entry count is not 1224")
    require(not observed["mtp_name_matches"] and mtp.get("mtp_name_matches") == 0,
            "committed GGUF manifest contains an MTP-like tensor")
    require(observed["max_block"] == mtp.get("max_block") == 47,
            "GGUF block range does not end at trunk block 47")

    later = summary.get("blocked_later_tasks")
    require(later == [5, 6, 7], "blocked dependent tasks must be [5, 6, 7]")
    for name in ("lease-discovery.json", "artifact-discovery.json", "checks.json", "README.md"):
        require((directory / name).is_file(), f"missing raw blocker evidence: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        validate(args.evidence_dir.resolve())
    except EvidenceError as exc:
        print(f"qwen38 tensorfold evidence: FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"qwen38 tensorfold evidence: PASS ({BLOCKED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
