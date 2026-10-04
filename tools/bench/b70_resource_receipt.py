#!/usr/bin/env python3
"""Run one focused worker with before/after cgroup receipts, without polling."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


FILES = (
    "memory.current", "memory.peak", "memory.max", "memory.high",
    "memory.events", "memory.swap.current", "memory.swap.peak",
    "memory.swap.max", "memory.swap.events", "cpu.max", "cpu.stat",
)


def snapshot(root):
    result = {}
    for name in FILES:
        try:
            raw = (Path(root) / name).read_text().strip()
        except OSError as error:
            result[name] = {"unavailable": str(error)}
            continue
        item = {"raw": raw}
        if name.endswith((".stat", ".events")):
            item["counters"] = {k: int(v) for k, v in
                                (line.split() for line in raw.splitlines())}
        result[name] = item
    return result


def counter_deltas(before, after):
    return {name: {key: value - before[name]["counters"][key]
                   for key, value in item["counters"].items()
                   if key in before.get(name, {}).get("counters", {})}
            for name, item in after.items()
            if "counters" in item and "counters" in before.get(name, {})}


def run(command, output, cgroup_root=Path("/sys/fs/cgroup")):
    # Reserve the receipt first so an existing artifact never starts a worker.
    with Path(output).open("x") as stream:
        before = snapshot(cgroup_root)
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        executable = Path(command[0]).resolve(strict=True)
        with executable.open("rb") as binary:
            executable_hash = hashlib.file_digest(binary, "sha256").hexdigest()
        start = time.monotonic()
        worker = subprocess.Popen(command)
        code = worker.wait()  # completion wait, no observation/polling loop
        elapsed = time.monotonic() - start
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        after = snapshot(cgroup_root)
        receipt = {
            "schema": "b70-exl3-worker-resource-v1", "command": command,
            "worker_pid": worker.pid, "worker_executable": str(executable),
            "worker_executable_sha256": executable_hash,
            "worker_exit_code": code, "worker_wall_s": elapsed,
            "cgroup_root": str(cgroup_root),
            "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
            "before": before, "after": after,
            "counter_deltas": counter_deltas(before, after),
            "child_major_fault_delta": usage.ru_majflt - usage_before.ru_majflt,
            "children_peak_rss_bytes": usage.ru_maxrss * 1024,
            "scope": "One worker; cgroup values include this wrapper and any "
                     "other members. Peaks are lifetime cgroup/process peaks, "
                     "not interval-exclusive. Missing counters are unavailable. "
                     "Wall time includes worker startup; this is not a warmed "
                     "serving score. No inference arithmetic is changed.",
        }
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a worker executable is required after --")
    code = run(command, args.out, args.cgroup_root)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
