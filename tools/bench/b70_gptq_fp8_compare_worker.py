"""Join B70 Python V2 worker timings with C++ forward step records.

This validates P/D/O, token hashes, actual scheduler phases and contexts before
printing diagnostic timings. It does not qualify parity without quality gates,
multiple sessions, and per-case route evidence.
"""

import argparse
import json
import statistics
from pathlib import Path


def records(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def tagged(path: Path, prefix: str):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            yield json.loads(line[len(prefix):])


def median_mad(values):
    median = statistics.median(values)
    return {"median": median,
            "median_absolute_deviation": statistics.median(
                abs(value - median) for value in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-log", type=Path, required=True)
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--python-worker", type=Path, required=True)
    parser.add_argument("--python-scheduler", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cpp = list(records(args.cpp_log))
    cpp_config = next(x for x in cpp if x.get("event") == "gptq4_benchmark_config")
    cpp_rounds = [x for x in cpp if x.get("event") == "gptq4_benchmark_round"]
    cpp_steps = [x for x in cpp if x.get("event") == "gptq4_benchmark_step"]
    py_config = next(tagged(args.python_log, "B70_CONFIG "))
    py_runs = [x for x in tagged(args.python_log, "B70_RUN ") if not x["warmup"]]
    workers = [x for x in records(args.python_worker)
               if x.get("event") == "b70_v2_worker_timing"]
    schedules = [x for x in records(args.python_scheduler)
                 if x.get("event") == "b70_scheduler_call"]

    p, d, o = py_config["P"], py_config["D"], py_config["O"]
    if (p, d, o) != (cpp_config["prompt_tokens"],
                      cpp_config["decode_forward_steps"],
                      cpp_config["generated_tokens"]):
        raise ValueError("P/D/O mismatch")
    if o != d + 1:
        raise ValueError("expected O=D+1")
    for name in ("prompt_ids_fnv1a64", "decode_ids_fnv1a64"):
        if py_config[name] != cpp_config[name]:
            raise ValueError(f"{name} mismatch")
    if py_config["actual_kv_block_size"] != cpp_config["kv_block_size"]:
        raise ValueError("effective KV page size mismatch")
    if len(py_runs) != len(cpp_rounds):
        raise ValueError("measured round count mismatch")

    def belongs(request_id, run_id):
        return request_id == run_id or request_id.startswith(run_id + "-")

    paired = []
    for run, cpp_round in zip(py_runs, cpp_rounds):
        run_id = run["request_id"]
        worker_steps = [x for x in workers if any(
            belongs(request_id, run_id)
            for request_id in x["query_tokens_by_request"])]
        schedule_steps = [call for entry in schedules for call in entry["calls"]
                          if belongs(call["request_id"], run_id)]
        cpp_run_steps = [x for x in cpp_steps if x["round"] == cpp_round["round"]]
        if len(worker_steps) != len(schedule_steps) or len(worker_steps) != len(cpp_run_steps):
            raise ValueError(f"model-call count differs for request {run_id}")
        steps = []
        for index, (worker, schedule, cxx) in enumerate(zip(
                worker_steps, schedule_steps, cpp_run_steps)):
            q = schedule["query_tokens"]
            prompt_q = schedule["prompt_query_tokens"]
            phase = "prefill" if prompt_q == q else (
                "decode" if prompt_q == 0 else "mixed")
            if (phase, q, schedule["context_before"], schedule["context_after"]) != (
                    cxx["phase"], cxx["query_tokens"], cxx["context_before"],
                    cxx["context_after"]):
                raise ValueError(f"scheduled step {index} differs for request {run_id}")
            if worker["query_tokens"] != q or "logits_complete_ns" not in worker:
                raise ValueError(f"worker step {index} is incomplete")
            steps.append({
                "phase": phase, "query_tokens": q,
                "context_before": schedule["context_before"],
                "context_after": schedule["context_after"],
                "python_worker_seconds": (
                    worker["logits_complete_ns"] - worker["execute_start_ns"]) / 1e9,
                "cpp_forward_seconds": cxx["seconds"],
                "python_runner_gap_seconds": (
                    worker["sample_start_ns"] - worker["execute_end_ns"]) / 1e9,
            })
        paired.append({"python_request_id": run_id,
                       "cpp_round": cpp_round["round"], "steps": steps})

    py_prefill = [sum(step["python_worker_seconds"] for step in pair["steps"]
                      if step["phase"] == "prefill") for pair in paired]
    cpp_prefill = [sum(step["cpp_forward_seconds"] for step in pair["steps"]
                       if step["phase"] == "prefill") for pair in paired]
    py_decode = [sum(step["python_worker_seconds"] for step in pair["steps"]
                     if step["phase"] == "decode") / d for pair in paired] if d else []
    cpp_decode = [sum(step["cpp_forward_seconds"] for step in pair["steps"]
                      if step["phase"] == "decode") / d for pair in paired] if d else []
    windows = {}
    for first, last in ((1, 16), (17, 64), (65, 256), (257, 1024),
                        (1025, 2048)):
        if d < first:
            continue
        label = f"{first}-{min(last, d)}"
        per_run = []
        for pair in paired:
            decode_steps = [step for step in pair["steps"]
                            if step["phase"] == "decode"]
            window_steps = decode_steps[first - 1:min(last, d)]
            per_run.append({
                "python": statistics.mean(step["python_worker_seconds"]
                                          for step in window_steps),
                "cpp": statistics.mean(step["cpp_forward_seconds"]
                                       for step in window_steps),
            })
        windows[label] = {
            "python_seconds_per_forward": median_mad([x["python"] for x in per_run]),
            "cpp_seconds_per_forward": median_mad([x["cpp"] for x in per_run]),
        }
    result = {
        "status": "diagnostic_only_quality_routes_and_independent_sessions_pending",
        "P": p, "D": d, "O": o, "rounds": len(paired),
        "token_hashes_match": True, "effective_kv_page_size": cpp_config["kv_block_size"],
        "schedule_matches": True, "paired_runs": paired,
        "prefill_seconds": {"python": median_mad(py_prefill),
                            "cpp": median_mad(cpp_prefill)},
        "decode_seconds_per_forward": {
            "python": median_mad(py_decode) if d else None,
            "cpp": median_mad(cpp_decode) if d else None},
        "decode_windows": windows,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
