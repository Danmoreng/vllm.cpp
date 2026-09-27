"""Summarize independent B70 worker-scope comparison sessions.

Use session medians as observations. Decode tokens within a request are
correlated and must not be treated as independent confidence samples.
"""

import argparse
import json
import statistics
from pathlib import Path


def summarize(values):
    return {
        "median_of_session_medians": statistics.median(values),
        "minimum_session_median": min(values),
        "maximum_session_median": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sessions = [json.loads(path.read_text(encoding="utf-8"))
                for path in args.sessions]
    anchor = (sessions[0]["P"], sessions[0]["D"], sessions[0]["O"],
              sessions[0]["effective_kv_page_size"])
    for path, session in zip(args.sessions, sessions):
        if (session["P"], session["D"], session["O"],
                session["effective_kv_page_size"]) != anchor:
            raise ValueError(f"case mismatch in {path}")
        if not session["token_hashes_match"] or not session["schedule_matches"]:
            raise ValueError(f"unmatched tokens or schedule in {path}")
        if session["rounds"] < 5:
            raise ValueError(f"fewer than five measured rounds in {path}")

    def scoped_values(key):
        return {
            engine: [session[key][engine]["median"] for session in sessions]
            for engine in ("python", "cpp")
        }

    prefill = scoped_values("prefill_seconds")
    decode = scoped_values("decode_seconds_per_forward") if anchor[1] else None
    window_labels = sessions[0]["decode_windows"].keys()
    if any(session["decode_windows"].keys() != window_labels
           for session in sessions):
        raise ValueError("decode window coverage differs between sessions")
    windows = {}
    for label in window_labels:
        windows[label] = {
            engine: summarize([
                session["decode_windows"][label][
                    f"{engine}_seconds_per_forward"]["median"]
                for session in sessions])
            for engine in ("python", "cpp")
        }
    result = {
        "status": "diagnostic_only_quality_routes_and_scope_review_pending",
        "P": anchor[0], "D": anchor[1], "O": anchor[2],
        "effective_kv_page_size": anchor[3],
        "independent_sessions": len(sessions),
        "measured_rounds_per_engine": sum(s["rounds"] for s in sessions),
        "session_files": [str(path) for path in args.sessions],
        "prefill_seconds": {engine: summarize(values)
                            for engine, values in prefill.items()},
        "prefill_cpp_throughput_fraction_of_python": summarize([
            py / cpp for py, cpp in zip(prefill["python"], prefill["cpp"])]),
        "decode_seconds_per_forward": {engine: summarize(values)
                                       for engine, values in decode.items()}
                                       if decode else None,
        "decode_cpp_throughput_fraction_of_python": summarize([
            py / cpp for py, cpp in zip(decode["python"], decode["cpp"])
        ]) if decode else None,
        "decode_windows_seconds_per_forward": windows,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
