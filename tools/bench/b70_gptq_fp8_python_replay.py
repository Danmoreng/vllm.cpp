"""Pinned B70 Python vLLM replay for GPTQ/FP8 P/D/O accounting.

This times complete engine steps, including scheduling, sampling, and IPC.
It does not report those durations as matched C++ forward times. The optional
trace forces generated token IDs so the first O-1 are identical to the C++
benchmark's D decode inputs.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams, TokensPrompt


def load_ids(path: Path | None, expected: int, default_start: int, period: int):
    ids = (
        json.loads(path.read_text(encoding="utf-8"))
        if path is not None
        else [default_start + i % period for i in range(expected)]
    )
    if not isinstance(ids, list) or len(ids) != expected:
        raise ValueError(f"expected {expected} token IDs in {path}")
    if any(type(token) is not int or token < 0 or token > 2**31 - 1
           for token in ids):
        raise ValueError(f"invalid token ID in {path}")
    return ids


def hash_ids(ids: list[int]) -> str:
    # Same byte-wise FNV-1a/LE-int32 digest as bench_gptq4_model.
    h = 14695981039346656037
    for token in ids:
        for byte in token.to_bytes(4, "little", signed=False):
            h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{h:016x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prompt-ids-file", type=Path)
    parser.add_argument("--decode-ids-file", type=Path)
    args = parser.parse_args()

    p, o = args.prompt_tokens, args.generated_tokens
    d = o - 1
    if (p < 1 or o < 1 or args.rounds < 1 or args.warmup_runs < 0 or
            p + o > args.max_model_len):
        parser.error("require P>=1, O>=1, rounds>=1, warmup-runs>=0, P+O<=max-model-len")
    prompt_ids = load_ids(args.prompt_ids_file, p, 100, 11)
    decode_ids = load_ids(args.decode_ids_file, d, 300, 248020)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        revision="a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
        quantization="gptq",
        dtype="float16",
        kv_cache_dtype="fp8",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=min(4096, args.max_model_len),
        block_size=64,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        enforce_eager=True,
        mamba_cache_mode="align",
        enable_trace_replay=d > 0,
    )
    config = llm.llm_engine.vllm_config
    print("B70_CONFIG " + json.dumps({
        "P": p, "D": d, "O": o,
        "requested_kv_block_size": 64,
        "actual_kv_block_size": config.cache_config.block_size,
        "max_model_len": config.model_config.max_model_len,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "warmup_runs": args.warmup_runs,
        "prompt_ids_fnv1a64": hash_ids(prompt_ids),
        "decode_ids_fnv1a64": hash_ids(decode_ids),
        "trace_replay": d > 0,
        "timing_scope": "full engine step with scheduler, sampling and IPC",
    }), flush=True)

    # Trace replay includes the final emitted token; only its first D tokens
    # become later model inputs and must match the C++ decode-ID file.
    trace_ids = decode_ids + [100] if d else None
    sampling = SamplingParams(
        temperature=0.0, max_tokens=o, min_tokens=o, ignore_eos=True,
        trace_decode_token_ids=trace_ids,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    engine = llm.llm_engine
    original_step = engine.step
    step_seconds: list[float] = []
    step_events: list[dict] = []
    request_start = 0.0

    def timed_step(*step_args, **step_kwargs):
        start = time.perf_counter()
        result = original_step(*step_args, **step_kwargs)
        ended = time.perf_counter()
        step_seconds.append(ended - start)
        emitted = 0
        if result and result[0].outputs:
            emitted = len(result[0].outputs[0].token_ids)
        step_events.append({
            "step_index": len(step_events),
            "seconds": ended - start,
            "elapsed_from_request_start": ended - request_start,
            "cumulative_output_tokens": emitted,
        })
        return result

    engine.step = timed_step
    for request_index in range(args.warmup_runs + args.rounds):
        step_seconds.clear()
        step_events.clear()
        start = request_start = time.perf_counter()
        result = llm.generate([prompt], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        output_ids = list(result.outputs[0].token_ids)
        if len(output_ids) != o or (trace_ids is not None and output_ids != trace_ids):
            raise RuntimeError("reference did not emit the requested trace IDs")
        first_api_output = next((event["elapsed_from_request_start"]
                                 for event in step_events
                                 if event["cumulative_output_tokens"] > 0), None)
        print("B70_RUN " + json.dumps({
            "round": request_index - args.warmup_runs,
            "warmup": request_index < args.warmup_runs,
            "P": p, "D": d, "O": o,
            "engine_steps": len(step_seconds),
            "engine_step_seconds": step_seconds,
            "engine_step_events": step_events,
            "first_nonempty_api_output_seconds": first_api_output,
            "ttft_seconds": None,
            "ttft_status": "not measured by the LLM.generate batch API",
            "request_wall_seconds": elapsed,
            "request_id": result.request_id,
            "output_ids_fnv1a64": hash_ids(output_ids),
            "phase_classification": "requires scheduler trace; engine ordinal is insufficient",
        }), flush=True)


if __name__ == "__main__":
    main()
