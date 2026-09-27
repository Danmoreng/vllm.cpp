"""B70 Python reference for the matching 4K GPTQ/FP8 C++ benchmark.

Run with the pinned vLLM XPU image and the same local checkpoint as
bench_gptq4_model. The reported engine-step times include scheduling,
sampling, and process communication; they are not model-forward times.
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams, TokensPrompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models")
    parser.add_argument("--decode-ids-out", required=True)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        revision="a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
        quantization="gptq",
        dtype="float16",
        kv_cache_dtype="fp8",
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        block_size=64,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        enforce_eager=True,
        mamba_cache_mode="align",
    )
    print(
        "CONFIG",
        json.dumps({"actual_kv_block_size": llm.llm_engine.vllm_config.cache_config.block_size}),
        flush=True,
    )
    prompt = TokensPrompt(prompt_token_ids=[100 + i % 11 for i in range(4096)])
    step_seconds = []
    original_step = llm.llm_engine.step

    def timed_step(*step_args, **step_kwargs):
        start = time.perf_counter()
        result = original_step(*step_args, **step_kwargs)
        step_seconds.append(time.perf_counter() - start)
        return result

    llm.llm_engine.step = timed_step
    for round_index, count in enumerate((1, 65, 1, 65)):
        sampling = SamplingParams(
            temperature=0.0, max_tokens=count, min_tokens=count, ignore_eos=True
        )
        step_seconds.clear()
        start = time.perf_counter()
        result = llm.generate([prompt], sampling, use_tqdm=False)[0]
        wall_seconds = time.perf_counter() - start
        ids = list(result.outputs[0].token_ids)
        if count == 65 and round_index == 3:
            with open(args.decode_ids_out, "w", encoding="utf-8") as file:
                json.dump(ids[:-1], file)
        print(
            "RUN",
            json.dumps(
                {
                    "round": round_index,
                    "output_tokens": len(ids),
                    "wall_seconds": wall_seconds,
                    "engine_steps": len(step_seconds),
                    "first_engine_step_seconds": step_seconds[0],
                    "remaining_engine_steps_seconds": sum(step_seconds[1:]),
                    "first_tokens": ids[:8],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
