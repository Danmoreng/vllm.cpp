"""Capture selected full-vocabulary logits from the pinned XPU V2 runner.

Load with PYTHONPATH and set B70_QUALITY_OUT, B70_QUALITY_PROMPT_TOKENS, and
B70_QUALITY_DECODE_STEPS (comma-separated additional decode-forward counts).
This copies logits to the CPU and is strictly a quality run, never a timing run.
"""

import functools
import hashlib
import json
import os
from pathlib import Path


_out_arg = os.environ.get("B70_QUALITY_OUT")
if _out_arg:
    import torch

    from vllm.v1.worker.xpu_model_runner import XPUModelRunnerV2

    _out = Path(_out_arg)
    _out.mkdir(parents=True, exist_ok=True)
    _prompt_tokens = int(os.environ["B70_QUALITY_PROMPT_TOKENS"])
    _step_args = os.environ.get("B70_QUALITY_DECODE_STEPS", "1").split(",")
    if any(not item.isdecimal() or int(item) < 1 for item in _step_args):
        raise ValueError("B70_QUALITY_DECODE_STEPS must contain positive integers")
    _steps = {int(item) for item in _step_args}
    if len(_steps) != len(_step_args):
        raise ValueError("B70_QUALITY_DECODE_STEPS must be unique")
    _sample = XPUModelRunnerV2.sample

    @functools.wraps(_sample)
    def _capture_sample(self, hidden_states, input_batch, grammar_output):
        if input_batch.num_reqs != 1:
            raise ValueError("B70 quality capture requires one request")
        request_id = input_batch.req_ids[0]
        if request_id.startswith("_warmup_"):
            return _sample(self, hidden_states, input_batch, grammar_output)
        prompt_tokens = int(input_batch.prefill_len_np[0])
        if prompt_tokens != _prompt_tokens:
            raise ValueError("quality prompt length differs from requested length")
        query_tokens = int(input_batch.num_scheduled_tokens[0])
        context_before = int(input_batch.num_computed_tokens_np[0])
        context_after = context_before + query_tokens
        prompt_query_tokens = min(
            query_tokens, max(0, prompt_tokens - context_before)
        )
        if prompt_query_tokens and context_after >= prompt_tokens:
            phase = "prefill"
            decode_forward = 0
            selected = True
        elif context_before >= prompt_tokens and query_tokens == 1:
            decode_forward = context_before - prompt_tokens + 1
            phase = "decode"
            selected = decode_forward in _steps
        else:
            phase = "other"
            decode_forward = None
            selected = False
        if not selected:
            return _sample(self, hidden_states, input_batch, grammar_output)

        model = self.model
        original_logits = model.compute_logits

        @functools.wraps(original_logits)
        def capture_logits(*args, **kwargs):
            logits = original_logits(*args, **kwargs)
            values = logits.detach().to(torch.float32).cpu().contiguous().numpy()
            if values.ndim != 2 or values.shape != (1, self.vocab_size):
                raise ValueError(f"unexpected quality logits shape {values.shape}")
            data = values.astype("<f4", copy=False).tobytes()
            label = "prefill" if phase == "prefill" else f"decode_{decode_forward}"
            output = _out / f"p{prompt_tokens}_{label}.f32"
            output.write_bytes(data)
            record = {
                "event": "b70_quality_logits",
                "request_id": request_id,
                "phase": phase,
                "decode_forward": decode_forward,
                "query_tokens": query_tokens,
                "context_before": context_before,
                "context_after": context_after,
                "model_logits_dtype": str(logits.dtype),
                "vocab": values.shape[1],
                "path": output.name,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            with (_out / "captures.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            return logits

        model.compute_logits = capture_logits
        try:
            return _sample(self, hidden_states, input_batch, grammar_output)
        finally:
            model.compute_logits = original_logits

    XPUModelRunnerV2.sample = _capture_sample
