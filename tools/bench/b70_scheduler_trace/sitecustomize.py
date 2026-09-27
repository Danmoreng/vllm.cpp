"""Diagnostic-only vLLM scheduler trace loaded through PYTHONPATH.

Set B70_SCHED_TRACE_PATH to a mounted JSONL file. Do not enable this hook in
scored timing runs: it writes synchronously once per scheduler invocation.
"""

import functools
import json
import os


_path = os.environ.get("B70_SCHED_TRACE_PATH")
if _path:
    from vllm.v1.core.sched.scheduler import Scheduler

    _schedule = Scheduler.schedule

    @functools.wraps(_schedule)
    def _traced_schedule(self, *args, **kwargs):
        before = {
            request_id: request.num_computed_tokens
            for request_id, request in self.requests.items()
        }
        output = _schedule(self, *args, **kwargs)
        calls = []
        for request_id, query_tokens in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            # schedule() has already advanced num_computed_tokens by the
            # selected query length, including any cache-restored prefix.
            context_after = request.num_computed_tokens
            context_before = context_after - query_tokens
            prompt_tokens = request.num_prompt_tokens
            prompt_query_tokens = min(
                query_tokens, max(0, prompt_tokens - context_before)
            )
            calls.append({
                "request_id": request_id,
                "query_tokens": query_tokens,
                "context_before": context_before,
                "context_after": context_after,
                "prompt_tokens": prompt_tokens,
                "prompt_query_tokens": prompt_query_tokens,
                "decode_query_tokens": query_tokens - prompt_query_tokens,
                "computed_before_scheduler": before.get(request_id, 0),
                "cache_hit_or_restored_tokens": context_before - before.get(request_id, 0),
                "phase": (
                    "prefill" if prompt_query_tokens == query_tokens else
                    "decode" if prompt_query_tokens == 0 else "mixed"
                ),
            })
        record = {
            "event": "b70_scheduler_call",
            "pid": os.getpid(),
            "total_query_tokens": output.total_num_scheduled_tokens,
            "calls": calls,
        }
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        fd = os.open(_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return output

    Scheduler.schedule = _traced_schedule
