import json
from pathlib import Path

import pytest

from tools.bench.qwen38_endpoint_bench import (
    comparison_verdict,
    load_corpus,
    run_workload,
    summarize,
)


def test_load_corpus_accepts_utf8_text_and_messages(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"id": "text", "text": "Caffè 東京", "max_tokens": 3},
        {"id": "chat", "messages": [{"role": "user", "content": "Olá 👋"}], "max_tokens": 2},
    ], ensure_ascii=False), encoding="utf-8")
    corpus = load_corpus(path)
    assert corpus[0]["text"] == "Caffè 東京"
    assert corpus[1]["messages"][0]["content"] == "Olá 👋"


class FakeTransport:
    tokenizer_identity = {"name": "fixture-tokenizer", "revision": "abc123"}

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        n = payload["max_tokens"]
        return {
            "events": [
                {"offset_s": 0.01 * (i + 1), "token_ids": [100 + i], "text": chr(97 + i)}
                for i in range(n)
            ],
            "prompt_tokens": 7,
            "generated_tokens": n,
            "cache": {"hits": 2, "queries": 3},
            "draft": {"accepted": 1, "proposed": 2},
            "memory_samples": [{"offset_s": 0, "bytes": 1234}],
        }


def test_workload_is_canonical_closed_loop_and_preserves_raw_repetitions():
    transport = FakeTransport()
    config = {
        "endpoint": "http://example.invalid/v1/completions",
        "model": "runtime-selected-model",
        "corpus": [{"id": "p", "text": "héllo", "max_tokens": 3}],
        "draft": "on",
        "concurrency": 2,
        "waves": 5,
    }
    result = run_workload(config, transport)
    assert len(result["samples"]) == 10
    assert [s["wave"] for s in result["samples"]].count(0) == 2
    assert result["raw_repetitions"] == result["samples"]
    assert result["tokenizer_identity"] == transport.tokenizer_identity
    assert len(result["canonical_payload_hash"]) == 64
    assert all(s["token_fingerprint"] for s in result["samples"])
    assert all(s["ttft_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["tpot_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["itl_s"] == pytest.approx([0.01, 0.01]) for s in result["samples"])
    assert all(s["cache"] == {"hits": 2, "queries": 3} for s in result["samples"])
    payload = transport.payloads[0]
    assert payload["model"] == "runtime-selected-model"
    assert payload["seed"] == 0
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True
    assert payload["stream"] is True
    assert payload["draft"] is True


def test_serving_refuses_fewer_than_five_waves():
    with pytest.raises(ValueError, match="five"):
        run_workload({
            "endpoint": "http://localhost/v1/completions", "model": "m",
            "corpus": [{"text": "x", "max_tokens": 1}], "waves": 4,
        }, FakeTransport())


def test_summary_uses_interpolated_percentiles_and_preserves_samples():
    samples = [
        {"ttft_s": x, "tpot_s": x / 10, "e2e_s": x + 1,
         "prompt_tokens": 2, "generated_tokens": 2, "itl_s": [x / 10]}
        for x in (1.0, 2.0, 3.0, 4.0)
    ]
    out = summarize(samples)
    assert out["request_count"] == 4
    assert out["metrics"]["ttft_s"]["p50"] == pytest.approx(2.5)
    assert out["metrics"]["ttft_s"]["p99"] == pytest.approx(3.97)
    assert out["request_rate"] == pytest.approx(4 / 5)
    assert out["token_throughput"] == pytest.approx(8 / 5)
    assert out["raw_repetitions"] == samples


def test_comparison_refuses_match_without_equal_token_fingerprints():
    base = {"token_fingerprints": ["a", "b"], "refusal_reason": None}
    assert comparison_verdict(base, dict(base))["verdict"] == "matched"
    mismatch = comparison_verdict(base, {"token_fingerprints": ["a", "c"]})
    assert mismatch["verdict"] == "refused"
    assert "fingerprint" in mismatch["refusal_reason"]
    missing = comparison_verdict({}, {})
    assert missing["verdict"] == "refused"
