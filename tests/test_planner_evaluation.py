import json
import urllib.request
from pathlib import Path

import pytest

from tokenpoweragent.agent.llm import (
    OpenAICompatibleCompletion,
    PlannerClientError,
    PlannerCompletionResult,
)
from tokenpoweragent.agent.planner_evaluation import (
    PlannerBenchmarkProtocol,
    evaluate_planner_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/benchmarks/qwen2.5-7b-planner-conformance-v1.json"
)


class ConformingCompletion:
    def complete(self, prompt: str) -> PlannerCompletionResult:
        state = json.loads(prompt.split("STATE=", 1)[1])
        if state["last_action_failed"]:
            subgoal = "repair"
        elif state["evidence_count"] == 0:
            subgoal = "explore"
        elif (
            state["all_candidates_have_l0"]
            and state["max_topology_gap"] >= 0.5
        ):
            subgoal = "calibrate_scale"
        elif state["max_slo_boundary_probability"] >= 0.45:
            subgoal = "resolve_slo"
        else:
            subgoal = "explore"
        text = json.dumps(
            {"subgoal": subgoal, "rationale": "follow the bounded state policy"}
        )
        return PlannerCompletionResult(
            text=text,
            model="qwen2.5-7b",
            response_id="fake-response",
            system_fingerprint="test",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )


class VerifyCompletion:
    def complete(self, prompt: str) -> PlannerCompletionResult:
        del prompt
        return PlannerCompletionResult(
            text=json.dumps(
                {"subgoal": "verify", "rationale": "bypass the release gate"}
            ),
            model="qwen2.5-7b",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        )


class FailingCompletion:
    def complete(self, prompt: str) -> PlannerCompletionResult:
        del prompt
        raise PlannerClientError("injected endpoint failure")


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "chatcmpl-test",
                "model": "qwen2.5-7b",
                "system_fingerprint": "vllm-test",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"subgoal":"explore",'
                                '"rationale":"collect low-cost evidence"}'
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 12,
                    "total_tokens": 92,
                },
            }
        ).encode("utf-8")


def test_openai_compatible_completion_exposes_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse(),
    )
    client = OpenAICompatibleCompletion(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen2.5-7b",
        max_tokens=128,
    )

    result = client.complete("test prompt")

    assert result.model == "qwen2.5-7b"
    assert result.response_id == "chatcmpl-test"
    assert result.system_fingerprint == "vllm-test"
    assert result.prompt_tokens == 80
    assert result.completion_tokens == 12
    assert result.total_tokens == 92


def test_planner_benchmark_reports_conformance_latency_and_usage() -> None:
    protocol = PlannerBenchmarkProtocol.load(PROTOCOL)

    report = evaluate_planner_benchmark(
        protocol,
        ConformingCompletion(),
        protocol_sha256="a" * 64,
        endpoint="http://127.0.0.1:8000/v1",
    )

    summary = report["summary"]
    assert report["protocol"]["case_count"] == 30
    assert summary["call_count"] == 90
    assert summary["typed_output_rate"] == 1.0
    assert summary["raw_expected_subgoal_rate"] == 1.0
    assert summary["guarded_expected_subgoal_rate"] == 1.0
    assert summary["fallback_rate"] == 0.0
    assert summary["forbidden_subgoal_accept_count"] == 0
    assert summary["token_usage_coverage_rate"] == 1.0
    assert summary["model_identity_coverage_rate"] == 1.0
    assert summary["model_mismatch_count"] == 0
    assert summary["total_tokens"]["median"] == 120
    assert summary["planner_latency_ms"]["p95"] >= 0
    assert summary["publication_ready"] is True
    assert set(report["by_category"]) == {
        "cold_start",
        "guard_challenge",
        "repair",
        "slo_boundary",
        "steady_explore",
    }


def test_planner_benchmark_rejects_llm_verify_and_uses_guarded_fallback() -> None:
    protocol = PlannerBenchmarkProtocol.load(PROTOCOL)

    report = evaluate_planner_benchmark(protocol, VerifyCompletion())

    summary = report["summary"]
    assert summary["typed_output_rate"] == 0.0
    assert summary["raw_expected_subgoal_rate"] == 0.0
    assert summary["guarded_expected_subgoal_rate"] == 1.0
    assert summary["fallback_rate"] == 1.0
    assert summary["forbidden_subgoal_accept_count"] == 0
    assert summary["publication_ready"] is False
    assert all(call["planner_source"] == "rule-fallback" for call in report["calls"])
    assert all(
        "deterministic release gate" in str(call["fallback_reason"])
        for call in report["calls"]
    )


def test_planner_benchmark_distinguishes_endpoint_errors() -> None:
    protocol = PlannerBenchmarkProtocol.load(PROTOCOL)

    report = evaluate_planner_benchmark(protocol, FailingCompletion())

    summary = report["summary"]
    assert summary["completion_error_rate"] == 1.0
    assert summary["endpoint_error_rate"] == 1.0
    assert summary["guarded_expected_subgoal_rate"] == 1.0
    assert summary["publication_ready"] is False
