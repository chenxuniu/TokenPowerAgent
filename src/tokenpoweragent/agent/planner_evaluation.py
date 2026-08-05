"""Preregistered conformance and overhead evaluation for the semantic planner."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.agent.llm import (
    PlannerClientError,
    PlannerCompletionResult,
)
from tokenpoweragent.agent.planner import (
    ConstrainedLLMPlanner,
    PlanningState,
    RuleBasedPlanner,
    Subgoal,
)


class PlannerEvaluationError(ValueError):
    """Raised when a planner benchmark protocol is invalid."""


@dataclass(frozen=True)
class PlannerBenchmarkCase:
    case_id: str
    category: str
    expected_subgoals: Tuple[Subgoal, ...]
    state: PlanningState


@dataclass(frozen=True)
class PlannerBenchmarkProtocol:
    schema_version: str
    benchmark_id: str
    planner_model: str
    temperature: float
    max_tokens: int
    repeats: int
    thresholds: Mapping[str, float]
    cases: Tuple[PlannerBenchmarkCase, ...]

    @classmethod
    def load(cls, path: Path) -> "PlannerBenchmarkProtocol":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlannerEvaluationError("cannot read planner protocol: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise PlannerEvaluationError("planner protocol root must be an object")

        planner = _mapping(raw, "planner")
        thresholds = _mapping(raw, "thresholds")
        templates_raw = raw.get("state_templates", {})
        if not isinstance(templates_raw, Mapping):
            raise PlannerEvaluationError("state_templates must be an object")
        cases_raw = raw.get("cases")
        if not isinstance(cases_raw, list) or not cases_raw:
            raise PlannerEvaluationError("planner protocol requires non-empty cases")

        cases = tuple(_parse_case(item, templates_raw) for item in cases_raw)
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise PlannerEvaluationError("planner case ids must be unique")

        repeats = _positive_int(raw, "repeats")
        model = str(planner.get("model", "")).strip()
        if not model:
            raise PlannerEvaluationError("planner.model cannot be empty")
        temperature = _nonnegative_float(planner, "temperature")
        max_tokens = _positive_int(planner, "max_tokens")

        parsed_thresholds = {
            "minimum_typed_output_rate": _unit_interval(
                thresholds, "minimum_typed_output_rate"
            ),
            "minimum_raw_expected_subgoal_rate": _unit_interval(
                thresholds, "minimum_raw_expected_subgoal_rate"
            ),
            "minimum_guarded_expected_subgoal_rate": _unit_interval(
                thresholds, "minimum_guarded_expected_subgoal_rate"
            ),
            "maximum_completion_error_rate": _unit_interval(
                thresholds, "maximum_completion_error_rate"
            ),
            "minimum_token_usage_coverage_rate": _unit_interval(
                thresholds, "minimum_token_usage_coverage_rate"
            ),
            "minimum_model_identity_coverage_rate": _unit_interval(
                thresholds, "minimum_model_identity_coverage_rate"
            ),
            "maximum_forbidden_subgoal_accept_count": float(
                _nonnegative_int(
                    thresholds, "maximum_forbidden_subgoal_accept_count"
                )
            ),
            "maximum_model_mismatch_count": float(
                _nonnegative_int(thresholds, "maximum_model_mismatch_count")
            ),
        }
        benchmark_id = str(raw.get("benchmark_id", "")).strip()
        if not benchmark_id:
            raise PlannerEvaluationError("benchmark_id cannot be empty")
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            benchmark_id=benchmark_id,
            planner_model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            repeats=repeats,
            thresholds=parsed_thresholds,
            cases=cases,
        )


class _CompletionCapture:
    def __init__(self, completion: Any) -> None:
        self.completion = completion
        self.result: Optional[PlannerCompletionResult] = None
        self.error: Optional[Exception] = None
        self.latency_ms: Optional[float] = None
        self.prompt_sha256: Optional[str] = None

    def __call__(self, prompt: str) -> str:
        self.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started = time.perf_counter_ns()
        try:
            self.result = self.completion.complete(prompt)
            return self.result.text
        except Exception as exc:
            self.error = exc
            raise
        finally:
            self.latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0


def evaluate_planner_benchmark(
    protocol: PlannerBenchmarkProtocol,
    completion: Any,
    protocol_sha256: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Evaluate LLM conformance behind the same deterministic planner guard."""

    rows: List[Dict[str, Any]] = []
    rule_planner = RuleBasedPlanner()
    for repeat in range(protocol.repeats):
        for case in protocol.cases:
            rule_started = time.perf_counter_ns()
            rule_plan = rule_planner.plan(case.state)
            rule_latency_ms = (
                time.perf_counter_ns() - rule_started
            ) / 1_000_000.0

            capture = _CompletionCapture(completion)
            planner = ConstrainedLLMPlanner(capture)
            planner_started = time.perf_counter_ns()
            plan = planner.plan(case.state)
            planner_latency_ms = (
                time.perf_counter_ns() - planner_started
            ) / 1_000_000.0

            expected = {subgoal.value for subgoal in case.expected_subgoals}
            accepted_typed_output = plan.planner == "llm"
            result = capture.result
            raw_response = None if result is None else result.text
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "repeat": repeat,
                    "state": case.state.to_dict(),
                    "expected_subgoals": sorted(expected),
                    "rule_subgoal": rule_plan.subgoal.value,
                    "selected_subgoal": plan.subgoal.value,
                    "planner_source": plan.planner,
                    "rationale": plan.rationale,
                    "rationale_chars": len(plan.rationale),
                    "fallback_reason": plan.fallback_reason,
                    "accepted_typed_output": accepted_typed_output,
                    "raw_expected_subgoal": (
                        accepted_typed_output and plan.subgoal.value in expected
                    ),
                    "guarded_expected_subgoal": plan.subgoal.value in expected,
                    "rule_expected_subgoal": rule_plan.subgoal.value in expected,
                    "guarded_rule_agreement": plan.subgoal == rule_plan.subgoal,
                    "raw_rule_agreement": (
                        accepted_typed_output and plan.subgoal == rule_plan.subgoal
                    ),
                    "forbidden_subgoal_accepted": (
                        accepted_typed_output and plan.subgoal == Subgoal.VERIFY
                    ),
                    "completion_error": capture.error is not None,
                    "endpoint_error": isinstance(capture.error, PlannerClientError),
                    "completion_error_type": (
                        None if capture.error is None else type(capture.error).__name__
                    ),
                    "planner_latency_ms": planner_latency_ms,
                    "completion_latency_ms": capture.latency_ms,
                    "rule_latency_ms": rule_latency_ms,
                    "prompt_sha256": capture.prompt_sha256,
                    "raw_response": raw_response,
                    "raw_response_sha256": (
                        None
                        if raw_response is None
                        else hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
                    ),
                    "response_id": None if result is None else result.response_id,
                    "reported_model": None if result is None else result.model,
                    "model_identity_present": (
                        result is not None and result.model is not None
                    ),
                    "reported_model_matches_requested": (
                        result is not None
                        and result.model == protocol.planner_model
                    ),
                    "system_fingerprint": (
                        None if result is None else result.system_fingerprint
                    ),
                    "prompt_tokens": None if result is None else result.prompt_tokens,
                    "completion_tokens": (
                        None if result is None else result.completion_tokens
                    ),
                    "total_tokens": None if result is None else result.total_tokens,
                }
            )

    aggregate = _aggregate(rows)
    by_category = {
        category: _aggregate([row for row in rows if row["category"] == category])
        for category in sorted({str(row["category"]) for row in rows})
    }
    checks = {
        "typed_output_rate": (
            aggregate["typed_output_rate"]
            >= protocol.thresholds["minimum_typed_output_rate"]
        ),
        "raw_expected_subgoal_rate": (
            aggregate["raw_expected_subgoal_rate"]
            >= protocol.thresholds["minimum_raw_expected_subgoal_rate"]
        ),
        "guarded_expected_subgoal_rate": (
            aggregate["guarded_expected_subgoal_rate"]
            >= protocol.thresholds["minimum_guarded_expected_subgoal_rate"]
        ),
        "completion_error_rate": (
            aggregate["completion_error_rate"]
            <= protocol.thresholds["maximum_completion_error_rate"]
        ),
        "token_usage_coverage_rate": (
            aggregate["token_usage_coverage_rate"]
            >= protocol.thresholds["minimum_token_usage_coverage_rate"]
        ),
        "model_identity_coverage_rate": (
            aggregate["model_identity_coverage_rate"]
            >= protocol.thresholds["minimum_model_identity_coverage_rate"]
        ),
        "forbidden_subgoal_accept_count": (
            aggregate["forbidden_subgoal_accept_count"]
            <= protocol.thresholds["maximum_forbidden_subgoal_accept_count"]
        ),
        "model_mismatch_count": (
            aggregate["model_mismatch_count"]
            <= protocol.thresholds["maximum_model_mismatch_count"]
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "1.0",
        "benchmark_id": protocol.benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256 or None,
        "planner": {
            "endpoint": endpoint or None,
            "requested_model": protocol.planner_model,
            "temperature": protocol.temperature,
            "max_tokens": protocol.max_tokens,
        },
        "protocol": {
            "case_count": len(protocol.cases),
            "repeats": protocol.repeats,
            "expected_call_count": len(protocol.cases) * protocol.repeats,
            "thresholds": dict(protocol.thresholds),
            "label_semantics": (
                "expected subgoals are frozen against the deterministic safety "
                "planner before querying the language model"
            ),
            "verify_semantics": (
                "verify is not an LLM subgoal and remains reserved for the "
                "deterministic release gate"
            ),
        },
        "summary": {
            **aggregate,
            "checks": checks,
            "publication_ready": not failed_checks,
            "warnings": ["failed threshold: %s" % name for name in failed_checks],
        },
        "by_category": by_category,
        "calls": rows,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise PlannerEvaluationError("cannot aggregate an empty planner result")
    count = len(rows)
    total_tokens = [
        int(row["total_tokens"])
        for row in rows
        if row.get("total_tokens") is not None
    ]
    return {
        "call_count": count,
        "typed_output_rate": _rate(rows, "accepted_typed_output"),
        "raw_expected_subgoal_rate": _rate(rows, "raw_expected_subgoal"),
        "guarded_expected_subgoal_rate": _rate(
            rows, "guarded_expected_subgoal"
        ),
        "rule_expected_subgoal_rate": _rate(rows, "rule_expected_subgoal"),
        "guarded_rule_agreement_rate": _rate(rows, "guarded_rule_agreement"),
        "raw_rule_agreement_rate": _rate(rows, "raw_rule_agreement"),
        "fallback_rate": 1.0 - _rate(rows, "accepted_typed_output"),
        "completion_error_rate": _rate(rows, "completion_error"),
        "endpoint_error_rate": _rate(rows, "endpoint_error"),
        "forbidden_subgoal_accept_count": sum(
            bool(row["forbidden_subgoal_accepted"]) for row in rows
        ),
        "token_usage_coverage_rate": len(total_tokens) / count,
        "model_identity_coverage_rate": _rate(rows, "model_identity_present"),
        "model_mismatch_count": sum(
            bool(row["model_identity_present"])
            and not bool(row["reported_model_matches_requested"])
            for row in rows
        ),
        "planner_latency_ms": _distribution(
            [float(row["planner_latency_ms"]) for row in rows]
        ),
        "completion_latency_ms": _distribution(
            [
                float(row["completion_latency_ms"])
                for row in rows
                if row.get("completion_latency_ms") is not None
            ]
        ),
        "rule_latency_ms": _distribution(
            [float(row["rule_latency_ms"]) for row in rows]
        ),
        "total_tokens": _distribution(total_tokens),
    }


def _distribution(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    materialized = sorted(float(value) for value in values)
    return {
        "mean": mean(materialized),
        "median": median(materialized),
        "p95": _nearest_rank(materialized, 0.95),
        "min": materialized[0],
        "max": materialized[-1],
    }


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return float(values[index])


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _parse_case(
    raw: Any, templates: Mapping[str, Any]
) -> PlannerBenchmarkCase:
    if not isinstance(raw, Mapping):
        raise PlannerEvaluationError("planner cases must be objects")
    case_id = str(raw.get("id", "")).strip()
    category = str(raw.get("category", "")).strip()
    if not case_id or not category:
        raise PlannerEvaluationError("planner cases require id and category")
    expected_raw = raw.get("expected_subgoals")
    if not isinstance(expected_raw, list) or not expected_raw:
        raise PlannerEvaluationError("case %s needs expected_subgoals" % case_id)
    try:
        expected = tuple(Subgoal(str(item).strip().lower()) for item in expected_raw)
    except ValueError as exc:
        raise PlannerEvaluationError(
            "case %s contains an unknown expected subgoal" % case_id
        ) from exc
    if Subgoal.VERIFY in expected:
        raise PlannerEvaluationError("case %s cannot expect verify" % case_id)

    if "state" in raw:
        state_raw = dict(_mapping(raw, "state"))
    else:
        template_id = str(raw.get("state_template", "")).strip()
        template = templates.get(template_id)
        if not isinstance(template, Mapping):
            raise PlannerEvaluationError(
                "case %s references an unknown state template" % case_id
            )
        state_raw = dict(template)
        state_raw["intent"] = str(raw.get("intent", "")).strip()
        overrides = raw.get("state_overrides", {})
        if not isinstance(overrides, Mapping):
            raise PlannerEvaluationError(
                "case %s state_overrides must be an object" % case_id
            )
        state_raw.update(overrides)
    try:
        state = PlanningState(
            intent=str(state_raw["intent"]).strip(),
            step=int(state_raw["step"]),
            evidence_count=int(state_raw["evidence_count"]),
            failure_count=int(state_raw["failure_count"]),
            last_action_failed=bool(state_raw["last_action_failed"]),
            all_candidates_have_l0=bool(state_raw["all_candidates_have_l0"]),
            max_slo_boundary_probability=float(
                state_raw["max_slo_boundary_probability"]
            ),
            max_topology_gap=float(state_raw["max_topology_gap"]),
            remaining_exploration_gpu_hours=float(
                state_raw["remaining_exploration_gpu_hours"]
            ),
            predicted_pareto_ids=tuple(
                str(item) for item in state_raw["predicted_pareto_ids"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerEvaluationError(
            "case %s has an invalid planning state" % case_id
        ) from exc
    if not state.intent:
        raise PlannerEvaluationError("case %s has an empty intent" % case_id)
    if state.step < 0 or state.evidence_count < 0 or state.failure_count < 0:
        raise PlannerEvaluationError("case %s has negative counters" % case_id)
    if state.remaining_exploration_gpu_hours < 0:
        raise PlannerEvaluationError("case %s has a negative budget" % case_id)
    for value in (
        state.max_slo_boundary_probability,
        state.max_topology_gap,
    ):
        if not 0.0 <= value <= 1.0:
            raise PlannerEvaluationError(
                "case %s probabilities and gaps must be in [0,1]" % case_id
            )
    rule_subgoal = RuleBasedPlanner().plan(state).subgoal
    if rule_subgoal not in expected:
        raise PlannerEvaluationError(
            "case %s expected labels exclude deterministic subgoal %s"
            % (case_id, rule_subgoal.value)
        )
    return PlannerBenchmarkCase(case_id, category, expected, state)


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise PlannerEvaluationError("%s must be an object" % key)
    return value


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_int(raw, key)
    if value < 1:
        raise PlannerEvaluationError("%s must be positive" % key)
    return value


def _nonnegative_int(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerEvaluationError("%s must be an integer" % key) from exc
    if value < 0:
        raise PlannerEvaluationError("%s cannot be negative" % key)
    return value


def _nonnegative_float(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerEvaluationError("%s must be numeric" % key) from exc
    if value < 0:
        raise PlannerEvaluationError("%s cannot be negative" % key)
    return value


def _unit_interval(raw: Mapping[str, Any], key: str) -> float:
    value = _nonnegative_float(raw, key)
    if value > 1:
        raise PlannerEvaluationError("%s must be in [0,1]" % key)
    return value
