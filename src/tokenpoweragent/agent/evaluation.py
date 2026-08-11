"""Replay evaluation for agent and non-agent acquisition policies."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tokenpoweragent.agent.controller import AgentRun, TokenPowerAgent
from tokenpoweragent.agent.planner import RuleBasedPlanner
from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus
from tokenpoweragent.executors.replay import BootstrapReplayExecutor
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.policy.ipig import (
    AcquisitionPolicy,
    CheapestFirstPolicy,
    CostBlindInformationPolicy,
    IPIGPolicy,
    RandomPolicy,
)
from tokenpoweragent.schema import EvidenceLevel, Scenario


POLICY_NAMES = ("ipig", "random", "cost-blind", "cheapest-first")


class ReplayEvaluationError(ValueError):
    """Raised when a replay corpus cannot support a fair evaluation."""


def build_policy(name: str, seed: int = 0) -> AcquisitionPolicy:
    normalized = name.strip().lower()
    if normalized == "ipig":
        return IPIGPolicy()
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized == "cost-blind":
        return CostBlindInformationPolicy()
    if normalized == "cheapest-first":
        return CheapestFirstPolicy()
    raise ReplayEvaluationError(
        "unknown policy %s; choose from %s" % (name, ", ".join(POLICY_NAMES))
    )


def evaluate_replay_policies(
    scenario: Scenario,
    records: Sequence[EvidenceRecord],
    policy_names: Sequence[str] = POLICY_NAMES,
    episodes: int = 10,
    max_steps: int = 20,
    min_ipig_score: float = 0.0,
    frontier_patience: int = 0,
    source_sha256: str = "",
    scenario_sha256: str = "",
) -> Dict[str, Any]:
    if episodes < 1:
        raise ReplayEvaluationError("episodes must be positive")
    if not policy_names:
        raise ReplayEvaluationError("at least one policy is required")
    oracle_metrics = _oracle_metrics(scenario, records)
    oracle_rows = [
        (candidate_id, metrics)
        for candidate_id, metrics in oracle_metrics.items()
        if scenario.slo.accepts(metrics)
    ]
    oracle_ids = pareto_front(oracle_rows, scenario.objectives)
    if not oracle_ids:
        raise ReplayEvaluationError(
            "replay corpus has no SLO-feasible L4 oracle candidate"
        )

    episode_rows: List[Dict[str, Any]] = []
    for policy_name in policy_names:
        for seed in range(episodes):
            policy = build_policy(policy_name, seed=seed)
            report = TokenPowerAgent(
                scenario=scenario,
                executor=BootstrapReplayExecutor(list(records)),
                planner=RuleBasedPlanner(),
                policy=policy,
            ).run(
                max_steps=max_steps,
                min_ipig_score=min_ipig_score,
                frontier_patience=frontier_patience,
                run_seed=seed,
            )
            episode_rows.append(
                _episode_metrics(report, scenario, oracle_ids, oracle_metrics)
            )

    aggregate = {
        policy_name: _aggregate_policy(
            [row for row in episode_rows if row["policy"] == policy_name]
        )
        for policy_name in policy_names
    }
    return {
        "schema_version": "1.0",
        "scenario": scenario.name,
        "intent": scenario.intent,
        "source_sha256": source_sha256 or None,
        "scenario_sha256": scenario_sha256 or None,
        "protocol": {
            "episodes_per_policy": episodes,
            "max_steps": max_steps,
            "min_ipig_score": min_ipig_score,
            "frontier_patience": frontier_patience,
            "policies": list(policy_names),
            "verification_required": True,
            "oracle_semantics": "median successful L4 metrics per candidate",
            "episode_semantics": (
                "candidate-level empirical bootstrap with common random numbers "
                "across policies"
            ),
            "resampling_scheme": "sha256(seed,candidate_id,evidence_level)",
        },
        "oracle": {
            "pareto_ids": oracle_ids,
            "candidate_metrics": oracle_metrics,
        },
        "aggregate": aggregate,
        "episodes": episode_rows,
    }


def _oracle_metrics(
    scenario: Scenario, records: Sequence[EvidenceRecord]
) -> Dict[str, Dict[str, float]]:
    required_metrics = set(scenario.objectives)
    if scenario.slo.ttft_ms is not None:
        required_metrics.add("ttft_ms")
    if scenario.slo.tpot_ms is not None:
        required_metrics.add("tpot_ms")
    if scenario.slo.min_goodput_req_s is not None:
        required_metrics.add("goodput_req_s")

    by_candidate: Dict[str, List[EvidenceRecord]] = {}
    for record in records:
        if (
            record.level == EvidenceLevel.L4
            and record.status == EvidenceStatus.SUCCEEDED
        ):
            by_candidate.setdefault(record.candidate_id, []).append(record)

    missing = [
        candidate.candidate_id
        for candidate in scenario.candidates
        if candidate.candidate_id not in by_candidate
    ]
    if missing:
        raise ReplayEvaluationError(
            "replay corpus lacks successful L4 evidence for: %s"
            % ", ".join(missing)
        )

    result: Dict[str, Dict[str, float]] = {}
    for candidate in scenario.candidates:
        rows = by_candidate[candidate.candidate_id]
        absent = [
            metric
            for metric in required_metrics
            if any(metric not in record.metrics for record in rows)
        ]
        if absent:
            raise ReplayEvaluationError(
                "L4 evidence for %s lacks metrics: %s"
                % (candidate.candidate_id, ", ".join(sorted(absent)))
            )
        result[candidate.candidate_id] = {
            metric: float(median(record.metrics[metric] for record in rows))
            for metric in sorted(required_metrics)
        }
    return result


def _episode_metrics(
    report: AgentRun,
    scenario: Scenario,
    oracle_ids: Sequence[str],
    oracle_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    predicted = set(report.verified_pareto_ids)
    oracle = set(oracle_ids)
    overlap = predicted & oracle
    verification_events = [
        event for event in report.events if event.phase == "verification"
    ]
    first_hit = next(
        (
            event.cumulative_gpu_hours
            for event in verification_events
            if event.candidate_id in oracle
            and event.status == EvidenceStatus.SUCCEEDED.value
            and scenario.slo.accepts(event.observed_metrics)
        ),
        None,
    )
    primary_metric, direction = next(iter(scenario.objectives.items()))
    oracle_best = _best(
        [oracle_metrics[candidate_id][primary_metric] for candidate_id in oracle],
        direction,
    )
    recommended_values = [
        oracle_metrics[candidate_id][primary_metric]
        for candidate_id in predicted
        if candidate_id in oracle_metrics
    ]
    regret = None
    if recommended_values:
        recommended_best = _best(recommended_values, direction)
        if direction == "min":
            regret = 100.0 * (recommended_best - oracle_best) / max(
                abs(oracle_best), 1e-12
            )
        else:
            regret = 100.0 * (oracle_best - recommended_best) / max(
                abs(oracle_best), 1e-12
            )
        regret = max(0.0, regret)

    return {
        "policy": report.policy,
        "semantic_guard_active": report.semantic_guard_active,
        "seed": report.run_seed,
        "status": report.status.value,
        "search_stop_reason": report.search_stop_reason.value,
        "spent_gpu_hours": report.spent_gpu_hours,
        "exploration_gpu_hours": report.exploration_gpu_hours,
        "verification_gpu_hours": report.verification_gpu_hours,
        "action_count": len(report.events),
        "failed_action_count": sum(
            event.status == EvidenceStatus.FAILED.value for event in report.events
        ),
        "verified_ids": list(report.verified_pareto_ids),
        "oracle_pareto_recall": len(overlap) / len(oracle),
        "oracle_pareto_precision": (
            len(overlap) / len(predicted) if predicted else 0.0
        ),
        "first_oracle_hit_gpu_hours": first_hit,
        "primary_metric": primary_metric,
        "best_primary_regret_pct": regret,
        "l4_verification_count": len(verification_events),
        "unnecessary_l4_verifications": sum(
            event.candidate_id not in oracle for event in verification_events
        ),
    }


def _aggregate_policy(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ReplayEvaluationError("cannot aggregate an empty policy result")
    first_hits = [
        float(row["first_oracle_hit_gpu_hours"])
        for row in rows
        if row["first_oracle_hit_gpu_hours"] is not None
    ]
    regrets = [
        float(row["best_primary_regret_pct"])
        for row in rows
        if row["best_primary_regret_pct"] is not None
    ]
    return {
        "episode_count": len(rows),
        "success_rate": mean(
            1.0 if float(row["oracle_pareto_recall"]) > 0 else 0.0
            for row in rows
        ),
        "mean_oracle_pareto_recall": mean(
            float(row["oracle_pareto_recall"]) for row in rows
        ),
        "mean_oracle_pareto_precision": mean(
            float(row["oracle_pareto_precision"]) for row in rows
        ),
        "mean_spent_gpu_hours": mean(
            float(row["spent_gpu_hours"]) for row in rows
        ),
        "mean_exploration_gpu_hours": mean(
            float(row["exploration_gpu_hours"]) for row in rows
        ),
        "median_first_oracle_hit_gpu_hours": (
            median(first_hits) if first_hits else None
        ),
        "mean_best_primary_regret_pct": mean(regrets) if regrets else None,
        "mean_unnecessary_l4_verifications": mean(
            float(row["unnecessary_l4_verifications"]) for row in rows
        ),
        "mean_failed_action_count": mean(
            float(row["failed_action_count"]) for row in rows
        ),
    }


def _best(values: Iterable[float], direction: str) -> float:
    materialized = list(values)
    return min(materialized) if direction == "min" else max(materialized)
