"""Matched-budget replay sweeps over a sealed empirical evidence corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.agent.evaluation import (
    POLICY_NAMES,
    evaluate_replay_policies,
)
from tokenpoweragent.evidence import EvidenceRecord
from tokenpoweragent.schema import Budget, EvidenceLevel, Scenario


class BudgetSweepError(ValueError):
    """Raised when a budget-sweep protocol or input corpus is invalid."""


@dataclass(frozen=True)
class BudgetSweepProtocol:
    schema_version: str
    benchmark_id: str
    scenario_sha256: str
    corpus_sha256: str
    budgets_gpu_hours: Tuple[float, ...]
    policy_names: Tuple[str, ...]
    episodes_per_policy: int
    max_steps: int
    min_ipig_score: float
    frontier_patience: int

    @classmethod
    def load(cls, path: Path) -> "BudgetSweepProtocol":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetSweepError("cannot read budget protocol: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise BudgetSweepError("budget protocol root must be an object")

        expected = _mapping(raw, "expected_inputs")
        replay = _mapping(raw, "replay")
        budgets_raw = raw.get("budgets_gpu_hours")
        if not isinstance(budgets_raw, list) or not budgets_raw:
            raise BudgetSweepError("budgets_gpu_hours must be a non-empty list")
        try:
            budgets = tuple(float(value) for value in budgets_raw)
        except (TypeError, ValueError) as exc:
            raise BudgetSweepError("budgets_gpu_hours must be numeric") from exc
        if any(value <= 0 for value in budgets):
            raise BudgetSweepError("budgets_gpu_hours must be positive")
        if tuple(sorted(budgets)) != budgets or len(set(budgets)) != len(budgets):
            raise BudgetSweepError(
                "budgets_gpu_hours must be unique and strictly increasing"
            )

        policies_raw = replay.get("policies")
        if not isinstance(policies_raw, list) or not policies_raw:
            raise BudgetSweepError("replay.policies must be a non-empty list")
        policies = tuple(str(value).strip().lower() for value in policies_raw)
        if len(set(policies)) != len(policies):
            raise BudgetSweepError("replay policies must be unique")
        unknown = [name for name in policies if name not in POLICY_NAMES]
        if unknown:
            raise BudgetSweepError(
                "unknown policies: %s" % ", ".join(sorted(unknown))
            )

        benchmark_id = str(raw.get("benchmark_id", "")).strip()
        if not benchmark_id:
            raise BudgetSweepError("benchmark_id cannot be empty")
        scenario_sha256 = _sha256_text(expected, "scenario_sha256")
        corpus_sha256 = _sha256_text(expected, "corpus_sha256")
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            benchmark_id=benchmark_id,
            scenario_sha256=scenario_sha256,
            corpus_sha256=corpus_sha256,
            budgets_gpu_hours=budgets,
            policy_names=policies,
            episodes_per_policy=_positive_int(replay, "episodes_per_policy"),
            max_steps=_positive_int(replay, "max_steps"),
            min_ipig_score=_nonnegative_float(replay, "min_ipig_score"),
            frontier_patience=_nonnegative_int(replay, "frontier_patience"),
        )


def evaluate_budget_sweep(
    scenario: Scenario,
    records: Sequence[EvidenceRecord],
    protocol: BudgetSweepProtocol,
    source_sha256: str,
    scenario_sha256: str,
    protocol_sha256: str = "",
) -> Dict[str, Any]:
    if source_sha256 != protocol.corpus_sha256:
        raise BudgetSweepError(
            "corpus SHA-256 does not match the preregistered protocol"
        )
    if scenario_sha256 != protocol.scenario_sha256:
        raise BudgetSweepError(
            "scenario SHA-256 does not match the preregistered protocol"
        )
    verification_reserve = (
        scenario.budget.verify_top_k
        * scenario.level_cost_gpu_hours[EvidenceLevel.L4]
    )
    invalid_budgets = [
        budget
        for budget in protocol.budgets_gpu_hours
        if budget + 1e-12 < verification_reserve
    ]
    if invalid_budgets:
        raise BudgetSweepError(
            "budgets below the %.6f GPU-hour verification reserve: %s"
            % (
                verification_reserve,
                ", ".join("%.6f" % value for value in invalid_budgets),
            )
        )

    points: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    oracle: Optional[Mapping[str, Any]] = None
    for budget_gpu_hours in protocol.budgets_gpu_hours:
        budget_scenario = replace(
            scenario,
            budget=Budget(
                gpu_hours=budget_gpu_hours,
                verify_top_k=scenario.budget.verify_top_k,
            ),
        )
        result = evaluate_replay_policies(
            scenario=budget_scenario,
            records=records,
            policy_names=protocol.policy_names,
            episodes=protocol.episodes_per_policy,
            max_steps=protocol.max_steps,
            min_ipig_score=protocol.min_ipig_score,
            frontier_patience=protocol.frontier_patience,
            source_sha256=source_sha256,
            scenario_sha256=scenario_sha256,
        )
        if oracle is None:
            oracle = result["oracle"]
        elif result["oracle"] != oracle:
            raise BudgetSweepError("the replay oracle changed across budget points")
        point_episodes = [
            {"budget_gpu_hours": budget_gpu_hours, **row}
            for row in result["episodes"]
        ]
        episodes.extend(point_episodes)
        points.append(
            {
                "budget_gpu_hours": budget_gpu_hours,
                "exploration_budget_gpu_hours": (
                    budget_gpu_hours - verification_reserve
                ),
                "aggregate": result["aggregate"],
                "paired_ipig_comparisons": _paired_ipig(point_episodes),
            }
        )

    budget_response = {
        policy: _budget_response_policy(points, policy)
        for policy in protocol.policy_names
    }
    return {
        "schema_version": "1.0",
        "benchmark_id": protocol.benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "scenario_sha256": scenario_sha256,
        "protocol_sha256": protocol_sha256 or None,
        "scenario": scenario.name,
        "intent": scenario.intent,
        "protocol": {
            "budgets_gpu_hours": list(protocol.budgets_gpu_hours),
            "verification_reserve_gpu_hours": verification_reserve,
            "policies": list(protocol.policy_names),
            "episodes_per_policy_per_budget": protocol.episodes_per_policy,
            "total_episode_count": len(episodes),
            "max_steps": protocol.max_steps,
            "min_ipig_score": protocol.min_ipig_score,
            "frontier_patience": protocol.frontier_patience,
            "verification_required": True,
            "oracle_semantics": "median successful L4 metrics per candidate",
            "episode_semantics": (
                "candidate-level empirical bootstrap with common random numbers "
                "across policies and matched seed ids across budgets"
            ),
            "resampling_scheme": "sha256(seed,candidate_id,evidence_level)",
            "auc_semantics": (
                "trapezoidal area normalized by the preregistered budget range"
            ),
        },
        "oracle": oracle,
        "points": points,
        "budget_response": budget_response,
        "episodes": episodes,
    }


def _budget_response_policy(
    points: Sequence[Mapping[str, Any]], policy: str
) -> Dict[str, Any]:
    budgets = [float(point["budget_gpu_hours"]) for point in points]
    aggregates = [point["aggregate"][policy] for point in points]
    success = [float(row["success_rate"]) for row in aggregates]
    recall = [float(row["mean_oracle_pareto_recall"]) for row in aggregates]
    precision = [float(row["mean_oracle_pareto_precision"]) for row in aggregates]
    spent = [float(row["mean_spent_gpu_hours"]) for row in aggregates]
    regrets = [row["mean_best_primary_regret_pct"] for row in aggregates]
    maximum = max(success)
    success_drops = [
        success[index] - success[index + 1]
        for index in range(len(success) - 1)
        if success[index + 1] < success[index]
    ]
    return {
        "normalized_success_auc": _normalized_auc(budgets, success),
        "normalized_recall_auc": _normalized_auc(budgets, recall),
        "normalized_precision_auc": _normalized_auc(budgets, precision),
        "normalized_mean_spent_gpu_hours_auc": _normalized_auc(budgets, spent),
        "normalized_conditional_regret_auc": (
            None
            if any(value is None for value in regrets)
            else _normalized_auc(budgets, [float(value) for value in regrets])
        ),
        "maximum_success_rate": maximum,
        "success_monotonic_nondecreasing": not success_drops,
        "success_rate_decrease_count": len(success_drops),
        "largest_success_rate_drop": max(success_drops, default=0.0),
        "first_budget_at_maximum_success_gpu_hours": next(
            budget for budget, value in zip(budgets, success) if value == maximum
        ),
        "first_budget_at_50pct_success_gpu_hours": next(
            (budget for budget, value in zip(budgets, success) if value >= 0.5),
            None,
        ),
        "point_count": len(points),
    }


def _paired_ipig(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_policy_seed = {
        (str(row["policy"]), int(row["seed"])): row for row in rows
    }
    if not any(policy == "ipig" for policy, _ in by_policy_seed):
        return {}
    seeds = sorted(seed for policy, seed in by_policy_seed if policy == "ipig")
    policies = sorted(
        {policy for policy, _ in by_policy_seed if policy != "ipig"}
    )
    comparisons: Dict[str, Any] = {}
    for policy in policies:
        counts = {
            "both_hit": 0,
            "ipig_only_hit": 0,
            "baseline_only_hit": 0,
            "neither_hit": 0,
        }
        for seed in seeds:
            ipig = by_policy_seed[("ipig", seed)]
            baseline = by_policy_seed[(policy, seed)]
            ipig_hit = float(ipig["oracle_pareto_recall"]) > 0
            baseline_hit = float(baseline["oracle_pareto_recall"]) > 0
            if ipig_hit and baseline_hit:
                counts["both_hit"] += 1
            elif ipig_hit:
                counts["ipig_only_hit"] += 1
            elif baseline_hit:
                counts["baseline_only_hit"] += 1
            else:
                counts["neither_hit"] += 1
        comparisons[policy] = counts
    return comparisons


def _normalized_auc(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or not x_values:
        raise BudgetSweepError("AUC inputs must have equal non-zero length")
    if len(x_values) == 1:
        return float(y_values[0])
    area = sum(
        (x_values[index + 1] - x_values[index])
        * (y_values[index + 1] + y_values[index])
        / 2.0
        for index in range(len(x_values) - 1)
    )
    return area / (x_values[-1] - x_values[0])


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise BudgetSweepError("%s must be an object" % key)
    return value


def _sha256_text(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise BudgetSweepError("%s must be a SHA-256 hex digest" % key)
    return value


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_int(raw, key)
    if value < 1:
        raise BudgetSweepError("%s must be positive" % key)
    return value


def _nonnegative_int(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetSweepError("%s must be an integer" % key) from exc
    if value < 0:
        raise BudgetSweepError("%s cannot be negative" % key)
    return value


def _nonnegative_float(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetSweepError("%s must be numeric" % key) from exc
    if value < 0:
        raise BudgetSweepError("%s cannot be negative" % key)
    return value
