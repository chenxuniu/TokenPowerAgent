"""Deterministic Pareto utilities shared by evaluation and recommendation."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple


def dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    objectives: Mapping[str, str],
) -> bool:
    weakly_better = True
    strictly_better = False
    for metric, direction in objectives.items():
        if metric not in left or metric not in right:
            return False
        l_value, r_value = left[metric], right[metric]
        if direction == "min":
            weakly_better &= l_value <= r_value
            strictly_better |= l_value < r_value
        else:
            weakly_better &= l_value >= r_value
            strictly_better |= l_value > r_value
    return weakly_better and strictly_better


def pareto_front(
    rows: Iterable[Tuple[str, Mapping[str, float]]],
    objectives: Mapping[str, str],
) -> List[str]:
    materialized = list(rows)
    return [
        candidate_id
        for candidate_id, metrics in materialized
        if not any(
            other_id != candidate_id and dominates(other_metrics, metrics, objectives)
            for other_id, other_metrics in materialized
        )
    ]
