"""Information-per-GPU-hour action selection.

The policy is exact with respect to a supplied information-gain estimator. The
default estimator is a documented proxy for the runnable scaffold; experiments
should inject the nested-posterior Pareto-set mutual-information estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Protocol

from tokenpoweragent.schema import EvidenceLevel


@dataclass(frozen=True)
class Action:
    candidate_id: str
    level: EvidenceLevel
    expected_gpu_hours: float


@dataclass(frozen=True)
class CandidateBelief:
    uncertainty: float
    frontier_probability: float
    slo_boundary_probability: float
    topology_gap: float


class InformationGainEstimator(Protocol):
    def estimate(
        self, action: Action, belief: CandidateBelief, subgoal: str
    ) -> float:
        pass


class PosteriorUncertaintyProxy:
    """Auditable proxy used until nested posterior sampling is connected."""

    _FIDELITY_GAIN = {
        EvidenceLevel.L0: 0.25,
        EvidenceLevel.L1: 0.45,
        EvidenceLevel.L2: 0.70,
        EvidenceLevel.L3: 0.90,
        EvidenceLevel.L4: 1.00,
    }

    def estimate(
        self, action: Action, belief: CandidateBelief, subgoal: str
    ) -> float:
        relevance = (
            0.45 * belief.frontier_probability
            + 0.35 * belief.slo_boundary_probability
            + 0.20 * belief.topology_gap
        )
        if subgoal == "calibrate_scale":
            relevance += 0.20 * belief.topology_gap
        elif subgoal == "resolve_slo":
            relevance += 0.20 * belief.slo_boundary_probability
        return max(0.0, belief.uncertainty * relevance) * self._FIDELITY_GAIN[action.level]


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    score: float
    information_gain: float
    rationale: str


class IPIGPolicy:
    def __init__(
        self,
        estimator: InformationGainEstimator = PosteriorUncertaintyProxy(),
        gamma: float = 0.7,
        epsilon: float = 1e-6,
    ) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.estimator = estimator
        self.gamma = gamma
        self.epsilon = epsilon

    def rank(
        self,
        actions: Iterable[Action],
        beliefs: Mapping[str, CandidateBelief],
        subgoal: str,
    ) -> Dict[Action, float]:
        scores: Dict[Action, float] = {}
        for action in actions:
            information = self.estimator.estimate(
                action, beliefs[action.candidate_id], subgoal
            )
            denominator = (action.expected_gpu_hours + self.epsilon) ** self.gamma
            scores[action] = information / denominator
        return scores

    def select(
        self,
        actions: Iterable[Action],
        beliefs: Mapping[str, CandidateBelief],
        subgoal: str,
    ) -> PolicyDecision:
        materialized = list(actions)
        if not materialized:
            raise ValueError("IPIGPolicy requires at least one action")
        scores = self.rank(materialized, beliefs, subgoal)
        action = max(materialized, key=lambda item: scores[item])
        information = self.estimator.estimate(action, beliefs[action.candidate_id], subgoal)
        return PolicyDecision(
            action=action,
            score=scores[action],
            information_gain=information,
            rationale=(
                "%s selected %s/%s: estimated Pareto information %.4f "
                "for %.3f GPU-hours"
                % (
                    subgoal,
                    action.candidate_id,
                    action.level.name,
                    information,
                    action.expected_gpu_hours,
                )
            ),
        )
