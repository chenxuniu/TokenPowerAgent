"""Typed semantic subgoals for a bounded LLM or deterministic planner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Subgoal(str, Enum):
    EXPLORE = "explore"
    RESOLVE_SLO = "resolve_slo"
    CALIBRATE_SCALE = "calibrate_scale"
    REPAIR = "repair"
    VERIFY = "verify"


@dataclass(frozen=True)
class PlanningState:
    evidence_count: int
    failure_count: int
    all_candidates_have_l0: bool
    max_slo_boundary_probability: float
    max_topology_gap: float
    remaining_exploration_gpu_hours: float


@dataclass(frozen=True)
class Plan:
    subgoal: Subgoal
    rationale: str


class SemanticPlanner(ABC):
    @abstractmethod
    def plan(self, state: PlanningState) -> Plan:
        pass


class RuleBasedPlanner(SemanticPlanner):
    """Deterministic baseline with the same output schema as an LLM planner."""

    def plan(self, state: PlanningState) -> Plan:
        if state.failure_count:
            return Plan(Subgoal.REPAIR, "repair or avoid the failed configuration region")
        if state.evidence_count == 0:
            return Plan(Subgoal.EXPLORE, "establish broad low-cost evidence")
        if state.all_candidates_have_l0 and state.max_topology_gap >= 0.5:
            return Plan(Subgoal.CALIBRATE_SCALE, "resolve the largest topology transfer gap")
        if state.max_slo_boundary_probability >= 0.45:
            return Plan(Subgoal.RESOLVE_SLO, "reduce uncertainty at the SLO boundary")
        return Plan(Subgoal.EXPLORE, "cover candidates with the cheapest useful evidence")
