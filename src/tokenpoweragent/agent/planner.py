"""Typed semantic subgoals for a bounded LLM or deterministic planner."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Sequence


class Subgoal(str, Enum):
    EXPLORE = "explore"
    RESOLVE_SLO = "resolve_slo"
    CALIBRATE_SCALE = "calibrate_scale"
    REPAIR = "repair"
    VERIFY = "verify"


@dataclass(frozen=True)
class PlanningState:
    intent: str
    step: int
    evidence_count: int
    failure_count: int
    last_action_failed: bool
    all_candidates_have_l0: bool
    max_slo_boundary_probability: float
    max_topology_gap: float
    remaining_exploration_gpu_hours: float
    predicted_pareto_ids: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent,
            "step": self.step,
            "evidence_count": self.evidence_count,
            "failure_count": self.failure_count,
            "last_action_failed": self.last_action_failed,
            "all_candidates_have_l0": self.all_candidates_have_l0,
            "max_slo_boundary_probability": self.max_slo_boundary_probability,
            "max_topology_gap": self.max_topology_gap,
            "remaining_exploration_gpu_hours": (
                self.remaining_exploration_gpu_hours
            ),
            "predicted_pareto_ids": list(self.predicted_pareto_ids),
        }


@dataclass(frozen=True)
class Plan:
    subgoal: Subgoal
    rationale: str
    planner: str = "rule"
    fallback_reason: Optional[str] = None


class SemanticPlanner(ABC):
    @abstractmethod
    def plan(self, state: PlanningState) -> Plan:
        pass


class RuleBasedPlanner(SemanticPlanner):
    """Deterministic baseline with the same output schema as an LLM planner."""

    def plan(self, state: PlanningState) -> Plan:
        if state.last_action_failed:
            return Plan(
                Subgoal.REPAIR,
                "repair or avoid the most recently failed configuration region",
            )
        if state.evidence_count == 0:
            return Plan(Subgoal.EXPLORE, "establish broad low-cost evidence")
        if state.all_candidates_have_l0 and state.max_topology_gap >= 0.5:
            return Plan(Subgoal.CALIBRATE_SCALE, "resolve the largest topology transfer gap")
        if state.max_slo_boundary_probability >= 0.45:
            return Plan(Subgoal.RESOLVE_SLO, "reduce uncertainty at the SLO boundary")
        return Plan(Subgoal.EXPLORE, "cover candidates with the cheapest useful evidence")


Completion = Callable[[str], str]


class ConstrainedLLMPlanner(SemanticPlanner):
    """Let an LLM select a semantic subgoal behind a strict typed boundary.

    Candidate selection, budget enforcement, execution, SLO checks, and Pareto
    computation remain deterministic. Invalid or unavailable model output is
    recorded and falls back to the rule-based planner.
    """

    def __init__(
        self,
        complete: Completion,
        fallback: Optional[SemanticPlanner] = None,
        max_rationale_chars: int = 400,
    ) -> None:
        self.complete = complete
        self.fallback = fallback or RuleBasedPlanner()
        self.max_rationale_chars = max_rationale_chars

    def plan(self, state: PlanningState) -> Plan:
        prompt = self._prompt(state)
        try:
            raw = self.complete(prompt)
            payload = self._parse_json_object(raw)
            subgoal = Subgoal(str(payload["subgoal"]).strip().lower())
            if subgoal == Subgoal.VERIFY:
                raise ValueError("verify is reserved for the deterministic release gate")
            rationale = str(payload["rationale"]).strip()
            if not rationale:
                raise ValueError("rationale cannot be empty")
            if len(rationale) > self.max_rationale_chars:
                rationale = rationale[: self.max_rationale_chars]
            return Plan(subgoal, rationale, planner="llm")
        except Exception as exc:
            fallback = self.fallback.plan(state)
            return Plan(
                fallback.subgoal,
                fallback.rationale,
                planner="rule-fallback",
                fallback_reason="%s: %s" % (type(exc).__name__, exc),
            )

    @staticmethod
    def _prompt(state: PlanningState) -> str:
        allowed = ", ".join(item.value for item in Subgoal if item != Subgoal.VERIFY)
        return (
            "You are the bounded semantic planner inside TokenPowerAgent. "
            "Choose only the next semantic subgoal. Deterministic code will "
            "select configurations, enforce budgets, execute tools, and verify "
            "recommendations. Use repair after an execution failure; use explore "
            "when no evidence exists or broad low-cost coverage is needed; use "
            "calibrate_scale only when the state reports a material transfer gap; "
            "and use resolve_slo when evidence exists near an SLO boundary. "
            "Apply that priority order when conditions overlap. Never choose "
            "verify, a candidate, an evidence level, or a shell command. Return "
            "one JSON object with exactly two keys: subgoal and rationale. "
            "Allowed subgoals: %s.\nSTATE=%s"
            % (allowed, json.dumps(state.to_dict(), sort_keys=True))
        )

    @staticmethod
    def _parse_json_object(raw: str) -> Dict[str, object]:
        text = str(raw).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("planner response must be a JSON object")
        if set(payload) != {"subgoal", "rationale"}:
            raise ValueError("planner response must contain exactly subgoal and rationale")
        return payload
