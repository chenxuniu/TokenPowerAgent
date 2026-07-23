"""Budgeted plan-act-observe-update loop with mandatory L4 verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tokenpoweragent.agent.planner import (
    Plan,
    PlanningState,
    RuleBasedPlanner,
    SemanticPlanner,
    Subgoal,
)
from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.executors.base import Executor
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.policy.ipig import Action, CandidateBelief, IPIGPolicy, PolicyDecision
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario
from tokenpoweragent.twin.base import EnergyTwin, Prediction
from tokenpoweragent.twin.simple import EmpiricalEnergyTwin


class BudgetViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionEvent:
    phase: str
    subgoal: str
    candidate_id: str
    level: str
    score: Optional[float]
    gpu_hours: float
    status: str
    rationale: str


@dataclass(frozen=True)
class AgentRun:
    scenario: str
    spent_gpu_hours: float
    exploration_gpu_hours: float
    verification_gpu_hours: float
    events: Sequence[DecisionEvent]
    verified_pareto_ids: Sequence[str]
    evidence: Sequence[EvidenceRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "spent_gpu_hours": round(self.spent_gpu_hours, 6),
            "exploration_gpu_hours": round(self.exploration_gpu_hours, 6),
            "verification_gpu_hours": round(self.verification_gpu_hours, 6),
            "verified_pareto_ids": list(self.verified_pareto_ids),
            "events": [asdict(event) for event in self.events],
            "evidence": [record.to_dict() for record in self.evidence],
        }


class TokenPowerAgent:
    def __init__(
        self,
        scenario: Scenario,
        executor: Executor,
        twin: Optional[EnergyTwin] = None,
        planner: Optional[SemanticPlanner] = None,
        policy: Optional[IPIGPolicy] = None,
        initial_evidence: Sequence[EvidenceRecord] = (),
    ) -> None:
        self.scenario = scenario
        self.executor = executor
        self.twin = twin or EmpiricalEnergyTwin()
        self.planner = planner or RuleBasedPlanner()
        self.policy = policy or IPIGPolicy()
        self.store = EvidenceStore(initial_evidence)
        for record in initial_evidence:
            self.twin.update(record)

    def run(self, max_steps: int = 20, min_ipig_score: float = 0.0) -> AgentRun:
        verification_cap = (
            self.scenario.budget.verify_top_k
            * self.scenario.level_cost_gpu_hours[EvidenceLevel.L4]
        )
        if verification_cap > self.scenario.budget.gpu_hours:
            raise BudgetViolation("verification reserve exceeds the total budget")

        exploration_remaining = self.scenario.budget.gpu_hours - verification_cap
        exploration_spent = 0.0
        verification_spent = 0.0
        events: List[DecisionEvent] = []

        for step in range(max_steps):
            beliefs = self._beliefs()
            plan = self.planner.plan(self._planning_state(beliefs, exploration_remaining))
            actions = self._exploration_actions(exploration_remaining)
            guarded = self._guard(actions, plan)
            if not guarded:
                break

            decision = self.policy.select(guarded, beliefs, plan.subgoal.value)
            if decision.score <= min_ipig_score:
                break
            record = self._execute(decision.action, step, exploration_remaining)
            self.store.append(record)
            self.twin.update(record)
            exploration_spent += record.gpu_hours
            exploration_remaining -= record.gpu_hours
            events.append(self._event("search", plan, decision, record))

        verification_remaining = self.scenario.budget.gpu_hours - exploration_spent
        verification_records = []
        for offset, candidate in enumerate(self._verification_candidates()):
            expected = self.scenario.level_cost_gpu_hours[EvidenceLevel.L4]
            if expected > verification_remaining + 1e-12:
                break
            action = Action(candidate.candidate_id, EvidenceLevel.L4, expected)
            record = self._execute(action, max_steps + offset, verification_remaining)
            self.store.append(record)
            self.twin.update(record)
            verification_spent += record.gpu_hours
            verification_remaining -= record.gpu_hours
            plan = Plan(Subgoal.VERIFY, "target-scale verification of a predicted Pareto candidate")
            decision = PolicyDecision(action, 0.0, 0.0, plan.rationale)
            events.append(self._event("verification", plan, decision, record))
            if record.verified and record.satisfies(self.scenario.slo):
                verification_records.append(record)

        verified_rows = [
            (record.candidate_id, record.metrics) for record in verification_records
        ]
        verified_ids = pareto_front(verified_rows, self.scenario.objectives)
        spent = exploration_spent + verification_spent
        if spent > self.scenario.budget.gpu_hours + 1e-9:
            raise BudgetViolation("agent exceeded the scenario budget")
        return AgentRun(
            scenario=self.scenario.name,
            spent_gpu_hours=spent,
            exploration_gpu_hours=exploration_spent,
            verification_gpu_hours=verification_spent,
            events=events,
            verified_pareto_ids=verified_ids,
            evidence=self.store.records,
        )

    def _predictions(self) -> Mapping[str, Prediction]:
        return {
            candidate.candidate_id: self.twin.predict(candidate)
            for candidate in self.scenario.candidates
        }

    def _beliefs(self) -> Mapping[str, CandidateBelief]:
        predictions = self._predictions()
        rows = [
            (candidate_id, prediction.metrics)
            for candidate_id, prediction in predictions.items()
        ]
        frontier = set(pareto_front(rows, self.scenario.objectives))
        beliefs: Dict[str, CandidateBelief] = {}
        for candidate in self.scenario.candidates:
            prediction = predictions[candidate.candidate_id]
            beliefs[candidate.candidate_id] = CandidateBelief(
                uncertainty=prediction.uncertainty,
                frontier_probability=0.75 if candidate.candidate_id in frontier else 0.20,
                slo_boundary_probability=self._slo_boundary_probability(prediction.metrics),
                topology_gap=(
                    1.0
                    if candidate.target_nodes > 1 and prediction.source_level < EvidenceLevel.L3
                    else 0.0
                ),
            )
        return beliefs

    def _planning_state(
        self,
        beliefs: Mapping[str, CandidateBelief],
        exploration_remaining: float,
    ) -> PlanningState:
        records = self.store.records
        return PlanningState(
            evidence_count=len(records),
            failure_count=sum(
                record.status == EvidenceStatus.FAILED for record in records
            ),
            all_candidates_have_l0=all(
                self.store.has(candidate.candidate_id, EvidenceLevel.L0)
                for candidate in self.scenario.candidates
            ),
            max_slo_boundary_probability=max(
                belief.slo_boundary_probability for belief in beliefs.values()
            ),
            max_topology_gap=max(belief.topology_gap for belief in beliefs.values()),
            remaining_exploration_gpu_hours=exploration_remaining,
        )

    def _exploration_actions(self, remaining: float) -> List[Action]:
        actions = []
        for candidate in self.scenario.candidates:
            for level in self.scenario.available_levels:
                if level == EvidenceLevel.L4 or self.store.has(candidate.candidate_id, level):
                    continue
                cost = self.scenario.level_cost_gpu_hours[level]
                if cost <= remaining + 1e-12:
                    actions.append(Action(candidate.candidate_id, level, cost))
        return actions

    @staticmethod
    def _guard(actions: Sequence[Action], plan: Plan) -> List[Action]:
        if plan.subgoal in {Subgoal.EXPLORE, Subgoal.REPAIR}:
            guarded = [action for action in actions if action.level <= EvidenceLevel.L1]
        elif plan.subgoal == Subgoal.CALIBRATE_SCALE:
            guarded = [
                action
                for action in actions
                if EvidenceLevel.L2 <= action.level <= EvidenceLevel.L3
            ]
        else:
            guarded = list(actions)
        return guarded or list(actions)

    def _execute(self, action: Action, seed: int, remaining: float) -> EvidenceRecord:
        candidate = self.scenario.candidate(action.candidate_id)
        record = self.executor.execute(candidate, action.level, seed)
        if record.candidate_id != action.candidate_id or record.level != action.level:
            raise ValueError("executor returned evidence for a different action")
        if record.gpu_hours > action.expected_gpu_hours + 1e-9:
            raise BudgetViolation("actual cost exceeded the action's declared upper bound")
        if record.gpu_hours > remaining + 1e-9:
            raise BudgetViolation("evidence acquisition exceeded the remaining budget")
        return record

    def _verification_candidates(self) -> List[Candidate]:
        predictions = self._predictions()
        feasible = [
            candidate
            for candidate in self.scenario.candidates
            if self.scenario.slo.accepts(predictions[candidate.candidate_id].metrics)
        ]
        rows = [
            (candidate.candidate_id, predictions[candidate.candidate_id].metrics)
            for candidate in feasible
        ]
        frontier = set(pareto_front(rows, self.scenario.objectives))
        candidates = [candidate for candidate in feasible if candidate.candidate_id in frontier]
        primary_metric, direction = next(iter(self.scenario.objectives.items()))
        candidates.sort(
            key=lambda candidate: predictions[candidate.candidate_id].metrics[primary_metric],
            reverse=direction == "max",
        )
        return candidates[: self.scenario.budget.verify_top_k]

    def _slo_boundary_probability(self, metrics: Mapping[str, float]) -> float:
        distances = []
        for metric, limit in (
            ("ttft_ms", self.scenario.slo.ttft_ms),
            ("tpot_ms", self.scenario.slo.tpot_ms),
        ):
            if limit is not None and metric in metrics:
                relative_distance = abs(metrics[metric] - limit) / max(abs(limit), 1.0)
                distances.append(max(0.0, 1.0 - 5.0 * relative_distance))
        if self.scenario.slo.min_goodput_req_s is not None and "goodput_req_s" in metrics:
            limit = self.scenario.slo.min_goodput_req_s
            relative_distance = abs(metrics["goodput_req_s"] - limit) / max(abs(limit), 1.0)
            distances.append(max(0.0, 1.0 - 5.0 * relative_distance))
        return max(distances) if distances else 0.1

    @staticmethod
    def _event(
        phase: str,
        plan: Plan,
        decision: PolicyDecision,
        record: EvidenceRecord,
    ) -> DecisionEvent:
        return DecisionEvent(
            phase=phase,
            subgoal=plan.subgoal.value,
            candidate_id=record.candidate_id,
            level=record.level.name,
            score=decision.score,
            gpu_hours=record.gpu_hours,
            status=record.status.value,
            rationale=decision.rationale,
        )
