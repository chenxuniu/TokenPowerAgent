"""Budgeted plan-act-observe-update loop with mandatory L4 verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tokenpoweragent.agent.planner import (
    Plan,
    PlanningState,
    RuleBasedPlanner,
    SemanticPlanner,
    Subgoal,
)
from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
)
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.policy.ipig import (
    AcquisitionPolicy,
    Action,
    CandidateBelief,
    IPIGPolicy,
    PolicyDecision,
)
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario
from tokenpoweragent.twin.base import EnergyTwin, Prediction
from tokenpoweragent.twin.simple import EmpiricalEnergyTwin


class BudgetViolation(RuntimeError):
    pass


class RunStatus(str, Enum):
    VERIFIED = "verified"
    ABSTAINED = "abstained"


class SearchStopReason(str, Enum):
    MAX_STEPS = "max_steps"
    NO_ACTIONS = "no_actions"
    LOW_INFORMATION_GAIN = "low_information_gain"
    FRONTIER_STABLE = "frontier_stable"


@dataclass(frozen=True)
class DecisionEvent:
    step: int
    phase: str
    subgoal: str
    planner: str
    candidate_id: str
    level: str
    score: Optional[float]
    information_gain: Optional[float]
    expected_gpu_hours: float
    gpu_hours: float
    cumulative_gpu_hours: float
    remaining_gpu_hours: float
    status: str
    evidence_kind: str
    uncertainty_before: Optional[float]
    uncertainty_after: Optional[float]
    predicted_pareto_before: Sequence[str]
    predicted_pareto_after: Sequence[str]
    observed_metrics: Mapping[str, float]
    rationale: str
    planner_fallback_reason: Optional[str] = None
    planner_proposed_subgoal: Optional[str] = None
    planner_guard_intervened: bool = False


@dataclass(frozen=True)
class AgentRun:
    scenario: str
    intent: str
    policy: str
    semantic_guard_active: bool
    planner_state_guard_active: bool
    run_seed: int
    status: RunStatus
    outcome_reason: str
    search_stop_reason: SearchStopReason
    spent_gpu_hours: float
    exploration_gpu_hours: float
    verification_gpu_hours: float
    events: Sequence[DecisionEvent]
    predicted_pareto_ids: Sequence[str]
    verified_pareto_ids: Sequence[str]
    evidence: Sequence[EvidenceRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "intent": self.intent,
            "policy": self.policy,
            "semantic_guard_active": self.semantic_guard_active,
            "planner_state_guard_active": self.planner_state_guard_active,
            "run_seed": self.run_seed,
            "status": self.status.value,
            "outcome_reason": self.outcome_reason,
            "search_stop_reason": self.search_stop_reason.value,
            "spent_gpu_hours": round(self.spent_gpu_hours, 6),
            "exploration_gpu_hours": round(self.exploration_gpu_hours, 6),
            "verification_gpu_hours": round(self.verification_gpu_hours, 6),
            "predicted_pareto_ids": list(self.predicted_pareto_ids),
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
        policy: Optional[AcquisitionPolicy] = None,
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

    def run(
        self,
        max_steps: int = 20,
        min_ipig_score: float = 0.0,
        frontier_patience: int = 0,
        run_seed: int = 0,
    ) -> AgentRun:
        if max_steps < 0:
            raise ValueError("max_steps cannot be negative")
        if frontier_patience < 0:
            raise ValueError("frontier_patience cannot be negative")
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
        search_stop_reason = SearchStopReason.MAX_STEPS
        stable_frontier_steps = 0
        last_frontier = tuple(self._predicted_frontier())

        for step in range(max_steps):
            beliefs = self._beliefs()
            frontier_before = tuple(self._predicted_frontier())
            plan = self.planner.plan(
                self._planning_state(
                    beliefs,
                    exploration_remaining,
                    step,
                    frontier_before,
                )
            )
            actions = self._exploration_actions(exploration_remaining)
            guarded = (
                self._guard(actions, plan)
                if getattr(self.policy, "uses_semantic_guard", True)
                else list(actions)
            )
            if not guarded:
                search_stop_reason = SearchStopReason.NO_ACTIONS
                break

            decision = self.policy.select(guarded, beliefs, plan.subgoal.value)
            if decision.score <= min_ipig_score:
                search_stop_reason = SearchStopReason.LOW_INFORMATION_GAIN
                break
            try:
                record = self._execute(
                    decision.action,
                    run_seed,
                    exploration_remaining,
                )
            except ExecutionError as exc:
                record = self._failed_evidence(decision.action, exc, run_seed)
            self.store.append(record)
            self.twin.update(record)
            exploration_spent += record.gpu_hours
            exploration_remaining -= record.gpu_hours
            beliefs_after = self._beliefs()
            frontier_after = tuple(self._predicted_frontier())
            events.append(
                self._event(
                    step=step,
                    phase="search",
                    plan=plan,
                    decision=decision,
                    record=record,
                    cumulative_gpu_hours=exploration_spent,
                    remaining_gpu_hours=(
                        self.scenario.budget.gpu_hours - exploration_spent
                    ),
                    beliefs_before=beliefs,
                    beliefs_after=beliefs_after,
                    frontier_before=frontier_before,
                    frontier_after=frontier_after,
                )
            )

            if frontier_after == last_frontier:
                stable_frontier_steps += 1
            else:
                stable_frontier_steps = 0
            last_frontier = frontier_after
            if frontier_patience and stable_frontier_steps >= frontier_patience:
                search_stop_reason = SearchStopReason.FRONTIER_STABLE
                break

        verification_remaining = self.scenario.budget.gpu_hours - exploration_spent
        verification_records: List[EvidenceRecord] = []
        verification_candidates = self._verification_candidates()
        verification_attempts = 0
        verification_start_step = len(events)
        for offset, candidate in enumerate(verification_candidates):
            expected = self.scenario.level_cost_gpu_hours[EvidenceLevel.L4]
            if expected > verification_remaining + 1e-12:
                break
            action = Action(candidate.candidate_id, EvidenceLevel.L4, expected)
            beliefs_before = self._beliefs()
            frontier_before = tuple(self._predicted_frontier())
            seed = run_seed
            try:
                record = self._execute(action, seed, verification_remaining)
            except ExecutionError as exc:
                record = self._failed_evidence(action, exc, seed)
            verification_attempts += 1
            self.store.append(record)
            self.twin.update(record)
            verification_spent += record.gpu_hours
            verification_remaining -= record.gpu_hours
            plan = Plan(
                Subgoal.VERIFY,
                "target-scale verification of a predicted Pareto candidate",
                planner="deterministic-guard",
            )
            decision = PolicyDecision(action, 0.0, 0.0, plan.rationale)
            events.append(
                self._event(
                    step=verification_start_step + offset,
                    phase="verification",
                    plan=plan,
                    decision=decision,
                    record=record,
                    cumulative_gpu_hours=exploration_spent + verification_spent,
                    remaining_gpu_hours=verification_remaining,
                    beliefs_before=beliefs_before,
                    beliefs_after=self._beliefs(),
                    frontier_before=frontier_before,
                    frontier_after=tuple(self._predicted_frontier()),
                )
            )
            if record.verified and record.satisfies(self.scenario.slo):
                verification_records.append(record)

        verified_rows = [
            (record.candidate_id, record.metrics) for record in verification_records
        ]
        verified_ids = pareto_front(verified_rows, self.scenario.objectives)
        if verified_ids:
            outcome_reason = "one or more SLO-feasible candidates passed L4 verification"
        elif not verification_candidates:
            outcome_reason = "no predicted SLO-feasible Pareto candidate was available"
        elif verification_attempts == 0:
            outcome_reason = "remaining budget could not fund required L4 verification"
        else:
            outcome_reason = "all attempted L4 candidates failed or violated the SLO"
        spent = exploration_spent + verification_spent
        if spent > self.scenario.budget.gpu_hours + 1e-9:
            raise BudgetViolation("agent exceeded the scenario budget")
        return AgentRun(
            scenario=self.scenario.name,
            intent=self.scenario.intent,
            policy=getattr(self.policy, "name", type(self.policy).__name__),
            semantic_guard_active=getattr(
                self.policy, "uses_semantic_guard", True
            ),
            planner_state_guard_active=(
                getattr(self.planner, "state_guard", None) is not None
            ),
            run_seed=run_seed,
            status=(RunStatus.VERIFIED if verified_ids else RunStatus.ABSTAINED),
            outcome_reason=outcome_reason,
            search_stop_reason=search_stop_reason,
            spent_gpu_hours=spent,
            exploration_gpu_hours=exploration_spent,
            verification_gpu_hours=verification_spent,
            events=events,
            predicted_pareto_ids=self._predicted_frontier(slo_feasible_only=True),
            verified_pareto_ids=verified_ids,
            evidence=self.store.records,
        )

    def _predictions(self) -> Mapping[str, Prediction]:
        return {
            candidate.candidate_id: self.twin.predict(candidate)
            for candidate in self.scenario.candidates
        }

    def _predicted_frontier(
        self, slo_feasible_only: bool = False
    ) -> List[str]:
        predictions = self._predictions()
        rows = []
        for candidate in self.scenario.candidates:
            metrics = predictions[candidate.candidate_id].metrics
            if slo_feasible_only and not self.scenario.slo.accepts(metrics):
                continue
            rows.append((candidate.candidate_id, metrics))
        return pareto_front(rows, self.scenario.objectives)

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
        step: int,
        predicted_pareto_ids: Sequence[str],
    ) -> PlanningState:
        records = self.store.records
        latest = self.store.latest
        return PlanningState(
            intent=self.scenario.intent,
            step=step,
            evidence_count=len(records),
            failure_count=sum(
                record.status == EvidenceStatus.FAILED for record in records
            ),
            last_action_failed=(
                latest is not None and latest.status == EvidenceStatus.FAILED
            ),
            all_candidates_have_l0=all(
                self.store.has_successful(candidate.candidate_id, EvidenceLevel.L0)
                for candidate in self.scenario.candidates
            ),
            max_slo_boundary_probability=max(
                belief.slo_boundary_probability for belief in beliefs.values()
            ),
            max_topology_gap=max(belief.topology_gap for belief in beliefs.values()),
            remaining_exploration_gpu_hours=exploration_remaining,
            predicted_pareto_ids=tuple(predicted_pareto_ids),
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

    @staticmethod
    def _failed_evidence(
        action: Action, error: ExecutionError, seed: int
    ) -> EvidenceRecord:
        kinds = {
            EvidenceLevel.L0: EvidenceKind.SIMULATED,
            EvidenceLevel.L1: EvidenceKind.MEASURED,
            EvidenceLevel.L2: EvidenceKind.EXTRAPOLATED,
            EvidenceLevel.L3: EvidenceKind.MEASURED,
            EvidenceLevel.L4: EvidenceKind.VERIFIED,
        }
        return EvidenceRecord(
            candidate_id=action.candidate_id,
            level=action.level,
            metrics={},
            gpu_hours=action.expected_gpu_hours,
            kind=kinds[action.level],
            status=EvidenceStatus.FAILED,
            provenance={
                "executor_error_type": type(error).__name__,
                "seed": seed,
                "budget_charge": "declared upper bound",
            },
            failure_reason=str(error),
        )

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
        step: int,
        phase: str,
        plan: Plan,
        decision: PolicyDecision,
        record: EvidenceRecord,
        cumulative_gpu_hours: float,
        remaining_gpu_hours: float,
        beliefs_before: Mapping[str, CandidateBelief],
        beliefs_after: Mapping[str, CandidateBelief],
        frontier_before: Sequence[str],
        frontier_after: Sequence[str],
    ) -> DecisionEvent:
        before = beliefs_before.get(record.candidate_id)
        after = beliefs_after.get(record.candidate_id)
        return DecisionEvent(
            step=step,
            phase=phase,
            subgoal=plan.subgoal.value,
            planner=plan.planner,
            candidate_id=record.candidate_id,
            level=record.level.name,
            score=decision.score,
            information_gain=decision.information_gain,
            expected_gpu_hours=decision.action.expected_gpu_hours,
            gpu_hours=record.gpu_hours,
            cumulative_gpu_hours=cumulative_gpu_hours,
            remaining_gpu_hours=remaining_gpu_hours,
            status=record.status.value,
            evidence_kind=record.kind.value,
            uncertainty_before=(None if before is None else before.uncertainty),
            uncertainty_after=(None if after is None else after.uncertainty),
            predicted_pareto_before=tuple(frontier_before),
            predicted_pareto_after=tuple(frontier_after),
            observed_metrics=dict(record.metrics),
            rationale=plan.rationale + "; " + decision.rationale,
            planner_fallback_reason=plan.fallback_reason,
            planner_proposed_subgoal=(
                None
                if plan.proposed_subgoal is None
                else plan.proposed_subgoal.value
            ),
            planner_guard_intervened=plan.guard_intervened,
        )
