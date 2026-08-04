from pathlib import Path

from tokenpoweragent.agent.controller import RunStatus, TokenPowerAgent
from tokenpoweragent.agent.evaluation import evaluate_replay_policies
from tokenpoweragent.agent.planner import (
    ConstrainedLLMPlanner,
    PlanningState,
    Subgoal,
)
from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
)
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.schema import EvidenceLevel, Scenario


ROOT = Path(__file__).resolve().parents[1]


def planning_state() -> PlanningState:
    return PlanningState(
        intent="minimize energy under the latency SLO",
        step=2,
        evidence_count=3,
        failure_count=0,
        last_action_failed=False,
        all_candidates_have_l0=True,
        max_slo_boundary_probability=0.8,
        max_topology_gap=0.0,
        remaining_exploration_gpu_hours=1.5,
        predicted_pareto_ids=("cfg-a",),
    )


def test_constrained_llm_planner_accepts_typed_json() -> None:
    planner = ConstrainedLLMPlanner(
        lambda prompt: (
            '```json\n{"subgoal":"resolve_slo",'
            '"rationale":"measure the uncertain SLO boundary"}\n```'
        )
    )

    plan = planner.plan(planning_state())

    assert plan.subgoal == Subgoal.RESOLVE_SLO
    assert plan.planner == "llm"
    assert plan.fallback_reason is None


def test_constrained_llm_planner_falls_back_on_untyped_output() -> None:
    planner = ConstrainedLLMPlanner(lambda prompt: "run cfg-a on eight GPUs")

    plan = planner.plan(planning_state())

    assert plan.subgoal == Subgoal.RESOLVE_SLO
    assert plan.planner == "rule-fallback"
    assert "JSON" in str(plan.fallback_reason)


def test_constrained_llm_planner_rejects_json_wrapped_in_prose() -> None:
    planner = ConstrainedLLMPlanner(
        lambda prompt: (
            'I suggest {"subgoal":"resolve_slo",'
            '"rationale":"measure the boundary"} next.'
        )
    )

    plan = planner.plan(planning_state())

    assert plan.planner == "rule-fallback"
    assert "JSONDecodeError" in str(plan.fallback_reason)


class FailFirstExecutor(Executor):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, candidate, level, seed):
        self.calls += 1
        if self.calls == 1:
            raise ExecutionError("injected launch failure")
        if level == EvidenceLevel.L4:
            return EvidenceRecord(
                candidate_id=candidate.candidate_id,
                level=level,
                metrics={"energy": 90, "ttft_ms": 90},
                gpu_hours=0.5,
                kind=EvidenceKind.VERIFIED,
            )
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=level,
            metrics={"energy": 100, "ttft_ms": 100},
            gpu_hours=0.1,
            kind=EvidenceKind.SIMULATED,
        )


def test_agent_records_failure_then_enters_repair_subgoal() -> None:
    scenario = Scenario.from_dict(
        {
            "name": "failure-repair",
            "intent": "minimize energy",
            "model": "test/model",
            "objectives": {"energy": "min"},
            "slo": {"ttft_ms": 200},
            "budget": {"gpu_hours": 1.0, "verify_top_k": 1},
            "available_levels": ["L0", "L4"],
            "level_cost_gpu_hours": {"L0": 0.1, "L4": 0.5},
            "candidates": [
                {
                    "id": "cfg-a",
                    "prior_metrics": {"energy": 110, "ttft_ms": 110},
                },
                {
                    "id": "cfg-b",
                    "prior_metrics": {"energy": 120, "ttft_ms": 120},
                },
            ],
        }
    )

    report = TokenPowerAgent(scenario, FailFirstExecutor()).run(max_steps=2)

    assert report.status == RunStatus.VERIFIED
    assert report.events[0].status == EvidenceStatus.FAILED.value
    assert report.events[0].gpu_hours == 0.1
    assert report.events[1].subgoal == Subgoal.REPAIR.value
    assert report.events[-1].phase == "verification"


def test_replay_benchmark_reports_agent_and_baseline_metrics() -> None:
    scenario = Scenario.load(ROOT / "configs/scenarios/replay_demo.json")
    records = EvidenceStore.read_jsonl(
        ROOT / "configs/replay/demo_records.jsonl"
    ).records

    report = evaluate_replay_policies(
        scenario,
        records,
        policy_names=("ipig", "random"),
        episodes=2,
        max_steps=5,
    )

    assert report["oracle"]["pareto_ids"]
    assert set(report["aggregate"]) == {"ipig", "random"}
    assert len(report["episodes"]) == 4
    assert report["aggregate"]["ipig"]["success_rate"] == 1.0


def test_replay_episode_uses_one_matched_seed_for_every_action() -> None:
    scenario = Scenario.load(ROOT / "configs/scenarios/replay_demo.json")
    records = EvidenceStore.read_jsonl(
        ROOT / "configs/replay/demo_records.jsonl"
    ).records

    report = TokenPowerAgent(
        scenario,
        executor=ReplayExecutor(records),
    ).run(max_steps=5, run_seed=7)

    replay_seeds = {
        record.provenance["seed"]
        for record in report.evidence
        if record.provenance.get("executor") == "replay"
    }
    assert replay_seeds == {7}
