from pathlib import Path

from tokenpoweragent.agent import ServeCompass
from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.schema import Scenario


def test_servecompass_public_name_preserves_legacy_controller() -> None:
    assert ServeCompass is TokenPowerAgent


ROOT = Path(__file__).resolve().parents[1]


def test_replay_agent_reserves_budget_and_verifies_recommendations() -> None:
    scenario = Scenario.load(ROOT / "configs/scenarios/replay_demo.json")
    executor = ReplayExecutor.from_jsonl(ROOT / "configs/replay/demo_records.jsonl")
    report = TokenPowerAgent(scenario, executor).run(max_steps=5)

    assert report.spent_gpu_hours <= scenario.budget.gpu_hours
    assert report.verification_gpu_hours == 2.0
    assert report.verified_pareto_ids
    verified = {
        record.candidate_id
        for record in report.evidence
        if record.verified and record.satisfies(scenario.slo)
    }
    assert set(report.verified_pareto_ids) <= verified
    assert any(event.subgoal == "calibrate_scale" for event in report.events)
