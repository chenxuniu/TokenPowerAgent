import hashlib
import json
from pathlib import Path

import pytest

from tokenpoweragent.agent.budget_sweep import (
    BudgetSweepError,
    BudgetSweepProtocol,
    evaluate_budget_sweep,
)
from tokenpoweragent.evidence import EvidenceStore
from tokenpoweragent.schema import Scenario


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "configs/scenarios/replay_demo.json"
RECORDS_PATH = ROOT / "configs/replay/demo_records.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protocol(tmp_path: Path, budgets=(2.0, 3.0)) -> Path:
    path = tmp_path / "budget-protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "benchmark_id": "test-budget-sweep",
                "expected_inputs": {
                    "scenario_sha256": sha256_file(SCENARIO_PATH),
                    "corpus_sha256": sha256_file(RECORDS_PATH),
                },
                "budgets_gpu_hours": list(budgets),
                "replay": {
                    "policies": ["ipig", "random"],
                    "episodes_per_policy": 2,
                    "max_steps": 5,
                    "min_ipig_score": 0,
                    "frontier_patience": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_budget_sweep_reports_matched_points_and_budget_response(
    tmp_path: Path,
) -> None:
    protocol_path = write_protocol(tmp_path)
    protocol = BudgetSweepProtocol.load(protocol_path)
    scenario = Scenario.load(SCENARIO_PATH)
    records = EvidenceStore.read_jsonl(RECORDS_PATH).records

    report = evaluate_budget_sweep(
        scenario,
        records,
        protocol,
        source_sha256=sha256_file(RECORDS_PATH),
        scenario_sha256=sha256_file(SCENARIO_PATH),
        protocol_sha256=sha256_file(protocol_path),
    )

    assert report["protocol"]["total_episode_count"] == 8
    assert report["protocol"]["verification_reserve_gpu_hours"] == 2.0
    assert len(report["points"]) == 2
    assert len(report["episodes"]) == 8
    assert set(report["budget_response"]) == {"ipig", "random"}
    assert report["budget_response"]["ipig"]["point_count"] == 2
    assert "success_monotonic_nondecreasing" in report["budget_response"]["ipig"]
    assert (
        0
        <= report["budget_response"]["ipig"]["normalized_success_auc"]
        <= 1
    )
    assert set(report["points"][0]["paired_ipig_comparisons"]) == {"random"}
    assert report["oracle"]["pareto_ids"]


def test_budget_sweep_rejects_hash_mismatch(tmp_path: Path) -> None:
    protocol = BudgetSweepProtocol.load(write_protocol(tmp_path))

    with pytest.raises(BudgetSweepError, match="corpus SHA-256"):
        evaluate_budget_sweep(
            Scenario.load(SCENARIO_PATH),
            EvidenceStore.read_jsonl(RECORDS_PATH).records,
            protocol,
            source_sha256="0" * 64,
            scenario_sha256=sha256_file(SCENARIO_PATH),
        )


def test_budget_sweep_rejects_budget_below_verification_reserve(
    tmp_path: Path,
) -> None:
    protocol = BudgetSweepProtocol.load(write_protocol(tmp_path, budgets=(1.5, 2.0)))

    with pytest.raises(BudgetSweepError, match="verification reserve"):
        evaluate_budget_sweep(
            Scenario.load(SCENARIO_PATH),
            EvidenceStore.read_jsonl(RECORDS_PATH).records,
            protocol,
            source_sha256=sha256_file(RECORDS_PATH),
            scenario_sha256=sha256_file(SCENARIO_PATH),
        )
