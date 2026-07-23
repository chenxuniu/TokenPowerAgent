from pathlib import Path

import pytest

from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.schema import EvidenceLevel, Scenario


ROOT = Path(__file__).resolve().parents[1]


def test_demo_scenario_is_versioned_and_verifiable() -> None:
    scenario = Scenario.load(ROOT / "configs/scenarios/replay_demo.json")
    assert scenario.schema_version == "1.0"
    assert scenario.budget.verify_top_k == 2
    assert EvidenceLevel.L4 in scenario.available_levels
    assert scenario.candidate("cfg-fast").target_nodes == 2


def test_only_l4_can_be_verified() -> None:
    with pytest.raises(ValueError, match="only L4"):
        EvidenceRecord(
            candidate_id="bad",
            level=EvidenceLevel.L2,
            metrics={"energy": 1.0},
            gpu_hours=0.1,
            kind=EvidenceKind.VERIFIED,
        )


def test_pareto_front_respects_mixed_directions() -> None:
    rows = [
        ("efficient", {"energy": 1.0, "goodput": 2.0}),
        ("balanced", {"energy": 1.2, "goodput": 3.0}),
        ("dominated", {"energy": 1.3, "goodput": 2.5}),
    ]
    assert pareto_front(rows, {"energy": "min", "goodput": "max"}) == [
        "efficient",
        "balanced",
    ]
