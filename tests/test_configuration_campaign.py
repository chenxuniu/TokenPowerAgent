import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    ConfigurationCampaignError,
    freeze_configuration_campaign,
    sha256_file,
)
from tokenpoweragent.evidence import EvidenceStore
from tokenpoweragent.schema import EvidenceLevel, Scenario


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-config-search-v1.json"
)
PROFILE_PATH = (
    ROOT
    / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
)


def _outputs(tmp_path: Path):
    return {
        "scenario_path": tmp_path / "scenario.json",
        "predictions_path": tmp_path / "predictions.jsonl",
        "schedule_path": tmp_path / "schedule.json",
        "summary_path": tmp_path / "summary.json",
        "manifest_path": tmp_path / "freeze.sha256",
    }


def test_configuration_campaign_locks_controlled_search_space() -> None:
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)

    assert len(campaign.candidates) == 12
    assert campaign.target_workload.input_tokens == 2048
    assert campaign.target_workload.output_tokens == 128
    assert campaign.target_workload.concurrency == 32
    assert campaign.target_workload.num_requests == 256
    assert campaign.probe_requests == 64
    assert campaign.repeats == 3
    assert campaign.objectives == {
        "energy_j_per_1k_output_tokens": "min",
        "throughput_tok_s": "max",
    }
    assert campaign.slo == {"ttft_ms": 1600.0, "tpot_ms": 20.0}
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L0] == 0
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L1] == 0.01
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L4] == 0.02
    assert campaign.runtime["server_image"].endswith(
        "6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
    )

    baselines = [point for point in campaign.candidates if point.role == "expert-baseline"]
    assert [point.point_id for point in baselines] == [
        "expert-seq256-bt8192-chunk"
    ]
    assert all(point.configuration["prefix_caching"] is False for point in campaign.candidates)
    assert all(point.configuration["tensor_parallel"] == 1 for point in campaign.candidates)
    assert all(point.configuration["pipeline_parallel"] == 1 for point in campaign.candidates)
    assert all(point.configuration["power_limit_w"] == 700 for point in campaign.candidates)


def test_freeze_writes_balanced_hash_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    paths = _outputs(tmp_path)

    summary = freeze_configuration_campaign(
        campaign_path=CAMPAIGN_PATH.relative_to(ROOT),
        calibration_path=PROFILE_PATH.relative_to(ROOT),
        **paths,
    )
    predictions = EvidenceStore.read_jsonl(paths["predictions_path"]).records
    scenario = Scenario.load(paths["scenario_path"])
    schedule = json.loads(paths["schedule_path"].read_text(encoding="utf-8"))

    assert summary["candidate_count"] == 12
    assert summary["l0_prediction_count"] == 12
    assert summary["measurement_count"] == 72
    assert len(predictions) == 12
    assert all(record.level == EvidenceLevel.L0 for record in predictions)
    assert len(scenario.candidates) == 12
    assert scenario.budget.gpu_hours == 0.12

    actions = schedule["actions"]
    assert [action["action_index"] for action in actions] == list(range(72))
    assert Counter(
        (action["candidate_id"], action["level"]) for action in actions
    ) == Counter(
        {
            (point.point_id, level): 3
            for point in ConfigurationCampaign.load(CAMPAIGN_PATH).candidates
            for level in ("L1", "L4")
        }
    )

    blocks = defaultdict(list)
    for action in actions:
        blocks[action["block_index"]].append(action)
    assert len(blocks) == 36
    for block in blocks.values():
        assert len(block) == 2
        assert block[0]["candidate_id"] == block[1]["candidate_id"]
        assert block[0]["repeat"] == block[1]["repeat"]
        assert {action["level"] for action in block} == {"L1", "L4"}
        assert [action["restart_server"] for action in block] == [True, False]
        assert {
            action["level"]: action["num_prompts"] for action in block
        } == {"L1": 64, "L4": 256}

    manifest_lines = paths["manifest_path"].read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 6
    for line in manifest_lines:
        expected, artifact = line.split("  ", 1)
        assert sha256_file(Path(artifact)) == expected

    with pytest.raises(ConfigurationCampaignError, match="refusing to overwrite"):
        freeze_configuration_campaign(
            campaign_path=CAMPAIGN_PATH.relative_to(ROOT),
            calibration_path=PROFILE_PATH.relative_to(ROOT),
            **paths,
        )


def test_configuration_campaign_rejects_prefix_cache_bias(tmp_path: Path) -> None:
    raw = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    raw["candidates"][0]["serving_configuration"]["prefix_caching"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationCampaignError, match="disable prefix caching"):
        ConfigurationCampaign.load(path)


def test_configuration_campaign_rejects_unpinned_server_image(tmp_path: Path) -> None:
    raw = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    raw["runtime"]["server_image"] = "vllm/vllm-openai:v0.23.0"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationCampaignError, match="immutable sha256"):
        ConfigurationCampaign.load(path)
