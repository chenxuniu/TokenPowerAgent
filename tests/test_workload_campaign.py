import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStore,
)
from tokenpoweragent.executors.serving import ServerEnvironment
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.workload_campaign import (
    WorkloadCampaign,
    WorkloadCampaignError,
    campaign_schedule,
    freeze_workload_campaign,
    run_serving_campaign,
    verify_frozen_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-workload-transfer-validation-v1.json"
)
PROFILE_PATH = (
    ROOT / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
)


def _freeze(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    summary = tmp_path / "predictions.summary.json"
    manifest = tmp_path / "freeze.sha256"
    freeze_workload_campaign(
        CAMPAIGN_PATH,
        PROFILE_PATH,
        predictions,
        summary,
        manifest,
    )
    return predictions, summary, manifest


class FakeServingExecutor:
    def __init__(self, configuration, sample_ms=100):
        self.configuration = dict(configuration)
        self.sample_ms = sample_ms
        self.calls = []

    def inspect_environment(self, candidate):
        return ServerEnvironment(
            image="vllm/vllm-openai:v0.23.0",
            image_id="sha256:server",
            command=("vllm", "serve"),
            configuration=self.configuration,
        )

    def execute(self, candidate, level, seed):
        self.calls.append((candidate.candidate_id, level, seed, dict(candidate.config)))
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=EvidenceLevel.L1,
            metrics={"throughput_tok_s": 1000.0},
            gpu_hours=0.01,
            kind=EvidenceKind.MEASURED,
            provenance={
                "dataset_split": candidate.config["dataset_split"],
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "client_image_id": "sha256:client",
                "server_image": "vllm/vllm-openai:v0.23.0",
                "server_image_id": "sha256:server",
                "server_command": ["vllm", "serve"],
                "power_limit_readback_w": candidate.config["power_limit_w"],
                "server_configuration": self.configuration,
                "workload": {
                    "input_len": candidate.config["input_len"],
                    "output_len": candidate.config["output_len"],
                    "max_concurrency": candidate.config["max_concurrency"],
                    "num_prompts": candidate.config["num_prompts"],
                    "num_warmups": candidate.config["num_warmups"],
                    "request_rate": candidate.config["request_rate"],
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )


class DriftingRecordExecutor(FakeServingExecutor):
    def execute(self, candidate, level, seed):
        record = super().execute(candidate, level, seed)
        provenance = dict(record.provenance)
        configuration = dict(provenance["server_configuration"])
        configuration["max_num_seqs"] = 128
        provenance["server_configuration"] = configuration
        return replace(record, provenance=provenance)


def _runtime():
    return {
        "image": "tokenpower-vllm-client:v0.23.0",
        "server_container": "tpa-vllm-qwen7b",
        "network": "tpa-serving-bench",
        "cache_volume": "tpa-hf-cache",
        "base_url": "http://tpa-vllm-qwen7b:8000",
        "served_model_name": "qwen2.5-7b",
    }


def _server_configuration(campaign: WorkloadCampaign):
    return {
        **campaign.configuration,
        "model_revision": campaign.model_revision,
        "tokenizer_revision": campaign.tokenizer_revision,
    }


def test_campaign_schema_and_balanced_schedule() -> None:
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    schedule = campaign_schedule(campaign.workloads, campaign.repeats)

    assert len(campaign.workloads) == 6
    assert len(schedule) == 18
    assert [item[0] for item in schedule] == list(range(18))
    assert [item[2].point_id for item in schedule[:6]] == [
        point.point_id for point in campaign.workloads
    ]
    assert schedule[6][2].point_id == campaign.workloads[2].point_id
    assert schedule[12][2].point_id == campaign.workloads[4].point_id
    assert {
        point.dataset_split for point in campaign.workloads
    } == {"validation"}
    for point in campaign.workloads:
        assert point.workload.num_requests == point.workload.concurrency * 8


def test_freeze_campaign_writes_verified_immutable_artifacts(tmp_path) -> None:
    predictions, summary, manifest = _freeze(tmp_path)
    campaign_hash, prediction_hash = verify_frozen_campaign(
        CAMPAIGN_PATH, predictions, manifest
    )
    records = EvidenceStore.read_jsonl(predictions).records
    rendered_summary = json.loads(summary.read_text(encoding="utf-8"))

    assert len(records) == 6
    assert all(record.level == EvidenceLevel.L0 for record in records)
    assert all(
        record.provenance["campaign_sha256"] == campaign_hash
        for record in records
    )
    assert len(prediction_hash) == 64
    assert rendered_summary["prediction_count"] == 6
    assert rendered_summary["level"] == "L0"
    assert rendered_summary["sandbox_backend"] == "L0-A"
    assert rendered_summary["projection_backend"] == "l0-a"

    with pytest.raises(WorkloadCampaignError, match="refusing to overwrite"):
        freeze_workload_campaign(
            CAMPAIGN_PATH,
            PROFILE_PATH,
            predictions,
            summary,
            manifest,
        )


def test_run_campaign_checkpoints_and_resumes(tmp_path) -> None:
    predictions, _, manifest = _freeze(tmp_path)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    executor = FakeServingExecutor(_server_configuration(campaign))
    output = tmp_path / "measurements.jsonl"

    first = run_serving_campaign(
        CAMPAIGN_PATH,
        predictions,
        manifest,
        output,
        executor,
        _runtime(),
    )
    second = run_serving_campaign(
        CAMPAIGN_PATH,
        predictions,
        manifest,
        output,
        executor,
        _runtime(),
        resume=True,
    )
    records = EvidenceStore.read_jsonl(output).records

    assert first["added_measurements"] == 18
    assert first["completed_measurements"] == 18
    assert second["added_measurements"] == 0
    assert second["existing_measurements"] == 18
    assert len(executor.calls) == 18
    assert len(records) == 18
    assert len(
        {
            (
                record.provenance["workload_id"],
                record.provenance["campaign_repeat"],
            )
            for record in records
        }
    ) == 18


def test_run_campaign_rejects_server_configuration_drift(tmp_path) -> None:
    predictions, _, manifest = _freeze(tmp_path)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    configuration = _server_configuration(campaign)
    configuration["max_num_seqs"] = 128
    executor = FakeServingExecutor(configuration)

    with pytest.raises(WorkloadCampaignError, match="max_num_seqs"):
        run_serving_campaign(
            CAMPAIGN_PATH,
            predictions,
            manifest,
            tmp_path / "measurements.jsonl",
            executor,
            _runtime(),
        )


def test_run_campaign_rejects_mid_campaign_record_drift(tmp_path) -> None:
    predictions, _, manifest = _freeze(tmp_path)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    executor = DriftingRecordExecutor(_server_configuration(campaign))

    with pytest.raises(WorkloadCampaignError, match="max_num_seqs"):
        run_serving_campaign(
            CAMPAIGN_PATH,
            predictions,
            manifest,
            tmp_path / "measurements.jsonl",
            executor,
            _runtime(),
        )
