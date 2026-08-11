from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStore,
)
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.validation import (
    WorkloadCampaignValidationError,
    build_workload_campaign_validation_report,
)
from tokenpoweragent.workload_campaign import (
    WorkloadCampaign,
    campaign_schedule,
    freeze_workload_campaign,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-workload-transfer-validation-v1.json"
)
HOLDOUT_CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-workload-transfer-holdout-v2.json"
)
PROFILE_PATH = (
    ROOT / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
)


def _write_manifest(path: Path, files) -> None:
    path.write_text(
        "".join("%s  %s\n" % (sha256_file(item), item) for item in files),
        encoding="utf-8",
    )


def _fixture_files(tmp_path: Path, campaign_path: Path = CAMPAIGN_PATH):
    predictions_path = tmp_path / "predictions.jsonl"
    prediction_summary_path = tmp_path / "predictions.summary.json"
    freeze_manifest_path = tmp_path / "freeze.sha256"
    measurements_path = tmp_path / "measurements.jsonl"
    artifact_manifest_path = tmp_path / "artifacts.sha256"
    freeze_workload_campaign(
        campaign_path=campaign_path,
        calibration_path=PROFILE_PATH,
        output_path=predictions_path,
        summary_path=prediction_summary_path,
        manifest_path=freeze_manifest_path,
    )

    campaign = WorkloadCampaign.load(campaign_path)
    predictions = EvidenceStore.read_jsonl(predictions_path).records
    prediction_map = {record.candidate_id: record for record in predictions}
    prediction_hash = sha256_file(predictions_path)
    campaign_hash = sha256_file(campaign_path)
    latest_prediction = max(
        datetime.fromisoformat(record.created_at) for record in predictions
    )
    throughput_factors = {
        point.point_id: 0.82 + 0.06 * index
        for index, point in enumerate(campaign.workloads)
    }
    power_factors = {
        point.point_id: 0.90 + 0.03 * index
        for index, point in enumerate(campaign.workloads)
    }
    records = []
    raw_artifacts = []
    configuration = {
        **campaign.configuration,
        "model_revision": campaign.model_revision,
        "tokenizer_revision": campaign.tokenizer_revision,
    }
    for campaign_index, repeat_index, point in campaign_schedule(
        campaign.workloads, campaign.repeats
    ):
        prediction = prediction_map[point.point_id]
        jitter = (0.99, 1.0, 1.01)[repeat_index]
        throughput = (
            prediction.metrics["throughput_tok_s"]
            * throughput_factors[point.point_id]
            * jitter
        )
        output_tokens = point.workload.num_requests * point.workload.output_tokens
        duration_s = output_tokens / throughput
        avg_power_w = (
            prediction.metrics["avg_power_w"]
            * power_factors[point.point_id]
            * jitter
        )
        energy_j = avg_power_w * duration_s
        telemetry_path = tmp_path / (
            "%s-seed%d.dcgm.txt" % (point.point_id, repeat_index)
        )
        client_path = tmp_path / (
            "%s-seed%d.client.txt" % (point.point_id, repeat_index)
        )
        telemetry_path.write_text("telemetry\n", encoding="utf-8")
        client_path.write_text("client output\n", encoding="utf-8")
        raw_artifacts.extend((telemetry_path, client_path))
        records.append(
            EvidenceRecord(
                candidate_id=point.point_id,
                level=EvidenceLevel.L1,
                metrics={
                    "energy_j": energy_j,
                    "j_per_output_token": energy_j / output_tokens,
                    "throughput_tok_s": throughput,
                    "ttft_ms": prediction.metrics["ttft_ms"]
                    * (1.08 - 0.02 * repeat_index),
                    "tpot_ms": prediction.metrics["tpot_ms"]
                    * (0.94 + 0.01 * repeat_index),
                    "avg_power_w": avg_power_w,
                    "benchmark_duration_s": duration_s,
                },
                gpu_hours=duration_s / 3600.0,
                kind=EvidenceKind.MEASURED,
                created_at=(
                    latest_prediction + timedelta(minutes=campaign_index + 1)
                ).isoformat(),
                provenance={
                    "dataset_split": point.dataset_split,
                    "model": campaign.model_id,
                    "served_model_name": "qwen2.5-7b",
                    "client_image_id": "sha256:client",
                    "server_image": "vllm/vllm-openai:v0.23.0",
                    "server_image_id": "sha256:server",
                    "server_command": ["vllm", "serve"],
                    "server_configuration": configuration,
                    "power_limit_readback_w": campaign.power_limit_w,
                    "workload": {
                        "input_len": point.workload.input_tokens,
                        "output_len": point.workload.output_tokens,
                        "num_prompts": point.workload.num_requests,
                        "num_warmups": campaign.num_warmups,
                        "request_rate": "inf",
                        "max_concurrency": point.workload.concurrency,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                    "telemetry_path": str(telemetry_path),
                    "client_output_path": str(client_path),
                    "campaign_id": campaign.campaign_id,
                    "campaign_sha256": campaign_hash,
                    "workload_id": point.point_id,
                    "campaign_repeat": repeat_index,
                    "campaign_schedule_index": campaign_index,
                    "frozen_prediction_sha256": prediction_hash,
                },
            )
        )
    EvidenceStore(records).write_jsonl(measurements_path)
    artifact_files = [
        campaign_path,
        predictions_path,
        freeze_manifest_path,
        measurements_path,
        *raw_artifacts,
    ]
    _write_manifest(artifact_manifest_path, artifact_files)
    return {
        "campaign": campaign_path,
        "predictions": predictions_path,
        "measurements": measurements_path,
        "freeze_manifest": freeze_manifest_path,
        "artifact_manifest": artifact_manifest_path,
    }


def _report(paths):
    return build_workload_campaign_validation_report(
        campaign_path=paths["campaign"],
        predictions_path=paths["predictions"],
        measurements_path=paths["measurements"],
        freeze_manifest_path=paths["freeze_manifest"],
        artifact_manifest_path=paths["artifact_manifest"],
    )


def test_build_workload_campaign_validation_report(tmp_path) -> None:
    paths = _fixture_files(tmp_path)

    report = _report(paths)

    assert report["protocol"]["preregistered_validation_valid"] is True
    assert report["protocol"]["workload_count"] == 6
    assert report["protocol"]["measurement_count"] == 18
    assert report["protocol"]["raw_artifact_manifest_verified"] is True
    assert report["protocol"]["raw_artifact_manifest_entry_count"] == 40
    assert len(report["workloads"]) == 6
    assert report["summary"]["metric_workload_pair_count"] == 24
    assert report["summary"]["analysis_ready"] is True
    assert report["summary"]["publication_ready"] is False
    energy = report["aggregate_metrics"]["energy_j_per_1k_output_tokens"]
    assert energy["interval_evaluated_workloads"] == 6
    assert energy["pairwise_comparable_pairs"] == 15
    assert energy["spearman_rank_correlation"] is not None


def test_campaign_validation_rejects_missing_measurement(tmp_path) -> None:
    paths = _fixture_files(tmp_path)
    lines = paths["measurements"].read_text(encoding="utf-8").splitlines()
    paths["measurements"].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(
        WorkloadCampaignValidationError, match="exactly 18 were pre-registered"
    ):
        _report(paths)


def test_campaign_validation_rejects_schedule_reordering(tmp_path) -> None:
    paths = _fixture_files(tmp_path)
    lines = paths["measurements"].read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    paths["measurements"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(
        WorkloadCampaignValidationError, match="frozen schedule"
    ):
        _report(paths)


def test_campaign_validation_rejects_incomplete_artifact_manifest(tmp_path) -> None:
    paths = _fixture_files(tmp_path)
    lines = paths["artifact_manifest"].read_text(encoding="utf-8").splitlines()
    paths["artifact_manifest"].write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        WorkloadCampaignValidationError, match="omits required evidence"
    ):
        _report(paths)


def test_campaign_validation_accepts_preregistered_final_holdout(tmp_path) -> None:
    paths = _fixture_files(tmp_path, HOLDOUT_CAMPAIGN_PATH)

    report = _report(paths)

    assert report["protocol"]["dataset_split"] == "holdout"
    assert report["protocol"]["preregistered_campaign_valid"] is True
    assert report["protocol"]["preregistered_holdout_valid"] is True
    assert report["protocol"]["preregistered_validation_valid"] is False
    assert report["summary"]["final_holdout_completed"] is True
    assert report["summary"]["final_holdout_required"] is False
