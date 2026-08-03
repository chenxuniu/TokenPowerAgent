import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.validation import (
    HoldoutValidationError,
    build_holdout_validation_report,
)


def _prediction(created_at: str) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id="tp1-pp1-seq256-bt8192",
        level=EvidenceLevel.L0,
        metrics={
            "energy_j_per_1k_output_tokens": 440.0,
            "energy_j_per_1k_output_tokens_lower": 250.0,
            "energy_j_per_1k_output_tokens_upper": 640.0,
            "throughput_tok_s": 1400.0,
            "throughput_tok_s_lower": 800.0,
            "throughput_tok_s_upper": 2000.0,
            "ttft_ms": 310.0,
            "ttft_ms_lower": 170.0,
            "ttft_ms_upper": 450.0,
            "tpot_ms": 11.0,
            "tpot_ms_lower": 6.0,
            "tpot_ms_upper": 16.0,
            "avg_power_w": 615.0,
            "duration_s": 11.85,
        },
        gpu_hours=0.0,
        kind=EvidenceKind.SIMULATED,
        created_at=created_at,
        provenance={
            "profile_id": "profile-v1",
            "profile_sha256": "profile-hash",
            "scenario_sha256": "scenario-hash",
            "profile_publication_eligible": False,
            "uncertainty_calibrated": False,
            "model": {"id": "Qwen/Qwen2.5-7B-Instruct"},
            "decomposition": {
                "workload": {
                    "input_tokens": 1024,
                    "output_tokens": 128,
                    "concurrency": 16,
                    "num_requests": 128,
                    "request_rate_req_s": "inf",
                },
                "configuration": {
                    "engine": "vllm",
                    "precision": "bfloat16",
                    "tensor_parallel": 1,
                    "pipeline_parallel": 1,
                    "data_parallel": 1,
                    "max_num_seqs": 256,
                    "max_num_batched_tokens": 8192,
                    "chunked_prefill": True,
                    "kv_cache_dtype": "bfloat16",
                },
            },
        },
    )


def _measurement(created_at: str, throughput: float, joules_per_token: float):
    configuration = {
        "engine": "vllm",
        "precision": "bfloat16",
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "data_parallel": 1,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "chunked_prefill": True,
        "kv_cache_dtype": "bfloat16",
    }
    return EvidenceRecord(
        candidate_id="qwen7b-serving-pl700",
        level=EvidenceLevel.L1,
        metrics={
            "j_per_output_token": joules_per_token,
            "throughput_tok_s": throughput,
            "ttft_ms": 370.0,
            "tpot_ms": 8.8,
            "avg_power_w": 549.0,
            "benchmark_duration_s": 9.53,
        },
        gpu_hours=0.01,
        kind=EvidenceKind.MEASURED,
        created_at=created_at,
        provenance={
            "dataset_split": "holdout",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "served_model_name": "qwen2.5-7b",
            "client_image_id": "sha256:client",
            "server_image_id": "sha256:image",
            "power_limit_readback_w": 700.0,
            "server_configuration": configuration,
            "workload": {
                "input_len": 1024,
                "output_len": 128,
                "max_concurrency": 16,
                "num_prompts": 128,
                "request_rate": "inf",
            },
        },
    )


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
        encoding="utf-8",
    )


def _fixture_files(tmp_path: Path):
    start = datetime(2026, 8, 3, tzinfo=timezone.utc)
    prediction_path = tmp_path / "prediction.jsonl"
    measurements_path = tmp_path / "holdout.jsonl"
    manifest_path = tmp_path / "freeze.sha256"
    _write_jsonl(prediction_path, [_prediction(start.isoformat())])
    _write_jsonl(
        measurements_path,
        [
            _measurement((start + timedelta(minutes=index + 1)).isoformat(), value, energy)
            for index, (value, energy) in enumerate(
                [(1710.0, 0.321), (1735.0, 0.317), (1719.0, 0.324)]
            )
        ],
    )
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        "%s  %s\n" % (digest, prediction_path), encoding="utf-8"
    )
    return prediction_path, measurements_path, manifest_path


def test_build_holdout_validation_report(tmp_path) -> None:
    prediction_path, measurements_path, manifest_path = _fixture_files(tmp_path)

    report = build_holdout_validation_report(
        prediction_path, measurements_path, manifest_path
    )

    assert report["protocol"]["blind_holdout_valid"] is True
    assert report["protocol"]["repeat_count"] == 3
    assert report["metrics"]["throughput_tok_s"]["observed_median"] == 1719.0
    assert (
        report["metrics"]["energy_j_per_1k_output_tokens"]["observed_median"]
        == 321.0
    )
    assert report["summary"]["interval_covered_metric_count"] == 4
    assert report["summary"]["publication_ready"] is False


def test_holdout_validation_rejects_non_holdout_records(tmp_path) -> None:
    prediction_path, measurements_path, manifest_path = _fixture_files(tmp_path)
    raw = json.loads(measurements_path.read_text(encoding="utf-8").splitlines()[0])
    raw["provenance"]["dataset_split"] = "calibration"
    measurements_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(HoldoutValidationError, match="dataset_split=holdout"):
        build_holdout_validation_report(
            prediction_path,
            measurements_path,
            manifest_path,
            min_repeats=1,
        )


def test_holdout_validation_rejects_modified_prediction(tmp_path) -> None:
    prediction_path, measurements_path, manifest_path = _fixture_files(tmp_path)
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(HoldoutValidationError, match="freeze manifest"):
        build_holdout_validation_report(
            prediction_path, measurements_path, manifest_path
        )
