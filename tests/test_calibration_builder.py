import json
from pathlib import Path

import pytest

from tokenpoweragent.calibration import (
    CalibrationBuildError,
    build_serving_calibration_profile,
)
from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.twin.topology import CalibrationProfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "configs/calibration/qwen2.5-7b-h100-profile-template.json"
)


def record(throughput: float, power_limit: float = 700.0) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id="qwen7b-serving-pl%d" % int(power_limit),
        level=EvidenceLevel.L1,
        metrics={
            "throughput_tok_s": throughput,
            "ttft_ms": 100.0 + throughput / 100.0,
            "tpot_ms": 10.0,
            "avg_power_w": 450.0,
            "j_per_output_token": 0.4,
            "j_per_total_token": 0.08,
        },
        gpu_hours=0.01,
        kind=EvidenceKind.MEASURED,
        provenance={
            "power_limit_readback_w": power_limit,
            "workload": {
                "input_len": 512,
                "output_len": 128,
                "max_concurrency": 8,
                "num_prompts": 64,
                "request_rate": "inf",
            },
            "server_command": [
                "vllm",
                "serve",
                "Qwen/Qwen2.5-7B-Instruct",
                "--revision",
                "model-revision",
                "--tokenizer-revision",
                "tokenizer-revision",
                "--dtype",
                "bfloat16",
                "--max-num-seqs",
                "256",
                "--max-num-batched-tokens",
                "8192",
                "--enable-chunked-prefill",
                "--no-enable-prefix-caching",
            ],
        },
    )


def write_records(path: Path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(item.to_dict()) for item in records) + "\n",
        encoding="utf-8",
    )


def test_build_calibration_aggregates_repeats_and_filters_power(tmp_path) -> None:
    records_path = tmp_path / "serving.jsonl"
    write_records(
        records_path,
        [record(1000), record(1200), record(1100), record(900, 490)],
    )

    raw = build_serving_calibration_profile(
        records_path=records_path,
        template_path=TEMPLATE,
        profile_id="measured-h100-test",
        power_limit_w=700,
        min_repeats=3,
    )
    profile = CalibrationProfile.from_mapping(raw)

    assert profile.profile_id == "measured-h100-test"
    assert profile.publication_eligible is False
    assert len(profile.points) == 1
    assert profile.points[0].metrics["throughput_tok_s"] == 1100
    assert raw["calibration_points"][0]["replicate_count"] == 3
    assert "sha256:" in raw["calibration_points"][0]["source_artifact"]
    assert raw["metadata"]["matching_records"] == 3
    assert len(raw["metadata"]["skipped_records"]) == 1


def test_publication_profile_rejects_insufficient_repeats(tmp_path) -> None:
    records_path = tmp_path / "serving.jsonl"
    write_records(records_path, [record(1000), record(1100)])

    with pytest.raises(CalibrationBuildError, match="lacks required repeats"):
        build_serving_calibration_profile(
            records_path=records_path,
            template_path=TEMPLATE,
            profile_id="not-ready",
            power_limit_w=700,
            min_repeats=3,
            publication_eligible=True,
        )
