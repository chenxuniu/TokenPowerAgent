import json
import math
from pathlib import Path

import pytest

from tokenpoweragent.residual_calibration import (
    ResidualCalibrationError,
    build_workload_residual_profile,
)
from tokenpoweragent.schema import Candidate, EvidenceLevel
from tokenpoweragent.twin.topology import (
    CalibrationProfile,
    InferenceWorkload,
    TopologyProjector,
)
from tokenpoweragent.workload_campaign import WorkloadCampaign, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-workload-transfer-validation-v1.json"
)
HOLDOUT_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-workload-transfer-holdout-v2.json"
)


def _features(workload, reference):
    context = math.log2(
        (workload.input_tokens + 0.5 * workload.output_tokens)
        / (reference.input_tokens + 0.5 * reference.output_tokens)
    )
    concurrency = math.log2(workload.concurrency / reference.concurrency)
    return (context, concurrency, context * concurrency, concurrency**2)


def _factor(coefficients, features):
    return math.exp(sum(a * b for a, b in zip(coefficients, features)))


def _write_development_report(tmp_path: Path) -> Path:
    profile = CalibrationProfile.load(PROFILE_PATH)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    reference = profile.points[0].workload
    projector = TopologyProjector(profile)
    coefficients = {
        "throughput_tok_s": (0.03, 0.14, -0.025, 0.018),
        "avg_power_w": (0.08, -0.20, 0.02, 0.025),
        "ttft_ms": (-0.05, 0.12, 0.02, 0.01),
        "tpot_ms": (0.02, -0.13, 0.018, -0.012),
    }
    workloads = []
    for point in campaign.workloads:
        estimate = projector.predict(
            campaign.candidate(point), point.workload, EvidenceLevel.L0
        )
        features = _features(point.workload, reference)
        factors = {
            metric: _factor(values, features)
            for metric, values in coefficients.items()
        }
        energy_factor = factors["avg_power_w"] / factors["throughput_tok_s"]
        workloads.append(
            {
                "workload_id": point.point_id,
                "dataset_split": "validation",
                "workload": point.workload.to_dict(),
                "metrics": {
                    "throughput_tok_s": {
                        "prediction": estimate.metrics["throughput_tok_s"],
                        "observed_median": estimate.metrics["throughput_tok_s"]
                        * factors["throughput_tok_s"],
                    },
                    "ttft_ms": {
                        "prediction": estimate.metrics["ttft_ms"],
                        "observed_median": estimate.metrics["ttft_ms"]
                        * factors["ttft_ms"],
                    },
                    "tpot_ms": {
                        "prediction": estimate.metrics["tpot_ms"],
                        "observed_median": estimate.metrics["tpot_ms"]
                        * factors["tpot_ms"],
                    },
                    "energy_j_per_1k_output_tokens": {
                        "prediction": estimate.metrics[
                            "energy_j_per_1k_output_tokens"
                        ],
                        "observed_median": estimate.metrics[
                            "energy_j_per_1k_output_tokens"
                        ]
                        * energy_factor,
                    },
                },
                "diagnostics": {
                    "avg_power_w": {
                        "prediction": estimate.metrics["avg_power_w"],
                        "observed_median": estimate.metrics["avg_power_w"]
                        * factors["avg_power_w"],
                    }
                },
            }
        )
    report = {
        "schema_version": "1.0",
        "protocol": {
            "preregistered_validation_valid": True,
            "prediction_manifest_verified": True,
            "predictions_precede_measurements": True,
            "raw_artifact_manifest_verified": True,
            "dataset_split": "validation",
            "campaign_sha256": sha256_file(CAMPAIGN_PATH),
            "workload_count": len(campaign.workloads),
            "repeats_per_workload": 3,
            "prediction_sha256": "1" * 64,
            "measurement_sha256": "2" * 64,
            "raw_artifact_manifest_sha256": "3" * 64,
        },
        "experiment": {
            "profile_id": profile.profile_id,
            "profile_sha256": sha256_file(PROFILE_PATH),
            "server_image_id": "sha256:server",
            "client_image_id": "sha256:client",
        },
        "workloads": workloads,
    }
    path = tmp_path / "validation-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _fit(tmp_path: Path):
    report_path = _write_development_report(tmp_path)
    raw, diagnostics = build_workload_residual_profile(
        profile_path=PROFILE_PATH,
        campaign_path=CAMPAIGN_PATH,
        report_path=report_path,
        profile_id="synthetic-workload-v2",
    )
    return CalibrationProfile.from_mapping(raw), diagnostics


def test_fit_residual_profile_preserves_anchor_and_energy_identity(tmp_path) -> None:
    profile, diagnostics = _fit(tmp_path)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    reference = profile.points[0].workload
    reference_candidate = Candidate(
        candidate_id="reference",
        config={**campaign.configuration},
        required_gpus=1,
        target_nodes=1,
    )

    estimate = TopologyProjector(profile).predict(
        reference_candidate, reference, EvidenceLevel.L0
    )

    assert profile.workload_residual_model is not None
    assert estimate.decomposition["workload_residual"]["applied"] is True
    assert estimate.metrics["throughput_tok_s"] == pytest.approx(1800)
    assert estimate.metrics["avg_power_w"] == pytest.approx(460)
    assert estimate.metrics["energy_j"] == pytest.approx(
        estimate.metrics["avg_power_w"] * estimate.metrics["duration_s"]
    )
    assert diagnostics["derived_energy_j_per_1k_output_tokens"][
        "mean_absolute_percentage_error_pct"
    ] < 10


def test_residual_model_improves_training_workload_and_is_scope_gated(tmp_path) -> None:
    fitted, _ = _fit(tmp_path)
    base = CalibrationProfile.load(PROFILE_PATH)
    campaign = WorkloadCampaign.load(CAMPAIGN_PATH)
    point = campaign.workloads[0]
    base_estimate = TopologyProjector(base).predict(
        campaign.candidate(point), point.workload, EvidenceLevel.L0
    )
    fitted_estimate = TopologyProjector(fitted).predict(
        campaign.candidate(point), point.workload, EvidenceLevel.L0
    )

    assert fitted_estimate.decomposition["workload_residual"]["applied"] is True
    assert fitted_estimate.metrics["avg_power_w"] != pytest.approx(
        base_estimate.metrics["avg_power_w"]
    )

    tp2 = Candidate(
        candidate_id="tp2",
        config={
            **campaign.configuration,
            "tensor_parallel": 2,
        },
        required_gpus=2,
        target_nodes=1,
    )
    scoped_out = TopologyProjector(fitted).predict(
        tp2, point.workload, EvidenceLevel.L0
    )
    assert scoped_out.decomposition["workload_residual"]["applied"] is False
    assert "one GPU" in scoped_out.decomposition["workload_residual"]["reason"]


def test_fit_rejects_unverified_report(tmp_path) -> None:
    report_path = _write_development_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["protocol"]["raw_artifact_manifest_verified"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ResidualCalibrationError, match="verified preregistered"):
        build_workload_residual_profile(
            profile_path=PROFILE_PATH,
            campaign_path=CAMPAIGN_PATH,
            report_path=report_path,
            profile_id="should-fail",
        )


def test_final_holdout_uses_only_disjoint_validation_combinations() -> None:
    validation = WorkloadCampaign.load(CAMPAIGN_PATH)
    holdout = WorkloadCampaign.load(HOLDOUT_PATH)
    validation_pairs = {
        (point.workload.input_tokens, point.workload.concurrency)
        for point in validation.workloads
    }
    holdout_pairs = {
        (point.workload.input_tokens, point.workload.concurrency)
        for point in holdout.workloads
    }

    assert len(holdout.workloads) == 8
    assert not validation_pairs.intersection(holdout_pairs)
    assert {point.dataset_split for point in holdout.workloads} == {"holdout"}
