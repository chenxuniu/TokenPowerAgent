import hashlib
import json

import pytest

from tokenpoweragent.scope_analysis import (
    ScopeConfirmationAnalysisError,
    build_scope_confirmation_decision,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _metric(absolute_percentage_error_pct: float):
    return {"absolute_percentage_error_pct": absolute_percentage_error_pct}


def _workload(workload_id: str, concurrency: int, ttft_ape: float):
    return {
        "workload_id": workload_id,
        "workload": {
            "input_tokens": 384 * concurrency,
            "concurrency": concurrency,
        },
        "metrics": {
            "energy_j_per_1k_output_tokens": _metric(5.0 + concurrency),
            "throughput_tok_s": _metric(8.0 + concurrency),
            "tpot_ms": _metric(6.0 + concurrency),
            "ttft_ms": _metric(ttft_ape),
        },
    }


def _artifacts(tmp_path):
    campaign_path = tmp_path / "campaign.json"
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "artifacts.sha256"
    manifest_path.write_text("sealed evidence\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    _write_json(
        campaign_path,
        {
            "campaign_id": "scope-confirmation-v3",
            "preregistered_analysis": {
                "model_update": "none",
                "primary_endpoint": {
                    "metric": "energy_j_per_1k_output_tokens",
                    "population": "all_workloads",
                    "success_criterion": "mape_pct_le_15",
                },
                "scope_decision": {
                    "supported_stratum": "concurrency_eq_4",
                    "sparse_stratum": "concurrency_le_2",
                    "latency_metric": "ttft_ms",
                    "acceptable_supported_mape_pct": 20,
                    "abstain_if_sparse_mape_exceeds_pct": 20,
                },
            },
        },
    )
    campaign_sha256 = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
    _write_json(
        report_path,
        {
            "protocol": {
                "campaign_id": "scope-confirmation-v3",
                "campaign_sha256": campaign_sha256,
                "preregistered_holdout_valid": True,
                "predictions_precede_measurements": True,
                "raw_artifact_manifest_verified": True,
                "raw_artifact_manifest_sha256": manifest_sha256,
                "workload_count": 3,
                "measurement_count": 9,
            },
            "aggregate_metrics": {
                "energy_j_per_1k_output_tokens": {
                    "mean_absolute_percentage_error_pct": 7.0
                }
            },
            "workloads": [
                _workload("k1", 1, 80.0),
                _workload("k2", 2, 40.0),
                _workload("k4", 4, 10.0),
            ],
        },
    )
    return campaign_path, report_path, manifest_path


def test_scope_confirmation_applies_frozen_decision_rules(tmp_path) -> None:
    campaign_path, report_path, manifest_path = _artifacts(tmp_path)

    decision = build_scope_confirmation_decision(
        campaign_path, report_path, manifest_path
    )

    assert decision["primary_endpoint"]["passed"] is True
    assert decision["latency_scope"]["supported_stratum"]["observed_mape_pct"] == 10
    assert decision["latency_scope"]["supported_stratum"]["passed"] is True
    assert decision["latency_scope"]["sparse_stratum"]["observed_mape_pct"] == 60
    assert decision["latency_scope"]["sparse_stratum"]["exceeds_threshold"] is True
    assert decision["latency_scope"]["decision"] == (
        "support_latency_at_concurrency_ge_4_abstain_below_4"
    )
    assert [row["concurrency"] for row in decision["by_concurrency"]] == [1, 2, 4]


def test_scope_confirmation_rejects_changed_artifact_manifest(tmp_path) -> None:
    campaign_path, report_path, manifest_path = _artifacts(tmp_path)
    manifest_path.write_text("changed evidence\n", encoding="utf-8")

    with pytest.raises(
        ScopeConfirmationAnalysisError,
        match="artifact manifest hash does not match",
    ):
        build_scope_confirmation_decision(
            campaign_path, report_path, manifest_path
        )
