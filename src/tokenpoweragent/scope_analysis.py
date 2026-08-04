"""Pre-registered scope analysis for workload-confirmation campaigns."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence


class ScopeConfirmationAnalysisError(ValueError):
    """Raised when a scope-confirmation report violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeConfirmationAnalysisError(
            "cannot read %s JSON: %s" % (label, exc)
        ) from exc
    if not isinstance(value, Mapping):
        raise ScopeConfirmationAnalysisError("%s root must be an object" % label)
    return value


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ScopeConfirmationAnalysisError("%s must be numeric" % label) from exc
    if number <= 0:
        raise ScopeConfirmationAnalysisError("%s must be positive" % label)
    return number


def _parse_primary_threshold(value: Any) -> float:
    match = re.fullmatch(r"mape_pct_le_([0-9]+(?:\.[0-9]+)?)", str(value))
    if match is None:
        raise ScopeConfirmationAnalysisError(
            "primary success criterion must use mape_pct_le_N"
        )
    return _positive_number(match.group(1), "primary MAPE threshold")


def _parse_concurrency(value: Any, operator: str, label: str) -> int:
    match = re.fullmatch(r"concurrency_%s_([0-9]+)" % operator, str(value))
    if match is None:
        raise ScopeConfirmationAnalysisError(
            "%s must use concurrency_%s_N" % (label, operator)
        )
    concurrency = int(match.group(1))
    if concurrency <= 0:
        raise ScopeConfirmationAnalysisError("%s must be positive" % label)
    return concurrency


def _metric_apes(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> Sequence[float]:
    values = []
    for row in rows:
        try:
            value = float(
                row["metrics"][metric]["absolute_percentage_error_pct"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScopeConfirmationAnalysisError(
                "workload %s lacks %s absolute percentage error"
                % (row.get("workload_id", "<unknown>"), metric)
            ) from exc
        values.append(value)
    if not values:
        raise ScopeConfirmationAnalysisError(
            "scope stratum has no workloads for metric %s" % metric
        )
    return values


def _metric_mape(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(mean(_metric_apes(rows, metric)))


def build_scope_confirmation_decision(
    campaign_path: Path,
    report_path: Path,
    artifact_manifest_path: Path,
) -> Dict[str, Any]:
    """Apply a campaign's frozen energy and latency-scope decision rules."""

    campaign = _load_object(campaign_path, "campaign")
    report = _load_object(report_path, "report")
    preregistered = campaign.get("preregistered_analysis")
    if not isinstance(preregistered, Mapping):
        raise ScopeConfirmationAnalysisError(
            "campaign lacks preregistered_analysis"
        )
    primary = preregistered.get("primary_endpoint")
    scope = preregistered.get("scope_decision")
    if not isinstance(primary, Mapping) or not isinstance(scope, Mapping):
        raise ScopeConfirmationAnalysisError(
            "campaign lacks primary_endpoint or scope_decision"
        )

    protocol = report.get("protocol")
    rows_raw = report.get("workloads")
    aggregate = report.get("aggregate_metrics")
    if not isinstance(protocol, Mapping):
        raise ScopeConfirmationAnalysisError("report lacks protocol")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ScopeConfirmationAnalysisError("report lacks workload rows")
    if not isinstance(aggregate, Mapping):
        raise ScopeConfirmationAnalysisError("report lacks aggregate metrics")
    rows = [row for row in rows_raw if isinstance(row, Mapping)]
    if len(rows) != len(rows_raw):
        raise ScopeConfirmationAnalysisError("report has a malformed workload row")

    campaign_id = str(campaign.get("campaign_id", ""))
    if protocol.get("campaign_id") != campaign_id:
        raise ScopeConfirmationAnalysisError(
            "report campaign_id does not match the frozen campaign"
        )
    campaign_sha256 = _sha256(campaign_path)
    if protocol.get("campaign_sha256") != campaign_sha256:
        raise ScopeConfirmationAnalysisError(
            "campaign hash does not match the validated report"
        )
    if protocol.get("preregistered_holdout_valid") is not True:
        raise ScopeConfirmationAnalysisError("report is not a valid holdout")
    if protocol.get("predictions_precede_measurements") is not True:
        raise ScopeConfirmationAnalysisError(
            "report predictions do not precede measurements"
        )
    if protocol.get("raw_artifact_manifest_verified") is not True:
        raise ScopeConfirmationAnalysisError(
            "report raw artifact manifest is not verified"
        )

    manifest_sha256 = _sha256(artifact_manifest_path)
    if protocol.get("raw_artifact_manifest_sha256") != manifest_sha256:
        raise ScopeConfirmationAnalysisError(
            "artifact manifest hash does not match the validated report"
        )
    if int(protocol.get("workload_count", -1)) != len(rows):
        raise ScopeConfirmationAnalysisError(
            "report workload_count does not match its workload rows"
        )

    primary_metric = str(primary.get("metric", ""))
    if primary.get("population") != "all_workloads":
        raise ScopeConfirmationAnalysisError(
            "scope analysis supports the all_workloads primary population"
        )
    primary_threshold = _parse_primary_threshold(
        primary.get("success_criterion")
    )
    try:
        primary_mape = float(
            aggregate[primary_metric]["mean_absolute_percentage_error_pct"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScopeConfirmationAnalysisError(
            "report lacks the pre-registered primary metric"
        ) from exc

    supported_concurrency = _parse_concurrency(
        scope.get("supported_stratum"), "eq", "supported stratum"
    )
    sparse_max_concurrency = _parse_concurrency(
        scope.get("sparse_stratum"), "le", "sparse stratum"
    )
    latency_metric = str(scope.get("latency_metric", ""))
    supported_threshold = _positive_number(
        scope.get("acceptable_supported_mape_pct"),
        "supported-stratum MAPE threshold",
    )
    sparse_threshold = _positive_number(
        scope.get("abstain_if_sparse_mape_exceeds_pct"),
        "sparse-stratum abstention threshold",
    )

    supported_rows = [
        row
        for row in rows
        if int(row["workload"]["concurrency"]) == supported_concurrency
    ]
    sparse_rows = [
        row
        for row in rows
        if int(row["workload"]["concurrency"]) <= sparse_max_concurrency
    ]
    supported_apes = _metric_apes(supported_rows, latency_metric)
    sparse_apes = _metric_apes(sparse_rows, latency_metric)
    supported_mape = float(mean(supported_apes))
    sparse_mape = float(mean(sparse_apes))

    if supported_mape > supported_threshold:
        decision = "broaden_latency_abstention_supported_stratum_failed"
    elif sparse_mape > sparse_threshold:
        decision = "support_latency_at_concurrency_ge_%d_abstain_below_%d" % (
            supported_concurrency,
            supported_concurrency,
        )
    else:
        decision = "latency_supported_across_tested_concurrency_range"

    by_concurrency = []
    for concurrency in sorted(
        {int(row["workload"]["concurrency"]) for row in rows}
    ):
        stratum = [
            row
            for row in rows
            if int(row["workload"]["concurrency"]) == concurrency
        ]
        by_concurrency.append(
            {
                "concurrency": concurrency,
                "workload_count": len(stratum),
                "energy_mape_pct": _metric_mape(
                    stratum, "energy_j_per_1k_output_tokens"
                ),
                "throughput_mape_pct": _metric_mape(
                    stratum, "throughput_tok_s"
                ),
                "tpot_mape_pct": _metric_mape(stratum, "tpot_ms"),
                "ttft_mape_pct": _metric_mape(stratum, "ttft_ms"),
            }
        )

    per_workload = []
    for row in rows:
        per_workload.append(
            {
                "workload_id": row["workload_id"],
                "context_tokens": int(row["workload"]["input_tokens"]),
                "concurrency": int(row["workload"]["concurrency"]),
                "energy_ape_pct": _metric_apes(
                    [row], "energy_j_per_1k_output_tokens"
                )[0],
                "throughput_ape_pct": _metric_apes(
                    [row], "throughput_tok_s"
                )[0],
                "tpot_ape_pct": _metric_apes([row], "tpot_ms")[0],
                "ttft_ape_pct": _metric_apes([row], "ttft_ms")[0],
            }
        )

    return {
        "schema_version": "1.0",
        "evidence": {
            "campaign_id": campaign_id,
            "campaign_sha256": campaign_sha256,
            "report_sha256": _sha256(report_path),
            "raw_artifact_manifest_sha256": manifest_sha256,
            "workload_count": len(rows),
            "measurement_count": int(protocol["measurement_count"]),
            "holdout_valid": True,
            "predictions_precede_measurements": True,
            "model_update_contract": preregistered.get("model_update"),
        },
        "primary_endpoint": {
            "metric": primary_metric,
            "population": "all_workloads",
            "threshold_mape_pct": primary_threshold,
            "observed_mape_pct": primary_mape,
            "passed": primary_mape <= primary_threshold,
        },
        "latency_scope": {
            "metric": latency_metric,
            "supported_stratum": {
                "concurrency": supported_concurrency,
                "workload_count": len(supported_rows),
                "threshold_mape_pct": supported_threshold,
                "observed_mape_pct": supported_mape,
                "maximum_ape_pct": max(supported_apes),
                "passed": supported_mape <= supported_threshold,
            },
            "sparse_stratum": {
                "maximum_concurrency": sparse_max_concurrency,
                "workload_count": len(sparse_rows),
                "abstention_threshold_mape_pct": sparse_threshold,
                "observed_mape_pct": sparse_mape,
                "maximum_ape_pct": max(sparse_apes),
                "exceeds_threshold": sparse_mape > sparse_threshold,
            },
            "decision": decision,
        },
        "by_concurrency": by_concurrency,
        "per_workload": per_workload,
    }
