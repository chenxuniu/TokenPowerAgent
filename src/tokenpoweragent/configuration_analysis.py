"""Validation and publication analysis for frozen configuration campaigns."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    sha256_file,
)
from tokenpoweragent.configuration_runner import (
    FrozenConfigurationArtifacts,
    ConfigurationRunnerError,
    verify_configuration_measurement_record,
    verify_frozen_configuration_campaign,
)
from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.schema import EvidenceLevel


class ConfigurationAnalysisError(ValueError):
    """Raised when a measured configuration corpus is not publication-safe."""


CORE_METRICS = (
    "energy_j_per_1k_output_tokens",
    "throughput_tok_s",
    "ttft_ms",
    "tpot_ms",
)

SUMMARY_METRICS = CORE_METRICS + (
    "avg_power_w",
    "measurement_window_ms",
)

RAW_ARTIFACT_KEYS = (
    "telemetry_path",
    "client_output_path",
    "server_log_path",
)


def build_configuration_campaign_report(
    campaign_path: Path,
    predictions_path: Path,
    schedule_path: Path,
    summary_path: Path,
    freeze_manifest_path: Path,
    measurements_path: Path,
    report_path: Path,
    corpus_path: Path,
    artifact_list_path: Optional[Path] = None,
    artifact_manifest_path: Optional[Path] = None,
    include_artifacts: Sequence[Path] = (),
) -> Mapping[str, Any]:
    """Validate a complete campaign and emit its oracle and replay corpus."""

    if (artifact_list_path is None) != (artifact_manifest_path is None):
        raise ConfigurationAnalysisError(
            "artifact list and artifact manifest must be requested together"
        )
    campaign = ConfigurationCampaign.load(campaign_path)
    try:
        artifacts = verify_frozen_configuration_campaign(
            campaign_path=campaign_path,
            predictions_path=predictions_path,
            schedule_path=schedule_path,
            summary_path=summary_path,
            manifest_path=freeze_manifest_path,
        )
    except ConfigurationRunnerError as exc:
        raise ConfigurationAnalysisError(str(exc)) from exc

    schedule = _read_object(schedule_path, "schedule")
    freeze_summary = _read_object(summary_path, "summary")
    predictions = EvidenceStore.read_jsonl(predictions_path).records
    measurements = EvidenceStore.read_jsonl(measurements_path).records
    actions = list(schedule.get("actions", ()))
    if len(measurements) != len(actions):
        raise ConfigurationAnalysisError(
            "measurement count %d does not match the frozen schedule count %d"
            % (len(measurements), len(actions))
        )

    point_map = {point.point_id: point for point in campaign.candidates}
    for action_index, (action, record) in enumerate(zip(actions, measurements)):
        if not isinstance(action, Mapping):
            raise ConfigurationAnalysisError("schedule action must be an object")
        if int(action.get("action_index", -1)) != action_index:
            raise ConfigurationAnalysisError(
                "schedule action indices are not contiguous"
            )
        point = point_map.get(str(action.get("candidate_id", "")))
        if point is None:
            raise ConfigurationAnalysisError(
                "schedule references an unknown candidate"
            )
        try:
            verify_configuration_measurement_record(
                campaign, point, action, record, artifacts
            )
        except ConfigurationRunnerError as exc:
            raise ConfigurationAnalysisError(
                "measurement action %d is invalid: %s" % (action_index, exc)
            ) from exc
        _assert_raw_artifact_provenance(record, action_index)

    prediction_times = [_parse_time(row.created_at) for row in predictions]
    measurement_times = [_parse_time(row.created_at) for row in measurements]
    predictions_precede_measurements = (
        bool(prediction_times)
        and bool(measurement_times)
        and max(prediction_times) < min(measurement_times)
    )
    if not predictions_precede_measurements:
        raise ConfigurationAnalysisError(
            "frozen predictions must precede every configuration measurement"
        )

    grouped = _group_measurements(measurements)
    candidate_rows = []
    for point in campaign.candidates:
        l0 = _single_prediction(predictions, point.point_id)
        l1 = grouped.get((point.point_id, EvidenceLevel.L1), ())
        l4 = grouped.get((point.point_id, EvidenceLevel.L4), ())
        if len(l1) != campaign.repeats or len(l4) != campaign.repeats:
            raise ConfigurationAnalysisError(
                "%s does not have %d repeats at both L1 and L4"
                % (point.point_id, campaign.repeats)
            )
        candidate_rows.append(
            {
                "candidate_id": point.point_id,
                "role": point.role,
                "configuration": dict(point.configuration),
                "l0": {
                    "metrics": {
                        metric: float(l0.metrics[metric])
                        for metric in CORE_METRICS
                    },
                    "relative_uncertainty": l0.metrics.get(
                        "relative_uncertainty"
                    ),
                },
                "l1": _summarize_level(l1),
                "l4": _summarize_level(l4),
            }
        )

    l4_feasible = [
        row
        for row in candidate_rows
        if _slo_accepts(row["l4"]["metrics"], campaign.slo)
    ]
    l4_pareto_ids = pareto_front(
        [
            (row["candidate_id"], row["l4"]["metrics"])
            for row in l4_feasible
        ],
        campaign.objectives,
    )
    if not l4_pareto_ids:
        raise ConfigurationAnalysisError(
            "the measured corpus has no SLO-feasible L4 Pareto candidate"
        )

    l0_validation = _fidelity_validation(
        candidate_rows, "l0", campaign, l4_pareto_ids
    )
    l1_validation = _fidelity_validation(
        candidate_rows, "l1", campaign, l4_pareto_ids
    )
    restart_counts = Counter(
        str(record.provenance.get("server_restart_reason", "missing"))
        for record in measurements
    )
    level_counts = Counter(record.level.name for record in measurements)
    status_counts = Counter(record.status.value for record in measurements)
    action_indices = [
        int(record.provenance["campaign_action_index"])
        for record in measurements
    ]
    repeats = Counter(
        (
            record.candidate_id,
            record.level.name,
            int(record.provenance["campaign_repeat"]),
        )
        for record in measurements
    )
    unique_repeat_actions = all(count == 1 for count in repeats.values())

    raw_artifact = {
        "verified": False,
        "entry_count": 0,
        "sha256": None,
    }
    if artifact_list_path is not None and artifact_manifest_path is not None:
        artifact_paths = _raw_artifact_paths(
            freeze_manifest_path,
            measurements_path,
            measurements,
            include_artifacts,
        )
        _write_artifact_manifests(
            artifact_paths, artifact_list_path, artifact_manifest_path
        )
        raw_artifact = {
            "verified": True,
            "entry_count": len(artifact_paths),
            "sha256": sha256_file(artifact_manifest_path),
            "list_path": str(artifact_list_path),
            "manifest_path": str(artifact_manifest_path),
        }

    corpus = EvidenceStore([*predictions, *measurements])
    Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
    corpus.write_jsonl(corpus_path)
    successful = status_counts.get(EvidenceStatus.SUCCEEDED.value, 0)
    publication_ready = (
        len(measurements) == len(actions)
        and successful == len(measurements)
        and predictions_precede_measurements
        and unique_repeat_actions
        and (artifact_manifest_path is None or raw_artifact["verified"])
    )
    warnings = []
    if artifact_manifest_path is None:
        warnings.append(
            "Raw artifact hashes were not requested; publication archival is incomplete."
        )

    expert_id = str(freeze_summary["expert_baseline_id"])
    expert = next(
        row for row in candidate_rows if row["candidate_id"] == expert_id
    )
    oracle_rows = [
        row for row in candidate_rows if row["candidate_id"] in l4_pareto_ids
    ]
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "campaign_id": campaign.campaign_id,
        "protocol": {
            "campaign_complete": len(measurements) == len(actions),
            "freeze_manifest_verified": True,
            "predictions_precede_measurements": predictions_precede_measurements,
            "candidate_count": len(campaign.candidates),
            "measurement_count": len(measurements),
            "expected_measurement_count": len(actions),
            "level_counts": dict(sorted(level_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "restart_reason_counts": dict(sorted(restart_counts.items())),
            "action_indices_contiguous": action_indices == list(range(len(actions))),
            "unique_candidate_level_repeat_actions": unique_repeat_actions,
            "campaign_sha256": artifacts.campaign_sha256,
            "prediction_sha256": artifacts.prediction_sha256,
            "schedule_sha256": artifacts.schedule_sha256,
            "summary_sha256": artifacts.summary_sha256,
            "freeze_manifest_sha256": artifacts.manifest_sha256,
            "measurement_sha256": sha256_file(measurements_path),
            "replay_corpus_sha256": sha256_file(corpus_path),
            "raw_artifacts": raw_artifact,
        },
        "experiment": {
            "model": campaign.model_id,
            "model_revision": campaign.model_revision,
            "hardware": dict(campaign.hardware),
            "workload": campaign.workload_dict(
                campaign.target_workload.num_requests
            ),
            "objectives": dict(campaign.objectives),
            "slo": dict(campaign.slo),
            "power_limit_w": campaign.power_limit_w,
            "repeats": campaign.repeats,
            "probe_requests": campaign.probe_requests,
            "target_requests": campaign.target_workload.num_requests,
        },
        "oracle": {
            "semantics": "median successful L4 metrics per candidate",
            "slo_feasible_ids": [row["candidate_id"] for row in l4_feasible],
            "pareto_ids": l4_pareto_ids,
            "expert_baseline_id": expert_id,
            "expert_metrics": dict(expert["l4"]["metrics"]),
            "pareto_metrics": {
                row["candidate_id"]: dict(row["l4"]["metrics"])
                for row in oracle_rows
            },
        },
        "fidelity_validation": {
            "l0_vs_l4": l0_validation,
            "l1_vs_l4": l1_validation,
        },
        "candidates": candidate_rows,
        "cost": {
            "actual_gpu_hours_total": sum(
                record.gpu_hours for record in measurements
            ),
            "actual_gpu_hours_by_level": {
                level.name: sum(
                    record.gpu_hours
                    for record in measurements
                    if record.level == level
                )
                for level in (EvidenceLevel.L1, EvidenceLevel.L4)
            },
            "declared_gpu_hours_per_level": {
                level.name: campaign.level_cost_gpu_hours[level]
                for level in (EvidenceLevel.L0, EvidenceLevel.L1, EvidenceLevel.L4)
            },
        },
        "summary": {
            "publication_ready": publication_ready,
            "warnings": warnings,
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationAnalysisError("cannot read %s: %s" % (label, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationAnalysisError("%s must be a JSON object" % label)
    return raw


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationAnalysisError(
            "evidence has an invalid created_at timestamp"
        ) from exc


def _assert_raw_artifact_provenance(
    record: EvidenceRecord, action_index: int
) -> None:
    for key in RAW_ARTIFACT_KEYS:
        value = record.provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationAnalysisError(
                "measurement action %d lacks %s" % (action_index, key)
            )


def _group_measurements(
    measurements: Sequence[EvidenceRecord],
) -> Mapping[tuple[str, EvidenceLevel], Sequence[EvidenceRecord]]:
    grouped: Dict[tuple[str, EvidenceLevel], List[EvidenceRecord]] = defaultdict(list)
    for record in measurements:
        grouped[(record.candidate_id, record.level)].append(record)
    return grouped


def _single_prediction(
    predictions: Sequence[EvidenceRecord], candidate_id: str
) -> EvidenceRecord:
    matches = [row for row in predictions if row.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ConfigurationAnalysisError(
            "%s must have exactly one frozen L0 prediction" % candidate_id
        )
    record = matches[0]
    missing = [metric for metric in CORE_METRICS if metric not in record.metrics]
    if missing:
        raise ConfigurationAnalysisError(
            "L0 prediction for %s lacks %s"
            % (candidate_id, ", ".join(missing))
        )
    return record


def _summarize_level(records: Sequence[EvidenceRecord]) -> Mapping[str, Any]:
    successful = [
        record for record in records if record.status == EvidenceStatus.SUCCEEDED
    ]
    if len(successful) != len(records):
        raise ConfigurationAnalysisError(
            "configuration corpus contains a failed measurement"
        )
    metrics: Dict[str, float] = {}
    dispersion: Dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        if any(metric not in record.metrics for record in successful):
            if metric in CORE_METRICS:
                raise ConfigurationAnalysisError(
                    "successful configuration measurement lacks %s" % metric
                )
            continue
        values = [float(record.metrics[metric]) for record in successful]
        center = float(median(values))
        metrics[metric] = center
        dispersion[metric + "_mad"] = float(
            median(abs(value - center) for value in values)
        )
    return {
        "repeat_count": len(records),
        "metrics": metrics,
        "dispersion": dispersion,
        "gpu_hours": {
            "total": sum(record.gpu_hours for record in records),
            "median": float(median(record.gpu_hours for record in records)),
        },
    }


def _slo_accepts(metrics: Mapping[str, float], slo: Mapping[str, float]) -> bool:
    for metric, limit in slo.items():
        if metric == "min_goodput_req_s":
            if float(metrics.get("goodput_req_s", float("-inf"))) < limit:
                return False
        elif float(metrics.get(metric, float("inf"))) > limit:
            return False
    return True


def _fidelity_validation(
    candidate_rows: Sequence[Mapping[str, Any]],
    fidelity: str,
    campaign: ConfigurationCampaign,
    oracle_ids: Sequence[str],
) -> Mapping[str, Any]:
    metric_rows = {}
    for metric in CORE_METRICS:
        predicted = [
            float(row[fidelity]["metrics"][metric]) for row in candidate_rows
        ]
        observed = [
            float(row["l4"]["metrics"][metric]) for row in candidate_rows
        ]
        errors = [
            100.0 * abs(left - right) / max(abs(right), 1e-12)
            for left, right in zip(predicted, observed)
        ]
        ordering = _pairwise_ordering(predicted, observed)
        metric_rows[metric] = {
            "mean_absolute_percentage_error_pct": sum(errors) / len(errors),
            "median_absolute_percentage_error_pct": float(median(errors)),
            "maximum_absolute_percentage_error_pct": max(errors),
            "spearman_rank_correlation": _spearman(predicted, observed),
            "pairwise_order_accuracy": ordering["accuracy"],
            "pairwise_concordant_pairs": ordering["concordant"],
            "pairwise_discordant_pairs": ordering["discordant"],
            "pairwise_comparable_pairs": ordering["comparable"],
        }

    feasible = [
        row
        for row in candidate_rows
        if _slo_accepts(row[fidelity]["metrics"], campaign.slo)
    ]
    frontier = pareto_front(
        [
            (row["candidate_id"], row[fidelity]["metrics"])
            for row in feasible
        ],
        campaign.objectives,
    )
    predicted = set(frontier)
    oracle = set(oracle_ids)
    overlap = predicted & oracle
    return {
        "metrics": metric_rows,
        "slo_feasible_ids": [row["candidate_id"] for row in feasible],
        "predicted_pareto_ids": frontier,
        "oracle_pareto_recall": len(overlap) / len(oracle),
        "oracle_pareto_precision": (
            len(overlap) / len(predicted) if predicted else 0.0
        ),
        "pareto_jaccard": len(overlap) / len(predicted | oracle),
    }


def _average_ranks(values: Sequence[float]) -> Sequence[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    ranked_left = _average_ranks(left)
    ranked_right = _average_ranks(right)
    left_mean = sum(ranked_left) / len(ranked_left)
    right_mean = sum(ranked_right) / len(ranked_right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(ranked_left, ranked_right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in ranked_left)
        * sum((value - right_mean) ** 2 for value in ranked_right)
    )
    return numerator / denominator if denominator else None


def _pairwise_ordering(
    predicted: Sequence[float], observed: Sequence[float]
) -> Mapping[str, Any]:
    concordant = 0
    discordant = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_delta = predicted[left] - predicted[right]
            observed_delta = observed[left] - observed[right]
            if predicted_delta == 0 or observed_delta == 0:
                continue
            if predicted_delta * observed_delta > 0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "accuracy": concordant / comparable if comparable else None,
        "concordant": concordant,
        "discordant": discordant,
        "comparable": comparable,
    }


def _raw_artifact_paths(
    freeze_manifest_path: Path,
    measurements_path: Path,
    measurements: Sequence[EvidenceRecord],
    include_artifacts: Sequence[Path],
) -> Sequence[Path]:
    paths = {Path(freeze_manifest_path), Path(measurements_path)}
    for line in Path(freeze_manifest_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ConfigurationAnalysisError("freeze manifest has an invalid line")
        paths.add(Path(parts[1].lstrip("* ")))
    paths.update(Path(path) for path in include_artifacts)
    for record in measurements:
        for key in RAW_ARTIFACT_KEYS:
            paths.add(Path(str(record.provenance[key])))
    ordered = sorted(paths, key=lambda path: str(path))
    missing = [str(path) for path in ordered if not path.is_file()]
    if missing:
        raise ConfigurationAnalysisError(
            "raw artifact files are missing: %s" % ", ".join(missing[:5])
        )
    return ordered


def _write_artifact_manifests(
    paths: Sequence[Path], list_path: Path, manifest_path: Path
) -> None:
    Path(list_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(list_path).write_text(
        "".join("%s\n" % path for path in paths), encoding="utf-8"
    )
    Path(manifest_path).write_text(
        "".join("%s  %s\n" % (sha256_file(path), path) for path in paths),
        encoding="utf-8",
    )
