"""Blind holdout validation for frozen sandbox predictions."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import median, stdev
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
)
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.workload_campaign import (
    WorkloadCampaign,
    WorkloadCampaignError,
    campaign_schedule,
    verify_frozen_campaign,
)


class HoldoutValidationError(ValueError):
    """Raised when prediction and holdout evidence are not comparable."""


class WorkloadCampaignValidationError(ValueError):
    """Raised when a workload-transfer validation campaign is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HoldoutValidationError(
            "evidence has an invalid created_at timestamp"
        ) from exc
    if parsed.utcoffset() is None:
        raise HoldoutValidationError("evidence created_at must include a timezone")
    return parsed


def _manifest_prediction_hash(manifest_path: Path, prediction_path: Path) -> str:
    matches = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        digest, raw_path = parts
        listed_path = raw_path.lstrip("* ")
        if (
            listed_path == str(prediction_path)
            or Path(listed_path).name == prediction_path.name
        ):
            matches.append(digest.lower())
    if len(matches) != 1:
        raise HoldoutValidationError(
            "freeze manifest must contain exactly one entry for the prediction JSONL"
        )
    actual = _sha256(prediction_path)
    if matches[0] != actual:
        raise HoldoutValidationError("prediction JSONL no longer matches freeze manifest")
    return actual


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HoldoutValidationError("missing %s" % name)
    return value


def _validate_comparability(
    prediction: EvidenceRecord, measurements: Sequence[EvidenceRecord]
) -> None:
    decomposition = _require_mapping(
        prediction.provenance.get("decomposition"), "prediction decomposition"
    )
    predicted_workload = _require_mapping(
        decomposition.get("workload"), "predicted workload"
    )
    predicted_config = _require_mapping(
        decomposition.get("configuration"), "predicted configuration"
    )

    first = measurements[0]
    measured_workload = _require_mapping(
        first.provenance.get("workload"), "measurement workload"
    )
    measured_config = _require_mapping(
        first.provenance.get("server_configuration"),
        "measurement server configuration",
    )
    expected_workload = {
        "input_tokens": measured_workload.get("input_len"),
        "output_tokens": measured_workload.get("output_len"),
        "concurrency": measured_workload.get("max_concurrency"),
        "num_requests": measured_workload.get("num_prompts"),
        "request_rate_req_s": measured_workload.get("request_rate"),
    }
    if dict(predicted_workload) != expected_workload:
        raise HoldoutValidationError("prediction and measurement workloads differ")

    for key, predicted_value in predicted_config.items():
        if key not in measured_config or measured_config[key] != predicted_value:
            raise HoldoutValidationError(
                "prediction and measurement configurations differ at %s" % key
            )

    stable_fields = (
        "model",
        "served_model_name",
        "client_image_id",
        "server_image_id",
        "power_limit_readback_w",
    )
    for record in measurements[1:]:
        if record.provenance.get("workload") != measured_workload:
            raise HoldoutValidationError("holdout repeats use different workloads")
        if record.provenance.get("server_configuration") != measured_config:
            raise HoldoutValidationError(
                "holdout repeats use different server configurations"
            )
        for field in stable_fields:
            if record.provenance.get(field) != first.provenance.get(field):
                raise HoldoutValidationError(
                    "holdout repeats differ at provenance.%s" % field
                )

    predicted_model = _require_mapping(
        prediction.provenance.get("model"), "prediction model"
    )
    if predicted_model.get("id") != first.provenance.get("model"):
        raise HoldoutValidationError("prediction and measurement models differ")
    if (
        predicted_model.get("revision") is not None
        and predicted_model.get("revision") != measured_config.get("model_revision")
    ):
        raise HoldoutValidationError(
            "prediction and measurement model revisions differ"
        )


def _metric_summary(
    prediction: EvidenceRecord,
    measurements: Sequence[EvidenceRecord],
    prediction_key: str,
    observed: Callable[[EvidenceRecord], float],
    observed_semantics: str,
    interval_aliases: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    values = [float(observed(record)) for record in measurements]
    if any(not math.isfinite(value) for value in values):
        raise HoldoutValidationError("holdout metric %s is not finite" % prediction_key)
    observed_median = float(median(values))
    if observed_median == 0:
        raise HoldoutValidationError("holdout metric %s has zero median" % prediction_key)
    observed_mad = float(median(abs(value - observed_median) for value in values))
    observed_mean = sum(values) / len(values)
    sample_sd = float(stdev(values)) if len(values) > 1 else 0.0
    predicted = float(prediction.metrics[prediction_key])
    signed_error_pct = (predicted - observed_median) / observed_median * 100.0
    interval_source = None
    lower = None
    upper = None
    for interval_key in (prediction_key,) + interval_aliases:
        candidate_lower = prediction.metrics.get(interval_key + "_lower")
        candidate_upper = prediction.metrics.get(interval_key + "_upper")
        if candidate_lower is not None and candidate_upper is not None:
            interval_source = interval_key
            lower = candidate_lower
            upper = candidate_upper
            break
    covered = None
    interval = None
    if lower is not None and upper is not None:
        interval = [float(lower), float(upper)]
        covered = interval[0] <= observed_median <= interval[1]
    return {
        "prediction": predicted,
        "observations": values,
        "observed_median": observed_median,
        "observed_mad": observed_mad,
        "observed_sample_sd": sample_sd,
        "observed_cv_pct": sample_sd / observed_mean * 100.0,
        "signed_prediction_error_pct": signed_error_pct,
        "absolute_percentage_error_pct": abs(signed_error_pct),
        "prediction_interval": interval,
        "prediction_interval_source": interval_source,
        "interval_covers_observed_median": covered,
        "observed_semantics": observed_semantics,
    }


def build_holdout_validation_report(
    predictions_path: Path,
    measurements_path: Path,
    freeze_manifest_path: Path,
    min_repeats: int = 3,
) -> Dict[str, Any]:
    """Validate one frozen prediction against repeated blind measurements."""

    if min_repeats < 1:
        raise HoldoutValidationError("min_repeats must be positive")
    predictions = EvidenceStore.read_jsonl(predictions_path).records
    measurements = EvidenceStore.read_jsonl(measurements_path).records
    if len(predictions) != 1:
        raise HoldoutValidationError("validation requires exactly one prediction")
    if len(measurements) < min_repeats:
        raise HoldoutValidationError(
            "holdout has %d repeats; at least %d are required"
            % (len(measurements), min_repeats)
        )

    prediction = predictions[0]
    if prediction.status != EvidenceStatus.SUCCEEDED:
        raise HoldoutValidationError("prediction did not succeed")
    legacy_l2_prediction = prediction.level == EvidenceLevel.L2
    if prediction.level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
        raise HoldoutValidationError("prediction must be CPU-side L0 evidence")
    if prediction.kind not in {
        EvidenceKind.SIMULATED,
        EvidenceKind.INTERPOLATED,
        EvidenceKind.EXTRAPOLATED,
    }:
        raise HoldoutValidationError("prediction has an invalid evidence kind")
    if legacy_l2_prediction and not (
        prediction.kind == EvidenceKind.EXTRAPOLATED
        and prediction.provenance.get("executor") == "topology-sandbox"
    ):
        raise HoldoutValidationError(
            "L2 predictions are accepted only as legacy topology-sandbox artifacts"
        )

    for record in measurements:
        if record.status != EvidenceStatus.SUCCEEDED:
            raise HoldoutValidationError("holdout contains a failed measurement")
        if record.level not in {EvidenceLevel.L1, EvidenceLevel.L4}:
            raise HoldoutValidationError("holdout must contain L1 or L4 evidence")
        if record.provenance.get("dataset_split") != "holdout":
            raise HoldoutValidationError(
                "every measurement must declare dataset_split=holdout"
            )

    prediction_time = _timestamp(prediction.created_at)
    measurement_times = [_timestamp(record.created_at) for record in measurements]
    first_measurement_time = min(measurement_times)
    prediction_precedes_measurements = prediction_time < first_measurement_time
    if not prediction_precedes_measurements:
        raise HoldoutValidationError("prediction was not frozen before measurements")

    prediction_hash = _manifest_prediction_hash(
        freeze_manifest_path, predictions_path
    )
    _validate_comparability(prediction, measurements)

    metrics = {
        "energy_j_per_1k_output_tokens": _metric_summary(
            prediction,
            measurements,
            "energy_j_per_1k_output_tokens",
            lambda record: record.metrics["j_per_output_token"] * 1000.0,
            "GPU joules per 1,000 output tokens",
            interval_aliases=("energy_j_per_1k_tokens",),
        ),
        "throughput_tok_s": _metric_summary(
            prediction,
            measurements,
            "throughput_tok_s",
            lambda record: record.metrics["throughput_tok_s"],
            "output-token throughput",
        ),
        "ttft_ms": _metric_summary(
            prediction,
            measurements,
            "ttft_ms",
            lambda record: record.metrics["ttft_ms"],
            "P95 TTFT per repeat, then median across repeats",
        ),
        "tpot_ms": _metric_summary(
            prediction,
            measurements,
            "tpot_ms",
            lambda record: record.metrics["tpot_ms"],
            "P95 TPOT per repeat, then median across repeats",
        ),
    }
    diagnostics = {
        "avg_power_w": _metric_summary(
            prediction,
            measurements,
            "avg_power_w",
            lambda record: record.metrics["avg_power_w"],
            "DCGM active-window average GPU power",
        ),
        "duration_s": _metric_summary(
            prediction,
            measurements,
            "duration_s",
            lambda record: record.metrics["benchmark_duration_s"],
            "vLLM main benchmark duration",
        ),
    }
    covered = [
        metric["interval_covers_observed_median"] for metric in metrics.values()
    ]
    absolute_errors = [
        metric["absolute_percentage_error_pct"] for metric in metrics.values()
    ]
    profile_publication_eligible = bool(
        prediction.provenance.get("profile_publication_eligible", False)
    )
    uncertainty_calibrated = bool(
        prediction.provenance.get("uncertainty_calibrated", False)
    )
    warnings = [
        "One holdout workload cannot establish statistical interval coverage."
    ]
    if not profile_publication_eligible:
        warnings.append("The calibration profile is not publication eligible.")
    if not uncertainty_calibrated:
        warnings.append(
            "Prediction intervals have not been calibrated on validation data."
        )
    if legacy_l2_prediction:
        warnings.append(
            "Legacy CPU extrapolation used level=L2; new CPU predictions use "
            "level=L0 with sandbox_backend=L0-T."
        )

    first = measurements[0]
    return {
        "schema_version": "1.0",
        "protocol": {
            "blind_holdout_valid": True,
            "dataset_split": "holdout",
            "prediction_created_at": prediction.created_at,
            "first_measurement_created_at": first_measurement_time.isoformat(),
            "prediction_precedes_measurements": prediction_precedes_measurements,
            "prediction_manifest_verified": True,
            "prediction_sha256": prediction_hash,
            "measurement_sha256": _sha256(measurements_path),
            "freeze_manifest_sha256": _sha256(freeze_manifest_path),
            "repeat_count": len(measurements),
            "legacy_l2_prediction_compatibility": legacy_l2_prediction,
        },
        "prediction": {
            "candidate_id": prediction.candidate_id,
            "level": prediction.level.name,
            "kind": prediction.kind.value,
            "scenario_sha256": prediction.provenance.get("scenario_sha256"),
            "profile_id": prediction.provenance.get("profile_id"),
            "profile_sha256": prediction.provenance.get("profile_sha256"),
            "profile_publication_eligible": profile_publication_eligible,
            "uncertainty_calibrated": uncertainty_calibrated,
        },
        "measurement": {
            "candidate_id": first.candidate_id,
            "level": first.level.name,
            "server_image_id": first.provenance.get("server_image_id"),
            "server_configuration": first.provenance.get("server_configuration"),
            "workload": first.provenance.get("workload"),
            "power_limit_w": first.provenance.get("power_limit_readback_w"),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "summary": {
            "metric_count": len(metrics),
            "interval_covered_metric_count": sum(value is True for value in covered),
            "observed_median_interval_coverage": sum(
                value is True for value in covered
            )
            / len(covered),
            "mean_absolute_percentage_error_pct": sum(absolute_errors)
            / len(absolute_errors),
            "maximum_absolute_percentage_error_pct": max(absolute_errors),
            "publication_ready": (
                profile_publication_eligible and uncertainty_calibrated
            ),
            "warnings": warnings,
        },
    }


_CAMPAIGN_METRICS = {
    "energy_j_per_1k_output_tokens": {
        "direction": "min",
        "prediction_key": "energy_j_per_1k_output_tokens",
        "observed": lambda record: record.metrics["j_per_output_token"] * 1000.0,
        "semantics": "GPU joules per 1,000 output tokens",
        "interval_aliases": ("energy_j_per_1k_tokens",),
    },
    "throughput_tok_s": {
        "direction": "max",
        "prediction_key": "throughput_tok_s",
        "observed": lambda record: record.metrics["throughput_tok_s"],
        "semantics": "output-token throughput",
        "interval_aliases": (),
    },
    "ttft_ms": {
        "direction": "min",
        "prediction_key": "ttft_ms",
        "observed": lambda record: record.metrics["ttft_ms"],
        "semantics": "P95 TTFT per repeat, then median across repeats",
        "interval_aliases": (),
    },
    "tpot_ms": {
        "direction": "min",
        "prediction_key": "tpot_ms",
        "observed": lambda record: record.metrics["tpot_ms"],
        "semantics": "P95 TPOT per repeat, then median across repeats",
        "interval_aliases": (),
    },
}

_CAMPAIGN_DIAGNOSTICS = {
    "avg_power_w": {
        "prediction_key": "avg_power_w",
        "observed": lambda record: record.metrics["avg_power_w"],
        "semantics": "DCGM active-window average GPU power",
    },
    "duration_s": {
        "prediction_key": "duration_s",
        "observed": lambda record: record.metrics["benchmark_duration_s"],
        "semantics": "vLLM main benchmark duration",
    },
}


def _average_ranks(values: Sequence[float]) -> Tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return tuple(ranks)


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _pairwise_ordering(
    predicted: Sequence[float], observed: Sequence[float]
) -> Mapping[str, Any]:
    concordant = 0
    discordant = 0
    skipped_ties = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_delta = predicted[left] - predicted[right]
            observed_delta = observed[left] - observed[right]
            if predicted_delta == 0 or observed_delta == 0:
                skipped_ties += 1
            elif predicted_delta * observed_delta > 0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "accuracy": concordant / comparable if comparable else None,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "comparable_pairs": comparable,
        "skipped_ties": skipped_ties,
    }


def _best_workload_id(
    workload_ids: Sequence[str], values: Sequence[float], direction: str
) -> str:
    selector = min if direction == "min" else max
    target = selector(values)
    return workload_ids[values.index(target)]


def _aggregate_campaign_metric(
    workload_rows: Sequence[Mapping[str, Any]], metric_name: str, direction: str
) -> Mapping[str, Any]:
    workload_ids = [str(row["workload_id"]) for row in workload_rows]
    summaries = [row["metrics"][metric_name] for row in workload_rows]
    predicted = [float(summary["prediction"]) for summary in summaries]
    observed = [float(summary["observed_median"]) for summary in summaries]
    absolute_errors = [
        float(summary["absolute_percentage_error_pct"]) for summary in summaries
    ]
    signed_errors = [
        float(summary["signed_prediction_error_pct"]) for summary in summaries
    ]
    cvs = [float(summary["observed_cv_pct"]) for summary in summaries]
    covered = [
        summary["interval_covers_observed_median"]
        for summary in summaries
        if summary["interval_covers_observed_median"] is not None
    ]
    ordering = _pairwise_ordering(predicted, observed)
    predicted_best = _best_workload_id(workload_ids, predicted, direction)
    observed_best = _best_workload_id(workload_ids, observed, direction)
    return {
        "direction": direction,
        "workload_count": len(workload_rows),
        "mean_absolute_percentage_error_pct": sum(absolute_errors)
        / len(absolute_errors),
        "median_absolute_percentage_error_pct": float(median(absolute_errors)),
        "maximum_absolute_percentage_error_pct": max(absolute_errors),
        "root_mean_squared_percentage_error_pct": math.sqrt(
            sum(value**2 for value in signed_errors) / len(signed_errors)
        ),
        "mean_signed_prediction_error_pct": sum(signed_errors)
        / len(signed_errors),
        "spearman_rank_correlation": _spearman(predicted, observed),
        "pairwise_order_accuracy": ordering["accuracy"],
        "pairwise_concordant_pairs": ordering["concordant_pairs"],
        "pairwise_discordant_pairs": ordering["discordant_pairs"],
        "pairwise_comparable_pairs": ordering["comparable_pairs"],
        "pairwise_skipped_ties": ordering["skipped_ties"],
        "predicted_best_workload_id": predicted_best,
        "observed_best_workload_id": observed_best,
        "best_workload_match": predicted_best == observed_best,
        "interval_covered_workloads": sum(value is True for value in covered),
        "interval_evaluated_workloads": len(covered),
        "observed_median_interval_coverage": (
            sum(value is True for value in covered) / len(covered)
            if covered
            else None
        ),
        "median_repeat_cv_pct": float(median(cvs)),
        "maximum_repeat_cv_pct": max(cvs),
    }


def _verify_artifact_manifest(
    manifest_path: Path, required_paths: Sequence[Path]
) -> Mapping[str, Any]:
    entries: Dict[Path, str] = {}
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise WorkloadCampaignValidationError(
                "artifact manifest contains an invalid line"
            )
        listed = Path(parts[1].lstrip("* ")).resolve()
        if listed in entries:
            raise WorkloadCampaignValidationError(
                "artifact manifest contains a duplicate path"
            )
        if not listed.is_file():
            raise WorkloadCampaignValidationError(
                "artifact manifest path does not exist: %s" % listed
            )
        actual = _sha256(listed)
        if actual != parts[0].lower():
            raise WorkloadCampaignValidationError(
                "artifact hash mismatch: %s" % listed
            )
        entries[listed] = actual

    required = {Path(path).resolve() for path in required_paths}
    missing = sorted(str(path) for path in required if path not in entries)
    if missing:
        raise WorkloadCampaignValidationError(
            "artifact manifest omits required evidence: %s" % ", ".join(missing)
        )
    return {
        "verified": True,
        "entry_count": len(entries),
        "sha256": _sha256(manifest_path),
    }


def _validate_campaign_measurements(
    campaign: WorkloadCampaign,
    campaign_hash: str,
    prediction_hash: str,
    measurements: Sequence[EvidenceRecord],
) -> Mapping[str, Tuple[EvidenceRecord, ...]]:
    schedule = campaign_schedule(campaign.workloads, campaign.repeats)
    if len(measurements) != len(schedule):
        raise WorkloadCampaignValidationError(
            "campaign has %d measurements; exactly %d were pre-registered"
            % (len(measurements), len(schedule))
        )
    first = measurements[0]
    server_image_id = str(first.provenance.get("server_image_id", ""))
    client_image_id = str(first.provenance.get("client_image_id", ""))
    if not server_image_id or not client_image_id:
        raise WorkloadCampaignValidationError(
            "measurements must identify server and client images"
        )

    grouped: Dict[str, list[EvidenceRecord]] = {
        point.point_id: [] for point in campaign.workloads
    }
    for file_index, (expected_index, expected_repeat, point) in enumerate(schedule):
        record = measurements[file_index]
        if record.status != EvidenceStatus.SUCCEEDED:
            raise WorkloadCampaignValidationError(
                "campaign contains a failed measurement"
            )
        if record.level not in {EvidenceLevel.L1, EvidenceLevel.L4}:
            raise WorkloadCampaignValidationError(
                "campaign measurements must be L1 or L4 evidence"
            )
        if record.kind not in {EvidenceKind.MEASURED, EvidenceKind.VERIFIED}:
            raise WorkloadCampaignValidationError(
                "campaign measurement has an invalid evidence kind"
            )
        provenance = record.provenance
        if record.candidate_id != point.point_id:
            raise WorkloadCampaignValidationError(
                "measurement file order differs from the frozen schedule"
            )
        if provenance.get("campaign_id") != campaign.campaign_id:
            raise WorkloadCampaignValidationError("measurement campaign_id drifted")
        if provenance.get("campaign_sha256") != campaign_hash:
            raise WorkloadCampaignValidationError("measurement campaign hash drifted")
        if provenance.get("frozen_prediction_sha256") != prediction_hash:
            raise WorkloadCampaignValidationError(
                "measurement frozen prediction hash drifted"
            )
        try:
            observed_repeat = int(provenance["campaign_repeat"])
            observed_index = int(provenance["campaign_schedule_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkloadCampaignValidationError(
                "measurement lacks campaign schedule annotations"
            ) from exc
        if observed_repeat != expected_repeat or observed_index != expected_index:
            raise WorkloadCampaignValidationError(
                "measurement annotations differ from the frozen schedule"
            )
        try:
            campaign.assert_measurement_record(
                point,
                record,
                server_image_id=server_image_id,
                client_image_id=client_image_id,
            )
        except WorkloadCampaignError as exc:
            raise WorkloadCampaignValidationError(str(exc)) from exc
        grouped[point.point_id].append(record)

    return {key: tuple(value) for key, value in grouped.items()}


def build_workload_campaign_validation_report(
    campaign_path: Path,
    predictions_path: Path,
    measurements_path: Path,
    freeze_manifest_path: Path,
    artifact_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate and summarize a frozen workload-transfer campaign."""

    try:
        campaign = WorkloadCampaign.load(campaign_path)
        campaign_hash, prediction_hash = verify_frozen_campaign(
            campaign_path, predictions_path, freeze_manifest_path
        )
    except WorkloadCampaignError as exc:
        raise WorkloadCampaignValidationError(str(exc)) from exc
    dataset_splits = {point.dataset_split for point in campaign.workloads}
    if len(dataset_splits) != 1 or next(iter(dataset_splits)) not in {
        "validation",
        "holdout",
    }:
        raise WorkloadCampaignValidationError(
            "workload-transfer analysis requires one validation or holdout split"
        )
    dataset_split = next(iter(dataset_splits))

    predictions = EvidenceStore.read_jsonl(predictions_path).records
    measurements = EvidenceStore.read_jsonl(measurements_path).records
    prediction_map = {record.candidate_id: record for record in predictions}
    if len(prediction_map) != len(predictions):
        raise WorkloadCampaignValidationError(
            "frozen predictions contain duplicate workload ids"
        )
    grouped = _validate_campaign_measurements(
        campaign, campaign_hash, prediction_hash, measurements
    )

    try:
        prediction_times = [_timestamp(record.created_at) for record in predictions]
        measurement_times = [_timestamp(record.created_at) for record in measurements]
    except HoldoutValidationError as exc:
        raise WorkloadCampaignValidationError(str(exc)) from exc
    last_prediction_time = max(prediction_times)
    first_measurement_time = min(measurement_times)
    if last_prediction_time >= first_measurement_time:
        raise WorkloadCampaignValidationError(
            "campaign predictions were not frozen before measurement"
        )

    workload_rows = []
    for point in campaign.workloads:
        prediction = prediction_map[point.point_id]
        repeats = grouped[point.point_id]
        try:
            _validate_comparability(prediction, repeats)
            metrics = {
                name: _metric_summary(
                    prediction,
                    repeats,
                    str(specification["prediction_key"]),
                    specification["observed"],
                    str(specification["semantics"]),
                    interval_aliases=specification["interval_aliases"],
                )
                for name, specification in _CAMPAIGN_METRICS.items()
            }
            diagnostics = {
                name: _metric_summary(
                    prediction,
                    repeats,
                    str(specification["prediction_key"]),
                    specification["observed"],
                    str(specification["semantics"]),
                )
                for name, specification in _CAMPAIGN_DIAGNOSTICS.items()
            }
        except (HoldoutValidationError, KeyError, TypeError, ValueError) as exc:
            raise WorkloadCampaignValidationError(
                "cannot compare workload %s: %s" % (point.point_id, exc)
            ) from exc
        workload_rows.append(
            {
                "workload_id": point.point_id,
                "dataset_split": point.dataset_split,
                "workload": point.workload.to_dict(),
                "repeat_count": len(repeats),
                "prediction_created_at": prediction.created_at,
                "first_measurement_created_at": min(
                    _timestamp(record.created_at) for record in repeats
                ).isoformat(),
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
        )

    raw_paths = []
    for record in measurements:
        for key in ("telemetry_path", "client_output_path"):
            value = record.provenance.get(key)
            if not isinstance(value, str) or not value:
                raise WorkloadCampaignValidationError(
                    "measurement lacks provenance.%s" % key
                )
            raw_paths.append(Path(value))
    artifact_verification = {
        "verified": False,
        "entry_count": 0,
        "sha256": None,
    }
    if artifact_manifest_path is not None:
        artifact_verification = _verify_artifact_manifest(
            artifact_manifest_path,
            (
                campaign_path,
                predictions_path,
                freeze_manifest_path,
                measurements_path,
                *raw_paths,
            ),
        )

    aggregate = {
        name: _aggregate_campaign_metric(
            workload_rows, name, str(specification["direction"])
        )
        for name, specification in _CAMPAIGN_METRICS.items()
    }
    all_absolute_errors = [
        float(row["metrics"][name]["absolute_percentage_error_pct"])
        for row in workload_rows
        for name in _CAMPAIGN_METRICS
    ]
    covered = [
        row["metrics"][name]["interval_covers_observed_median"]
        for row in workload_rows
        for name in _CAMPAIGN_METRICS
    ]
    profile_publication_eligible = all(
        bool(record.provenance.get("profile_publication_eligible", False))
        for record in predictions
    )
    uncertainty_calibrated = all(
        bool(record.provenance.get("uncertainty_calibrated", False))
        for record in predictions
    )
    if dataset_split == "validation":
        warnings = [
            "Validation workloads may calibrate the sandbox and cannot be reused "
            "as final holdout evidence.",
            "A separate frozen final holdout campaign is required for publication claims.",
        ]
    else:
        warnings = [
            "Final holdout workloads must not be used to refit the reported model.",
            "Any post-holdout model change requires a newly frozen disjoint holdout.",
        ]
    if artifact_manifest_path is None:
        warnings.append("A sealed raw-artifact manifest was not supplied.")
    if not profile_publication_eligible:
        warnings.append("The source calibration profile is not publication eligible.")
    if not uncertainty_calibrated:
        warnings.append("Prediction intervals were not calibrated before validation.")

    interval_evaluated = sum(value is not None for value in covered)
    first_prediction = predictions[0]
    first_measurement = measurements[0]
    return {
        "schema_version": "1.0",
        "protocol": {
            "preregistered_campaign_valid": True,
            "preregistered_validation_valid": dataset_split == "validation",
            "preregistered_holdout_valid": dataset_split == "holdout",
            "dataset_split": dataset_split,
            "campaign_id": campaign.campaign_id,
            "campaign_sha256": campaign_hash,
            "prediction_sha256": prediction_hash,
            "measurement_sha256": _sha256(measurements_path),
            "freeze_manifest_sha256": _sha256(freeze_manifest_path),
            "prediction_manifest_verified": True,
            "last_prediction_created_at": last_prediction_time.isoformat(),
            "first_measurement_created_at": first_measurement_time.isoformat(),
            "predictions_precede_measurements": True,
            "schedule": campaign.schedule,
            "workload_count": len(campaign.workloads),
            "repeats_per_workload": campaign.repeats,
            "measurement_count": len(measurements),
            "raw_artifact_manifest_verified": artifact_verification["verified"],
            "raw_artifact_manifest_entry_count": artifact_verification["entry_count"],
            "raw_artifact_manifest_sha256": artifact_verification["sha256"],
        },
        "experiment": {
            "model": campaign.model_id,
            "model_revision": campaign.model_revision,
            "server_image_id": first_measurement.provenance.get("server_image_id"),
            "client_image_id": first_measurement.provenance.get("client_image_id"),
            "serving_configuration": dict(campaign.configuration),
            "power_limit_w": campaign.power_limit_w,
            "profile_id": first_prediction.provenance.get("profile_id"),
            "profile_sha256": first_prediction.provenance.get("profile_sha256"),
            "profile_publication_eligible": profile_publication_eligible,
            "uncertainty_calibrated": uncertainty_calibrated,
        },
        "workloads": workload_rows,
        "aggregate_metrics": aggregate,
        "summary": {
            "workload_count": len(workload_rows),
            "measurement_count": len(measurements),
            "metric_workload_pair_count": len(all_absolute_errors),
            "mean_absolute_percentage_error_pct": sum(all_absolute_errors)
            / len(all_absolute_errors),
            "median_absolute_percentage_error_pct": float(
                median(all_absolute_errors)
            ),
            "maximum_absolute_percentage_error_pct": max(all_absolute_errors),
            "interval_covered_metric_workload_pairs": sum(
                value is True for value in covered
            ),
            "interval_evaluated_metric_workload_pairs": sum(
                value is not None for value in covered
            ),
            "observed_interval_coverage": (
                sum(value is True for value in covered) / interval_evaluated
                if interval_evaluated
                else None
            ),
            "analysis_ready": True,
            "publication_ready": False,
            "final_holdout_required": dataset_split != "holdout",
            "final_holdout_completed": dataset_split == "holdout",
            "warnings": warnings,
        },
    }
