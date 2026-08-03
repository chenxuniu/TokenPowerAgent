"""Blind holdout validation for frozen sandbox predictions."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from statistics import median, stdev
from typing import Any, Callable, Dict, Mapping, Sequence

from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
)
from tokenpoweragent.schema import EvidenceLevel


class HoldoutValidationError(ValueError):
    """Raised when prediction and holdout evidence are not comparable."""


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
    lower = prediction.metrics.get(prediction_key + "_lower")
    upper = prediction.metrics.get(prediction_key + "_upper")
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
    if prediction.level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
        raise HoldoutValidationError("prediction must be L0 or L2 evidence")
    if prediction.kind not in {
        EvidenceKind.SIMULATED,
        EvidenceKind.INTERPOLATED,
        EvidenceKind.EXTRAPOLATED,
    }:
        raise HoldoutValidationError("prediction has an invalid evidence kind")

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
