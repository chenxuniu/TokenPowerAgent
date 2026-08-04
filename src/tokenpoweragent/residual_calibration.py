"""Fit a sparse, auditable workload-transfer correction for the L0 projector."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tokenpoweragent.twin.topology import (
    WORKLOAD_RESIDUAL_FEATURES,
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
)
from tokenpoweragent.workload_campaign import (
    WorkloadCampaign,
    WorkloadCampaignError,
    sha256_file,
)


class ResidualCalibrationError(ValueError):
    """Raised when development evidence cannot support a residual model."""


_METRIC_LOCATIONS = {
    "throughput_tok_s": ("metrics", "throughput_tok_s"),
    "avg_power_w": ("diagnostics", "avg_power_w"),
    "ttft_ms": ("metrics", "ttft_ms"),
    "tpot_ms": ("metrics", "tpot_ms"),
}

_RIDGE_LAMBDA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def build_workload_residual_profile(
    profile_path: Path,
    campaign_path: Path,
    report_path: Path,
    profile_id: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return a v2 profile and development-only fit diagnostics.

    The feature family is fixed before fitting. Ridge strength is selected per
    metric by leave-one-workload-out MAPE. Energy is never fitted independently;
    its diagnostic prediction is derived from corrected power and throughput.
    """

    profile_id = profile_id.strip()
    if not profile_id:
        raise ResidualCalibrationError("profile_id is required")
    profile_path = Path(profile_path)
    campaign_path = Path(campaign_path)
    report_path = Path(report_path)
    raw_profile = _load_object(profile_path, "calibration profile")
    report = _load_object(report_path, "validation report")
    if "workload_residual_model" in raw_profile:
        raise ResidualCalibrationError(
            "source profile already contains a workload residual model"
        )
    try:
        profile = CalibrationProfile.from_mapping(raw_profile)
        campaign = WorkloadCampaign.load(campaign_path)
        campaign.assert_profile(profile)
    except (CalibrationError, WorkloadCampaignError) as exc:
        raise ResidualCalibrationError(str(exc)) from exc
    if len(profile.points) != 1:
        raise ResidualCalibrationError(
            "residual fitting currently requires exactly one measured L1 anchor"
        )

    profile_hash = sha256_file(profile_path)
    campaign_hash = sha256_file(campaign_path)
    report_hash = sha256_file(report_path)
    _verify_report_contract(report, profile, campaign, profile_hash, campaign_hash)
    reference = profile.points[0].workload
    rows = _development_rows(report, campaign, reference)
    feature_matrix = [row["features"] for row in rows]
    feature_scales = _feature_scales(feature_matrix)

    metric_models: Dict[str, Any] = {}
    metric_diagnostics: Dict[str, Any] = {}
    selected_loo_factors: Dict[str, List[float]] = {}
    for metric, (section, key) in _METRIC_LOCATIONS.items():
        targets = [
            math.log(row[section][key]["observed"] / row[section][key]["base"])
            for row in rows
        ]
        selection = _select_ridge_lambda(feature_matrix, targets)
        selected_lambda = selection["selected_lambda"]
        coefficients = _ridge_fit(
            _scale_matrix(feature_matrix, feature_scales),
            targets,
            selected_lambda,
        )
        fitted_factors = [
            math.exp(_dot(coefficients, _scale_row(row, feature_scales)))
            for row in feature_matrix
        ]
        fit_errors = [
            100.0
            * abs(
                row[section][key]["base"] * factor
                / row[section][key]["observed"]
                - 1.0
            )
            for row, factor in zip(rows, fitted_factors)
        ]
        selected_fold = next(
            item
            for item in selection["lambda_grid"]
            if item["ridge_lambda"] == selected_lambda
        )
        selected_loo_factors[metric] = selected_fold.pop("_factors")
        for item in selection["lambda_grid"]:
            item.pop("_factors", None)
        metric_models[metric] = {
            "coefficients": coefficients,
            "ridge_lambda": selected_lambda,
            "development_loo_mape_pct": selected_fold[
                "mean_absolute_percentage_error_pct"
            ],
            "development_loo_max_ape_pct": selected_fold[
                "maximum_absolute_percentage_error_pct"
            ],
        }
        metric_diagnostics[metric] = {
            **selection,
            "coefficients": coefficients,
            "full_development_fit_mape_pct": _mean(fit_errors),
            "full_development_fit_max_ape_pct": max(fit_errors),
            "leave_one_workload_out": [
                {
                    "workload_id": row["workload_id"],
                    "base_prediction": row[section][key]["base"],
                    "corrected_prediction": (
                        row[section][key]["base"]
                        * selected_loo_factors[metric][index]
                    ),
                    "observed_median": row[section][key]["observed"],
                    "absolute_percentage_error_pct": selected_fold[
                        "absolute_percentage_errors_pct"
                    ][index],
                }
                for index, row in enumerate(rows)
            ],
        }

    energy_diagnostics = _derived_energy_diagnostics(rows, selected_loo_factors)
    context_values = [row["features"][0] for row in rows]
    concurrency_values = [row["features"][1] for row in rows]
    experiment = report["experiment"]
    protocol = report["protocol"]
    workload_residual_model = {
        "schema_version": "1.0",
        "model_id": profile_id + "-workload-residual",
        "feature_schema": "context-concurrency-quadratic-log-residual-v1",
        "feature_names": list(WORKLOAD_RESIDUAL_FEATURES),
        "feature_scales": feature_scales,
        "reference_workload": reference.to_dict(),
        "scope": {
            "required_gpus": 1,
            "target_nodes": 1,
            "fixed_output_tokens": reference.output_tokens,
            "request_rate": "inf",
            "context_log2_range": [min(context_values), max(context_values)],
            "concurrency_log2_range": [
                min(concurrency_values),
                max(concurrency_values),
            ],
            "serving_configuration": dict(campaign.configuration),
        },
        "metrics": metric_models,
        "training": {
            "development_only": True,
            "final_holdout_required": True,
            "dataset_split": "validation",
            "workload_ids": [row["workload_id"] for row in rows],
            "workload_count": len(rows),
            "repeats_per_workload": protocol["repeats_per_workload"],
            "source_profile_sha256": profile_hash,
            "source_campaign_sha256": campaign_hash,
            "source_report_sha256": report_hash,
            "source_prediction_sha256": protocol["prediction_sha256"],
            "source_measurement_sha256": protocol["measurement_sha256"],
            "source_raw_artifact_manifest_sha256": protocol[
                "raw_artifact_manifest_sha256"
            ],
            "server_image_id": experiment["server_image_id"],
            "client_image_id": experiment["client_image_id"],
        },
    }

    output_profile = copy.deepcopy(raw_profile)
    output_profile["schema_version"] = "1.1"
    output_profile["profile_id"] = profile_id
    output_profile["publication_eligible"] = False
    output_profile["uncertainty_calibrated"] = False
    output_profile["workload_residual_model"] = workload_residual_model
    metadata = dict(output_profile.get("metadata", {}))
    metadata.update(
        {
            "generated_by": "tokenpoweragent fit-workload-residuals",
            "base_profile_id": profile.profile_id,
            "base_profile_sha256": profile_hash,
            "development_campaign_id": campaign.campaign_id,
            "development_campaign_sha256": campaign_hash,
            "development_report_sha256": report_hash,
            "development_only": True,
            "final_holdout_required": True,
        }
    )
    output_profile["metadata"] = metadata

    diagnostics = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "model_id": workload_residual_model["model_id"],
        "status": "development-only; independent final holdout required",
        "selection_protocol": {
            "target": "log(observed_median / base_prediction)",
            "intercept": False,
            "feature_names": list(WORKLOAD_RESIDUAL_FEATURES),
            "feature_scaling": "training-fold RMS; full-fit RMS stored in profile",
            "ridge_lambda_grid": list(_RIDGE_LAMBDA_GRID),
            "selection_metric": "leave-one-workload-out MAPE",
            "energy_model": "derived from corrected power and throughput",
        },
        "source": workload_residual_model["training"],
        "feature_scales": feature_scales,
        "training_rows": [
            {
                "workload_id": row["workload_id"],
                "workload": row["workload"],
                "features": row["features"],
            }
            for row in rows
        ],
        "metrics": metric_diagnostics,
        "derived_energy_j_per_1k_output_tokens": energy_diagnostics,
    }
    return output_profile, diagnostics


def _verify_report_contract(
    report: Mapping[str, Any],
    profile: CalibrationProfile,
    campaign: WorkloadCampaign,
    profile_hash: str,
    campaign_hash: str,
) -> None:
    protocol = report.get("protocol", {})
    experiment = report.get("experiment", {})
    workloads = report.get("workloads", ())
    if not isinstance(protocol, Mapping) or not isinstance(experiment, Mapping):
        raise ResidualCalibrationError("validation report lacks protocol metadata")
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)):
        raise ResidualCalibrationError("validation report workloads must be an array")
    required_flags = (
        "preregistered_validation_valid",
        "prediction_manifest_verified",
        "predictions_precede_measurements",
        "raw_artifact_manifest_verified",
    )
    if not all(protocol.get(flag) is True for flag in required_flags):
        raise ResidualCalibrationError(
            "residual fitting requires a verified preregistered validation report"
        )
    if protocol.get("dataset_split") != "validation":
        raise ResidualCalibrationError(
            "residual fitting consumes validation evidence, not final holdout evidence"
        )
    if protocol.get("campaign_sha256") != campaign_hash:
        raise ResidualCalibrationError("validation report campaign hash differs")
    if experiment.get("profile_sha256") != profile_hash:
        raise ResidualCalibrationError("validation report profile hash differs")
    if experiment.get("profile_id") != profile.profile_id:
        raise ResidualCalibrationError("validation report profile id differs")
    if protocol.get("workload_count") != len(campaign.workloads):
        raise ResidualCalibrationError("validation report workload count differs")
    if protocol.get("repeats_per_workload", 0) < 3:
        raise ResidualCalibrationError("at least three repeats per workload are required")
    if len(workloads) < 6:
        raise ResidualCalibrationError("at least six validation workloads are required")


def _development_rows(
    report: Mapping[str, Any],
    campaign: WorkloadCampaign,
    reference: InferenceWorkload,
) -> List[Dict[str, Any]]:
    campaign_points = {point.point_id: point for point in campaign.workloads}
    rows = []
    for raw in report["workloads"]:
        if not isinstance(raw, Mapping):
            raise ResidualCalibrationError("validation workload entry must be an object")
        workload_id = str(raw.get("workload_id", ""))
        point = campaign_points.get(workload_id)
        if point is None or raw.get("dataset_split") != "validation":
            raise ResidualCalibrationError("validation workload identity differs")
        if point.dataset_split != "validation":
            raise ResidualCalibrationError("campaign contains a non-validation workload")
        workload = InferenceWorkload.from_mapping(raw.get("workload", {}))
        if workload != point.workload:
            raise ResidualCalibrationError("validation workload dimensions differ")
        if workload.output_tokens != reference.output_tokens:
            raise ResidualCalibrationError(
                "residual v1 requires a fixed output-token length"
            )
        if workload.request_rate_req_s is not None:
            raise ResidualCalibrationError("residual v1 requires request_rate=inf")
        row: Dict[str, Any] = {
            "workload_id": workload_id,
            "workload": workload.to_dict(),
            "features": list(_features(workload, reference)),
            "metrics": {},
            "diagnostics": {},
        }
        for section, key in _METRIC_LOCATIONS.values():
            source = raw.get(section, {}).get(key, {})
            if not isinstance(source, Mapping):
                raise ResidualCalibrationError("validation metric %s is missing" % key)
            try:
                base = float(source["prediction"])
                observed = float(source["observed_median"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ResidualCalibrationError(
                    "validation metric %s is incomplete" % key
                ) from exc
            if not all(math.isfinite(value) and value > 0 for value in (base, observed)):
                raise ResidualCalibrationError(
                    "validation metric %s must be positive" % key
                )
            row[section][key] = {"base": base, "observed": observed}
        energy = raw.get("metrics", {}).get(
            "energy_j_per_1k_output_tokens", {}
        )
        try:
            energy_base = float(energy["prediction"])
            energy_observed = float(energy["observed_median"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResidualCalibrationError("validation energy metric is incomplete") from exc
        if not all(
            math.isfinite(value) and value > 0
            for value in (energy_base, energy_observed)
        ):
            raise ResidualCalibrationError("validation energy metric must be positive")
        row["metrics"]["energy_j_per_1k_output_tokens"] = {
            "base": energy_base,
            "observed": energy_observed,
        }
        rows.append(row)
    if {row["workload_id"] for row in rows} != set(campaign_points):
        raise ResidualCalibrationError("validation report workloads differ from campaign")
    return rows


def _features(
    workload: InferenceWorkload, reference: InferenceWorkload
) -> Tuple[float, ...]:
    context = math.log(
        (workload.input_tokens + 0.5 * workload.output_tokens)
        / (reference.input_tokens + 0.5 * reference.output_tokens),
        2.0,
    )
    concurrency = math.log(workload.concurrency / reference.concurrency, 2.0)
    return (
        context,
        concurrency,
        context * concurrency,
        concurrency * concurrency,
    )


def _select_ridge_lambda(
    features: Sequence[Sequence[float]], targets: Sequence[float]
) -> Dict[str, Any]:
    grid = []
    for ridge_lambda in _RIDGE_LAMBDA_GRID:
        factors = _leave_one_out_factors(features, targets, ridge_lambda)
        actual_factors = [math.exp(value) for value in targets]
        errors = [
            100.0 * abs(predicted / observed - 1.0)
            for predicted, observed in zip(factors, actual_factors)
        ]
        grid.append(
            {
                "ridge_lambda": ridge_lambda,
                "mean_absolute_percentage_error_pct": _mean(errors),
                "maximum_absolute_percentage_error_pct": max(errors),
                "absolute_percentage_errors_pct": errors,
                "_factors": factors,
            }
        )
    selected = min(
        grid,
        key=lambda item: (
            item["mean_absolute_percentage_error_pct"],
            item["maximum_absolute_percentage_error_pct"],
            -item["ridge_lambda"],
        ),
    )
    return {"selected_lambda": selected["ridge_lambda"], "lambda_grid": grid}


def _leave_one_out_factors(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    ridge_lambda: float,
) -> List[float]:
    factors = []
    for held_out in range(len(features)):
        train_features = [
            row for index, row in enumerate(features) if index != held_out
        ]
        train_targets = [
            value for index, value in enumerate(targets) if index != held_out
        ]
        scales = _feature_scales(train_features)
        coefficients = _ridge_fit(
            _scale_matrix(train_features, scales), train_targets, ridge_lambda
        )
        factors.append(
            math.exp(
                _dot(coefficients, _scale_row(features[held_out], scales))
            )
        )
    return factors


def _derived_energy_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    factors: Mapping[str, Sequence[float]],
) -> Mapping[str, Any]:
    rendered = []
    errors = []
    for index, row in enumerate(rows):
        energy = row["metrics"]["energy_j_per_1k_output_tokens"]
        prediction = (
            energy["base"]
            * factors["avg_power_w"][index]
            / factors["throughput_tok_s"][index]
        )
        error = 100.0 * abs(prediction / energy["observed"] - 1.0)
        errors.append(error)
        rendered.append(
            {
                "workload_id": row["workload_id"],
                "base_prediction": energy["base"],
                "corrected_prediction": prediction,
                "observed_median": energy["observed"],
                "absolute_percentage_error_pct": error,
            }
        )
    return {
        "semantics": "derived from leave-one-out power and throughput corrections",
        "mean_absolute_percentage_error_pct": _mean(errors),
        "maximum_absolute_percentage_error_pct": max(errors),
        "leave_one_workload_out": rendered,
    }


def _feature_scales(features: Sequence[Sequence[float]]) -> List[float]:
    if not features:
        raise ResidualCalibrationError("cannot scale an empty feature matrix")
    scales = []
    for column in range(len(WORKLOAD_RESIDUAL_FEATURES)):
        rms = math.sqrt(_mean([row[column] * row[column] for row in features]))
        scales.append(max(rms, 1e-12))
    return scales


def _scale_matrix(
    features: Sequence[Sequence[float]], scales: Sequence[float]
) -> List[List[float]]:
    return [_scale_row(row, scales) for row in features]


def _scale_row(row: Sequence[float], scales: Sequence[float]) -> List[float]:
    return [value / scale for value, scale in zip(row, scales)]


def _ridge_fit(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    ridge_lambda: float,
) -> List[float]:
    width = len(WORKLOAD_RESIDUAL_FEATURES)
    matrix = [[0.0 for _ in range(width)] for _ in range(width)]
    vector = [0.0 for _ in range(width)]
    for row, target in zip(features, targets):
        for left in range(width):
            vector[left] += row[left] * target
            for right in range(width):
                matrix[left][right] += row[left] * row[right]
    for index in range(width):
        matrix[index][index] += ridge_lambda
    return _solve(matrix, vector)


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    size = len(vector)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ResidualCalibrationError("ridge system is numerically singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ResidualCalibrationError("cannot average an empty sequence")
    return sum(values) / len(values)


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualCalibrationError("cannot read %s: %s" % (label, exc)) from exc
    if not isinstance(raw, dict):
        raise ResidualCalibrationError("%s root must be an object" % label)
    return raw
