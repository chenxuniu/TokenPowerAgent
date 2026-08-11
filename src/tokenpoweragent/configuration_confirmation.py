"""Independent paired confirmation for a locked serving configuration."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Mapping, Optional, Sequence

from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    sha256_file,
)
from tokenpoweragent.configuration_runner import (
    ConfigurationRunnerError,
    verify_configuration_measurement_record,
    verify_frozen_configuration_campaign,
)
from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.schema import EvidenceLevel


class ConfigurationConfirmationError(ValueError):
    """Raised when an independent confirmation contract is invalid."""


METRIC_DIRECTIONS = {
    "energy_j_per_1k_output_tokens": "lower",
    "throughput_tok_s": "higher",
    "ttft_ms": "lower",
    "tpot_ms": "lower",
}

RAW_ARTIFACT_KEYS = (
    "telemetry_path",
    "client_output_path",
    "server_log_path",
)


def build_configuration_confirmation_report(
    campaign_path: Path,
    predictions_path: Path,
    schedule_path: Path,
    summary_path: Path,
    freeze_manifest_path: Path,
    measurements_path: Path,
    report_path: Path,
    artifact_list_path: Optional[Path] = None,
    artifact_manifest_path: Optional[Path] = None,
    include_artifacts: Sequence[Path] = (),
) -> Mapping[str, Any]:
    """Validate a pre-locked, paired L4 confirmation campaign."""

    if (artifact_list_path is None) != (artifact_manifest_path is None):
        raise ConfigurationConfirmationError(
            "artifact list and artifact manifest must be requested together"
        )
    campaign = ConfigurationCampaign.load(campaign_path)
    contract = _load_confirmation_contract(campaign_path, campaign)
    try:
        artifacts = verify_frozen_configuration_campaign(
            campaign_path=campaign_path,
            predictions_path=predictions_path,
            schedule_path=schedule_path,
            summary_path=summary_path,
            manifest_path=freeze_manifest_path,
        )
    except ConfigurationRunnerError as exc:
        raise ConfigurationConfirmationError(str(exc)) from exc

    schedule = _read_object(schedule_path, "schedule")
    predictions = EvidenceStore.read_jsonl(predictions_path).records
    measurements = EvidenceStore.read_jsonl(measurements_path).records
    actions = list(schedule.get("actions", ()))
    if len(measurements) != len(actions):
        raise ConfigurationConfirmationError(
            "measurement count %d does not match the frozen schedule count %d"
            % (len(measurements), len(actions))
        )

    point_map = {point.point_id: point for point in campaign.candidates}
    for action_index, (action, record) in enumerate(zip(actions, measurements)):
        if not isinstance(action, Mapping):
            raise ConfigurationConfirmationError(
                "schedule action must be an object"
            )
        if int(action.get("action_index", -1)) != action_index:
            raise ConfigurationConfirmationError(
                "schedule action indices are not contiguous"
            )
        point = point_map.get(str(action.get("candidate_id", "")))
        if point is None:
            raise ConfigurationConfirmationError(
                "schedule references an unknown candidate"
            )
        try:
            verify_configuration_measurement_record(
                campaign, point, action, record, artifacts
            )
        except ConfigurationRunnerError as exc:
            raise ConfigurationConfirmationError(
                "measurement action %d is invalid: %s"
                % (action_index, exc)
            ) from exc
        _assert_raw_artifact_provenance(record, action_index)

    prediction_times = [_parse_time(record.created_at) for record in predictions]
    measurement_times = [
        _parse_time(record.created_at) for record in measurements
    ]
    predictions_precede_measurements = (
        bool(prediction_times)
        and bool(measurement_times)
        and max(prediction_times) < min(measurement_times)
    )
    if not predictions_precede_measurements:
        raise ConfigurationConfirmationError(
            "frozen predictions must precede every confirmation measurement"
        )

    winner_id = contract["winner_id"]
    baseline_id = contract["baseline_id"]
    pairs = _build_pairs(
        campaign=campaign,
        actions=actions,
        measurements=measurements,
        winner_id=winner_id,
        baseline_id=baseline_id,
    )
    metric_results = {
        metric: _paired_metric_summary(
            pairs, metric, direction, winner_id, baseline_id
        )
        for metric, direction in METRIC_DIRECTIONS.items()
    }
    candidate_results = {
        candidate_id: _candidate_summary(
            [
                record
                for record in measurements
                if record.candidate_id == candidate_id
            ]
        )
        for candidate_id in (winner_id, baseline_id)
    }

    all_measurements_succeeded = all(
        record.status == EvidenceStatus.SUCCEEDED for record in measurements
    )
    all_measurements_l4 = all(
        record.level == EvidenceLevel.L4 for record in measurements
    )
    slo_pass = {
        candidate_id: all(
            _slo_accepts(record.metrics, campaign.slo)
            for record in measurements
            if record.candidate_id == candidate_id
        )
        for candidate_id in (winner_id, baseline_id)
    }

    raw_artifact: Dict[str, Any] = {
        "verified": False,
        "entry_count": 0,
        "sha256": None,
    }
    if artifact_list_path is not None and artifact_manifest_path is not None:
        paths = _raw_artifact_paths(
            freeze_manifest_path,
            measurements_path,
            measurements,
            include_artifacts,
        )
        _write_artifact_manifests(
            paths, artifact_list_path, artifact_manifest_path
        )
        raw_artifact = {
            "verified": True,
            "entry_count": len(paths),
            "sha256": sha256_file(artifact_manifest_path),
            "list_path": str(artifact_list_path),
            "manifest_path": str(artifact_manifest_path),
        }

    acceptance_contract = contract["acceptance"]
    energy = metric_results["energy_j_per_1k_output_tokens"]
    ttft = metric_results["ttft_ms"]
    checks = {
        "minimum_pair_count": len(pairs)
        >= int(acceptance_contract["minimum_pair_count"]),
        "minimum_energy_pair_win_rate": energy["win_rate"]
        >= float(acceptance_contract["minimum_energy_pair_win_rate"]),
        "energy_mean_saving_ci_lower_gt_pct": energy[
            "mean_improvement_pct_ci95"
        ][0]
        > float(acceptance_contract["energy_mean_saving_ci_lower_gt_pct"]),
        "ttft_median_reduction_gt_pct": ttft["median_improvement_pct"]
        > float(acceptance_contract["ttft_median_reduction_gt_pct"]),
        "all_measurements_satisfy_slo": all(slo_pass.values()),
        "all_measurements_succeeded": all_measurements_succeeded,
    }
    protocol_valid = (
        campaign.dataset_split == "configuration-confirmation"
        and campaign.measurement_schedule == "paired-alternating"
        and campaign.measurement_levels == (EvidenceLevel.L4,)
        and len(pairs) == campaign.repeats
        and len(measurements) == 2 * campaign.repeats
        and all_measurements_l4
        and predictions_precede_measurements
        and contract["selection_locked_before_measurement"]
        and contract["no_refit_or_reselection"]
    )
    confirmation_passed = protocol_valid and all(checks.values())
    publication_ready = confirmation_passed and raw_artifact["verified"]
    warnings = []
    if not raw_artifact["verified"]:
        warnings.append(
            "Raw artifact hashes were not requested; publication archival is incomplete."
        )
    if not confirmation_passed:
        warnings.append(
            "The preregistered confirmation rule did not pass; report the result without reselection."
        )

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "campaign_id": campaign.campaign_id,
        "protocol": {
            "independent_confirmation_valid": protocol_valid,
            "freeze_manifest_verified": True,
            "predictions_precede_measurements": predictions_precede_measurements,
            "selection_locked_before_measurement": contract[
                "selection_locked_before_measurement"
            ],
            "no_refit_or_reselection": contract[
                "no_refit_or_reselection"
            ],
            "schedule": campaign.measurement_schedule,
            "dataset_split": campaign.dataset_split,
            "pair_count": len(pairs),
            "measurement_count": len(measurements),
            "expected_measurement_count": 2 * campaign.repeats,
            "level_counts": dict(
                sorted(
                    Counter(
                        record.level.name for record in measurements
                    ).items()
                )
            ),
            "status_counts": dict(
                sorted(
                    Counter(
                        record.status.value for record in measurements
                    ).items()
                )
            ),
            "campaign_sha256": artifacts.campaign_sha256,
            "prediction_sha256": artifacts.prediction_sha256,
            "schedule_sha256": artifacts.schedule_sha256,
            "summary_sha256": artifacts.summary_sha256,
            "freeze_manifest_sha256": artifacts.manifest_sha256,
            "measurement_sha256": sha256_file(measurements_path),
            "raw_artifacts": raw_artifact,
        },
        "selection_lock": {
            "winner_id": winner_id,
            "baseline_id": baseline_id,
            "source_artifact_sha256": contract["source_artifact_sha256"],
        },
        "experiment": {
            "model": campaign.model_id,
            "model_revision": campaign.model_revision,
            "hardware": dict(campaign.hardware),
            "workload": campaign.workload_dict(
                campaign.target_workload.num_requests
            ),
            "power_limit_w": campaign.power_limit_w,
            "repeats": campaign.repeats,
            "seeds": sorted(
                {int(action["seed"]) for action in actions}
            ),
            "slo": dict(campaign.slo),
        },
        "candidates": candidate_results,
        "pairs": pairs,
        "paired_results": metric_results,
        "slo": {
            "all_repeats_pass": slo_pass,
            "thresholds": dict(campaign.slo),
        },
        "acceptance": {
            "preregistered_thresholds": acceptance_contract,
            "checks": checks,
            "confirmation_passed": confirmation_passed,
        },
        "cost": {
            "actual_gpu_hours_total": sum(
                record.gpu_hours for record in measurements
            ),
            "actual_gpu_hours_by_candidate": {
                candidate_id: sum(
                    record.gpu_hours
                    for record in measurements
                    if record.candidate_id == candidate_id
                )
                for candidate_id in (winner_id, baseline_id)
            },
        },
        "summary": {
            "publication_ready": publication_ready,
            "confirmation_passed": confirmation_passed,
            "headline_energy_saving_pct": energy["mean_improvement_pct"],
            "headline_energy_saving_ci95_pct": energy[
                "mean_improvement_pct_ci95"
            ],
            "energy_pair_win_rate": energy["win_rate"],
            "energy_sign_test_p_one_sided": energy[
                "exact_sign_test_p_one_sided"
            ],
            "headline_ttft_reduction_pct": ttft[
                "median_improvement_pct"
            ],
            "warnings": warnings,
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_confirmation_contract(
    campaign_path: Path, campaign: ConfigurationCampaign
) -> Mapping[str, Any]:
    raw = _read_object(campaign_path, "campaign")
    contract = raw.get("preregistered_confirmation")
    if not isinstance(contract, Mapping):
        raise ConfigurationConfirmationError(
            "campaign lacks preregistered_confirmation"
        )
    winner_id = str(contract.get("winner_id", "")).strip()
    baseline_id = str(contract.get("baseline_id", "")).strip()
    point_map = {point.point_id: point for point in campaign.candidates}
    if winner_id not in point_map or baseline_id not in point_map:
        raise ConfigurationConfirmationError(
            "confirmation winner and baseline must be frozen candidates"
        )
    if winner_id == baseline_id:
        raise ConfigurationConfirmationError(
            "confirmation winner and baseline must differ"
        )
    if point_map[baseline_id].role != "expert-baseline":
        raise ConfigurationConfirmationError(
            "confirmation baseline must use the expert-baseline role"
        )
    source_hashes = contract.get("source_artifact_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ConfigurationConfirmationError(
            "confirmation must lock source artifact hashes"
        )
    normalized_hashes = {}
    for key, value in source_hashes.items():
        digest = str(value).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ConfigurationConfirmationError(
                "source artifact %s lacks a SHA-256 digest" % key
            )
        normalized_hashes[str(key)] = digest
    acceptance = contract.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ConfigurationConfirmationError(
            "confirmation lacks preregistered acceptance rules"
        )
    required = {
        "minimum_pair_count",
        "minimum_energy_pair_win_rate",
        "energy_mean_saving_ci_lower_gt_pct",
        "ttft_median_reduction_gt_pct",
    }
    if set(acceptance) != required:
        raise ConfigurationConfirmationError(
            "confirmation acceptance rules must be exactly %s"
            % ", ".join(sorted(required))
        )
    try:
        normalized_acceptance = {
            "minimum_pair_count": int(acceptance["minimum_pair_count"]),
            "minimum_energy_pair_win_rate": float(
                acceptance["minimum_energy_pair_win_rate"]
            ),
            "energy_mean_saving_ci_lower_gt_pct": float(
                acceptance["energy_mean_saving_ci_lower_gt_pct"]
            ),
            "ttft_median_reduction_gt_pct": float(
                acceptance["ttft_median_reduction_gt_pct"]
            ),
        }
    except (TypeError, ValueError) as exc:
        raise ConfigurationConfirmationError(
            "confirmation acceptance values must be numeric"
        ) from exc
    if normalized_acceptance["minimum_pair_count"] != campaign.repeats:
        raise ConfigurationConfirmationError(
            "minimum_pair_count must equal the frozen repeat count"
        )
    if not 0 <= normalized_acceptance["minimum_energy_pair_win_rate"] <= 1:
        raise ConfigurationConfirmationError(
            "minimum_energy_pair_win_rate must be in [0, 1]"
        )
    selection_locked = contract.get("selection_locked_before_measurement") is True
    no_reselection = contract.get("no_refit_or_reselection") is True
    if not selection_locked or not no_reselection:
        raise ConfigurationConfirmationError(
            "confirmation must lock selection and prohibit refitting/reselection"
        )
    return {
        "winner_id": winner_id,
        "baseline_id": baseline_id,
        "source_artifact_sha256": normalized_hashes,
        "selection_locked_before_measurement": selection_locked,
        "no_refit_or_reselection": no_reselection,
        "acceptance": normalized_acceptance,
    }


def _build_pairs(
    campaign: ConfigurationCampaign,
    actions: Sequence[Mapping[str, Any]],
    measurements: Sequence[EvidenceRecord],
    winner_id: str,
    baseline_id: str,
) -> Sequence[Mapping[str, Any]]:
    grouped: Dict[
        int, list[tuple[Mapping[str, Any], EvidenceRecord]]
    ] = defaultdict(list)
    for action, record in zip(actions, measurements):
        repeat = int(action.get("repeat", -1))
        if int(action.get("pair_index", -1)) != repeat:
            raise ConfigurationConfirmationError(
                "paired schedule has inconsistent pair_index"
            )
        grouped[repeat].append((action, record))
    if sorted(grouped) != list(range(campaign.repeats)):
        raise ConfigurationConfirmationError(
            "confirmation repeat indices are not contiguous"
        )

    pairs = []
    for repeat in range(campaign.repeats):
        rows = grouped[repeat]
        if len(rows) != 2:
            raise ConfigurationConfirmationError(
                "confirmation repeat %d does not contain exactly two actions"
                % repeat
            )
        ids = [record.candidate_id for _, record in rows]
        if set(ids) != {winner_id, baseline_id}:
            raise ConfigurationConfirmationError(
                "confirmation repeat %d does not contain the locked pair"
                % repeat
            )
        positions = [
            int(action.get("pair_order_position", -1))
            for action, _ in rows
        ]
        if positions != [0, 1]:
            raise ConfigurationConfirmationError(
                "confirmation pair order positions are invalid"
            )
        winner = next(
            record for _, record in rows if record.candidate_id == winner_id
        )
        baseline = next(
            record for _, record in rows if record.candidate_id == baseline_id
        )
        seed_values = {int(action["seed"]) for action, _ in rows}
        if len(seed_values) != 1:
            raise ConfigurationConfirmationError(
                "paired actions must share one seed"
            )
        pairs.append(
            {
                "pair_index": repeat,
                "seed": seed_values.pop(),
                "run_order": ids,
                "winner_metrics": {
                    metric: float(winner.metrics[metric])
                    for metric in METRIC_DIRECTIONS
                },
                "baseline_metrics": {
                    metric: float(baseline.metrics[metric])
                    for metric in METRIC_DIRECTIONS
                },
                "winner_gpu_hours": winner.gpu_hours,
                "baseline_gpu_hours": baseline.gpu_hours,
            }
        )
    return pairs


def _paired_metric_summary(
    pairs: Sequence[Mapping[str, Any]],
    metric: str,
    direction: str,
    winner_id: str,
    baseline_id: str,
) -> Mapping[str, Any]:
    winner_values = [
        float(pair["winner_metrics"][metric]) for pair in pairs
    ]
    baseline_values = [
        float(pair["baseline_metrics"][metric]) for pair in pairs
    ]
    improvements = []
    beneficial_differences = []
    for winner, baseline in zip(winner_values, baseline_values):
        if direction == "lower":
            beneficial = baseline - winner
        else:
            beneficial = winner - baseline
        beneficial_differences.append(beneficial)
        improvements.append(100.0 * beneficial / max(abs(baseline), 1e-12))
    wins = sum(value > 0 for value in beneficial_differences)
    ties = sum(value == 0 for value in beneficial_differences)
    non_ties = len(beneficial_differences) - ties
    return {
        "metric": metric,
        "direction": direction,
        "winner_id": winner_id,
        "baseline_id": baseline_id,
        "winner_values": winner_values,
        "baseline_values": baseline_values,
        "winner_median": float(median(winner_values)),
        "baseline_median": float(median(baseline_values)),
        "paired_improvement_pct": improvements,
        "mean_improvement_pct": float(mean(improvements)),
        "median_improvement_pct": float(median(improvements)),
        "minimum_improvement_pct": min(improvements),
        "maximum_improvement_pct": max(improvements),
        "mean_improvement_pct_ci95": _exact_bootstrap_mean_ci(improvements),
        "win_count": wins,
        "tie_count": ties,
        "pair_count": len(improvements),
        "win_rate": wins / len(improvements),
        "exact_sign_test_p_one_sided": _sign_test_p_one_sided(wins, non_ties),
    }


def _candidate_summary(records: Sequence[EvidenceRecord]) -> Mapping[str, Any]:
    if not records:
        raise ConfigurationConfirmationError(
            "confirmation candidate has no records"
        )
    metrics = {}
    for metric in METRIC_DIRECTIONS:
        values = [float(record.metrics[metric]) for record in records]
        center = float(median(values))
        metrics[metric] = {
            "values": values,
            "mean": float(mean(values)),
            "median": center,
            "mad": float(median(abs(value - center) for value in values)),
            "sample_sd": float(stdev(values)) if len(values) > 1 else 0.0,
        }
    return {
        "repeat_count": len(records),
        "metrics": metrics,
        "gpu_hours_total": sum(record.gpu_hours for record in records),
    }


def _exact_bootstrap_mean_ci(values: Sequence[float]) -> Sequence[float]:
    if not values:
        raise ConfigurationConfirmationError("cannot bootstrap an empty sample")
    if len(values) > 6:
        raise ConfigurationConfirmationError(
            "exact bootstrap is limited to six paired repeats"
        )
    means = sorted(
        sum(values[index] for index in sample) / len(values)
        for sample in itertools.product(range(len(values)), repeat=len(values))
    )
    return [
        _percentile(means, 0.025),
        _percentile(means, 0.975),
    ]


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _sign_test_p_one_sided(wins: int, non_ties: int) -> Optional[float]:
    if non_ties == 0:
        return None
    return sum(
        math.comb(non_ties, count)
        for count in range(wins, non_ties + 1)
    ) / (2.0**non_ties)


def _slo_accepts(metrics: Mapping[str, float], slo: Mapping[str, float]) -> bool:
    for metric, limit in slo.items():
        if metric == "min_goodput_req_s":
            if float(metrics.get("goodput_req_s", float("-inf"))) < limit:
                return False
        elif float(metrics.get(metric, float("inf"))) > limit:
            return False
    return True


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationConfirmationError(
            "cannot read %s: %s" % (label, exc)
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationConfirmationError(
            "%s must be a JSON object" % label
        )
    return raw


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationConfirmationError(
            "evidence has an invalid created_at timestamp"
        ) from exc


def _assert_raw_artifact_provenance(
    record: EvidenceRecord, action_index: int
) -> None:
    for key in RAW_ARTIFACT_KEYS:
        value = record.provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationConfirmationError(
                "measurement action %d lacks %s" % (action_index, key)
            )


def _raw_artifact_paths(
    freeze_manifest_path: Path,
    measurements_path: Path,
    measurements: Sequence[EvidenceRecord],
    include_artifacts: Sequence[Path],
) -> Sequence[Path]:
    paths = {Path(freeze_manifest_path), Path(measurements_path)}
    manifest_lines = Path(freeze_manifest_path).read_text(
        encoding="utf-8"
    ).splitlines()
    for line in manifest_lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ConfigurationConfirmationError(
                "freeze manifest has an invalid line"
            )
        paths.add(Path(parts[1].lstrip("* ")))
    paths.update(Path(path) for path in include_artifacts)
    for record in measurements:
        for key in RAW_ARTIFACT_KEYS:
            paths.add(Path(str(record.provenance[key])))
    ordered = sorted(paths, key=lambda path: str(path))
    missing = [str(path) for path in ordered if not path.is_file()]
    if missing:
        raise ConfigurationConfirmationError(
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
