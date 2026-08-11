#!/usr/bin/env python3
"""Export ServeCompass evidence archives as analysis-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "plot-data"


@dataclass(frozen=True)
class Archive:
    source_id: str
    path: Path

    def read_bytes(self, member: str) -> bytes:
        with tarfile.open(self.path, "r:gz") as handle:
            stream = handle.extractfile(member)
            if stream is None:
                raise FileNotFoundError(f"{member} is not in {self.path}")
            return stream.read()

    def read_text(self, member: str) -> str:
        return self.read_bytes(member).decode("utf-8")

    def read_json(self, member: str) -> Dict[str, Any]:
        value = json.loads(self.read_text(member))
        if not isinstance(value, dict):
            raise ValueError(f"expected object in {self.path}:{member}")
        return value

    def read_jsonl(self, member: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line_number, line in enumerate(self.read_text(member).splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected object at {self.path}:{member}:{line_number}"
                )
            rows.append(value)
        return rows

    def matching_members(self, suffix: str) -> List[str]:
        with tarfile.open(self.path, "r:gz") as handle:
            return sorted(
                member.name
                for member in handle.getmembers()
                if member.isfile() and member.name.endswith(suffix)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path.home() / "Downloads",
        help="directory searched recursively for tpa-*.tar.gz archives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory receiving CSV, Markdown, and raw JSON files",
    )
    parser.add_argument(
        "--budget-report",
        type=Path,
        help="full deterministic budget-sweep JSON containing all episodes",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_archives(downloads: Path) -> List[Path]:
    paths = set(downloads.rglob("tpa-*.tar.gz"))
    paths.update((REPO_ROOT / "artifacts").glob("tpa-*.tar.gz"))
    paths.update((REPO_ROOT / "experiments" / "evidence").rglob("tpa-*.tar.gz"))
    return sorted(path.resolve() for path in paths)


def choose_archive(paths: Sequence[Path], filename: str, downloads: Path) -> Path:
    matches = [path for path in paths if path.name == filename]
    if not matches:
        raise FileNotFoundError(f"required archive not found: {filename}")

    def priority(path: Path) -> tuple[int, int, str]:
        direct_download = path.parent.resolve() == downloads.resolve()
        return (0 if direct_download else 1, len(path.parts), str(path))

    return sorted(matches, key=priority)[0]


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred: Sequence[str] = (),
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = set()
    for row in rows:
        fields.update(row.keys())
    fieldnames = list(dict.fromkeys([*preferred, *sorted(fields - set(preferred))]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    return len(rows)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def metric_unit(metric: str) -> str:
    if metric.endswith(("_ms", "_ms_lower", "_ms_upper")):
        return "ms"
    if metric.endswith(("_w", "_w_lower", "_w_upper")):
        return "W"
    if metric.endswith(("_j", "_j_lower", "_j_upper")):
        return "J"
    if "j_per_1k" in metric:
        return "J/1k_tokens"
    if metric.startswith("j_per_"):
        return "J/token"
    if metric.endswith("_tok_s"):
        return "tokens/s"
    if metric.endswith("_req_s"):
        return "requests/s"
    if metric.endswith("_tflops") or metric == "throughput_tflops":
        return "TFLOP/s"
    if metric.endswith("_pct") or metric.endswith("_rate"):
        return "percent_or_fraction"
    if metric.endswith("_gib"):
        return "GiB"
    if metric.endswith("_mib"):
        return "MiB"
    if metric.endswith("_s"):
        return "s"
    return ""


def workload_from(record: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = mapping(record.get("provenance"))
    direct = mapping(provenance.get("workload"))
    if direct:
        return direct
    return mapping(mapping(provenance.get("decomposition")).get("workload"))


def configuration_from(record: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = mapping(record.get("provenance"))
    for key in ("server_configuration", "frozen_serving_configuration"):
        value = mapping(provenance.get(key))
        if value:
            return value
    return mapping(mapping(provenance.get("decomposition")).get("configuration"))


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def evidence_row(
    record: Mapping[str, Any],
    experiment_id: str,
    source_archive: str,
    source_member: str,
) -> Dict[str, Any]:
    provenance = mapping(record.get("provenance"))
    workload = workload_from(record)
    configuration = configuration_from(record)
    row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "source_archive": source_archive,
        "source_member": source_member,
        "candidate_id": record.get("candidate_id"),
        "created_at": record.get("created_at"),
        "status": record.get("status"),
        "failure_reason": record.get("failure_reason"),
        "kind": record.get("kind"),
        "level": record.get("level"),
        "gpu_hours": record.get("gpu_hours"),
        "campaign_id": provenance.get("campaign_id"),
        "dataset_split": provenance.get("dataset_split"),
        "candidate_role": provenance.get("candidate_role"),
        "executor": provenance.get("executor"),
        "workload_id": provenance.get("workload_id"),
        "repeat": provenance.get("campaign_repeat"),
        "seed": provenance.get("seed"),
        "gpu_id": provenance.get("gpu_id"),
        "power_limit_w": provenance.get("power_limit_readback_w")
        or provenance.get("power_limit_requested_w"),
        "telemetry_path": provenance.get("telemetry_path"),
        "input_tokens": workload.get("input_len", workload.get("input_tokens")),
        "output_tokens": workload.get("output_len", workload.get("output_tokens")),
        "concurrency": workload.get(
            "max_concurrency", workload.get("concurrency")
        ),
        "num_requests": workload.get("num_prompts", workload.get("num_requests")),
        "request_rate": workload.get(
            "request_rate", workload.get("request_rate_req_s")
        ),
        "max_num_seqs": configuration.get("max_num_seqs"),
        "max_num_batched_tokens": configuration.get("max_num_batched_tokens"),
        "chunked_prefill": configuration.get("chunked_prefill"),
        "prefix_caching": configuration.get("prefix_caching"),
        "tensor_parallel": configuration.get("tensor_parallel"),
        "pipeline_parallel": configuration.get("pipeline_parallel"),
        "data_parallel": configuration.get("data_parallel"),
        "precision": configuration.get("precision"),
        "kv_cache_dtype": configuration.get("kv_cache_dtype"),
    }
    for key, value in mapping(record.get("metrics")).items():
        row[key] = value
    return row


def evidence_long_rows(
    records: Sequence[Mapping[str, Any]],
    experiment_id: str,
    source_archive: str,
    source_member: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        provenance = mapping(record.get("provenance"))
        workload = workload_from(record)
        for name, value in mapping(record.get("metrics")).items():
            if not numeric(value):
                continue
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "source_archive": source_archive,
                    "source_member": source_member,
                    "candidate_id": record.get("candidate_id"),
                    "status": record.get("status"),
                    "kind": record.get("kind"),
                    "level": record.get("level"),
                    "campaign_id": provenance.get("campaign_id"),
                    "dataset_split": provenance.get("dataset_split"),
                    "workload_id": provenance.get("workload_id"),
                    "repeat": provenance.get("campaign_repeat"),
                    "seed": provenance.get("seed"),
                    "input_tokens": workload.get(
                        "input_len", workload.get("input_tokens")
                    ),
                    "output_tokens": workload.get(
                        "output_len", workload.get("output_tokens")
                    ),
                    "concurrency": workload.get(
                        "max_concurrency", workload.get("concurrency")
                    ),
                    "metric": name,
                    "value": value,
                    "unit": metric_unit(name),
                }
            )
    return rows


def descriptive_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "n": len(values),
        "mean": mean,
        "median": median,
        "sample_sd": sample_sd,
        "mad": mad,
        "cv_pct": None if mean == 0 else sample_sd / abs(mean) * 100.0,
        "min": min(values),
        "max": max(values),
    }


def grouped_metric_summary(
    records: Sequence[Mapping[str, Any]],
    experiment_id: str,
    group_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        flat = evidence_row(record, experiment_id, "", "")
        grouped[tuple(flat.get(field) for field in group_fields)].append(record)
    rows: List[Dict[str, Any]] = []
    for group_key, group_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        metric_names = sorted(
            {
                key
                for record in group_records
                for key, value in mapping(record.get("metrics")).items()
                if numeric(value)
            }
        )
        for metric in metric_names:
            values = [
                float(mapping(record.get("metrics"))[metric])
                for record in group_records
                if numeric(mapping(record.get("metrics")).get(metric))
            ]
            row = {
                "experiment_id": experiment_id,
                **dict(zip(group_fields, group_key)),
                "metric": metric,
                "unit": metric_unit(metric),
                **descriptive_stats(values),
            }
            rows.append(row)
    return rows


def validation_rows(
    report: Mapping[str, Any], experiment_id: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for workload in report.get("workloads", []):
        if not isinstance(workload, Mapping):
            continue
        descriptor = mapping(workload.get("workload"))
        for metric_group in ("metrics", "diagnostics"):
            for metric, result in mapping(workload.get(metric_group)).items():
                if not isinstance(result, Mapping):
                    continue
                interval = result.get("prediction_interval")
                lower = interval[0] if isinstance(interval, list) and len(interval) == 2 else None
                upper = interval[1] if isinstance(interval, list) and len(interval) == 2 else None
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset_split": workload.get("dataset_split"),
                        "workload_id": workload.get("workload_id"),
                        "input_tokens": descriptor.get("input_tokens"),
                        "output_tokens": descriptor.get("output_tokens"),
                        "concurrency": descriptor.get("concurrency"),
                        "num_requests": descriptor.get("num_requests"),
                        "repeat_count": workload.get("repeat_count"),
                        "metric_group": metric_group,
                        "metric": metric,
                        "unit": metric_unit(metric),
                        "prediction": result.get("prediction"),
                        "observed_median": result.get("observed_median"),
                        "signed_prediction_error_pct": result.get(
                            "signed_prediction_error_pct"
                        ),
                        "absolute_percentage_error_pct": result.get(
                            "absolute_percentage_error_pct"
                        ),
                        "prediction_interval_lower": lower,
                        "prediction_interval_upper": upper,
                        "interval_covers_observed_median": result.get(
                            "interval_covers_observed_median"
                        ),
                        "observed_cv_pct": result.get("observed_cv_pct"),
                        "observed_mad": result.get("observed_mad"),
                        "observed_sample_sd": result.get("observed_sample_sd"),
                        "observations": result.get("observations"),
                        "observed_semantics": result.get("observed_semantics"),
                    }
                )
    return rows


def validation_aggregate_rows(
    report: Mapping[str, Any], experiment_id: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, values in mapping(report.get("aggregate_metrics")).items():
        if not isinstance(values, Mapping):
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "metric": metric,
                "unit": metric_unit(metric),
                **values,
            }
        )
    return rows


def winner_confirmation_pair_rows(
    report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    paired_results = mapping(report.get("paired_results"))
    selection = mapping(report.get("selection_lock"))
    rows: List[Dict[str, Any]] = []
    for pair in report.get("pairs", []):
        if not isinstance(pair, Mapping):
            continue
        pair_index = int(pair.get("pair_index", len(rows)))
        baseline_metrics = mapping(pair.get("baseline_metrics"))
        winner_metrics = mapping(pair.get("winner_metrics"))
        row: Dict[str, Any] = {
            "experiment_id": "winner_confirmation_v1",
            "pair_index": pair_index,
            "seed": pair.get("seed"),
            "run_order": pair.get("run_order"),
            "baseline_id": selection.get("baseline_id"),
            "winner_id": selection.get("winner_id"),
            "baseline_gpu_hours": pair.get("baseline_gpu_hours"),
            "winner_gpu_hours": pair.get("winner_gpu_hours"),
        }
        for metric, result in paired_results.items():
            result_map = mapping(result)
            improvements = result_map.get("paired_improvement_pct", [])
            row[f"baseline_{metric}"] = baseline_metrics.get(metric)
            row[f"winner_{metric}"] = winner_metrics.get(metric)
            row[f"improvement_{metric}_pct"] = (
                improvements[pair_index]
                if isinstance(improvements, list) and pair_index < len(improvements)
                else None
            )
        rows.append(row)
    return rows


def winner_confirmation_summary_rows(
    report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, result in sorted(mapping(report.get("paired_results")).items()):
        value = mapping(result)
        ci = value.get("mean_improvement_pct_ci95", [])
        rows.append(
            {
                "experiment_id": "winner_confirmation_v1",
                "metric": metric,
                "direction": value.get("direction"),
                "baseline_id": value.get("baseline_id"),
                "winner_id": value.get("winner_id"),
                "pair_count": value.get("pair_count"),
                "baseline_median": value.get("baseline_median"),
                "winner_median": value.get("winner_median"),
                "mean_improvement_pct": value.get("mean_improvement_pct"),
                "median_improvement_pct": value.get("median_improvement_pct"),
                "minimum_improvement_pct": value.get("minimum_improvement_pct"),
                "maximum_improvement_pct": value.get("maximum_improvement_pct"),
                "mean_improvement_ci95_lower_pct": (
                    ci[0] if isinstance(ci, list) and len(ci) == 2 else None
                ),
                "mean_improvement_ci95_upper_pct": (
                    ci[1] if isinstance(ci, list) and len(ci) == 2 else None
                ),
                "win_count": value.get("win_count"),
                "tie_count": value.get("tie_count"),
                "win_rate": value.get("win_rate"),
                "exact_sign_test_p_one_sided": value.get(
                    "exact_sign_test_p_one_sided"
                ),
            }
        )
    return rows


def winner_confirmation_candidate_rows(
    report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    selection = mapping(report.get("selection_lock"))
    rows: List[Dict[str, Any]] = []
    for candidate_id, candidate in sorted(mapping(report.get("candidates")).items()):
        candidate_map = mapping(candidate)
        row: Dict[str, Any] = {
            "experiment_id": "winner_confirmation_v1",
            "candidate_id": candidate_id,
            "role": (
                "winner"
                if candidate_id == selection.get("winner_id")
                else "expert_baseline"
            ),
            "repeat_count": candidate_map.get("repeat_count"),
            "gpu_hours_total": candidate_map.get("gpu_hours_total"),
        }
        for metric, result in sorted(mapping(candidate_map.get("metrics")).items()):
            metric_result = mapping(result)
            for statistic in ("mean", "median", "sample_sd", "mad"):
                row[f"{metric}_{statistic}"] = metric_result.get(statistic)
        rows.append(row)
    return rows


def config_candidate_rows(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    by_candidate_level: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_candidate_level[(str(record.get("candidate_id")), str(record.get("level")))].append(record)
    candidates = sorted({candidate for candidate, _ in by_candidate_level})
    key_metrics = (
        "energy_j_per_1k_output_tokens",
        "throughput_tok_s",
        "ttft_ms",
        "tpot_ms",
        "avg_power_w",
    )
    pareto_ids = set(oracle.get("pareto_ids", []))
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        all_records = [
            record
            for (candidate_id, _), level_records in by_candidate_level.items()
            if candidate_id == candidate
            for record in level_records
        ]
        config = configuration_from(all_records[0]) if all_records else {}
        row: Dict[str, Any] = {
            "candidate_id": candidate,
            "oracle_pareto": candidate in pareto_ids,
            "max_num_seqs": config.get("max_num_seqs"),
            "max_num_batched_tokens": config.get("max_num_batched_tokens"),
            "chunked_prefill": config.get("chunked_prefill"),
            "prefix_caching": config.get("prefix_caching"),
            "tensor_parallel": config.get("tensor_parallel"),
            "pipeline_parallel": config.get("pipeline_parallel"),
            "data_parallel": config.get("data_parallel"),
        }
        for level in ("L0", "L1", "L4"):
            level_records = by_candidate_level.get((candidate, level), [])
            row[f"{level.lower()}_repeat_count"] = len(level_records)
            gpu_hours = [
                float(record["gpu_hours"])
                for record in level_records
                if numeric(record.get("gpu_hours"))
            ]
            row[f"{level.lower()}_median_gpu_hours"] = (
                statistics.median(gpu_hours) if gpu_hours else None
            )
            for metric in key_metrics:
                values = [
                    float(mapping(record.get("metrics"))[metric])
                    for record in level_records
                    if numeric(mapping(record.get("metrics")).get(metric))
                ]
                row[f"{level.lower()}_{metric}"] = (
                    statistics.median(values) if values else None
                )
        for metric in key_metrics[:4]:
            predicted = row.get(f"l0_{metric}")
            observed = row.get(f"l4_{metric}")
            row[f"l0_vs_l4_{metric}_error_pct"] = (
                None
                if not numeric(predicted) or not numeric(observed) or observed == 0
                else (float(predicted) - float(observed)) / abs(float(observed)) * 100.0
            )
        rows.append(row)
    return rows


def policy_episode_rows(
    report: Mapping[str, Any], experiment_id: str, default_budget: Optional[float]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in report.get("episodes", []):
        if not isinstance(episode, Mapping):
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "budget_gpu_hours": episode.get("budget_gpu_hours", default_budget),
                **episode,
            }
        )
    return rows


def policy_summary_rows(
    report: Mapping[str, Any], experiment_id: str, default_budget: Optional[float]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    points = report.get("points")
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, Mapping):
                continue
            for policy, metrics in mapping(point.get("aggregate")).items():
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "budget_gpu_hours": point.get("budget_gpu_hours"),
                        "exploration_budget_gpu_hours": point.get(
                            "exploration_budget_gpu_hours"
                        ),
                        "policy": policy,
                        **mapping(metrics),
                    }
                )
        return rows
    for policy, metrics in mapping(report.get("aggregate")).items():
        rows.append(
            {
                "experiment_id": experiment_id,
                "budget_gpu_hours": default_budget,
                "policy": policy,
                **mapping(metrics),
            }
        )
    return rows


def budget_response_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"experiment_id": "budget_sweep_v1", "policy": policy, **mapping(values)}
        for policy, values in sorted(mapping(report.get("budget_response")).items())
    ]


def paired_policy_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for point in report.get("points", []):
        if not isinstance(point, Mapping):
            continue
        for baseline, counts in mapping(point.get("paired_ipig_comparisons")).items():
            rows.append(
                {
                    "experiment_id": "budget_sweep_v1",
                    "budget_gpu_hours": point.get("budget_gpu_hours"),
                    "baseline_policy": baseline,
                    **mapping(counts),
                }
            )
    return rows


def planner_call_rows(
    report: Mapping[str, Any], experiment_id: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for call in report.get("calls", []):
        if not isinstance(call, Mapping):
            continue
        state = mapping(call.get("state"))
        row = {"experiment_id": experiment_id}
        for key, value in call.items():
            if key == "state":
                continue
            row[key] = value
        for key, value in state.items():
            row[f"state_{key}"] = value
        rows.append(row)
    return rows


def flatten_summary(prefix: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if not isinstance(nested_value, (dict, list)):
                    row[f"{prefix}{key}_{nested_key}"] = nested_value
        elif not isinstance(value, list):
            row[f"{prefix}{key}"] = value
    return row


def planner_summary_rows(
    report: Mapping[str, Any], experiment_id: str
) -> List[Dict[str, Any]]:
    rows = [
        {
            "experiment_id": experiment_id,
            "scope": "overall",
            **flatten_summary("", mapping(report.get("summary"))),
        }
    ]
    for category, summary in sorted(mapping(report.get("by_category")).items()):
        rows.append(
            {
                "experiment_id": experiment_id,
                "scope": "category",
                "category": category,
                **flatten_summary("", mapping(summary)),
            }
        )
    return rows


DCGM_FIELD_NAMES = {
    "SMCLK": "sm_clock_mhz",
    "MMCLK": "memory_clock_mhz",
    "TMPTR": "temperature_c",
    "POWER": "power_w",
    "TOTEC": "total_energy_mj",
    "POWINST": "instant_power_w",
    "PMLMT": "power_limit_w",
    "GPUTL": "gpu_util_pct",
    "MCUTL": "memory_util_pct",
    "DVCCTR": "device_counter",
    "PVIOL": "power_violation_counter",
    "TVIOL": "thermal_violation_counter",
}


def telemetry_record_index(
    experiment_records: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]]
) -> Dict[tuple[str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for source_id, experiment_id, records in experiment_records:
        for record in records:
            provenance = mapping(record.get("provenance"))
            telemetry_path = provenance.get("telemetry_path")
            if not telemetry_path:
                continue
            index[(source_id, Path(str(telemetry_path)).name)] = {
                "experiment_id": experiment_id,
                "candidate_id": record.get("candidate_id"),
                "level": record.get("level"),
                "repeat": provenance.get("campaign_repeat"),
                "seed": provenance.get("seed"),
                "workload_id": provenance.get("workload_id"),
                "sample_ms": provenance.get("sample_ms"),
                "energy_start_mj": provenance.get("energy_start_mj"),
                "energy_end_mj": provenance.get("energy_end_mj"),
            }
    return index


def parse_dcgm(
    archive: Archive,
    member: str,
    metadata: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    sample_index = 0
    trace_start_energy: Optional[float] = None
    field_labels: List[str] = []
    effective_metadata = dict(metadata)
    if not effective_metadata.get("experiment_id") and archive.source_id == "gemm":
        match = re.match(r"(?P<candidate>.+)-seed(?P<seed>\d+)-", Path(member).stem)
        effective_metadata.update(
            {
                "experiment_id": "gemm_pilot_unsealed",
                "candidate_id": match.group("candidate") if match else None,
                "seed": int(match.group("seed")) if match else None,
            }
        )
    for line in archive.read_text(member).splitlines():
        parts = line.split()
        if parts and parts[0] == "#Entity":
            field_labels = parts[1:]
            continue
        if len(parts) < 3 or parts[0] != "GPU" or not field_labels:
            continue
        try:
            values = [float(value) for value in parts[2 : 2 + len(field_labels)]]
        except ValueError:
            continue
        metrics = {
            DCGM_FIELD_NAMES[label]: value
            for label, value in zip(field_labels, values)
            if label in DCGM_FIELD_NAMES
        }
        total_energy = metrics.get("total_energy_mj")
        if total_energy is None:
            continue
        if trace_start_energy is None:
            trace_start_energy = total_energy
        energy_start = effective_metadata.get("energy_start_mj")
        energy_end = effective_metadata.get("energy_end_mj")
        in_window = (
            numeric(energy_start)
            and numeric(energy_end)
            and float(energy_start) <= total_energy <= float(energy_end)
        )
        sample_ms = effective_metadata.get("sample_ms")
        row: Dict[str, Any] = {
            "source_archive": archive.source_id,
            "source_member": member,
            "trace_file": Path(member).name,
            "sample_index": sample_index,
            "sample_elapsed_ms": (
                sample_index * float(sample_ms) if numeric(sample_ms) else None
            ),
            "entity": parts[0],
            "gpu_id": int(parts[1]),
            "in_measurement_window": bool(in_window),
            "energy_since_trace_start_j": (
                total_energy - trace_start_energy
            )
            / 1000.0,
            "energy_since_measurement_start_j": (
                (total_energy - float(energy_start)) / 1000.0
                if in_window and numeric(energy_start)
                else None
            ),
            **effective_metadata,
            **metrics,
        }
        yield row
        sample_index += 1


def calibration_rows(profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for point in profile.get("calibration_points", []):
        if not isinstance(point, Mapping):
            continue
        workload = mapping(point.get("workload"))
        configuration = mapping(point.get("configuration"))
        row: Dict[str, Any] = {
            "profile_id": profile.get("profile_id"),
            "point_id": point.get("id"),
            "replicate_count": point.get("replicate_count"),
            **{f"workload_{key}": value for key, value in workload.items()},
            **{f"config_{key}": value for key, value in configuration.items()},
            **{f"metric_{key}": value for key, value in mapping(point.get("metrics")).items()},
            **{
                f"dispersion_{key}": value
                for key, value in mapping(point.get("dispersion")).items()
            },
            "source_artifact": point.get("source_artifact"),
        }
        rows.append(row)
    return rows


def residual_coefficient_rows(profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = mapping(profile.get("workload_residual_model"))
    features = list(model.get("feature_names", []))
    scales = list(model.get("feature_scales", []))
    rows: List[Dict[str, Any]] = []
    for metric, result in sorted(mapping(model.get("metrics")).items()):
        coefficients = list(mapping(result).get("coefficients", []))
        for index, feature in enumerate(features):
            rows.append(
                {
                    "profile_id": profile.get("profile_id"),
                    "model_id": model.get("model_id"),
                    "metric": metric,
                    "feature_index": index,
                    "feature": feature,
                    "feature_scale": scales[index] if index < len(scales) else None,
                    "coefficient": (
                        coefficients[index] if index < len(coefficients) else None
                    ),
                    "ridge_lambda": mapping(result).get("ridge_lambda"),
                    "development_loo_mape_pct": mapping(result).get(
                        "development_loo_mape_pct"
                    ),
                    "development_loo_max_ape_pct": mapping(result).get(
                        "development_loo_max_ape_pct"
                    ),
                }
            )
    return rows


def raw_write(
    raw_dir: Path,
    output_name: str,
    data: bytes,
    origin: str,
    manifest: List[Dict[str, Any]],
) -> None:
    path = raw_dir / output_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest.append(
        {
            "file": str(path.relative_to(raw_dir.parent)),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "origin": origin,
        }
    )


def archive_manifest_rows(
    all_paths: Sequence[Path], selected: Mapping[str, Archive]
) -> List[Dict[str, Any]]:
    selected_by_path = {archive.path.resolve(): source_id for source_id, archive in selected.items()}
    rows: List[Dict[str, Any]] = []
    for path in all_paths:
        rows.append(
            {
                "source_id": selected_by_path.get(path.resolve()),
                "filename": path.name,
                "path": str(path),
                "available": True,
                "selected_for_export": path.resolve() in selected_by_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "notes": "",
            }
        )
    if "config_search_full" not in selected:
        rows.append(
            {
                "source_id": "config_search_full_raw_archive",
                "filename": "tpa-qwen7b-config-search-v1-final-20260805.tar.gz",
                "path": "not found on local workstation",
                "available": False,
                "selected_for_export": False,
                "bytes": None,
                "sha256": None,
                "notes": (
                    "The sealed replay corpus still contains all 12 L0 plus 72 "
                    "L1/L4 evidence records; only the original per-sample config-search "
                    "DCGM/client files are absent locally."
                ),
            }
        )
    rows.append(
        {
            "source_id": "calibration_measurements_raw",
            "filename": "qwen7b-calibration-c512-o128-c8.jsonl",
            "path": "not found on local workstation",
            "available": False,
            "selected_for_export": False,
            "bytes": None,
            "sha256": "5ab2b1d39d86f6f90bcf1af1197b4872ba0912b9fc3f54f1457cc68412e5b42e",
            "notes": "The three-repeat aggregate and dispersion survive in the calibration profile.",
        }
    )
    return rows


def archive_member_rows(selected: Mapping[str, Archive]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_id, archive in sorted(selected.items()):
        with tarfile.open(archive.path, "r:gz") as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                stream = handle.extractfile(member)
                if stream is None:
                    continue
                data = stream.read()
                rows.append(
                    {
                        "source_id": source_id,
                        "archive": archive.path.name,
                        "member": member.name,
                        "bytes": member.size,
                        "sha256": sha256_bytes(data),
                    }
                )
    return rows


def output_manifest_rows(
    output: Path, descriptions: Mapping[str, str], row_counts: Mapping[str, int]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for filename, description in sorted(descriptions.items()):
        path = output / filename
        if not path.exists():
            continue
        rows.append(
            {
                "file": filename,
                "rows": row_counts.get(filename),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "description": description,
            }
        )
    return rows


def write_readme(
    output: Path,
    counts: Mapping[str, int],
    validation_aggregates: Sequence[Mapping[str, Any]],
    planner_summaries: Sequence[Mapping[str, Any]],
    winner_report: Mapping[str, Any],
    config_telemetry_available: bool,
) -> None:
    energy_rows = [
        row
        for row in validation_aggregates
        if row.get("metric") == "energy_j_per_1k_output_tokens"
    ]
    energy_lines = "\n".join(
        "| {experiment_id} | {mean_absolute_percentage_error_pct:.2f} | "
        "{spearman_rank_correlation:.3f} | {pairwise_order_accuracy:.3f} |".format(
            **row
        )
        for row in energy_rows
    )
    planner_overall = [row for row in planner_summaries if row.get("scope") == "overall"]
    planner_lines = "\n".join(
        "| {experiment_id} | {call_count} | {raw_expected_subgoal_rate:.3f} | "
        "{guarded_expected_subgoal_rate:.3f} | {planner_latency_ms_median:.1f} | "
        "{planner_latency_ms_p95:.1f} |".format(**row)
        for row in planner_overall
    )
    winner_summary = mapping(winner_report.get("summary"))
    winner_ci = winner_summary.get("headline_energy_saving_ci95_pct", [])
    config_gap = (
        "- The complete configuration-search archive was found, hashed, and its "
        "DCGM traces are included in `dcgm_telemetry.csv`."
        if config_telemetry_available
        else "- The complete configuration-search archive was not downloaded to this\n"
        "  workstation. Its sealed replay corpus contains all 12 L0 predictions and all\n"
        "  72 L1/L4 measurement records, so candidate-level plots are complete; only its\n"
        "  original DCGM/client time-series files are unavailable here."
    )
    text = f"""# ServeCompass Plot Data

This directory is a self-contained, analysis-ready export of the evidence
available on the local workstation on 2026-08-05. CSV files distinguish real
H100 measurements, L0 predictions, empirical replay episodes, and planner
calls. The `raw/` directory retains the source JSON/JSONL used to build them.

## Quick Inventory

| Data layer | Rows |
|---|---:|
| GEMM H100 trials | {counts['gemm_trials.csv']} |
| Real Qwen serving trials | {counts['serving_trials.csv']} |
| Frozen serving predictions | {counts['serving_predictions.csv']} |
| Independent winner-confirmation pairs | {counts['winner_confirmation_pairs.csv']} |
| Configuration-search evidence (L0/L1/L4) | {counts['config_search_evidence.csv']} |
| Numeric evidence values in long format | {counts['evidence_metrics_long.csv']} |
| Validation workload-metric rows | {counts['workload_validation.csv']} |
| Policy replay episodes | {counts['policy_episodes.csv']} |
| Planner calls | {counts['planner_calls.csv']} |
| DCGM telemetry samples | {counts['dcgm_telemetry.csv']} |

## Headline Validation Results

| Campaign | Energy MAPE (%) | Spearman | Pairwise order accuracy |
|---|---:|---:|---:|
{energy_lines}

## Independent Winner Confirmation

Five locked, paired L4 repeats confirm the selected configuration without
refitting or reselection. Mean energy saving is
{winner_summary.get('headline_energy_saving_pct', 0):.2f}% (95% CI
[{winner_ci[0]:.2f}%, {winner_ci[1]:.2f}%); all five energy pairs favor the
winner, and median paired TTFT falls
{winner_summary.get('headline_ttft_reduction_pct', 0):.2f}%.

## Planner Results

| Benchmark | Calls | Raw accuracy | Guarded accuracy | P50 latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|---:|---:|
{planner_lines}

## Recommended Plot Inputs

- `workload_validation.csv`: predicted-versus-observed scatter plots, error by
  context/concurrency, and interval coverage.
- `serving_trials.csv`: repeat-level energy, throughput, TTFT, TPOT, and power.
- `winner_confirmation_pairs.csv`: the five seed-matched baseline-versus-winner
  comparisons and per-metric improvements.
- `winner_confirmation_summary.csv`: confidence intervals, win rates, and exact
  sign-test results for the independent confirmation.
- `config_candidate_summary.csv`: L0/L1/L4 comparison for each serving knob
  configuration and the observed oracle Pareto label.
- `policy_summary.csv` and `policy_episodes.csv`: success/cost/regret curves and
  confidence intervals over replay seeds.
- `planner_calls.csv`: latency, token usage, raw decisions, guard interventions,
  and state-dependent accuracy.
- `dcgm_telemetry.csv`: trace-level power, clocks, utilization, temperature, and
  cumulative energy. Filter `in_measurement_window == true` for the benchmark
  interval.

## Provenance and Known Gaps

- `source_manifest.csv` hashes every discovered local archive and explicitly
  marks missing artifacts.
- `archive_members.csv` hashes every member of each selected evidence archive.
- `raw_file_manifest.csv` hashes every copied or regenerated raw JSON/JSONL.
- `dataset_manifest.csv` hashes every generated table.
{config_gap}
- The original three calibration JSONL rows are not local. Their aggregate,
  MAD values, source hash, and fitted residual model are preserved in
  `calibration_points.csv` and `residual_model_coefficients.csv`.
- `raw/config-search-v1-budget-sweep-regenerated.json` was deterministically
  regenerated from the sealed replay corpus and frozen protocol. Its 10,000
  episodes reproduce every compact paper summary value exactly.

## Pandas Example

```python
import pandas as pd

df = pd.read_csv("workload_validation.csv")
energy = df[df.metric == "energy_j_per_1k_output_tokens"]
ax = energy.plot.scatter(x="observed_median", y="prediction")
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_data_dictionary(output: Path) -> None:
    text = """# Data Dictionary

## Evidence Semantics

| Value | Meaning |
|---|---|
| `L0` | CPU-side sandbox prediction. No target GPU trial was run for that row. |
| `L1` | Low-cost real H100 calibration or proxy-workload measurement. |
| `L4` | Real H100 verification on the frozen target workload and configuration. |
| `measured` | Metric comes from a hardware measurement record. |
| `predicted` / simulated kind | Metric comes from the topology/workload model. |

Do not average L0, L1, and L4 rows together. Use `level` as a facet or compare
the aligned columns in `config_candidate_summary.csv`.

## Common Identity Columns

| Column | Meaning |
|---|---|
| `experiment_id` | Stable export identifier for the campaign or benchmark. |
| `candidate_id` | Serving configuration or GEMM power-cap identifier. |
| `workload_id` | Frozen workload identifier encoding context/output/concurrency. |
| `repeat` | Campaign repeat index, normally 0, 1, or 2. |
| `seed` | Random seed recorded by the runner or replay. |
| `source_archive` | Selected evidence archive identifier. |
| `source_member` | Original path inside the source tar.gz. |
| `dataset_split` | Development, validation, holdout, or configuration-search split. |

## Serving Metrics

| Column | Unit | Meaning |
|---|---:|---|
| `energy_j` | J | GPU energy in the synchronized measurement window. |
| `energy_j_per_1k_output_tokens` | J/1k output tokens | Primary energy objective. |
| `j_per_output_token` | J/token | Same energy normalized per output token. |
| `throughput_tok_s` | output tokens/s | Output-token serving throughput. |
| `ttft_ms` | ms | P95 time to first token used by the evaluation. |
| `tpot_ms` | ms | P95 time per output token used by the evaluation. |
| `avg_power_w` | W | DCGM average GPU power during the active window. |
| `benchmark_duration_s` | s | vLLM main benchmark duration. |
| `gpu_hours` | GPU-h | Evidence acquisition cost assigned to the trial. |

The energy scope is GPU-only. Host CPU, storage, and network energy are not
included.

## Validation Tables

`workload_validation.csv` is one row per campaign, workload, and metric.

| Column | Meaning |
|---|---|
| `prediction` | Frozen prediction produced before holdout measurements. |
| `observed_median` | Median across the three hardware repeats. |
| `observations` | JSON array containing every repeat value. |
| `signed_prediction_error_pct` | `(prediction-observed)/abs(observed)*100`. |
| `absolute_percentage_error_pct` | Absolute value of signed error. |
| `prediction_interval_lower/upper` | Frozen prediction interval. |
| `interval_covers_observed_median` | Whether the observed median lies in that interval. |
| `observed_cv_pct` | Repeat coefficient of variation in percent. |

`validation_aggregate.csv` contains campaign-level MAPE, Spearman rank
correlation, pairwise ordering accuracy, repeat CV, and interval coverage.

## Configuration Search

`config_search_evidence.csv` retains every L0/L1/L4 record. The compact
`config_candidate_summary.csv` aligns median values by candidate:

- `l0_*`, `l1_*`, `l4_*`: prediction or median measurement at each level.
- `l0_vs_l4_*_error_pct`: signed L0 prediction error relative to L4.
- `oracle_pareto`: membership in the Pareto set built from median L4 evidence.
- `max_num_seqs`, `max_num_batched_tokens`, `chunked_prefill`: serving knobs.

## Independent Winner Confirmation

`winner_confirmation_pairs.csv` contains one row per seed-matched L4 pair.
The `baseline_*` and `winner_*` columns are measured values; corresponding
`improvement_*_pct` columns are positive when the locked winner improves the
metric. `winner_confirmation_summary.csv` records the paired mean and median
improvements, 95% confidence interval, win rate, and one-sided exact sign-test
p-value. `winner_confirmation_candidates.csv` contains aggregate statistics
for the two locked configurations.

## Policy Replay

`policy_episodes.csv` contains one row per replay seed, policy, and budget.
These are empirical bootstrap/replay episodes over measured evidence, not new
GPU executions.

| Column | Meaning |
|---|---|
| `success_rate` | Aggregate fraction of episodes that verify the oracle Pareto candidate. |
| `oracle_pareto_recall/precision` | Episode-level frontier recovery. |
| `spent_gpu_hours` | Exploration plus final L4 verification cost. |
| `first_oracle_hit_gpu_hours` | Cost when an oracle candidate was first acquired. |
| `best_primary_regret_pct` | Energy regret of the best verified feasible candidate. |
| `unnecessary_l4_verifications` | L4 checks not belonging to the oracle frontier. |

The five-budget benchmark uses common random numbers: the same seed selects the
same candidate-level empirical repeat for every policy and budget.

## Planner Calls

`planner_calls.csv` retains all 180 calls, including raw responses.

| Column | Meaning |
|---|---|
| `proposed_subgoal` | Parsed LLM proposal before the state guard. |
| `selected_subgoal` | Subgoal admitted after deterministic guarding. |
| `raw_expected_subgoal` | Whether the raw proposal matches the frozen expected set. |
| `guarded_expected_subgoal` | Whether the admitted subgoal matches the expected set. |
| `semantic_guard_intervened` | Whether the guard replaced the LLM proposal. |
| `planner_latency_ms` | Full planner call latency. |
| `completion_latency_ms` | Inference endpoint latency. |
| `rule_latency_ms` | Deterministic state-rule latency. |
| `state_*` | Frozen agent state shown to the planner. |

## DCGM Telemetry

`dcgm_telemetry.csv` is one row per DCGM sample and trace.

| Column | Unit | Meaning |
|---|---:|---|
| `sample_index` | sample | Order within one trace. |
| `sample_elapsed_ms` | ms | Index multiplied by the recorded sampling period. |
| `in_measurement_window` | boolean | Sample lies between synchronized energy start/end counters. |
| `power_w`, `instant_power_w` | W | DCGM power fields. |
| `total_energy_mj` | mJ | Monotonic cumulative GPU energy counter. |
| `energy_since_measurement_start_j` | J | Counter delta from the benchmark boundary. |
| `sm_clock_mhz`, `memory_clock_mhz` | MHz | GPU clocks. |
| `gpu_util_pct`, `memory_util_pct` | percent | DCGM utilization fields. |
| `temperature_c` | C | GPU temperature. |

Two early GEMM pilot traces use a shorter DCGM schema and are labeled
`gemm_pilot_unsealed`. Missing fields are empty; their available samples are
still retained.

## Null Values

An empty CSV cell means the source did not define that field. It does not mean
zero. Examples include unavailable prediction intervals for diagnostics,
missing first-hit cost in unsuccessful policy episodes, and unavailable DCGM
fields in the early pilot schema.
"""
    (output / "DATA_DICTIONARY.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    raw_dir = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_archive_paths = discover_archives(args.downloads.expanduser().resolve())
    archive_names = {
        "gemm": "tpa-h100-gemm-20260803.tar.gz",
        "workload_v1": "tpa-qwen7b-workload-transfer-v1-20260804.tar.gz",
        "holdout_v2": "tpa-qwen7b-holdout-v2-final-20260804.tar.gz",
        "scope_v3": "tpa-qwen7b-scope-v3-final-20260804.tar.gz",
        "policy_v1": "tpa-qwen7b-policy-benchmark-v1-20260805.tar.gz",
        "planner": "tpa-qwen7b-planner-guard-v1-v2-20260805.tar.gz",
        "policy_bootstrap_v2": "tpa-qwen7b-policy-bootstrap-v2-20260805.tar.gz",
        "winner_confirmation": "tpa-qwen7b-winner-confirmation-v1-final-20260805.tar.gz",
    }
    archives = {
        source_id: Archive(
            source_id,
            choose_archive(all_archive_paths, filename, args.downloads.expanduser()),
        )
        for source_id, filename in archive_names.items()
    }
    optional_config_archive = "tpa-qwen7b-config-search-v1-final-20260805.tar.gz"
    if any(path.name == optional_config_archive for path in all_archive_paths):
        archives["config_search_full"] = Archive(
            "config_search_full",
            choose_archive(
                all_archive_paths,
                optional_config_archive,
                args.downloads.expanduser(),
            ),
        )

    members = {
        "gemm": "experiments/results/h100-gemm-power-sweep.jsonl",
        "workload_measurements": "experiments/results/workload-transfer-validation-v1-measurements.jsonl",
        "workload_predictions": "experiments/results/workload-transfer-validation-v1-predictions.jsonl",
        "workload_report": "experiments/results/workload-transfer-validation-v1-report.json",
        "holdout_measurements": "experiments/results/workload-transfer-holdout-v2-measurements.jsonl",
        "holdout_predictions": "experiments/results/workload-transfer-holdout-v2-predictions.jsonl",
        "holdout_report": "experiments/results/workload-transfer-holdout-v2-report.json",
        "profile": "experiments/results/qwen2.5-7b-h100-workload-v2.json",
        "profile_fit": "experiments/results/qwen2.5-7b-h100-workload-v2-fit.json",
        "scope_measurements": "experiments/results/scope-confirmation-v3-measurements.jsonl",
        "scope_predictions": "experiments/results/scope-confirmation-v3-predictions.jsonl",
        "scope_report": "experiments/results/scope-confirmation-v3-report.json",
        "scope_decision": "experiments/results/scope-confirmation-v3-preregistered-decision.json",
        "scenario": "experiments/results/config-search-v1-scenario.json",
        "config_corpus": "experiments/results/config-search-v1-replay-corpus.jsonl",
        "policy_v1": "experiments/results/config-search-v1-policy-benchmark.json",
        "policy_bootstrap_v2": "experiments/results/config-search-v1-policy-bootstrap-v2.json",
        "planner_v1": "experiments/results/planner-conformance-v1.json",
        "planner_v2": "experiments/results/planner-conformance-v2-holdout.json",
        "planner_v1_diagnostics": "experiments/results/planner-conformance-v1-diagnostics.json",
        "planner_v2_diagnostics": "experiments/results/planner-conformance-v2-holdout-diagnostics.json",
        "winner_measurements": "experiments/results/winner-confirmation-v1-measurements.jsonl",
        "winner_predictions": "experiments/results/winner-confirmation-v1-l0-predictions.jsonl",
        "winner_report": "experiments/results/winner-confirmation-v1-report.json",
        "winner_scenario": "experiments/results/winner-confirmation-v1-scenario.json",
        "winner_schedule": "experiments/results/winner-confirmation-v1-schedule.json",
        "winner_freeze_summary": "experiments/results/winner-confirmation-v1-freeze-summary.json",
    }

    gemm_records = archives["gemm"].read_jsonl(members["gemm"])
    workload_records = archives["workload_v1"].read_jsonl(members["workload_measurements"])
    workload_predictions = archives["workload_v1"].read_jsonl(members["workload_predictions"])
    workload_report = archives["workload_v1"].read_json(members["workload_report"])
    holdout_records = archives["holdout_v2"].read_jsonl(members["holdout_measurements"])
    holdout_predictions = archives["holdout_v2"].read_jsonl(members["holdout_predictions"])
    holdout_report = archives["holdout_v2"].read_json(members["holdout_report"])
    profile = archives["holdout_v2"].read_json(members["profile"])
    scope_records = archives["scope_v3"].read_jsonl(members["scope_measurements"])
    scope_predictions = archives["scope_v3"].read_jsonl(members["scope_predictions"])
    scope_report = archives["scope_v3"].read_json(members["scope_report"])
    config_records = archives["policy_v1"].read_jsonl(members["config_corpus"])
    scenario = archives["policy_v1"].read_json(members["scenario"])
    policy_v1 = archives["policy_v1"].read_json(members["policy_v1"])
    policy_bootstrap_v2 = archives["policy_bootstrap_v2"].read_json(
        members["policy_bootstrap_v2"]
    )
    planner_v1 = archives["planner"].read_json(members["planner_v1"])
    planner_v2 = archives["planner"].read_json(members["planner_v2"])
    winner_records = archives["winner_confirmation"].read_jsonl(
        members["winner_measurements"]
    )
    winner_report = archives["winner_confirmation"].read_json(
        members["winner_report"]
    )

    budget_path = args.budget_report
    if budget_path is None:
        existing = raw_dir / "config-search-v1-budget-sweep-regenerated.json"
        budget_path = existing if existing.exists() else None
    if budget_path is None or not budget_path.exists():
        raise FileNotFoundError(
            "full budget report missing; pass --budget-report with the regenerated JSON"
        )
    budget_report = json.loads(budget_path.read_text(encoding="utf-8"))
    if not isinstance(budget_report, dict):
        raise ValueError("budget report must be a JSON object")

    raw_manifest: List[Dict[str, Any]] = []
    raw_specs = [
        ("gemm", members["gemm"], "h100-gemm-power-sweep.jsonl"),
        ("workload_v1", members["workload_measurements"], "workload-transfer-v1-measurements.jsonl"),
        ("workload_v1", members["workload_predictions"], "workload-transfer-v1-predictions.jsonl"),
        ("workload_v1", members["workload_report"], "workload-transfer-v1-report.json"),
        ("holdout_v2", members["holdout_measurements"], "workload-holdout-v2-measurements.jsonl"),
        ("holdout_v2", members["holdout_predictions"], "workload-holdout-v2-predictions.jsonl"),
        ("holdout_v2", members["holdout_report"], "workload-holdout-v2-report.json"),
        ("holdout_v2", members["profile"], "qwen2.5-7b-h100-workload-v2.json"),
        ("holdout_v2", members["profile_fit"], "qwen2.5-7b-h100-workload-v2-fit.json"),
        ("scope_v3", members["scope_measurements"], "scope-confirmation-v3-measurements.jsonl"),
        ("scope_v3", members["scope_predictions"], "scope-confirmation-v3-predictions.jsonl"),
        ("scope_v3", members["scope_report"], "scope-confirmation-v3-report.json"),
        ("scope_v3", members["scope_decision"], "scope-confirmation-v3-decision.json"),
        ("policy_v1", members["scenario"], "config-search-v1-scenario.json"),
        ("policy_v1", members["config_corpus"], "config-search-v1-replay-corpus.jsonl"),
        ("policy_v1", members["policy_v1"], "config-search-v1-policy-benchmark.json"),
        ("policy_bootstrap_v2", members["policy_bootstrap_v2"], "config-search-v1-policy-bootstrap-v2.json"),
        ("planner", members["planner_v1"], "planner-conformance-v1.json"),
        ("planner", members["planner_v2"], "planner-conformance-v2-holdout.json"),
        ("planner", members["planner_v1_diagnostics"], "planner-conformance-v1-diagnostics.json"),
        ("planner", members["planner_v2_diagnostics"], "planner-conformance-v2-holdout-diagnostics.json"),
        ("winner_confirmation", members["winner_measurements"], "winner-confirmation-v1-measurements.jsonl"),
        ("winner_confirmation", members["winner_predictions"], "winner-confirmation-v1-l0-predictions.jsonl"),
        ("winner_confirmation", members["winner_report"], "winner-confirmation-v1-report.json"),
        ("winner_confirmation", members["winner_scenario"], "winner-confirmation-v1-scenario.json"),
        ("winner_confirmation", members["winner_schedule"], "winner-confirmation-v1-schedule.json"),
        ("winner_confirmation", members["winner_freeze_summary"], "winner-confirmation-v1-freeze-summary.json"),
    ]
    for source_id, member, output_name in raw_specs:
        raw_write(
            raw_dir,
            output_name,
            archives[source_id].read_bytes(member),
            f"{archives[source_id].path.name}:{member}",
            raw_manifest,
        )
    raw_write(
        raw_dir,
        "config-search-v1-budget-sweep-regenerated.json",
        budget_path.read_bytes(),
        "deterministic replay from frozen protocol and sealed corpus",
        raw_manifest,
    )

    gemm_wide = [
        evidence_row(record, "gemm_power_sweep", "gemm", members["gemm"])
        for record in gemm_records
    ]
    serving_sources = [
        ("workload_transfer_v1", "workload_v1", members["workload_measurements"], workload_records),
        ("workload_holdout_v2", "holdout_v2", members["holdout_measurements"], holdout_records),
        ("scope_confirmation_v3", "scope_v3", members["scope_measurements"], scope_records),
        ("winner_confirmation_v1", "winner_confirmation", members["winner_measurements"], winner_records),
    ]
    serving_wide = [
        evidence_row(record, experiment_id, source_id, member)
        for experiment_id, source_id, member, records in serving_sources
        for record in records
    ]
    prediction_sources = [
        ("workload_transfer_v1", "workload_v1", members["workload_predictions"], workload_predictions),
        ("workload_holdout_v2", "holdout_v2", members["holdout_predictions"], holdout_predictions),
        ("scope_confirmation_v3", "scope_v3", members["scope_predictions"], scope_predictions),
    ]
    serving_prediction_wide = [
        evidence_row(record, experiment_id, source_id, member)
        for experiment_id, source_id, member, records in prediction_sources
        for record in records
    ]
    config_wide = [
        evidence_row(record, "config_search_v1", "policy_v1", members["config_corpus"])
        for record in config_records
    ]
    evidence_long = evidence_long_rows(
        gemm_records, "gemm_power_sweep", "gemm", members["gemm"]
    )
    for experiment_id, source_id, member, records in [*serving_sources, *prediction_sources]:
        evidence_long.extend(
            evidence_long_rows(records, experiment_id, source_id, member)
        )
    evidence_long.extend(
        evidence_long_rows(
            config_records, "config_search_v1", "policy_v1", members["config_corpus"]
        )
    )

    serving_summary: List[Dict[str, Any]] = []
    for experiment_id, _, _, records in serving_sources:
        serving_summary.extend(
            grouped_metric_summary(records, experiment_id, ("workload_id",))
        )
    config_level_summary = grouped_metric_summary(
        config_records, "config_search_v1", ("candidate_id", "level")
    )
    config_candidates = config_candidate_rows(config_records, mapping(policy_v1.get("oracle")))
    winner_pairs = winner_confirmation_pair_rows(winner_report)
    winner_summaries = winner_confirmation_summary_rows(winner_report)
    winner_candidates = winner_confirmation_candidate_rows(winner_report)

    validation = []
    validation_aggregates = []
    for experiment_id, report in (
        ("workload_transfer_v1", workload_report),
        ("workload_holdout_v2", holdout_report),
        ("scope_confirmation_v3", scope_report),
    ):
        validation.extend(validation_rows(report, experiment_id))
        validation_aggregates.extend(
            validation_aggregate_rows(report, experiment_id)
        )

    default_budget = float(mapping(scenario.get("budget")).get("gpu_hours", 0.12))
    policy_reports = (
        ("policy_benchmark_v1", policy_v1, default_budget),
        ("policy_bootstrap_v2", policy_bootstrap_v2, default_budget),
        ("budget_sweep_v1", budget_report, None),
    )
    policy_episodes = [
        row
        for experiment_id, report, budget in policy_reports
        for row in policy_episode_rows(report, experiment_id, budget)
    ]
    policy_summaries = [
        row
        for experiment_id, report, budget in policy_reports
        for row in policy_summary_rows(report, experiment_id, budget)
    ]

    planner_calls = [
        *planner_call_rows(planner_v1, "planner_conformance_v1_development"),
        *planner_call_rows(planner_v2, "planner_conformance_v2_holdout"),
    ]
    planner_summaries = [
        *planner_summary_rows(planner_v1, "planner_conformance_v1_development"),
        *planner_summary_rows(planner_v2, "planner_conformance_v2_holdout"),
    ]

    telemetry_index = telemetry_record_index(
        [
            ("gemm", "gemm_power_sweep", gemm_records),
            ("workload_v1", "workload_transfer_v1", workload_records),
            ("holdout_v2", "workload_holdout_v2", holdout_records),
            ("scope_v3", "scope_confirmation_v3", scope_records),
            ("winner_confirmation", "winner_confirmation_v1", winner_records),
            *(
                [("config_search_full", "config_search_v1", config_records)]
                if "config_search_full" in archives
                else []
            ),
        ]
    )
    telemetry_rows: List[Dict[str, Any]] = []
    telemetry_sources = [
        "gemm",
        "workload_v1",
        "holdout_v2",
        "scope_v3",
        "winner_confirmation",
    ]
    if "config_search_full" in archives:
        telemetry_sources.append("config_search_full")
    for source_id in telemetry_sources:
        archive = archives[source_id]
        for member in archive.matching_members(".dcgm.txt"):
            metadata = telemetry_index.get((source_id, Path(member).name), {})
            telemetry_rows.extend(parse_dcgm(archive, member, metadata))

    archive_rows = archive_manifest_rows(all_archive_paths, archives)
    member_rows = archive_member_rows(archives)
    calibration = calibration_rows(profile)
    residual_coefficients = residual_coefficient_rows(profile)

    catalog = [
        {"experiment_id": "gemm_power_sweep", "category": "hardware_smoke", "evidence": "real_H100", "record_count": len(gemm_records), "unique_items": len({r.get('candidate_id') for r in gemm_records}), "notes": "FP16 cuBLAS GEMM at 350/500/700 W"},
        {"experiment_id": "workload_transfer_v1", "category": "workload_validation", "evidence": "real_H100", "record_count": len(workload_records), "unique_items": len(workload_report.get('workloads', [])), "notes": "Development workload-transfer campaign"},
        {"experiment_id": "workload_holdout_v2", "category": "blind_holdout", "evidence": "real_H100", "record_count": len(holdout_records), "unique_items": len(holdout_report.get('workloads', [])), "notes": "Preregistered final workload holdout"},
        {"experiment_id": "scope_confirmation_v3", "category": "scope_holdout", "evidence": "real_H100", "record_count": len(scope_records), "unique_items": len(scope_report.get('workloads', [])), "notes": "Preregistered scope and abstention confirmation"},
        {"experiment_id": "winner_confirmation_v1", "category": "independent_configuration_confirmation", "evidence": "paired_real_H100_L4", "record_count": len(winner_records), "unique_items": len(winner_pairs), "notes": "Five locked alternating pairs; no refit or reselection"},
        {"experiment_id": "config_search_v1", "category": "serving_configuration_search", "evidence": "L0_prediction_and_real_H100", "record_count": len(config_records), "unique_items": len(config_candidates), "notes": "12 candidates; 12 L0 plus 36 L1 and 36 L4 records"},
        {"experiment_id": "policy_benchmark_v1", "category": "agent_policy_replay", "evidence": "empirical_replay", "record_count": len(policy_v1.get('episodes', [])), "unique_items": len(mapping(policy_v1.get('aggregate'))), "notes": "Initial fixed-budget replay"},
        {"experiment_id": "policy_bootstrap_v2", "category": "agent_policy_replay", "evidence": "candidate_level_bootstrap", "record_count": len(policy_bootstrap_v2.get('episodes', [])), "unique_items": len(mapping(policy_bootstrap_v2.get('aggregate'))), "notes": "Candidate-level empirical bootstrap"},
        {"experiment_id": "budget_sweep_v1", "category": "agent_budget_response", "evidence": "deterministic_empirical_replay", "record_count": len(budget_report.get('episodes', [])), "unique_items": len(budget_report.get('points', [])), "notes": "Five budgets and four policies"},
        {"experiment_id": "planner_conformance_v1_development", "category": "planner_conformance", "evidence": "real_Qwen_planner", "record_count": len(planner_v1.get('calls', [])), "unique_items": len(mapping(planner_v1.get('by_category'))), "notes": "Development protocol"},
        {"experiment_id": "planner_conformance_v2_holdout", "category": "planner_conformance", "evidence": "real_Qwen_planner", "record_count": len(planner_v2.get('calls', [])), "unique_items": len(mapping(planner_v2.get('by_category'))), "notes": "Disjoint holdout with deterministic state guard"},
    ]

    descriptions = {
        "source_manifest.csv": "All discovered evidence archives, hashes, selection status, and known missing raw artifacts.",
        "archive_members.csv": "Per-member size and SHA-256 for every selected archive.",
        "raw_file_manifest.csv": "Hashes and origins for self-contained raw JSON/JSONL copies.",
        "experiment_catalog.csv": "One row per experiment with evidence semantics and record counts.",
        "gemm_trials.csv": "Nine repeat-level H100 cuBLAS GEMM power-cap trials.",
        "serving_trials.csv": "All 69 repeat-level real Qwen serving measurements across three workload campaigns.",
        "serving_predictions.csv": "Frozen L0 workload predictions for the three validation campaigns.",
        "serving_workload_summary.csv": "Long-format descriptive statistics by workload and metric.",
        "workload_validation.csv": "Prediction, observed repeat values, error, and interval coverage by workload and metric.",
        "validation_aggregate.csv": "Campaign-level MAPE, rank correlation, pairwise accuracy, and interval coverage.",
        "config_search_evidence.csv": "All 84 L0/L1/L4 configuration-search evidence records.",
        "config_search_level_summary.csv": "Descriptive statistics for each candidate, evidence level, and metric.",
        "config_candidate_summary.csv": "Wide L0/L1/L4 comparison with L0-vs-L4 errors and oracle Pareto label.",
        "winner_confirmation_pairs.csv": "Five seed-matched L4 baseline-versus-winner pairs and per-metric improvements.",
        "winner_confirmation_summary.csv": "Paired effect sizes, confidence intervals, win rates, and exact sign tests.",
        "winner_confirmation_candidates.csv": "Aggregate statistics for the locked expert baseline and selected winner.",
        "evidence_metrics_long.csv": "Every numeric evidence metric in tidy long format.",
        "calibration_points.csv": "Measured calibration aggregate and dispersion retained in the fitted profile.",
        "residual_model_coefficients.csv": "Workload-residual features, scales, coefficients, and development LOO errors.",
        "policy_episodes.csv": "All 14,000 policy replay episodes from v1, bootstrap v2, and the five-budget sweep.",
        "policy_summary.csv": "Policy aggregate metrics by benchmark and budget.",
        "policy_budget_response.csv": "Normalized budget-response AUC and monotonicity diagnostics.",
        "policy_paired_ipig.csv": "Matched-seed IPIG-versus-baseline success contingency counts by budget.",
        "planner_calls.csv": "All 180 planner calls with state, raw/guarded decisions, latency, and token usage.",
        "planner_summary.csv": "Overall and per-category planner conformance summaries.",
        "dcgm_telemetry.csv": "All locally available per-sample DCGM power, clock, utilization, temperature, and energy traces.",
    }
    row_counts: Dict[str, int] = {}

    def emit(filename: str, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> None:
        row_counts[filename] = write_csv(output / filename, rows, preferred)

    evidence_preferred = (
        "experiment_id", "candidate_id", "workload_id", "level", "kind", "status",
        "repeat", "seed", "input_tokens", "output_tokens", "concurrency",
        "max_num_seqs", "max_num_batched_tokens", "chunked_prefill", "gpu_hours",
    )
    emit("source_manifest.csv", archive_rows, ("source_id", "filename", "available", "selected_for_export"))
    emit("archive_members.csv", member_rows, ("source_id", "archive", "member", "bytes", "sha256"))
    emit("raw_file_manifest.csv", raw_manifest, ("file", "bytes", "sha256", "origin"))
    emit("experiment_catalog.csv", catalog, ("experiment_id", "category", "evidence", "record_count", "unique_items", "notes"))
    emit("gemm_trials.csv", gemm_wide, evidence_preferred)
    emit("serving_trials.csv", serving_wide, evidence_preferred)
    emit("serving_predictions.csv", serving_prediction_wide, evidence_preferred)
    emit("serving_workload_summary.csv", serving_summary, ("experiment_id", "workload_id", "metric", "unit", "n", "mean", "median"))
    emit("workload_validation.csv", validation, ("experiment_id", "workload_id", "input_tokens", "concurrency", "metric_group", "metric", "prediction", "observed_median"))
    emit("validation_aggregate.csv", validation_aggregates, ("experiment_id", "metric", "mean_absolute_percentage_error_pct", "spearman_rank_correlation", "pairwise_order_accuracy"))
    emit("config_search_evidence.csv", config_wide, evidence_preferred)
    emit("config_search_level_summary.csv", config_level_summary, ("experiment_id", "candidate_id", "level", "metric", "unit", "n", "mean", "median"))
    emit("config_candidate_summary.csv", config_candidates, ("candidate_id", "oracle_pareto", "max_num_seqs", "max_num_batched_tokens", "chunked_prefill"))
    emit("winner_confirmation_pairs.csv", winner_pairs, ("experiment_id", "pair_index", "seed", "run_order", "baseline_id", "winner_id"))
    emit("winner_confirmation_summary.csv", winner_summaries, ("experiment_id", "metric", "direction", "pair_count", "mean_improvement_pct", "median_improvement_pct"))
    emit("winner_confirmation_candidates.csv", winner_candidates, ("experiment_id", "candidate_id", "role", "repeat_count", "gpu_hours_total"))
    emit("evidence_metrics_long.csv", evidence_long, ("experiment_id", "candidate_id", "workload_id", "level", "repeat", "seed", "metric", "value", "unit"))
    emit("calibration_points.csv", calibration, ("profile_id", "point_id", "replicate_count"))
    emit("residual_model_coefficients.csv", residual_coefficients, ("profile_id", "model_id", "metric", "feature_index", "feature", "coefficient"))
    emit("policy_episodes.csv", policy_episodes, ("experiment_id", "budget_gpu_hours", "policy", "seed", "status", "spent_gpu_hours"))
    emit("policy_summary.csv", policy_summaries, ("experiment_id", "budget_gpu_hours", "policy", "episode_count", "success_rate"))
    emit("policy_budget_response.csv", budget_response_rows(budget_report), ("experiment_id", "policy", "normalized_success_auc", "maximum_success_rate"))
    emit("policy_paired_ipig.csv", paired_policy_rows(budget_report), ("experiment_id", "budget_gpu_hours", "baseline_policy"))
    emit("planner_calls.csv", planner_calls, ("experiment_id", "category", "case_id", "repeat", "proposed_subgoal", "selected_subgoal", "semantic_guard_intervened", "planner_latency_ms"))
    emit("planner_summary.csv", planner_summaries, ("experiment_id", "scope", "category", "call_count", "raw_expected_subgoal_rate", "guarded_expected_subgoal_rate"))
    emit("dcgm_telemetry.csv", telemetry_rows, ("experiment_id", "candidate_id", "workload_id", "level", "repeat", "seed", "trace_file", "sample_index", "sample_elapsed_ms", "in_measurement_window", "power_w", "total_energy_mj"))

    expected_counts = {
        "gemm_trials.csv": 9,
        "serving_trials.csv": 79,
        "serving_predictions.csv": 23,
        "config_search_evidence.csv": 84,
        "winner_confirmation_pairs.csv": 5,
        "winner_confirmation_summary.csv": 4,
        "winner_confirmation_candidates.csv": 2,
        "policy_episodes.csv": 14000,
        "planner_calls.csv": 180,
    }
    mismatches = {
        name: (row_counts.get(name), expected)
        for name, expected in expected_counts.items()
        if row_counts.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"unexpected export row counts: {mismatches}")

    compact_budget = json.loads(
        (REPO_ROOT / "paper-agenticai4hpc26" / "results" / "budget-response-summary.json").read_text(
            encoding="utf-8"
        )
    )
    if budget_report.get("budget_response") != compact_budget.get("budget_response"):
        raise ValueError("regenerated budget response differs from the paper summary")
    compact_points = compact_budget.get("points", [])
    full_points = budget_report.get("points", [])
    for compact, full in zip(compact_points, full_points):
        if compact.get("budget_gpu_hours") != full.get("budget_gpu_hours") or compact.get("aggregate") != full.get("aggregate"):
            raise ValueError("regenerated budget point differs from the paper summary")

    write_readme(
        output,
        row_counts,
        validation_aggregates,
        planner_summaries,
        winner_report,
        config_telemetry_available="config_search_full" in archives,
    )
    write_data_dictionary(output)
    descriptions["README.md"] = "Data dictionary, inventory, plotting guidance, provenance, and known gaps."
    descriptions["DATA_DICTIONARY.md"] = "Column semantics, evidence levels, metric units, and null handling."
    dataset_manifest = output_manifest_rows(output, descriptions, row_counts)
    row_counts["dataset_manifest.csv"] = write_csv(
        output / "dataset_manifest.csv",
        dataset_manifest,
        ("file", "rows", "bytes", "sha256", "description"),
    )

    print(json.dumps({"output": str(output), "row_counts": row_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
