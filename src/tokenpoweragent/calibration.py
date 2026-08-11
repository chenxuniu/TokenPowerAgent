"""Build immutable topology-sandbox calibration profiles from L1 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord, EvidenceStatus
from tokenpoweragent.executors.serving import (
    SandboxExecutionError,
    parse_vllm_server_configuration,
)
from tokenpoweragent.schema import EvidenceLevel
from tokenpoweragent.twin.topology import CalibrationError, CalibrationProfile


class CalibrationBuildError(CalibrationError):
    """Raised when serving evidence cannot form an auditable profile."""


_CALIBRATION_METRICS = (
    "throughput_tok_s",
    "ttft_ms",
    "tpot_ms",
    "avg_power_w",
    "energy_j_per_1k_tokens",
    "j_per_output_token",
    "j_per_total_token",
)


@dataclass(frozen=True)
class _Observation:
    line_number: int
    record: EvidenceRecord
    workload: Mapping[str, Any]
    configuration: Mapping[str, Any]

    @property
    def group_key(self) -> str:
        return json.dumps(
            {
                "workload": self.workload,
                "configuration": self.configuration,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_configuration(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = provenance.get("server_configuration")
    if isinstance(explicit, Mapping):
        config = dict(explicit)
    else:
        command = provenance.get("server_command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise CalibrationBuildError(
                "record lacks server_configuration and server_command"
            )
        try:
            config = dict(parse_vllm_server_configuration(command))
        except SandboxExecutionError as exc:
            raise CalibrationBuildError(str(exc)) from exc
    if config.get("prefix_caching") is not False:
        raise CalibrationBuildError(
            "calibration requires explicit prefix_caching=false"
        )
    return config


def _workload(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = provenance.get("workload")
    if not isinstance(raw, Mapping):
        raise CalibrationBuildError("record lacks a structured workload")
    aliases = {
        "input_tokens": ("input_tokens", "input_len"),
        "output_tokens": ("output_tokens", "output_len"),
        "concurrency": ("concurrency", "max_concurrency"),
        "num_requests": ("num_requests", "num_prompts"),
    }
    parsed: Dict[str, Any] = {}
    for target, names in aliases.items():
        value = next((raw[name] for name in names if name in raw), None)
        try:
            parsed[target] = int(value)
        except (TypeError, ValueError) as exc:
            raise CalibrationBuildError(
                "record workload lacks positive %s" % target
            ) from exc
        if parsed[target] < 1:
            raise CalibrationBuildError(
                "record workload has invalid %s" % target
            )
    parsed["request_rate"] = raw.get(
        "request_rate_req_s", raw.get("request_rate", "inf")
    )
    return parsed


def _load_observations(
    records_path: Path,
    power_limit_w: float,
) -> Tuple[List[_Observation], List[str]]:
    observations = []
    skipped = []
    for line_number, line in enumerate(
        Path(records_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record = EvidenceRecord.from_dict(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationBuildError(
                "invalid evidence at line %d: %s" % (line_number, exc)
            ) from exc
        if (
            record.level != EvidenceLevel.L1
            or record.kind != EvidenceKind.MEASURED
            or record.status != EvidenceStatus.SUCCEEDED
        ):
            skipped.append("line %d: not successful measured L1 evidence" % line_number)
            continue
        try:
            record_power = float(record.provenance["power_limit_readback_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationBuildError(
                "line %d lacks power_limit_readback_w" % line_number
            ) from exc
        if abs(record_power - power_limit_w) > 1.0:
            skipped.append(
                "line %d: power limit %.1f W does not match %.1f W"
                % (line_number, record_power, power_limit_w)
            )
            continue
        missing = [
            metric
            for metric in _CALIBRATION_METRICS[:4]
            if metric not in record.metrics
            or not math.isfinite(record.metrics[metric])
            or record.metrics[metric] <= 0
        ]
        if missing:
            raise CalibrationBuildError(
                "line %d lacks calibration metrics: %s"
                % (line_number, ", ".join(missing))
            )
        observations.append(
            _Observation(
                line_number=line_number,
                record=record,
                workload=_workload(record.provenance),
                configuration=_server_configuration(record.provenance),
            )
        )
    if not observations:
        raise CalibrationBuildError(
            "no successful L1 records match %.1f W" % power_limit_w
        )
    return observations, skipped


def _median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def build_serving_calibration_profile(
    records_path: Path,
    template_path: Path,
    profile_id: str,
    power_limit_w: float,
    min_repeats: int = 3,
    publication_eligible: bool = False,
) -> Mapping[str, Any]:
    """Aggregate repeated serving records into a validated profile mapping."""

    if not profile_id.strip():
        raise CalibrationBuildError("profile_id cannot be empty")
    if power_limit_w <= 0 or min_repeats < 1:
        raise CalibrationBuildError("power_limit_w and min_repeats must be positive")
    try:
        template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationBuildError("cannot load template %s" % template_path) from exc
    if not isinstance(template, Mapping):
        raise CalibrationBuildError("calibration template root must be an object")

    observations, skipped = _load_observations(records_path, power_limit_w)
    template_model = dict(template.get("model_architecture", {}))
    template_model_id = str(template_model.get("model_id", ""))
    mismatched_models = [
        observation.line_number
        for observation in observations
        if observation.record.provenance.get("model") not in {None, template_model_id}
    ]
    if mismatched_models:
        raise CalibrationBuildError(
            "record model does not match template at lines %s"
            % ",".join(str(line) for line in mismatched_models)
        )
    groups: Dict[str, List[_Observation]] = {}
    for observation in observations:
        groups.setdefault(observation.group_key, []).append(observation)

    digest = _sha256(records_path)
    points = []
    insufficient = []
    for group_key in sorted(groups):
        group = groups[group_key]
        first = group[0]
        if len(group) < min_repeats:
            insufficient.append(
                "%s has %d/%d repeats"
                % (first.record.candidate_id, len(group), min_repeats)
            )
        metrics: Dict[str, float] = {}
        dispersion: Dict[str, float] = {}
        for metric in _CALIBRATION_METRICS:
            values = [
                float(observation.record.metrics[metric])
                for observation in group
                if metric in observation.record.metrics
            ]
            if not values:
                continue
            metrics[metric] = float(statistics.median(values))
            dispersion[metric + "_mad"] = float(
                _median_absolute_deviation(values)
            )
        point_hash = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
        lines = [observation.line_number for observation in group]
        points.append(
            {
                "id": "l1-%s" % point_hash,
                "source_artifact": "%s@sha256:%s#lines=%s"
                % (
                    Path(records_path),
                    digest,
                    ",".join(str(line) for line in lines),
                ),
                "source_record_lines": lines,
                "candidate_ids": sorted(
                    {observation.record.candidate_id for observation in group}
                ),
                "replicate_count": len(group),
                "workload": dict(first.workload),
                "configuration": {
                    **dict(first.configuration),
                    "power_limit_w": power_limit_w,
                },
                "metrics": metrics,
                "dispersion": dispersion,
            }
        )

    if publication_eligible and insufficient:
        raise CalibrationBuildError(
            "publication profile lacks required repeats: %s"
            % "; ".join(insufficient)
        )
    if publication_eligible:
        topology_source = str(
            dict(template.get("hardware", {})).get("topology_source", "")
        ).lower()
        if not topology_source or any(
            marker in topology_source
            for marker in ("synthetic", "unspecified", "replace")
        ):
            raise CalibrationBuildError(
                "publication profile requires a measured topology_source"
            )
        if template.get("uncertainty_calibrated") is not True:
            raise CalibrationBuildError(
                "publication profile requires held-out uncertainty calibration"
            )
        wrong_split = [
            observation.line_number
            for observation in observations
            if observation.record.provenance.get("dataset_split") != "calibration"
        ]
        if wrong_split:
            raise CalibrationBuildError(
                "publication profile accepts only dataset_split=calibration; lines %s"
                % ",".join(str(line) for line in wrong_split)
            )
        missing_model = [
            observation.line_number
            for observation in observations
            if observation.record.provenance.get("model") != template_model_id
        ]
        if missing_model:
            raise CalibrationBuildError(
                "publication profile requires an exact model id at lines %s"
                % ",".join(str(line) for line in missing_model)
            )
        unpinned = [
            observation.line_number
            for observation in observations
            if observation.configuration.get("model_revision") in {None, "unrecorded"}
            or observation.configuration.get("tokenizer_revision")
            in {None, "unrecorded"}
        ]
        if unpinned:
            raise CalibrationBuildError(
                "publication profile has unpinned model/tokenizer revisions at lines %s"
                % ",".join(str(line) for line in unpinned)
            )
    result = dict(template)
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "generated_by": "tokenpoweragent build-calibration",
            "source_records": str(Path(records_path)),
            "source_records_sha256": digest,
            "fixed_power_limit_w": power_limit_w,
            "minimum_repeats": min_repeats,
            "matching_records": len(observations),
            "calibration_groups": len(points),
            "skipped_records": skipped,
            "insufficient_repeat_groups": insufficient,
        }
    )
    result.update(
        {
            "schema_version": str(result.get("schema_version", "1.0")),
            "profile_id": profile_id,
            "publication_eligible": publication_eligible,
            "metadata": metadata,
            "calibration_points": points,
        }
    )
    CalibrationProfile.from_mapping(result)
    return result
