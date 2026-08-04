"""Preregistered serving-configuration search campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.executors.topology import (
    TopologySandboxError,
    TopologySandboxExecutor,
)
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario, SchemaError
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
    ServingConfiguration,
)


class ConfigurationCampaignError(ValueError):
    """Raised when a configuration-search campaign violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationCampaignError(
            "%s must be a positive integer" % key
        ) from exc
    if value < 1:
        raise ConfigurationCampaignError("%s must be a positive integer" % key)
    return value


def _positive_float(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationCampaignError(
            "%s must be a positive number" % key
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationCampaignError("%s must be a positive number" % key)
    return value


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationCampaignError("campaign.%s must be an object" % key)
    return value


@dataclass(frozen=True)
class ConfigurationPoint:
    point_id: str
    configuration: Mapping[str, Any]
    role: str = "candidate"


@dataclass(frozen=True)
class ConfigurationCampaign:
    schema_version: str
    campaign_id: str
    intent: str
    model: Mapping[str, Any]
    hardware: Mapping[str, Any]
    runtime: Mapping[str, Any]
    target_workload: InferenceWorkload
    probe_requests: int
    objectives: Mapping[str, str]
    slo: Mapping[str, float]
    budget_gpu_hours: float
    verify_top_k: int
    level_cost_gpu_hours: Mapping[EvidenceLevel, float]
    power_limit_w: float
    repeats: int
    num_warmups: int
    sample_ms: int
    schedule_seed: int
    candidates: Tuple[ConfigurationPoint, ...]

    @classmethod
    def load(cls, path: Path) -> "ConfigurationCampaign":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationCampaignError(
                "cannot read configuration campaign: %s" % exc
            ) from exc
        if not isinstance(raw, Mapping):
            raise ConfigurationCampaignError("campaign root must be an object")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any]
    ) -> "ConfigurationCampaign":
        campaign_id = str(raw.get("campaign_id", "")).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", campaign_id):
            raise ConfigurationCampaignError(
                "campaign_id must be stable and lowercase"
            )
        intent = str(raw.get("intent", "")).strip()
        if not intent:
            raise ConfigurationCampaignError("campaign.intent is required")

        model = dict(_mapping(raw, "model"))
        runtime = dict(_mapping(raw, "runtime"))
        hardware = dict(_mapping(raw, "hardware"))
        measurement = _mapping(raw, "measurement")
        workload_raw = _mapping(raw, "workload")
        budget = _mapping(raw, "budget")
        costs_raw = _mapping(raw, "level_cost_gpu_hours")
        slo_raw = _mapping(raw, "slo")
        objectives_raw = _mapping(raw, "objectives")

        model_id = str(model.get("id", "")).strip()
        revision = str(model.get("revision", "")).strip()
        tokenizer_revision = str(model.get("tokenizer_revision", "")).strip()
        if not model_id or not revision or not tokenizer_revision:
            raise ConfigurationCampaignError(
                "campaign must pin model and tokenizer revisions"
            )
        max_model_len = _positive_int(model, "max_model_len")

        server_image = str(runtime.get("server_image", "")).strip()
        if not re.search(r"@sha256:[0-9a-f]{64}$", server_image):
            raise ConfigurationCampaignError(
                "runtime.server_image must use an immutable sha256 digest"
            )
        client_image_id = str(runtime.get("client_image_id", "")).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", client_image_id):
            raise ConfigurationCampaignError(
                "runtime.client_image_id must pin an immutable image ID"
            )
        for key in (
            "client_image",
            "server_container",
            "network",
            "cache_volume",
            "base_url",
            "served_model_name",
            "generation_config",
        ):
            if not str(runtime.get(key, "")).strip():
                raise ConfigurationCampaignError("runtime.%s is required" % key)
        try:
            memory_utilization = float(runtime["gpu_memory_utilization"])
            runtime_seed = int(runtime["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationCampaignError(
                "runtime must define gpu_memory_utilization and integer seed"
            ) from exc
        if not math.isfinite(memory_utilization) or not 0 < memory_utilization <= 1:
            raise ConfigurationCampaignError(
                "runtime.gpu_memory_utilization must be in (0, 1]"
            )
        runtime["gpu_memory_utilization"] = memory_utilization
        runtime["seed"] = runtime_seed

        try:
            gpu_count = int(hardware["gpu_count"])
            node_count = int(hardware["node_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationCampaignError(
                "hardware must define integer gpu_count and node_count"
            ) from exc
        if gpu_count != 1 or node_count != 1:
            raise ConfigurationCampaignError(
                "this controlled campaign requires one GPU on one node"
            )
        for key in ("gpu_name", "gpu_uuid", "driver_version", "hostname"):
            if not str(hardware.get(key, "")).strip():
                raise ConfigurationCampaignError("hardware.%s is required" % key)

        try:
            target_workload = InferenceWorkload.from_mapping(workload_raw)
        except CalibrationError as exc:
            raise ConfigurationCampaignError("invalid target workload: %s" % exc) from exc
        if target_workload.input_tokens + target_workload.output_tokens > max_model_len:
            raise ConfigurationCampaignError("target workload exceeds max_model_len")
        if target_workload.num_requests % target_workload.concurrency:
            raise ConfigurationCampaignError(
                "target num_requests must be divisible by concurrency"
            )
        probe_requests = _positive_int(measurement, "probe_requests")
        if (
            probe_requests >= target_workload.num_requests
            or probe_requests % target_workload.concurrency
        ):
            raise ConfigurationCampaignError(
                "probe_requests must be smaller than the target and divisible by concurrency"
            )

        objectives = {
            str(metric): str(direction).strip().lower()
            for metric, direction in objectives_raw.items()
        }
        expected_objectives = {
            "energy_j_per_1k_output_tokens": "min",
            "throughput_tok_s": "max",
        }
        if objectives != expected_objectives:
            raise ConfigurationCampaignError(
                "objectives must minimize output-token energy and maximize throughput"
            )
        try:
            slo = {
                str(metric): float(value) for metric, value in slo_raw.items()
            }
        except (TypeError, ValueError) as exc:
            raise ConfigurationCampaignError("slo values must be numeric") from exc
        supported_slo = {"ttft_ms", "tpot_ms", "min_goodput_req_s"}
        if (
            not slo
            or set(slo) - supported_slo
            or any(not math.isfinite(value) or value <= 0 for value in slo.values())
        ):
            raise ConfigurationCampaignError(
                "slo must use positive supported latency or goodput metrics"
            )

        costs: Dict[EvidenceLevel, float] = {}
        for level in (EvidenceLevel.L0, EvidenceLevel.L1, EvidenceLevel.L4):
            try:
                cost = float(costs_raw[level.name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationCampaignError(
                    "missing level cost for %s" % level.name
                ) from exc
            if not math.isfinite(cost) or cost < 0:
                raise ConfigurationCampaignError(
                    "level cost for %s must be non-negative" % level.name
                )
            costs[level] = cost
        if costs[EvidenceLevel.L0] != 0 or min(
            costs[EvidenceLevel.L1], costs[EvidenceLevel.L4]
        ) <= 0:
            raise ConfigurationCampaignError(
                "L0 must be free and L1/L4 costs must be positive"
            )

        budget_gpu_hours = _positive_float(budget, "gpu_hours")
        verify_top_k = _positive_int(budget, "verify_top_k")
        if verify_top_k * costs[EvidenceLevel.L4] > budget_gpu_hours:
            raise ConfigurationCampaignError(
                "verification reserve exceeds the agent budget"
            )

        power_limit_w = _positive_float(measurement, "power_limit_w")
        repeats = _positive_int(measurement, "repeats")
        sample_ms = _positive_int(measurement, "sample_ms")
        try:
            num_warmups = int(measurement.get("num_warmups", 0))
            schedule_seed = int(measurement.get("schedule_seed", 0))
        except (TypeError, ValueError) as exc:
            raise ConfigurationCampaignError(
                "measurement warmups and schedule seed must be integers"
            ) from exc
        if num_warmups < 0:
            raise ConfigurationCampaignError(
                "measurement.num_warmups must be non-negative"
            )

        candidates_raw = raw.get("candidates")
        if not isinstance(candidates_raw, Sequence) or isinstance(
            candidates_raw, (str, bytes)
        ) or not candidates_raw:
            raise ConfigurationCampaignError("campaign requires candidates")
        candidates = []
        seen_ids = set()
        seen_configs = set()
        baseline_count = 0
        for item in candidates_raw:
            if not isinstance(item, Mapping):
                raise ConfigurationCampaignError("candidate must be an object")
            point_id = str(item.get("id", "")).strip()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", point_id):
                raise ConfigurationCampaignError("candidate has invalid id")
            if point_id in seen_ids:
                raise ConfigurationCampaignError("candidate ids must be unique")
            seen_ids.add(point_id)
            raw_config = item.get("serving_configuration")
            if not isinstance(raw_config, Mapping):
                raise ConfigurationCampaignError(
                    "candidate %s lacks serving_configuration" % point_id
                )
            if raw_config.get("prefix_caching") is not False:
                raise ConfigurationCampaignError(
                    "candidate %s must disable prefix caching" % point_id
                )
            try:
                parsed = ServingConfiguration.from_mapping(
                    raw_config, target_workload
                )
            except CalibrationError as exc:
                raise ConfigurationCampaignError(
                    "candidate %s is invalid: %s" % (point_id, exc)
                ) from exc
            if parsed.required_gpus != 1:
                raise ConfigurationCampaignError(
                    "campaign candidates must use TP=PP=DP=1"
                )
            if (
                not parsed.chunked_prefill
                and parsed.max_num_batched_tokens < max_model_len
            ):
                raise ConfigurationCampaignError(
                    "non-chunked candidate %s needs max_num_batched_tokens >= max_model_len"
                    % point_id
                )
            normalized = parsed.to_dict()
            normalized.update(
                {
                    "prefix_caching": False,
                    "model_revision": revision,
                    "tokenizer_revision": tokenizer_revision,
                    "power_limit_w": power_limit_w,
                }
            )
            config_key = json.dumps(normalized, sort_keys=True)
            if config_key in seen_configs:
                raise ConfigurationCampaignError(
                    "candidate serving configurations must be unique"
                )
            seen_configs.add(config_key)
            role = str(item.get("role", "candidate")).strip().lower()
            if role not in {"candidate", "expert-baseline"}:
                raise ConfigurationCampaignError(
                    "candidate role must be candidate or expert-baseline"
                )
            baseline_count += int(role == "expert-baseline")
            candidates.append(ConfigurationPoint(point_id, normalized, role))
        if baseline_count != 1:
            raise ConfigurationCampaignError(
                "campaign requires exactly one expert-baseline candidate"
            )

        schedule = str(measurement.get("schedule", "")).strip().lower()
        if schedule != "cyclic-balanced":
            raise ConfigurationCampaignError(
                "measurement.schedule must be cyclic-balanced"
            )

        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            campaign_id=campaign_id,
            intent=intent,
            model=model,
            hardware=hardware,
            runtime=runtime,
            target_workload=target_workload,
            probe_requests=probe_requests,
            objectives=objectives,
            slo=slo,
            budget_gpu_hours=budget_gpu_hours,
            verify_top_k=verify_top_k,
            level_cost_gpu_hours=costs,
            power_limit_w=power_limit_w,
            repeats=repeats,
            num_warmups=num_warmups,
            sample_ms=sample_ms,
            schedule_seed=schedule_seed,
            candidates=tuple(candidates),
        )

    @property
    def model_id(self) -> str:
        return str(self.model["id"])

    @property
    def model_revision(self) -> str:
        return str(self.model["revision"])

    def candidate(self, point: ConfigurationPoint) -> Candidate:
        return Candidate(
            candidate_id=point.point_id,
            config=dict(point.configuration),
            required_gpus=1,
            target_nodes=1,
        )

    def assert_profile(self, profile: CalibrationProfile) -> None:
        if profile.model.model_id != self.model_id:
            raise ConfigurationCampaignError(
                "calibration profile model does not match campaign"
            )
        if profile.model.revision != self.model_revision:
            raise ConfigurationCampaignError(
                "calibration profile revision does not match campaign"
            )

    def workload_dict(self, num_requests: int) -> Dict[str, Any]:
        workload = self.target_workload
        return {
            "input_tokens": workload.input_tokens,
            "output_tokens": workload.output_tokens,
            "concurrency": workload.concurrency,
            "num_requests": num_requests,
            "request_rate": (
                "inf"
                if workload.request_rate_req_s is None
                else workload.request_rate_req_s
            ),
        }


def freeze_configuration_campaign(
    campaign_path: Path,
    calibration_path: Path,
    scenario_path: Path,
    predictions_path: Path,
    schedule_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    outputs = (
        scenario_path,
        predictions_path,
        schedule_path,
        summary_path,
        manifest_path,
    )
    for path in outputs:
        if Path(path).exists():
            raise ConfigurationCampaignError(
                "refusing to overwrite frozen artifact: %s" % path
            )

    campaign = ConfigurationCampaign.load(campaign_path)
    try:
        profile = CalibrationProfile.load(calibration_path)
    except CalibrationError as exc:
        raise ConfigurationCampaignError(
            "invalid calibration profile: %s" % exc
        ) from exc
    campaign.assert_profile(profile)
    campaign_hash = sha256_file(campaign_path)
    profile_hash = sha256_file(calibration_path)

    try:
        executor = TopologySandboxExecutor(
            profile,
            campaign.target_workload,
            expected_model=campaign.model_id,
            profile_path=calibration_path,
            scenario_path=campaign_path,
        )
    except TopologySandboxError as exc:
        raise ConfigurationCampaignError(
            "cannot initialize L0 configuration predictor: %s" % exc
        ) from exc
    predictions = EvidenceStore()
    scenario_candidates = []
    for point in campaign.candidates:
        try:
            record = executor.execute(
                campaign.candidate(point), EvidenceLevel.L0, seed=0
            )
        except TopologySandboxError as exc:
            raise ConfigurationCampaignError(
                "L0 prediction failed for %s: %s" % (point.point_id, exc)
            ) from exc
        if record.status != EvidenceStatus.SUCCEEDED:
            raise ConfigurationCampaignError(
                "L0 prediction failed for %s: %s"
                % (point.point_id, record.failure_reason or "unknown failure")
            )
        provenance = dict(record.provenance)
        provenance.update(
            {
                "campaign_id": campaign.campaign_id,
                "campaign_sha256": campaign_hash,
                "candidate_role": point.role,
                "dataset_split": "preregistered-configuration-search",
            }
        )
        record = replace(record, provenance=provenance)
        predictions.append(record)
        scenario_candidates.append(
            {
                "id": point.point_id,
                "required_gpus": 1,
                "target_nodes": 1,
                "config": dict(point.configuration),
                "prior_metrics": dict(record.metrics),
            }
        )

    scenario_raw = {
        "schema_version": campaign.schema_version,
        "name": campaign.campaign_id,
        "intent": campaign.intent,
        "model": campaign.model_id,
        "hardware": dict(campaign.hardware),
        "workload": campaign.workload_dict(
            campaign.target_workload.num_requests
        ),
        "objectives": dict(campaign.objectives),
        "slo": dict(campaign.slo),
        "budget": {
            "gpu_hours": campaign.budget_gpu_hours,
            "verify_top_k": campaign.verify_top_k,
        },
        "available_levels": ["L0", "L1", "L4"],
        "level_cost_gpu_hours": {
            level.name: campaign.level_cost_gpu_hours[level]
            for level in (EvidenceLevel.L0, EvidenceLevel.L1, EvidenceLevel.L4)
        },
        "candidates": scenario_candidates,
    }
    try:
        scenario = Scenario.from_dict(scenario_raw)
    except (SchemaError, KeyError, TypeError, ValueError) as exc:
        raise ConfigurationCampaignError(
            "generated agent scenario is invalid: %s" % exc
        ) from exc

    slo_feasible = [
        record
        for record in predictions.records
        if record.satisfies(scenario.slo)
    ]
    predicted_frontier = pareto_front(
        [(record.candidate_id, record.metrics) for record in slo_feasible],
        scenario.objectives,
    )
    robust_rows = []
    robust_slo_feasible_ids = []
    for record in predictions.records:
        pessimistic = dict(record.metrics)
        for metric, direction in scenario.objectives.items():
            suffix = "_upper" if direction == "min" else "_lower"
            pessimistic[metric] = record.metrics.get(
                metric + suffix, record.metrics[metric]
            )
        for metric in ("ttft_ms", "tpot_ms"):
            pessimistic[metric] = record.metrics.get(
                metric + "_upper", record.metrics.get(metric, float("inf"))
            )
        if scenario.slo.accepts(pessimistic):
            robust_slo_feasible_ids.append(record.candidate_id)
            robust_rows.append((record.candidate_id, pessimistic))
    robust_frontier = pareto_front(robust_rows, scenario.objectives)

    schedule = _measurement_schedule(campaign, campaign_hash)
    summary = {
        "schema_version": campaign.schema_version,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign_hash,
        "profile_id": profile.profile_id,
        "profile_sha256": profile_hash,
        "profile_publication_eligible": profile.publication_eligible,
        "uncertainty_calibrated": profile.uncertainty_calibrated,
        "candidate_count": len(campaign.candidates),
        "expert_baseline_id": next(
            point.point_id
            for point in campaign.candidates
            if point.role == "expert-baseline"
        ),
        "l0_prediction_count": len(predictions.records),
        "measurement_count": len(schedule["actions"]),
        "measurement_blocks": len(campaign.candidates) * campaign.repeats,
        "repeats_per_candidate_level": campaign.repeats,
        "levels": ["L1", "L4"],
        "probe_requests": campaign.probe_requests,
        "target_requests": campaign.target_workload.num_requests,
        "schedule_seed": campaign.schedule_seed,
        "l0_slo_feasible_ids": [
            record.candidate_id for record in slo_feasible
        ],
        "l0_predicted_pareto_ids": predicted_frontier,
        "l0_robust_slo_feasible_ids": robust_slo_feasible_ids,
        "l0_robust_predicted_pareto_ids": robust_frontier,
        "l0_frontier_is_provisional": True,
        "objectives": dict(campaign.objectives),
        "slo": dict(campaign.slo),
        "budget_gpu_hours": campaign.budget_gpu_hours,
        "verify_top_k": campaign.verify_top_k,
        "publication_claims_locked_until_measurement": True,
    }

    for path in outputs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(scenario_path).write_text(
        json.dumps(scenario_raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predictions.write_jsonl(predictions_path)
    Path(schedule_path).write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_entries = (
        campaign_path,
        calibration_path,
        scenario_path,
        predictions_path,
        schedule_path,
        summary_path,
    )
    Path(manifest_path).write_text(
        "".join(
            "%s  %s\n" % (sha256_file(path), path)
            for path in manifest_entries
        ),
        encoding="utf-8",
    )
    return summary


def _measurement_schedule(
    campaign: ConfigurationCampaign, campaign_hash: str
) -> Dict[str, Any]:
    base = [point.point_id for point in campaign.candidates]
    random.Random(campaign.schedule_seed).shuffle(base)
    shift_step = max(1, len(base) // campaign.repeats)
    actions = []
    block_index = 0
    action_index = 0
    for repeat_index in range(campaign.repeats):
        shift = (repeat_index * shift_step) % len(base)
        ordered = base[shift:] + base[:shift]
        if repeat_index % 2:
            ordered = list(reversed(ordered))
        for position, candidate_id in enumerate(ordered):
            levels = [EvidenceLevel.L1, EvidenceLevel.L4]
            if (position + repeat_index) % 2:
                levels.reverse()
            for level_position, level in enumerate(levels):
                actions.append(
                    {
                        "action_index": action_index,
                        "block_index": block_index,
                        "candidate_id": candidate_id,
                        "level": level.name,
                        "num_prompts": (
                            campaign.probe_requests
                            if level == EvidenceLevel.L1
                            else campaign.target_workload.num_requests
                        ),
                        "repeat": repeat_index,
                        "seed": repeat_index,
                        "restart_server": level_position == 0,
                    }
                )
                action_index += 1
            block_index += 1
    return {
        "schema_version": campaign.schema_version,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign_hash,
        "schedule": "cyclic-balanced",
        "schedule_seed": campaign.schedule_seed,
        "candidate_count": len(campaign.candidates),
        "repeats": campaign.repeats,
        "workload": campaign.workload_dict(
            campaign.target_workload.num_requests
        ),
        "measurement": {
            "probe_requests": campaign.probe_requests,
            "target_requests": campaign.target_workload.num_requests,
            "num_warmups": campaign.num_warmups,
            "sample_ms": campaign.sample_ms,
            "power_limit_w": campaign.power_limit_w,
        },
        "candidate_order_seeded": base,
        "actions": actions,
    }
