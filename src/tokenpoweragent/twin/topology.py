"""Calibrated, topology-aware projection for LLM serving configurations.

The projector is intentionally transparent.  It anchors every estimate to a
measured single-GPU serving point, scales compute from that anchor, and adds
explicit TP/PP communication and memory terms.  It is a ranking model for
choosing expensive validation runs, not a substitute for target-scale
measurement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus
from tokenpoweragent.schema import Candidate, EvidenceLevel
from tokenpoweragent.twin.base import EnergyTwin, Prediction


class CalibrationError(ValueError):
    """Raised when a calibration profile is incomplete or inconsistent."""


class ProjectionError(RuntimeError):
    """Raised when a candidate cannot be projected by the declared model."""


def _positive_float(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("missing positive numeric field: %s" % key) from exc
    if not math.isfinite(value) or value <= 0:
        raise CalibrationError("%s must be a positive finite number" % key)
    return value


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = _positive_float(raw, key)
    integer = int(value)
    if integer != value:
        raise CalibrationError("%s must be an integer" % key)
    return integer


def _optional_positive_float(
    raw: Mapping[str, Any], key: str, default: float
) -> float:
    if key not in raw:
        return default
    return _positive_float(raw, key)


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise CalibrationError("%s must be boolean" % key)


def _alias(
    raw: Mapping[str, Any], names: Sequence[str], default: Optional[Any] = None
) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _log_distance(left: float, right: float) -> float:
    return abs(math.log(max(left, 1e-12) / max(right, 1e-12), 2.0))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class InferenceWorkload:
    """Workload dimensions held fixed while the agent tunes serving knobs."""

    input_tokens: int
    output_tokens: int
    concurrency: int
    num_requests: int
    request_rate_req_s: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "concurrency", "num_requests"):
            if getattr(self, name) < 1:
                raise CalibrationError("workload.%s must be positive" % name)
        if self.request_rate_req_s is not None and (
            not math.isfinite(self.request_rate_req_s)
            or self.request_rate_req_s <= 0
        ):
            raise CalibrationError("workload.request_rate must be positive or 'inf'")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "InferenceWorkload":
        def integer(names: Sequence[str], default: Optional[int] = None) -> int:
            value = _alias(raw, names, default)
            if value is None:
                raise CalibrationError(
                    "workload requires one of: %s" % ", ".join(names)
                )
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise CalibrationError("invalid workload field %s" % names[0]) from exc
            if parsed < 1:
                raise CalibrationError("workload.%s must be positive" % names[0])
            return parsed

        raw_rate = _alias(raw, ("request_rate_req_s", "request_rate"), "inf")
        if str(raw_rate).strip().lower() == "inf":
            rate = None
        else:
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError) as exc:
                raise CalibrationError("invalid workload request_rate") from exc

        concurrency = integer(("concurrency", "max_concurrency"))
        return cls(
            input_tokens=integer(("input_tokens", "input_len")),
            output_tokens=integer(("output_tokens", "output_len")),
            concurrency=concurrency,
            num_requests=integer(("num_requests", "num_prompts"), concurrency * 8),
            request_rate_req_s=rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "request_rate_req_s": (
                "inf" if self.request_rate_req_s is None else self.request_rate_req_s
            ),
        }


@dataclass(frozen=True)
class ServingConfiguration:
    """Agent-controlled serving and parallelism knobs."""

    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    max_num_seqs: int
    max_num_batched_tokens: int
    chunked_prefill: bool
    precision: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"
    engine: str = "vllm"

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if getattr(self, name) < 1:
                raise CalibrationError("configuration.%s must be positive" % name)
        if not self.precision or not self.kv_cache_dtype or not self.engine:
            raise CalibrationError("configuration strings cannot be empty")

    @property
    def required_gpus(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], workload: InferenceWorkload
    ) -> "ServingConfiguration":
        def integer(names: Sequence[str], default: int) -> int:
            value = _alias(raw, names, default)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise CalibrationError("invalid configuration field %s" % names[0]) from exc
            if parsed < 1:
                raise CalibrationError("configuration.%s must be positive" % names[0])
            return parsed

        chunked = _as_bool(
            _alias(raw, ("chunked_prefill", "enable_chunked_prefill"), True),
            "chunked_prefill",
        )
        return cls(
            tensor_parallel=integer(
                ("tensor_parallel", "tensor_parallel_size", "tp"), 1
            ),
            pipeline_parallel=integer(
                ("pipeline_parallel", "pipeline_parallel_size", "pp"), 1
            ),
            data_parallel=integer(("data_parallel", "data_parallel_size", "dp"), 1),
            max_num_seqs=integer(("max_num_seqs", "max_batch_size"), workload.concurrency),
            max_num_batched_tokens=integer(
                ("max_num_batched_tokens", "batch_token_limit"),
                max(workload.input_tokens, workload.concurrency),
            ),
            chunked_prefill=chunked,
            precision=str(_alias(raw, ("precision", "dtype"), "bfloat16")).lower(),
            kv_cache_dtype=str(raw.get("kv_cache_dtype", "bfloat16")).lower(),
            engine=str(raw.get("engine", "vllm")).lower(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "data_parallel": self.data_parallel,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "chunked_prefill": self.chunked_prefill,
            "precision": self.precision,
            "kv_cache_dtype": self.kv_cache_dtype,
            "engine": self.engine,
        }


@dataclass(frozen=True)
class ModelArchitecture:
    model_id: str
    parameter_count_b: float
    num_layers: int
    hidden_size: int
    weight_bytes_per_parameter: float
    activation_bytes_per_element: float
    kv_bytes_per_token: float
    tp_collectives_per_layer: int = 2
    revision: str = "unrecorded"

    def __post_init__(self) -> None:
        numeric = (
            self.parameter_count_b,
            self.num_layers,
            self.hidden_size,
            self.weight_bytes_per_parameter,
            self.activation_bytes_per_element,
            self.kv_bytes_per_token,
            self.tp_collectives_per_layer,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise CalibrationError("model architecture values must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelArchitecture":
        model_id = str(raw.get("model_id", "")).strip()
        if not model_id:
            raise CalibrationError("model_architecture.model_id is required")

        kv_bytes = raw.get("kv_bytes_per_token")
        if kv_bytes is None:
            layers = _positive_int(raw, "num_layers")
            hidden = _positive_int(raw, "hidden_size")
            heads = _positive_int(raw, "num_attention_heads")
            kv_heads = _positive_int(raw, "num_kv_heads")
            if hidden % heads:
                raise CalibrationError("hidden_size must divide num_attention_heads")
            kv_dtype_bytes = _optional_positive_float(raw, "kv_dtype_bytes", 2.0)
            kv_bytes = 2.0 * layers * kv_heads * (hidden // heads) * kv_dtype_bytes

        return cls(
            model_id=model_id,
            parameter_count_b=_positive_float(raw, "parameter_count_b"),
            num_layers=_positive_int(raw, "num_layers"),
            hidden_size=_positive_int(raw, "hidden_size"),
            weight_bytes_per_parameter=_optional_positive_float(
                raw, "weight_bytes_per_parameter", 2.0
            ),
            activation_bytes_per_element=_optional_positive_float(
                raw, "activation_bytes_per_element", 2.0
            ),
            kv_bytes_per_token=float(kv_bytes),
            tp_collectives_per_layer=int(raw.get("tp_collectives_per_layer", 2)),
            revision=str(raw.get("revision", "unrecorded")),
        )


@dataclass(frozen=True)
class HardwareTopology:
    gpu_name: str
    gpu_memory_gib: float
    gpus_per_node: int
    max_nodes: int
    intra_node_bandwidth_gb_s: float
    inter_node_bandwidth_gb_s: float
    intra_node_latency_us: float
    inter_node_latency_us: float
    idle_power_w: float
    active_power_w: float
    memory_utilization: float = 0.90
    topology_source: str = "unspecified"

    def __post_init__(self) -> None:
        if not math.isfinite(self.memory_utilization) or not 0 < self.memory_utilization <= 1:
            raise CalibrationError("hardware.memory_utilization must be in (0, 1]")
        if self.active_power_w <= self.idle_power_w:
            raise CalibrationError("active_power_w must exceed idle_power_w")

    @property
    def available_gpus(self) -> int:
        return self.gpus_per_node * self.max_nodes

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HardwareTopology":
        gpu_name = str(raw.get("gpu_name", raw.get("gpu", ""))).strip()
        if not gpu_name:
            raise CalibrationError("hardware.gpu_name is required")
        return cls(
            gpu_name=gpu_name,
            gpu_memory_gib=_positive_float(raw, "gpu_memory_gib"),
            gpus_per_node=_positive_int(raw, "gpus_per_node"),
            max_nodes=_positive_int(raw, "max_nodes"),
            intra_node_bandwidth_gb_s=_positive_float(
                raw, "intra_node_bandwidth_gb_s"
            ),
            inter_node_bandwidth_gb_s=_positive_float(
                raw, "inter_node_bandwidth_gb_s"
            ),
            intra_node_latency_us=_positive_float(raw, "intra_node_latency_us"),
            inter_node_latency_us=_positive_float(raw, "inter_node_latency_us"),
            idle_power_w=_positive_float(raw, "idle_power_w"),
            active_power_w=_positive_float(raw, "active_power_w"),
            memory_utilization=float(raw.get("memory_utilization", 0.90)),
            topology_source=str(raw.get("topology_source", "unspecified")),
        )


@dataclass(frozen=True)
class ProjectionAssumptions:
    batch_saturation_concurrency: float = 8.0
    prefill_length_exponent: float = 0.90
    decode_context_exponent: float = 0.18
    chunk_overhead_fraction: float = 0.04
    runtime_overhead_gib: float = 4.0
    allreduce_efficiency: float = 0.75
    point_to_point_efficiency: float = 0.80
    communication_power_fraction: float = 0.65
    l0_relative_uncertainty: float = 0.38
    l2_relative_uncertainty: float = 0.22
    workload_distance_weight: float = 0.07
    scale_distance_weight: float = 0.04
    cross_node_penalty: float = 0.12
    pipeline_penalty: float = 0.04
    interval_coverage_target: float = 0.80

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProjectionAssumptions":
        defaults = cls()
        values: Dict[str, float] = {}
        for name in cls.__dataclass_fields__:
            value = float(raw.get(name, getattr(defaults, name)))
            if not math.isfinite(value) or value < 0:
                raise CalibrationError("assumptions.%s must be non-negative" % name)
            values[name] = value
        for fraction in (
            "allreduce_efficiency",
            "point_to_point_efficiency",
            "communication_power_fraction",
            "interval_coverage_target",
        ):
            if not 0 < values[fraction] <= 1:
                raise CalibrationError("assumptions.%s must be in (0, 1]" % fraction)
        return cls(**values)


_REQUIRED_CALIBRATION_METRICS = (
    "throughput_tok_s",
    "ttft_ms",
    "tpot_ms",
    "avg_power_w",
)


@dataclass(frozen=True)
class CalibrationPoint:
    point_id: str
    workload: InferenceWorkload
    configuration: ServingConfiguration
    metrics: Mapping[str, float]
    source_artifact: str

    def __post_init__(self) -> None:
        if self.configuration.required_gpus != 1:
            raise CalibrationError(
                "calibration point %s must be an L1 single-GPU probe" % self.point_id
            )
        for metric in _REQUIRED_CALIBRATION_METRICS:
            if (
                metric not in self.metrics
                or not math.isfinite(self.metrics[metric])
                or self.metrics[metric] <= 0
            ):
                raise CalibrationError(
                    "calibration point %s requires positive %s"
                    % (self.point_id, metric)
                )
        if not self.source_artifact:
            raise CalibrationError(
                "calibration point %s requires source_artifact" % self.point_id
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CalibrationPoint":
        point_id = str(raw.get("id", "")).strip()
        if not point_id:
            raise CalibrationError("calibration point id is required")
        workload_raw = raw.get("workload", {})
        config_raw = raw.get("configuration", raw.get("config", {}))
        metrics_raw = raw.get("metrics", {})
        if not isinstance(workload_raw, Mapping) or not isinstance(config_raw, Mapping):
            raise CalibrationError("calibration point workload/config must be objects")
        if not isinstance(metrics_raw, Mapping):
            raise CalibrationError("calibration point metrics must be an object")
        workload = InferenceWorkload.from_mapping(workload_raw)
        configuration = ServingConfiguration.from_mapping(config_raw, workload)
        return cls(
            point_id=point_id,
            workload=workload,
            configuration=configuration,
            metrics={str(key): float(value) for key, value in metrics_raw.items()},
            source_artifact=str(raw.get("source_artifact", "")).strip(),
        )


@dataclass(frozen=True)
class CalibrationProfile:
    schema_version: str
    profile_id: str
    publication_eligible: bool
    uncertainty_calibrated: bool
    model: ModelArchitecture
    hardware: HardwareTopology
    assumptions: ProjectionAssumptions
    points: Tuple[CalibrationPoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.publication_eligible:
            return
        if not self.uncertainty_calibrated:
            raise CalibrationError(
                "publication profiles require calibrated uncertainty"
            )
        topology_source = self.hardware.topology_source.lower()
        if any(
            marker in topology_source
            for marker in ("synthetic", "unspecified", "replace")
        ):
            raise CalibrationError(
                "publication profiles require measured topology provenance"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CalibrationProfile":
        model_raw = raw.get("model_architecture", {})
        hardware_raw = raw.get("hardware", {})
        assumptions_raw = raw.get("assumptions", {})
        points_raw = raw.get("calibration_points", [])
        if not isinstance(model_raw, Mapping) or not isinstance(hardware_raw, Mapping):
            raise CalibrationError("model_architecture and hardware must be objects")
        if not isinstance(assumptions_raw, Mapping):
            raise CalibrationError("assumptions must be an object")
        if not isinstance(points_raw, Sequence) or isinstance(points_raw, (str, bytes)):
            raise CalibrationError("calibration_points must be an array")
        points = tuple(CalibrationPoint.from_mapping(point) for point in points_raw)
        if not points:
            raise CalibrationError("at least one calibration point is required")
        ids = [point.point_id for point in points]
        if len(ids) != len(set(ids)):
            raise CalibrationError("calibration point ids must be unique")
        profile_id = str(raw.get("profile_id", "")).strip()
        if not profile_id:
            raise CalibrationError("profile_id is required")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CalibrationError("metadata must be an object")
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            profile_id=profile_id,
            publication_eligible=_as_bool(
                raw.get("publication_eligible", False), "publication_eligible"
            ),
            uncertainty_calibrated=_as_bool(
                raw.get("uncertainty_calibrated", False),
                "uncertainty_calibrated",
            ),
            model=ModelArchitecture.from_mapping(model_raw),
            hardware=HardwareTopology.from_mapping(hardware_raw),
            assumptions=ProjectionAssumptions.from_mapping(assumptions_raw),
            points=points,
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: Path) -> "CalibrationProfile":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError("cannot load calibration profile %s" % path) from exc
        if not isinstance(raw, Mapping):
            raise CalibrationError("calibration profile root must be an object")
        return cls.from_mapping(raw)

    def nearest_point(
        self,
        workload: InferenceWorkload,
        configuration: ServingConfiguration,
    ) -> Tuple[CalibrationPoint, float]:
        compatible = [
            point
            for point in self.points
            if point.configuration.precision == configuration.precision
            and point.configuration.kv_cache_dtype == configuration.kv_cache_dtype
            and point.configuration.engine == configuration.engine
        ]
        if not compatible:
            raise ProjectionError(
                "no calibration point matches engine=%s precision=%s kv_cache_dtype=%s"
                % (
                    configuration.engine,
                    configuration.precision,
                    configuration.kv_cache_dtype,
                )
            )

        def distance(point: CalibrationPoint) -> float:
            workload_distance = (
                0.45
                * _log_distance(workload.input_tokens, point.workload.input_tokens)
                + 0.20
                * _log_distance(workload.output_tokens, point.workload.output_tokens)
                + 0.35
                * _log_distance(workload.concurrency, point.workload.concurrency)
            )
            config_distance = (
                0.10
                * _log_distance(
                    configuration.max_num_seqs,
                    point.configuration.max_num_seqs,
                )
                + 0.10
                * _log_distance(
                    configuration.max_num_batched_tokens,
                    point.configuration.max_num_batched_tokens,
                )
                + (
                    0.15
                    if configuration.chunked_prefill
                    != point.configuration.chunked_prefill
                    else 0.0
                )
            )
            return workload_distance + config_distance

        selected = min(compatible, key=distance)
        return selected, distance(selected)


@dataclass(frozen=True)
class SandboxEstimate:
    feasible: bool
    metrics: Mapping[str, float]
    relative_uncertainty: float
    reference_point_id: str
    calibration_distance: float
    decomposition: Mapping[str, Any]
    failure_reason: Optional[str] = None


class TopologyProjector:
    """Project serving metrics from L1 anchors to candidate TP/PP topologies."""

    _INTERVAL_METRICS = (
        "energy_j_per_1k_tokens",
        "throughput_tok_s",
        "ttft_ms",
        "tpot_ms",
    )

    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def predict(
        self,
        candidate: Candidate,
        workload: InferenceWorkload,
        level: EvidenceLevel,
    ) -> SandboxEstimate:
        if level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
            raise ProjectionError("topology projection supports only L0 and L2")
        config = ServingConfiguration.from_mapping(candidate.config, workload)
        reference, calibration_distance = self.profile.nearest_point(workload, config)
        hardware = self.profile.hardware
        model = self.profile.model
        assumptions = self.profile.assumptions

        geometry_failure = self._geometry_failure(candidate, config)
        memory = self._memory_terms(config, workload)
        if geometry_failure or memory["memory_feasible"] is False:
            reason = geometry_failure or (
                "estimated per-GPU memory %.2f GiB exceeds usable %.2f GiB"
                % (memory["memory_required_gib"], memory["memory_usable_gib"])
            )
            return SandboxEstimate(
                feasible=False,
                metrics={},
                relative_uncertainty=1.0,
                reference_point_id=reference.point_id,
                calibration_distance=calibration_distance,
                decomposition={
                    "configuration": config.to_dict(),
                    "workload": workload.to_dict(),
                    **memory,
                },
                failure_reason=reason,
            )

        active_dp = min(config.data_parallel, workload.concurrency)
        active_sequences = min(
            config.max_num_seqs,
            int(math.ceil(workload.concurrency / float(active_dp))),
        )
        reference_sequences = min(
            reference.configuration.max_num_seqs,
            reference.workload.concurrency,
        )
        batch_efficiency = self._batch_efficiency(active_sequences)
        reference_batch_efficiency = self._batch_efficiency(reference_sequences)

        reference_step_s = (
            reference_sequences / reference.metrics["throughput_tok_s"]
        )
        target_context = workload.input_tokens + 0.5 * workload.output_tokens
        reference_context = (
            reference.workload.input_tokens
            + 0.5 * reference.workload.output_tokens
        )
        context_factor = (
            target_context / reference_context
        ) ** assumptions.decode_context_exponent
        serial_decode_step_s = (
            reference_step_s
            * (active_sequences / float(reference_sequences))
            * (reference_batch_efficiency / batch_efficiency)
            * context_factor
        )

        topology_enabled = level == EvidenceLevel.L2
        placed_gpus_per_node = min(
            hardware.gpus_per_node,
            int(math.ceil(config.required_gpus / float(candidate.target_nodes))),
        )
        tp_crosses_nodes = (
            config.tensor_parallel > placed_gpus_per_node
            and candidate.target_nodes > 1
        )
        pp_crosses_nodes = (
            config.pipeline_parallel > 1
            and config.tensor_parallel * config.pipeline_parallel
            > placed_gpus_per_node
            and candidate.target_nodes > 1
        )
        layers_per_stage = int(math.ceil(model.num_layers / config.pipeline_parallel))
        decode_payload_bytes = (
            active_sequences
            * model.hidden_size
            * model.activation_bytes_per_element
        )
        tp_collective_s = self._allreduce_seconds(
            decode_payload_bytes,
            config.tensor_parallel,
            tp_crosses_nodes,
            topology_enabled,
        )
        tp_comm_stage_s = (
            tp_collective_s
            * model.tp_collectives_per_layer
            * layers_per_stage
        )
        pp_comm_stage_s = self._point_to_point_seconds(
            decode_payload_bytes,
            pp_crosses_nodes,
            topology_enabled and config.pipeline_parallel > 1,
        )
        compute_stage_s = serial_decode_step_s / (
            config.tensor_parallel * config.pipeline_parallel
        )
        stage_cycle_s = compute_stage_s + tp_comm_stage_s + pp_comm_stage_s
        microbatches = max(1, min(active_sequences, config.max_num_seqs))
        pipeline_efficiency = microbatches / float(
            microbatches + config.pipeline_parallel - 1
        )
        capacity_tok_s = (
            active_dp
            * active_sequences
            * pipeline_efficiency
            / max(stage_cycle_s, 1e-12)
        )

        if workload.request_rate_req_s is None:
            throughput_tok_s = capacity_tok_s
            offered_load_fraction = 1.0
        else:
            offered_tok_s = workload.request_rate_req_s * workload.output_tokens
            throughput_tok_s = min(capacity_tok_s, offered_tok_s)
            offered_load_fraction = min(1.0, offered_tok_s / capacity_tok_s)

        end_to_end_decode_s = (
            compute_stage_s * config.pipeline_parallel
            + tp_comm_stage_s * config.pipeline_parallel
            + pp_comm_stage_s * max(0, config.pipeline_parallel - 1)
        )
        tpot_ms = reference.metrics["tpot_ms"] * (
            end_to_end_decode_s / reference_step_s
        )

        ttft_ms, prefill_terms = self._project_ttft(
            workload,
            config,
            reference,
            active_sequences,
            reference_sequences,
            tp_crosses_nodes,
            pp_crosses_nodes,
            topology_enabled,
        )
        queue_terms = self._queue_terms(
            workload, capacity_tok_s, ttft_ms
        )
        ttft_ms = queue_terms["ttft_with_queue_ms"]

        compute_fraction = compute_stage_s / max(stage_cycle_s, 1e-12)
        communication_fraction = (
            tp_comm_stage_s + pp_comm_stage_s
        ) / max(stage_cycle_s, 1e-12)
        reference_power_fraction = _clamp(
            (reference.metrics["avg_power_w"] - hardware.idle_power_w)
            / (hardware.active_power_w - hardware.idle_power_w),
            0.05,
            1.0,
        )
        active_power_fraction = _clamp(
            (
                reference_power_fraction
                * (batch_efficiency / reference_batch_efficiency)
                * pipeline_efficiency
                * compute_fraction
                + assumptions.communication_power_fraction
                * communication_fraction
            )
            * offered_load_fraction,
            0.02,
            1.0,
        )
        active_gpu_power_w = hardware.idle_power_w + active_power_fraction * (
            hardware.active_power_w - hardware.idle_power_w
        )
        active_gpus = active_dp * config.tensor_parallel * config.pipeline_parallel
        inactive_gpus = config.required_gpus - active_gpus
        total_power_w = (
            active_gpus * active_gpu_power_w
            + inactive_gpus * hardware.idle_power_w
        )
        request_throughput_req_s = throughput_tok_s / workload.output_tokens
        duration_s = workload.num_requests / max(request_throughput_req_s, 1e-12)
        energy_j = total_power_w * duration_s
        output_tokens = workload.num_requests * workload.output_tokens
        total_tokens = workload.num_requests * (
            workload.input_tokens + workload.output_tokens
        )

        metrics: Dict[str, float] = {
            "energy_j": energy_j,
            "gpu_energy_j": energy_j,
            "energy_j_per_1k_tokens": 1000.0 * energy_j / output_tokens,
            "energy_j_per_1k_output_tokens": 1000.0 * energy_j / output_tokens,
            "energy_j_per_1k_total_tokens": 1000.0 * energy_j / total_tokens,
            "j_per_output_token": energy_j / output_tokens,
            "j_per_total_token": energy_j / total_tokens,
            "avg_power_w": total_power_w,
            "total_gpu_power_w": total_power_w,
            "avg_power_per_gpu_w": total_power_w / config.required_gpus,
            "throughput_tok_s": throughput_tok_s,
            "capacity_tok_s": capacity_tok_s,
            "request_throughput_req_s": request_throughput_req_s,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "duration_s": duration_s,
            "estimated_memory_gib": float(memory["memory_required_gib"]),
        }
        uncertainty = self._uncertainty(
            level,
            calibration_distance,
            config,
            candidate.target_nodes,
            queue_terms["queue_utilization"],
        )
        metrics["relative_uncertainty"] = uncertainty
        for metric in self._INTERVAL_METRICS:
            estimate = metrics[metric]
            metrics[metric + "_lower"] = max(0.0, estimate * (1.0 - uncertainty))
            metrics[metric + "_upper"] = estimate * (1.0 + uncertainty)

        return SandboxEstimate(
            feasible=True,
            metrics=metrics,
            relative_uncertainty=uncertainty,
            reference_point_id=reference.point_id,
            calibration_distance=calibration_distance,
            decomposition={
                "configuration": config.to_dict(),
                "workload": workload.to_dict(),
                "reference_workload": reference.workload.to_dict(),
                "reference_configuration": reference.configuration.to_dict(),
                "reference_metrics": dict(reference.metrics),
                "active_data_parallel_replicas": active_dp,
                "active_sequences_per_replica": active_sequences,
                "batch_efficiency": batch_efficiency,
                "pipeline_efficiency": pipeline_efficiency,
                "tp_crosses_nodes": tp_crosses_nodes,
                "pp_crosses_nodes": pp_crosses_nodes,
                "placement_assumption": "TP groups packed before PP and DP",
                "placed_gpus_per_node": placed_gpus_per_node,
                "compute_stage_ms": 1000.0 * compute_stage_s,
                "tp_communication_stage_ms": 1000.0 * tp_comm_stage_s,
                "pp_communication_stage_ms": 1000.0 * pp_comm_stage_s,
                "communication_fraction": communication_fraction,
                "topology_terms_enabled": topology_enabled,
                "offered_load_fraction": offered_load_fraction,
                "active_gpu_power_w": active_gpu_power_w,
                "active_gpus": active_gpus,
                "inactive_gpus": inactive_gpus,
                **memory,
                **prefill_terms,
                **queue_terms,
            },
        )

    def _geometry_failure(
        self, candidate: Candidate, config: ServingConfiguration
    ) -> Optional[str]:
        hardware = self.profile.hardware
        if candidate.required_gpus != config.required_gpus:
            return (
                "candidate.required_gpus=%d but TP*PP*DP=%d"
                % (candidate.required_gpus, config.required_gpus)
            )
        minimum_nodes = int(
            math.ceil(config.required_gpus / float(hardware.gpus_per_node))
        )
        if candidate.target_nodes < minimum_nodes:
            return (
                "candidate needs at least %d nodes for %d GPUs"
                % (minimum_nodes, config.required_gpus)
            )
        if candidate.target_nodes > hardware.max_nodes:
            return (
                "candidate requests %d nodes but profile exposes %d"
                % (candidate.target_nodes, hardware.max_nodes)
            )
        if config.required_gpus > hardware.available_gpus:
            return (
                "candidate requests %d GPUs but profile exposes %d"
                % (config.required_gpus, hardware.available_gpus)
            )
        return None

    def _memory_terms(
        self, config: ServingConfiguration, workload: InferenceWorkload
    ) -> Dict[str, Any]:
        model = self.profile.model
        hardware = self.profile.hardware
        model_gib = (
            model.parameter_count_b
            * 1e9
            * model.weight_bytes_per_parameter
            / (config.tensor_parallel * config.pipeline_parallel)
            / (1024.0**3)
        )
        active_replicas = min(config.data_parallel, workload.concurrency)
        sequences_per_replica = min(
            config.max_num_seqs,
            int(math.ceil(workload.concurrency / float(active_replicas))),
        )
        kv_gib = (
            model.kv_bytes_per_token
            * (workload.input_tokens + workload.output_tokens)
            * sequences_per_replica
            / (config.tensor_parallel * config.pipeline_parallel)
            / (1024.0**3)
        )
        required = model_gib + kv_gib + self.profile.assumptions.runtime_overhead_gib
        usable = hardware.gpu_memory_gib * hardware.memory_utilization
        return {
            "model_memory_gib": model_gib,
            "kv_cache_memory_gib": kv_gib,
            "runtime_overhead_gib": self.profile.assumptions.runtime_overhead_gib,
            "memory_required_gib": required,
            "memory_usable_gib": usable,
            "memory_feasible": required <= usable,
        }

    def _batch_efficiency(self, concurrency: int) -> float:
        knee = max(self.profile.assumptions.batch_saturation_concurrency, 1e-6)
        return max(0.05, 1.0 - math.exp(-concurrency / knee))

    def _link(self, crosses_nodes: bool) -> Tuple[float, float]:
        hardware = self.profile.hardware
        if crosses_nodes:
            return (
                hardware.inter_node_bandwidth_gb_s,
                hardware.inter_node_latency_us,
            )
        return (
            hardware.intra_node_bandwidth_gb_s,
            hardware.intra_node_latency_us,
        )

    def _allreduce_seconds(
        self,
        payload_bytes: float,
        participants: int,
        crosses_nodes: bool,
        enabled: bool,
    ) -> float:
        if not enabled or participants <= 1:
            return 0.0
        bandwidth_gb_s, latency_us = self._link(crosses_nodes)
        ring_factor = 2.0 * (participants - 1) / participants
        transfer_s = ring_factor * payload_bytes / (
            bandwidth_gb_s
            * 1e9
            * self.profile.assumptions.allreduce_efficiency
        )
        latency_s = 2.0 * (participants - 1) * latency_us * 1e-6
        return transfer_s + latency_s

    def _point_to_point_seconds(
        self, payload_bytes: float, crosses_nodes: bool, enabled: bool
    ) -> float:
        if not enabled:
            return 0.0
        bandwidth_gb_s, latency_us = self._link(crosses_nodes)
        return (
            payload_bytes
            / (
                bandwidth_gb_s
                * 1e9
                * self.profile.assumptions.point_to_point_efficiency
            )
            + latency_us * 1e-6
        )

    def _chunk_factor(
        self,
        workload: InferenceWorkload,
        config: ServingConfiguration,
        active_sequences: int,
    ) -> Tuple[int, float]:
        prompt_tokens = workload.input_tokens * active_sequences
        chunks = max(1, int(math.ceil(prompt_tokens / config.max_num_batched_tokens)))
        multiplier = 1.0 + self.profile.assumptions.chunk_overhead_fraction * (
            chunks - 1
        )
        if not config.chunked_prefill and chunks > 1:
            multiplier += self.profile.assumptions.chunk_overhead_fraction * chunks
        return chunks, multiplier

    def _project_ttft(
        self,
        workload: InferenceWorkload,
        config: ServingConfiguration,
        reference: CalibrationPoint,
        active_sequences: int,
        reference_sequences: int,
        tp_crosses_nodes: bool,
        pp_crosses_nodes: bool,
        topology_enabled: bool,
    ) -> Tuple[float, Mapping[str, Any]]:
        model = self.profile.model
        assumptions = self.profile.assumptions
        target_prefill_efficiency = self._batch_efficiency(active_sequences)
        reference_prefill_efficiency = self._batch_efficiency(reference_sequences)
        input_factor = (
            workload.input_tokens / float(reference.workload.input_tokens)
        ) ** assumptions.prefill_length_exponent
        serial_prefill_s = (
            reference.metrics["ttft_ms"]
            / 1000.0
            * input_factor
            * (active_sequences / float(reference_sequences))
            * (reference_prefill_efficiency / target_prefill_efficiency)
        )
        target_chunks, target_chunk_factor = self._chunk_factor(
            workload, config, active_sequences
        )
        reference_chunks, reference_chunk_factor = self._chunk_factor(
            reference.workload,
            reference.configuration,
            reference_sequences,
        )
        prefill_payload = (
            active_sequences
            * workload.input_tokens
            * model.hidden_size
            * model.activation_bytes_per_element
        )
        tp_collective_s = self._allreduce_seconds(
            prefill_payload,
            config.tensor_parallel,
            tp_crosses_nodes,
            topology_enabled,
        )
        tp_comm_s = (
            tp_collective_s * model.tp_collectives_per_layer * model.num_layers
        )
        pp_link_s = self._point_to_point_seconds(
            prefill_payload,
            pp_crosses_nodes,
            topology_enabled and config.pipeline_parallel > 1,
        )
        pp_comm_s = pp_link_s * max(0, config.pipeline_parallel - 1)
        ttft_s = (
            serial_prefill_s / config.tensor_parallel + tp_comm_s + pp_comm_s
        ) * (target_chunk_factor / reference_chunk_factor)
        return 1000.0 * ttft_s, {
            "prefill_serial_ms": 1000.0 * serial_prefill_s,
            "prefill_tp_communication_ms": 1000.0 * tp_comm_s,
            "prefill_pp_communication_ms": 1000.0 * pp_comm_s,
            "prefill_chunks": target_chunks,
            "reference_prefill_chunks": reference_chunks,
            "prefill_chunk_factor": target_chunk_factor,
        }

    @staticmethod
    def _queue_terms(
        workload: InferenceWorkload,
        capacity_tok_s: float,
        base_ttft_ms: float,
    ) -> Mapping[str, float]:
        capacity_req_s = capacity_tok_s / workload.output_tokens
        if workload.request_rate_req_s is None:
            return {
                "queue_utilization": 0.0,
                "queue_delay_ms": 0.0,
                "ttft_with_queue_ms": base_ttft_ms,
            }
        rho = workload.request_rate_req_s / max(capacity_req_s, 1e-12)
        bounded_rho = min(rho, 0.98)
        service_ms = 1000.0 / max(capacity_req_s, 1e-12)
        queue_delay_ms = service_ms * bounded_rho / max(1.0 - bounded_rho, 0.02)
        return {
            "queue_utilization": rho,
            "queue_delay_ms": queue_delay_ms,
            "ttft_with_queue_ms": base_ttft_ms + queue_delay_ms,
        }

    def _uncertainty(
        self,
        level: EvidenceLevel,
        calibration_distance: float,
        config: ServingConfiguration,
        target_nodes: int,
        queue_utilization: float,
    ) -> float:
        assumptions = self.profile.assumptions
        base = (
            assumptions.l0_relative_uncertainty
            if level == EvidenceLevel.L0
            else assumptions.l2_relative_uncertainty
        )
        uncertainty = (
            base
            + assumptions.workload_distance_weight * calibration_distance
            + assumptions.scale_distance_weight
            * math.log(max(config.required_gpus, 1), 2.0)
            + (assumptions.cross_node_penalty if target_nodes > 1 else 0.0)
            + assumptions.pipeline_penalty
            * math.log(max(config.pipeline_parallel, 1), 2.0)
        )
        if queue_utilization >= 0.90:
            uncertainty += 0.10
        return _clamp(uncertainty, 0.05, 0.95)


class TopologyEnergyTwin(EnergyTwin):
    """Use sandbox projections as priors, then prefer acquired evidence.

    This adapter lets the controller start without hand-written
    ``prior_metrics``. Failed observations never overwrite a usable belief.
    """

    _OBSERVED_UNCERTAINTY = {
        EvidenceLevel.L0: 0.38,
        EvidenceLevel.L1: 0.28,
        EvidenceLevel.L2: 0.22,
        EvidenceLevel.L3: 0.12,
        EvidenceLevel.L4: 0.03,
    }

    def __init__(
        self,
        profile: CalibrationProfile,
        workload: InferenceWorkload,
        prior_level: EvidenceLevel = EvidenceLevel.L0,
    ) -> None:
        if prior_level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
            raise ValueError("TopologyEnergyTwin prior_level must be L0 or L2")
        self.projector = TopologyProjector(profile)
        self.workload = workload
        self.prior_level = prior_level
        self._records: Dict[str, List[EvidenceRecord]] = {}

    def update(self, evidence: EvidenceRecord) -> None:
        self._records.setdefault(evidence.candidate_id, []).append(evidence)

    def predict(self, candidate: Candidate) -> Prediction:
        successful = [
            record
            for record in self._records.get(candidate.candidate_id, [])
            if record.status == EvidenceStatus.SUCCEEDED
        ]
        if successful:
            best = max(successful, key=lambda record: int(record.level))
            raw_uncertainty = best.provenance.get("relative_uncertainty")
            uncertainty = (
                float(raw_uncertainty)
                if raw_uncertainty is not None
                else self._OBSERVED_UNCERTAINTY[best.level]
            )
            return Prediction(best.metrics, uncertainty, int(best.level))

        estimate = self.projector.predict(
            candidate, self.workload, self.prior_level
        )
        if not estimate.feasible:
            raise ValueError(
                "candidate %s is infeasible: %s"
                % (candidate.candidate_id, estimate.failure_reason)
            )
        return Prediction(
            estimate.metrics,
            estimate.relative_uncertainty,
            int(self.prior_level),
        )
