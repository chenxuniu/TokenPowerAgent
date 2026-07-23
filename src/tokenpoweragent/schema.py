"""Versioned scenario and candidate schemas used across all executors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


class SchemaError(ValueError):
    """Raised when a scenario violates a deterministic contract."""


class EvidenceLevel(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4

    @classmethod
    def parse(cls, value: Any) -> "EvidenceLevel":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip().upper()
        if not text.startswith("L"):
            text = "L" + text
        try:
            return cls[text]
        except KeyError as exc:
            raise SchemaError("unknown evidence level: %s" % value) from exc


@dataclass(frozen=True)
class SLO:
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    min_goodput_req_s: Optional[float] = None

    def accepts(self, metrics: Mapping[str, float]) -> bool:
        checks = []
        if self.ttft_ms is not None:
            checks.append(metrics.get("ttft_ms", float("inf")) <= self.ttft_ms)
        if self.tpot_ms is not None:
            checks.append(metrics.get("tpot_ms", float("inf")) <= self.tpot_ms)
        if self.min_goodput_req_s is not None:
            checks.append(
                metrics.get("goodput_req_s", float("-inf"))
                >= self.min_goodput_req_s
            )
        return all(checks)


@dataclass(frozen=True)
class Budget:
    gpu_hours: float
    verify_top_k: int = 1

    def __post_init__(self) -> None:
        if self.gpu_hours <= 0:
            raise SchemaError("budget.gpu_hours must be positive")
        if self.verify_top_k < 1:
            raise SchemaError("budget.verify_top_k must be at least one")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    config: Mapping[str, Any] = field(default_factory=dict)
    required_gpus: int = 1
    target_nodes: int = 1
    prior_metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise SchemaError("candidate_id cannot be empty")
        if self.required_gpus < 1 or self.target_nodes < 1:
            raise SchemaError("candidate resources must be positive")
        if self.required_gpus < self.target_nodes:
            raise SchemaError("required_gpus cannot be smaller than target_nodes")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Candidate":
        known = {"id", "required_gpus", "target_nodes", "prior_metrics"}
        config = dict(raw.get("config", {}))
        config.update({key: value for key, value in raw.items() if key not in known | {"config"}})
        return cls(
            candidate_id=str(raw["id"]),
            config=config,
            required_gpus=int(raw.get("required_gpus", 1)),
            target_nodes=int(raw.get("target_nodes", 1)),
            prior_metrics={
                str(key): float(value)
                for key, value in raw.get("prior_metrics", {}).items()
            },
        )


@dataclass(frozen=True)
class Scenario:
    schema_version: str
    name: str
    model: str
    hardware: Mapping[str, Any]
    workload: Mapping[str, Any]
    objectives: Mapping[str, str]
    slo: SLO
    budget: Budget
    candidates: Tuple[Candidate, ...]
    available_levels: Tuple[EvidenceLevel, ...]
    level_cost_gpu_hours: Mapping[EvidenceLevel, float]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Scenario":
        candidates = tuple(Candidate.from_dict(item) for item in raw["candidates"])
        ids = [candidate.candidate_id for candidate in candidates]
        if not candidates or len(ids) != len(set(ids)):
            raise SchemaError("scenario requires non-empty, unique candidates")

        objectives = {
            str(metric): str(direction).lower()
            for metric, direction in raw.get("objectives", {}).items()
        }
        if not objectives or any(value not in {"min", "max"} for value in objectives.values()):
            raise SchemaError("objectives must map metrics to 'min' or 'max'")

        levels = tuple(
            EvidenceLevel.parse(value)
            for value in raw.get("available_levels", [0, 1, 2, 3, 4])
        )
        if EvidenceLevel.L4 not in levels:
            raise SchemaError("L4 must be available for final verification")

        raw_costs = raw.get("level_cost_gpu_hours", {})
        costs: Dict[EvidenceLevel, float] = {}
        for level in levels:
            key = level.name
            value = raw_costs.get(key, raw_costs.get(str(int(level))))
            if value is None or float(value) < 0:
                raise SchemaError("missing non-negative cost for %s" % key)
            costs[level] = float(value)

        slo_raw = raw.get("slo", {})
        budget_raw = raw["budget"]
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            name=str(raw.get("name", "unnamed-scenario")),
            model=str(raw["model"]),
            hardware=dict(raw.get("hardware", {})),
            workload=dict(raw.get("workload", {})),
            objectives=objectives,
            slo=SLO(
                ttft_ms=_optional_float(slo_raw.get("ttft_ms")),
                tpot_ms=_optional_float(slo_raw.get("tpot_ms")),
                min_goodput_req_s=_optional_float(slo_raw.get("min_goodput_req_s")),
            ),
            budget=Budget(
                gpu_hours=float(budget_raw["gpu_hours"]),
                verify_top_k=int(budget_raw.get("verify_top_k", 1)),
            ),
            candidates=candidates,
            available_levels=levels,
            level_cost_gpu_hours=costs,
        )

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise SchemaError("install tokenpoweragent[yaml] for YAML scenarios") from exc
            raw = yaml.safe_load(text)
        else:
            raise SchemaError("scenario must be JSON or YAML")
        if not isinstance(raw, Mapping):
            raise SchemaError("scenario root must be an object")
        return cls.from_dict(raw)

    def candidate(self, candidate_id: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise SchemaError("unknown candidate: %s" % candidate_id)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
