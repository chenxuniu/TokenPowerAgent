"""Deterministic compiler for TP/PP/batching candidate grids."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.schema import Candidate
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
    ProjectionBackend,
    TopologyProjector,
)


def _unique_positive_ints(raw: Mapping[str, Any], key: str) -> Tuple[int, ...]:
    values = raw.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CalibrationError("search grid %s must be an array" % key)
    try:
        parsed = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("search grid %s must contain integers" % key) from exc
    if not parsed or any(value < 1 for value in parsed):
        raise CalibrationError("search grid %s must contain positive integers" % key)
    if len(parsed) != len(set(parsed)):
        raise CalibrationError("search grid %s must contain unique values" % key)
    return parsed


def _booleans(raw: Mapping[str, Any], key: str) -> Tuple[bool, ...]:
    values = raw.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CalibrationError("search grid %s must be an array" % key)
    if not values or any(not isinstance(value, bool) for value in values):
        raise CalibrationError("search grid %s must contain booleans" % key)
    parsed = tuple(values)
    if len(parsed) != len(set(parsed)):
        raise CalibrationError("search grid %s must contain unique values" % key)
    return parsed


@dataclass(frozen=True)
class RejectedCandidate:
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class CandidateCompilation:
    candidates: Tuple[Candidate, ...]
    rejected: Tuple[RejectedCandidate, ...]


@dataclass(frozen=True)
class CandidateGrid:
    schema_version: str
    grid_id: str
    gpus_per_node: int
    max_nodes: int
    max_candidates: int
    tensor_parallel: Tuple[int, ...]
    pipeline_parallel: Tuple[int, ...]
    data_parallel: Tuple[int, ...]
    max_num_seqs: Tuple[int, ...]
    max_num_batched_tokens: Tuple[int, ...]
    chunked_prefill: Tuple[bool, ...]
    target_nodes: Optional[Tuple[int, ...]]
    fixed: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CandidateGrid":
        grid = raw.get("grid", {})
        fixed = raw.get("fixed", {})
        if not isinstance(grid, Mapping) or not isinstance(fixed, Mapping):
            raise CalibrationError("search-space grid and fixed must be objects")
        try:
            gpus_per_node = int(raw["gpus_per_node"])
            max_nodes = int(raw["max_nodes"])
            max_candidates = int(raw.get("max_candidates", 10000))
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(
                "search space requires integer gpus_per_node and max_nodes"
            ) from exc
        if min(gpus_per_node, max_nodes, max_candidates) < 1:
            raise CalibrationError("search-space resource limits must be positive")
        target_nodes = (
            _unique_positive_ints(grid, "target_nodes")
            if "target_nodes" in grid
            else None
        )
        grid_id = str(raw.get("grid_id", "")).strip()
        if not grid_id:
            raise CalibrationError("search-space grid_id is required")
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            grid_id=grid_id,
            gpus_per_node=gpus_per_node,
            max_nodes=max_nodes,
            max_candidates=max_candidates,
            tensor_parallel=_unique_positive_ints(grid, "tensor_parallel"),
            pipeline_parallel=_unique_positive_ints(grid, "pipeline_parallel"),
            data_parallel=_unique_positive_ints(grid, "data_parallel"),
            max_num_seqs=_unique_positive_ints(grid, "max_num_seqs"),
            max_num_batched_tokens=_unique_positive_ints(
                grid, "max_num_batched_tokens"
            ),
            chunked_prefill=_booleans(grid, "chunked_prefill"),
            target_nodes=target_nodes,
            fixed=dict(fixed),
        )

    @classmethod
    def load(cls, path: Path) -> "CandidateGrid":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError("cannot load search space %s" % path) from exc
        if not isinstance(raw, Mapping):
            raise CalibrationError("search-space root must be an object")
        return cls.from_mapping(raw)

    def compile(
        self,
        profile: Optional[CalibrationProfile] = None,
        workload: Optional[InferenceWorkload] = None,
    ) -> CandidateCompilation:
        if (profile is None) != (workload is None):
            raise CalibrationError(
                "profile and workload must be supplied together for feasibility filtering"
            )
        dimensions = (
            self.tensor_parallel,
            self.pipeline_parallel,
            self.data_parallel,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.chunked_prefill,
        )
        raw_size = math.prod(len(values) for values in dimensions)
        node_factor = len(self.target_nodes) if self.target_nodes else 1
        if raw_size * node_factor > self.max_candidates:
            raise CalibrationError(
                "search grid expands to at least %d candidates; max_candidates=%d"
                % (raw_size * node_factor, self.max_candidates)
            )

        candidates = []
        rejected = []
        projector = TopologyProjector(profile) if profile is not None else None
        max_gpus = self.gpus_per_node * self.max_nodes
        for tp, pp, dp, max_seqs, max_tokens, chunked in itertools.product(
            *dimensions
        ):
            required_gpus = tp * pp * dp
            minimum_nodes = int(math.ceil(required_gpus / float(self.gpus_per_node)))
            node_values = self.target_nodes or (minimum_nodes,)
            for nodes in node_values:
                candidate_id = self._candidate_id(
                    tp, pp, dp, max_seqs, max_tokens, chunked, nodes
                )
                if required_gpus > max_gpus:
                    rejected.append(
                        RejectedCandidate(
                            candidate_id,
                            "TP*PP*DP=%d exceeds %d available GPUs"
                            % (required_gpus, max_gpus),
                        )
                    )
                    continue
                if (
                    nodes < minimum_nodes
                    or nodes > self.max_nodes
                    or nodes > required_gpus
                ):
                    rejected.append(
                        RejectedCandidate(
                            candidate_id,
                            "target_nodes=%d is invalid for %d GPUs"
                            % (nodes, required_gpus),
                        )
                    )
                    continue
                config: Dict[str, Any] = dict(self.fixed)
                config.update(
                    {
                        "tensor_parallel": tp,
                        "pipeline_parallel": pp,
                        "data_parallel": dp,
                        "max_num_seqs": max_seqs,
                        "max_num_batched_tokens": max_tokens,
                        "chunked_prefill": chunked,
                    }
                )
                candidate = Candidate(
                    candidate_id=candidate_id,
                    config=config,
                    required_gpus=required_gpus,
                    target_nodes=nodes,
                )
                if projector is not None and workload is not None:
                    estimate = projector.predict(
                        candidate, workload, ProjectionBackend.L0_A
                    )
                    if not estimate.feasible:
                        rejected.append(
                            RejectedCandidate(
                                candidate_id,
                                estimate.failure_reason or "infeasible",
                            )
                        )
                        continue
                candidates.append(candidate)
        return CandidateCompilation(tuple(candidates), tuple(rejected))

    @staticmethod
    def _candidate_id(
        tp: int,
        pp: int,
        dp: int,
        max_seqs: int,
        max_tokens: int,
        chunked: bool,
        nodes: int,
    ) -> str:
        return (
            "tp%d-pp%d-dp%d-seq%d-bt%d-chunk%d-n%d"
            % (tp, pp, dp, max_seqs, max_tokens, int(chunked), nodes)
        )
