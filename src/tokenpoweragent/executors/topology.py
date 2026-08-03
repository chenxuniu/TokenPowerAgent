"""Executor that turns topology projections into explicitly labeled evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord, EvidenceStatus
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.schema import Candidate, EvidenceLevel
from tokenpoweragent.twin.topology import (
    CalibrationProfile,
    InferenceWorkload,
    ProjectionError,
    TopologyProjector,
)


class TopologySandboxError(ExecutionError):
    """Raised when the topology sandbox cannot honor its input contract."""


class TopologySandboxExecutor(Executor):
    """Produce L0/L2 estimates without presenting them as hardware runs.

    L0 omits communication terms and is labeled ``simulated``.  L2 enables
    topology terms from the calibration profile and is labeled
    ``extrapolated``.  Both cost zero incremental GPU-hours; the one-time cost
    of collecting their calibration profile must be accounted for separately
    by an experiment manifest.
    """

    ESTIMATOR_VERSION = "topology-projector-v1"

    def __init__(
        self,
        profile: CalibrationProfile,
        workload: InferenceWorkload,
        expected_model: str = "",
        profile_path: Optional[Path] = None,
        scenario_path: Optional[Path] = None,
    ) -> None:
        if expected_model and profile.model.model_id != expected_model:
            raise TopologySandboxError(
                "profile model %s does not match scenario model %s"
                % (profile.model.model_id, expected_model)
            )
        self.profile = profile
        self.workload = workload
        self.projector = TopologyProjector(profile)
        self.artifacts = self._artifact_provenance(profile_path, scenario_path)

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        if level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
            raise TopologySandboxError(
                "TopologySandboxExecutor supports only L0 and L2"
            )
        try:
            estimate = self.projector.predict(candidate, self.workload, level)
        except (ProjectionError, ValueError) as exc:
            raise TopologySandboxError(str(exc)) from exc

        kind = (
            EvidenceKind.SIMULATED
            if level == EvidenceLevel.L0
            else EvidenceKind.EXTRAPOLATED
        )
        source = next(
            point.source_artifact
            for point in self.profile.points
            if point.point_id == estimate.reference_point_id
        )
        hardware = self.profile.hardware
        provenance: Dict[str, Any] = {
            "executor": "topology-sandbox",
            "estimator_version": self.ESTIMATOR_VERSION,
            "deterministic": True,
            "seed_recorded_but_unused": seed,
            "evidence_semantics": (
                "compute-only simulation"
                if level == EvidenceLevel.L0
                else "topology-aware extrapolation, not a serving measurement"
            ),
            "profile_id": self.profile.profile_id,
            "profile_schema_version": self.profile.schema_version,
            "profile_publication_eligible": self.profile.publication_eligible,
            "uncertainty_calibrated": self.profile.uncertainty_calibrated,
            "reference_point_id": estimate.reference_point_id,
            "reference_source_artifact": source,
            "calibration_distance": estimate.calibration_distance,
            "relative_uncertainty": estimate.relative_uncertainty,
            "interval_coverage_target": (
                self.profile.assumptions.interval_coverage_target
            ),
            "model": {
                "id": self.profile.model.model_id,
                "revision": self.profile.model.revision,
            },
            "hardware": {
                "gpu_name": hardware.gpu_name,
                "gpus_per_node": hardware.gpus_per_node,
                "max_nodes": hardware.max_nodes,
                "topology_source": hardware.topology_source,
            },
            "decomposition": dict(estimate.decomposition),
            "validation_required": True,
            "metric_semantics": {
                "energy_scope": "GPU energy only; host and network energy excluded",
                "energy_j_per_1k_tokens": "alias for 1,000 output tokens",
                "goodput": "not estimated without a request-level SLO classifier",
            },
            **self.artifacts,
        }
        if not estimate.feasible:
            return EvidenceRecord(
                candidate_id=candidate.candidate_id,
                level=level,
                metrics={},
                gpu_hours=0.0,
                kind=kind,
                status=EvidenceStatus.FAILED,
                provenance=provenance,
                failure_reason=estimate.failure_reason,
            )
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=level,
            metrics=estimate.metrics,
            gpu_hours=0.0,
            kind=kind,
            provenance=provenance,
        )

    @staticmethod
    def _artifact_provenance(
        profile_path: Optional[Path], scenario_path: Optional[Path]
    ) -> Dict[str, str]:
        provenance: Dict[str, str] = {}
        for label, path in (
            ("profile", profile_path),
            ("scenario", scenario_path),
        ):
            if path is None:
                continue
            materialized = Path(path)
            try:
                digest = hashlib.sha256(materialized.read_bytes()).hexdigest()
            except OSError as exc:
                raise TopologySandboxError(
                    "cannot hash %s artifact %s" % (label, materialized)
                ) from exc
            provenance[label + "_path"] = str(materialized)
            provenance[label + "_sha256"] = digest
        return provenance
