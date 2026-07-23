"""Transparent empirical Twin used by the runnable replay demonstration."""

from __future__ import annotations

from typing import Dict, List

from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus
from tokenpoweragent.schema import Candidate, EvidenceLevel
from tokenpoweragent.twin.base import EnergyTwin, Prediction


class EmpiricalEnergyTwin(EnergyTwin):
    """Uses the highest-fidelity observation and explicit uncertainty tiers.

    This baseline keeps the package executable while the paper's calibrated
    probabilistic Twin is implemented. It deliberately exposes that limitation
    instead of presenting a heuristic as a learned posterior.
    """

    _UNCERTAINTY = {
        EvidenceLevel.L0: 0.45,
        EvidenceLevel.L1: 0.34,
        EvidenceLevel.L2: 0.22,
        EvidenceLevel.L3: 0.12,
        EvidenceLevel.L4: 0.03,
    }

    def __init__(self) -> None:
        self._records: Dict[str, List[EvidenceRecord]] = {}

    def update(self, evidence: EvidenceRecord) -> None:
        self._records.setdefault(evidence.candidate_id, []).append(evidence)

    def predict(self, candidate: Candidate) -> Prediction:
        successful = [
            record
            for record in self._records.get(candidate.candidate_id, [])
            if record.status == EvidenceStatus.SUCCEEDED
        ]
        if not successful:
            if not candidate.prior_metrics:
                raise ValueError(
                    "candidate %s has no prior metrics" % candidate.candidate_id
                )
            return Prediction(candidate.prior_metrics, 0.70, -1)

        best = max(successful, key=lambda record: int(record.level))
        uncertainty = self._UNCERTAINTY[best.level]
        if candidate.target_nodes > 1 and best.level < EvidenceLevel.L3:
            uncertainty = min(1.0, uncertainty + 0.15)
        return Prediction(best.metrics, uncertainty, int(best.level))
