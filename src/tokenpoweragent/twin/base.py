"""Interface for cross-fidelity prediction and uncertainty updates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from tokenpoweragent.evidence import EvidenceRecord
from tokenpoweragent.schema import Candidate


@dataclass(frozen=True)
class Prediction:
    metrics: Mapping[str, float]
    uncertainty: float
    source_level: int


class EnergyTwin(ABC):
    @abstractmethod
    def update(self, evidence: EvidenceRecord) -> None:
        pass

    @abstractmethod
    def predict(self, candidate: Candidate) -> Prediction:
        pass
