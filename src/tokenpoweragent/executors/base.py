"""Common execution boundary used by offline and live experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tokenpoweragent.evidence import EvidenceRecord
from tokenpoweragent.schema import Candidate, EvidenceLevel


class Executor(ABC):
    @abstractmethod
    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        """Acquire one evidence record for a candidate-level action."""


class ExecutionError(RuntimeError):
    """Base class for executor failures visible to the controller."""
