"""Common execution boundary used by offline and live experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

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


class RoutedExecutor(Executor):
    """Route each evidence level to its owning acquisition backend."""

    def __init__(self, routes: Mapping[EvidenceLevel, Executor]) -> None:
        self.routes = dict(routes)
        if not self.routes:
            raise ValueError("RoutedExecutor requires at least one route")

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        try:
            executor = self.routes[level]
        except KeyError as exc:
            raise ExecutionError("no executor route for %s" % level.name) from exc
        return executor.execute(candidate, level, seed)
