"""Deterministic executor over recorded TokenPowerBench-style evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from tokenpoweragent.evidence import EvidenceRecord
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.schema import Candidate, EvidenceLevel


class EvidenceUnavailable(ExecutionError):
    pass


class ReplayExecutor(Executor):
    def __init__(self, records: List[EvidenceRecord]) -> None:
        self._index: Dict[Tuple[str, EvidenceLevel], List[EvidenceRecord]] = {}
        for record in records:
            self._index.setdefault((record.candidate_id, record.level), []).append(record)

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayExecutor":
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(EvidenceRecord.from_dict(json.loads(line)))
        return cls(records)

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        matches = self._index.get((candidate.candidate_id, level), [])
        if not matches:
            raise EvidenceUnavailable(
                "no replay evidence for %s at %s" % (candidate.candidate_id, level.name)
            )
        source = matches[seed % len(matches)]
        provenance = dict(source.provenance)
        provenance.update({"executor": "replay", "seed": seed})
        return replace(source, provenance=provenance)
