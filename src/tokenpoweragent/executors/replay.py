"""Deterministic executor over recorded TokenPowerBench-style evidence."""

from __future__ import annotations

import hashlib
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


class BootstrapReplayExecutor(ReplayExecutor):
    """Replay an empirical evidence world with common random numbers.

    A benchmark episode should not bind every candidate and fidelity to the
    same replicate index. Doing so creates only ``repeat_count`` distinct
    worlds regardless of the requested episode count. This executor instead
    derives a stable replicate index from the episode seed and action identity.
    The mapping is shared across policies, so policy comparisons still use
    common random numbers while candidate-level measurement noise is sampled
    independently.
    """

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        matches = self._index.get((candidate.candidate_id, level), [])
        if not matches:
            raise EvidenceUnavailable(
                "no replay evidence for %s at %s"
                % (candidate.candidate_id, level.name)
            )
        selection_key = "%d\0%s\0%s" % (seed, candidate.candidate_id, level.name)
        key_hash = hashlib.sha256(selection_key.encode("utf-8"))
        digest = key_hash.digest()
        selection_index = int.from_bytes(digest[:8], "big") % len(matches)
        source = matches[selection_index]
        provenance = dict(source.provenance)
        provenance.update(
            {
                "executor": "bootstrap-replay",
                "seed": seed,
                "replay_selection_index": selection_index,
                "replay_selection_key_sha256": key_hash.hexdigest(),
            }
        )
        return replace(source, provenance=provenance)
