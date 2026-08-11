"""Evidence records, provenance labels, and append-only storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tokenpoweragent.schema import EvidenceLevel, SLO


class EvidenceKind(str, Enum):
    SIMULATED = "simulated"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    MEASURED = "measured"
    VERIFIED = "verified"


class EvidenceStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class EvidenceRecord:
    candidate_id: str
    level: EvidenceLevel
    metrics: Mapping[str, float]
    gpu_hours: float
    kind: EvidenceKind
    status: EvidenceStatus = EvidenceStatus.SUCCEEDED
    provenance: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.gpu_hours < 0:
            raise ValueError("gpu_hours cannot be negative")
        if self.status == EvidenceStatus.SUCCEEDED and not self.metrics:
            raise ValueError("successful evidence requires metrics")
        if self.level == EvidenceLevel.L4 and self.kind != EvidenceKind.VERIFIED:
            raise ValueError("L4 evidence must be labeled verified")
        if self.kind == EvidenceKind.VERIFIED and self.level != EvidenceLevel.L4:
            raise ValueError("only L4 evidence can be labeled verified")

    @property
    def verified(self) -> bool:
        return self.level == EvidenceLevel.L4 and self.kind == EvidenceKind.VERIFIED

    def satisfies(self, slo: SLO) -> bool:
        return self.status == EvidenceStatus.SUCCEEDED and slo.accepts(self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["level"] = self.level.name
        raw["kind"] = self.kind.value
        raw["status"] = self.status.value
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            candidate_id=str(raw["candidate_id"]),
            level=EvidenceLevel.parse(raw["level"]),
            metrics={str(key): float(value) for key, value in raw.get("metrics", {}).items()},
            gpu_hours=float(raw.get("gpu_hours", 0.0)),
            kind=EvidenceKind(str(raw["kind"])),
            status=EvidenceStatus(str(raw.get("status", "succeeded"))),
            provenance=dict(raw.get("provenance", {})),
            failure_reason=raw.get("failure_reason"),
            created_at=str(raw.get("created_at", datetime.now(timezone.utc).isoformat())),
        )


class EvidenceStore:
    """Small in-memory store with optional JSONL persistence."""

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        self._records: List[EvidenceRecord] = list(records)

    @property
    def records(self) -> List[EvidenceRecord]:
        return list(self._records)

    def append(self, record: EvidenceRecord) -> None:
        self._records.append(record)

    def for_candidate(self, candidate_id: str) -> List[EvidenceRecord]:
        return [record for record in self._records if record.candidate_id == candidate_id]

    def has(self, candidate_id: str, level: EvidenceLevel) -> bool:
        return any(
            record.candidate_id == candidate_id and record.level == level
            for record in self._records
        )

    def has_successful(self, candidate_id: str, level: EvidenceLevel) -> bool:
        return any(
            record.candidate_id == candidate_id
            and record.level == level
            and record.status == EvidenceStatus.SUCCEEDED
            for record in self._records
        )

    @property
    def latest(self) -> Optional[EvidenceRecord]:
        return self._records[-1] if self._records else None

    def write_jsonl(self, path: Path) -> None:
        rows = [json.dumps(record.to_dict(), sort_keys=True) for record in self._records]
        Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    @classmethod
    def read_jsonl(cls, path: Path) -> "EvidenceStore":
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(EvidenceRecord.from_dict(json.loads(line)))
        return cls(records)
