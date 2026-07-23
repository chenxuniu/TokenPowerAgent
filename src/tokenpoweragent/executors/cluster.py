"""Slurm rendering and injectable live-cluster execution boundary."""

from __future__ import annotations

import math
from typing import Callable, Optional

from tokenpoweragent.evidence import EvidenceRecord
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.schema import Candidate, EvidenceLevel


class LiveExecutionDisabled(ExecutionError):
    """Raised when no site-specific Slurm/telemetry runner is configured."""

    def __init__(self, script: str) -> None:
        super().__init__("live execution is disabled; inspect the rendered script")
        self.script = script


Runner = Callable[[str, Candidate, EvidenceLevel, int], EvidenceRecord]


class ClusterExecutor(Executor):
    def __init__(
        self,
        partition: str = "gpu",
        account: Optional[str] = None,
        runner: Optional[Runner] = None,
    ) -> None:
        self.partition = partition
        self.account = account
        self.runner = runner

    def resources(self, candidate: Candidate, level: EvidenceLevel) -> tuple:
        if level == EvidenceLevel.L0:
            return (0, 0)
        if level == EvidenceLevel.L1:
            return (1, 1)
        if level == EvidenceLevel.L2:
            return (1, max(1, int(math.ceil(candidate.required_gpus / candidate.target_nodes))))
        return (candidate.target_nodes, candidate.required_gpus)

    def render_slurm(
        self, candidate: Candidate, level: EvidenceLevel, seed: int = 0
    ) -> str:
        nodes, total_gpus = self.resources(candidate, level)
        if nodes == 0:
            raise ExecutionError("L0 runs in the simulator/replay backend")
        gpus_per_node = int(math.ceil(total_gpus / nodes))
        command = str(
            candidate.config.get(
                "command",
                "tokenpowerbench run --config configs/%s.yaml" % candidate.candidate_id,
            )
        )
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=tpa-%s-%s" % (candidate.candidate_id, level.name.lower()),
            "#SBATCH --partition=%s" % self.partition,
            "#SBATCH --nodes=%d" % nodes,
            "#SBATCH --gpus-per-node=%d" % gpus_per_node,
            "#SBATCH --output=artifacts/%s-%s-%%j.out"
            % (candidate.candidate_id, level.name.lower()),
        ]
        if self.account:
            lines.append("#SBATCH --account=%s" % self.account)
        lines.extend(
            [
                "",
                "set -euo pipefail",
                "export TOKENPOWERAGENT_SEED=%d" % seed,
                command,
            ]
        )
        return "\n".join(lines) + "\n"

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        script = self.render_slurm(candidate, level, seed)
        if self.runner is None:
            raise LiveExecutionDisabled(script)
        return self.runner(script, candidate, level, seed)
