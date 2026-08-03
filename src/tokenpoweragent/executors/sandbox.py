"""Single-GPU Docker sandbox with host-side DCGM energy measurement."""

from __future__ import annotations

import json
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO, Tuple

from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
)
from tokenpoweragent.executors.base import ExecutionError, Executor
from tokenpoweragent.schema import Candidate, EvidenceLevel


class SandboxExecutionError(ExecutionError):
    """Raised when a sandbox contract or host-side command fails."""


@dataclass(frozen=True)
class PowerState:
    current_w: float
    minimum_w: float
    maximum_w: float


def parse_power_state(text: str) -> PowerState:
    """Parse nvidia-smi's current, minimum, and maximum power-limit row."""

    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(",")]
        if len(values) != 3:
            continue
        try:
            current, minimum, maximum = (float(value) for value in values)
        except ValueError as exc:
            raise SandboxExecutionError("invalid nvidia-smi power output") from exc
        return PowerState(current, minimum, maximum)
    raise SandboxExecutionError("nvidia-smi returned no power-limit row")


def parse_dcgmi_energy_mj(text: str) -> int:
    """Extract DCGM field 156 from a one-sample dmon response."""

    for line in reversed(text.splitlines()):
        columns = line.split()
        if len(columns) >= 3 and columns[0] == "GPU":
            try:
                return int(columns[-1])
            except ValueError as exc:
                raise SandboxExecutionError("invalid DCGM energy value") from exc
    raise SandboxExecutionError("DCGM returned no GPU energy sample")


def parse_workload_result(text: str) -> Mapping[str, Any]:
    """Find and validate the JSON object emitted by the sandbox workload."""

    for line in reversed(text.splitlines()):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, Mapping):
            continue
        required = ("elapsed_ms", "operations", "tflops")
        if not all(key in raw for key in required):
            continue
        try:
            if any(float(raw[key]) <= 0 for key in required):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise SandboxExecutionError("workload returned invalid metrics") from exc
        return raw
    raise SandboxExecutionError("workload returned no valid result JSON")


class SandboxExecutor(Executor):
    """Run a typed, non-networked container and measure it outside the sandbox."""

    TELEMETRY_FIELDS = "100,101,150,155,156,157,160,203,204,112,240,241"

    def __init__(
        self,
        telemetry_dir: Path = Path("experiments/results/telemetry"),
        gpu_id: int = 0,
        sample_ms: int = 100,
        use_sudo: bool = True,
        settle_seconds: float = 0.5,
        timeout_seconds: float = 300.0,
    ) -> None:
        if gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        if sample_ms < 1:
            raise ValueError("sample_ms must be positive")
        if settle_seconds < 0 or timeout_seconds <= 0:
            raise ValueError("invalid sandbox timing configuration")
        self.telemetry_dir = Path(telemetry_dir)
        self.gpu_id = gpu_id
        self.sample_ms = sample_ms
        self.privileged_prefix: Tuple[str, ...] = ("sudo", "-n") if use_sudo else ()
        self.settle_seconds = settle_seconds
        self.timeout_seconds = timeout_seconds

    def docker_command(self, candidate: Candidate) -> Sequence[str]:
        """Render the fixed security envelope around a typed workload command."""

        image, workload_args, _ = self._candidate_contract(candidate)
        return [
            *self.privileged_prefix,
            "docker",
            "run",
            "--rm",
            "--gpus",
            "device=%d" % self.gpu_id,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--env",
            "CUDA_CACHE_PATH=/tmp/cuda-cache",
            image,
            *workload_args,
        ]

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        if level == EvidenceLevel.L0:
            raise SandboxExecutionError("L0 belongs to replay or simulation")
        if candidate.required_gpus != 1 or candidate.target_nodes != 1:
            raise SandboxExecutionError("SandboxExecutor supports one GPU on one node")

        image, _, requested_power_w = self._candidate_contract(candidate)
        image_id = self._image_id(image)
        power_state = self._query_power_state()
        if not power_state.minimum_w <= requested_power_w <= power_state.maximum_w:
            raise SandboxExecutionError(
                "power limit %.1f W is outside [%.1f, %.1f] W"
                % (requested_power_w, power_state.minimum_w, power_state.maximum_w)
            )

        artifact_stem = self._artifact_stem(candidate.candidate_id, seed)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = self.telemetry_dir / (artifact_stem + ".dcgm.txt")
        stdout_path = self.telemetry_dir / (artifact_stem + ".stdout.txt")
        stderr_path = self.telemetry_dir / (artifact_stem + ".stderr.txt")

        telemetry_process = None
        telemetry_stream = None
        wall_seconds = 0.0
        energy_before_mj = 0
        energy_after_mj = 0
        power_readback_w = power_state.current_w
        docker_result: subprocess.CompletedProcess[str]

        try:
            self._set_power_limit(requested_power_w)
            time.sleep(self.settle_seconds)
            power_readback_w = self._query_power_state().current_w
            if abs(power_readback_w - requested_power_w) > 1.0:
                raise SandboxExecutionError(
                    "power-limit readback %.1f W does not match request %.1f W"
                    % (power_readback_w, requested_power_w)
                )

            telemetry_process, telemetry_stream = self._start_telemetry(telemetry_path)
            time.sleep(max(0.2, 2 * self.sample_ms / 1000.0))
            energy_before_mj = self._read_energy_mj()
            started = time.perf_counter()
            docker_result = subprocess.run(
                self.docker_command(candidate),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            wall_seconds = time.perf_counter() - started
            energy_after_mj = self._read_energy_mj()
            stdout_path.write_text(docker_result.stdout, encoding="utf-8")
            stderr_path.write_text(docker_result.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxExecutionError("sandbox command failed: %s" % exc) from exc
        finally:
            try:
                if telemetry_process is not None and telemetry_stream is not None:
                    self._stop_telemetry(telemetry_process, telemetry_stream)
            finally:
                self._restore_power_limit(power_state.current_w)

        provenance = {
            "executor": "docker-sandbox",
            "image": image,
            "image_id": image_id,
            "docker_command": list(self.docker_command(candidate)),
            "gpu_id": self.gpu_id,
            "seed": seed,
            "power_limit_requested_w": requested_power_w,
            "power_limit_readback_w": power_readback_w,
            "power_limit_original_w": power_state.current_w,
            "energy_counter": "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
            "energy_field_id": 156,
            "measurement_boundary": "docker-run-start-to-exit",
            "sample_ms": self.sample_ms,
            "telemetry_path": str(telemetry_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        if "campaign_index" in candidate.config:
            provenance["campaign_index"] = int(candidate.config["campaign_index"])

        kind = EvidenceKind.VERIFIED if level == EvidenceLevel.L4 else EvidenceKind.MEASURED
        gpu_hours = wall_seconds / 3600.0
        if docker_result.returncode != 0:
            reason = "container exited with status %d" % docker_result.returncode
            return EvidenceRecord(
                candidate_id=candidate.candidate_id,
                level=level,
                metrics={},
                gpu_hours=gpu_hours,
                kind=kind,
                status=EvidenceStatus.FAILED,
                provenance=provenance,
                failure_reason=reason,
            )

        try:
            workload = parse_workload_result(docker_result.stdout)
        except SandboxExecutionError as exc:
            return EvidenceRecord(
                candidate_id=candidate.candidate_id,
                level=level,
                metrics={},
                gpu_hours=gpu_hours,
                kind=kind,
                status=EvidenceStatus.FAILED,
                provenance=provenance,
                failure_reason=str(exc),
            )

        energy_j = (energy_after_mj - energy_before_mj) / 1000.0
        operations = float(workload["operations"])
        if energy_j <= 0 or wall_seconds <= 0:
            raise SandboxExecutionError("non-positive measured energy or duration")
        metrics = {
            "energy_j": energy_j,
            "elapsed_ms": wall_seconds * 1000.0,
            "kernel_elapsed_ms": float(workload["elapsed_ms"]),
            "throughput_tflops": float(workload["tflops"]),
            "avg_power_w": energy_j / wall_seconds,
            "gross_j_per_tflop": energy_j / (operations / 1.0e12),
        }
        provenance["workload"] = dict(workload)
        provenance["energy_start_mj"] = energy_before_mj
        provenance["energy_end_mj"] = energy_after_mj
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=level,
            metrics=metrics,
            gpu_hours=gpu_hours,
            kind=kind,
            provenance=provenance,
        )

    def _candidate_contract(
        self, candidate: Candidate
    ) -> Tuple[str, Tuple[str, ...], float]:
        image = str(candidate.config.get("image", "")).strip()
        valid_image = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image)
        if not image or image.startswith("-") or not valid_image:
            raise SandboxExecutionError("candidate requires a valid container image")

        raw_command = candidate.config.get("command", ())
        if isinstance(raw_command, (str, bytes)) or not isinstance(raw_command, Sequence):
            raise SandboxExecutionError("sandbox command must be a sequence of arguments")
        command = tuple(str(argument) for argument in raw_command)
        if any("\x00" in argument for argument in command):
            raise SandboxExecutionError("sandbox command contains a null byte")
        try:
            power_limit_w = float(candidate.config["power_limit_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SandboxExecutionError("candidate requires numeric power_limit_w") from exc
        return image, command, power_limit_w

    def _query_power_state(self) -> PowerState:
        output = self._run_checked(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_id),
                "--query-gpu=power.limit,power.min_limit,power.max_limit",
                "--format=csv,noheader,nounits",
            ]
        )
        return parse_power_state(output)

    def _image_id(self, image: str) -> str:
        return self._run_checked(
            [
                *self.privileged_prefix,
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image,
            ]
        ).strip()

    def _set_power_limit(self, watts: float) -> None:
        self._run_checked(
            [
                *self.privileged_prefix,
                "nvidia-smi",
                "-i",
                str(self.gpu_id),
                "-pl",
                "%g" % watts,
            ]
        )

    def _restore_power_limit(self, watts: float) -> None:
        try:
            self._set_power_limit(watts)
        except SandboxExecutionError as exc:
            warnings.warn(
                "failed to restore GPU power limit to %.1f W: %s" % (watts, exc),
                RuntimeWarning,
            )

    def _read_energy_mj(self) -> int:
        output = self._run_checked(
            [
                "dcgmi",
                "dmon",
                "-i",
                "gpu:%d" % self.gpu_id,
                "-e",
                "156",
                "-c",
                "1",
            ]
        )
        return parse_dcgmi_energy_mj(output)

    def _start_telemetry(
        self, path: Path
    ) -> Tuple[subprocess.Popen[str], TextIO]:
        stream = path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    "stdbuf",
                    "-oL",
                    "-eL",
                    "dcgmi",
                    "dmon",
                    "-i",
                    "gpu:%d" % self.gpu_id,
                    "-e",
                    self.TELEMETRY_FIELDS,
                    "-d",
                    str(self.sample_ms),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError:
            stream.close()
            raise
        return process, stream

    @staticmethod
    def _stop_telemetry(process: subprocess.Popen[str], stream: TextIO) -> None:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
        finally:
            stream.close()

    def _run_checked(self, command: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxExecutionError("command failed: %s" % exc) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise SandboxExecutionError(
                "command exited with status %d: %s" % (result.returncode, detail)
            )
        return result.stdout

    @staticmethod
    def _artifact_stem(candidate_id: str, seed: int) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate_id).strip("-") or "candidate"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return "%s-seed%d-%s" % (safe_id, seed, timestamp)
