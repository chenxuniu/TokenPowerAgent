"""Resumable execution of a frozen serving-configuration campaign."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    ConfigurationPoint,
    configuration_measurement_schedule,
    sha256_file,
)
from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.executors.sandbox import SandboxExecutionError
from tokenpoweragent.executors.serving import (
    ServerEnvironment,
    ServingSandboxExecutor,
)
from tokenpoweragent.schema import Candidate, EvidenceLevel


class ConfigurationRunnerError(ValueError):
    """Raised when live execution would violate the frozen campaign."""


@dataclass(frozen=True)
class FrozenConfigurationArtifacts:
    campaign_sha256: str
    prediction_sha256: str
    schedule_sha256: str
    summary_sha256: str
    manifest_sha256: str


def verify_frozen_configuration_campaign(
    campaign_path: Path,
    predictions_path: Path,
    schedule_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> FrozenConfigurationArtifacts:
    """Verify every frozen artifact and its cross-file campaign contract."""

    entries = _read_manifest(manifest_path)
    for listed, expected in entries.items():
        artifact = Path(listed)
        if not artifact.exists():
            raise ConfigurationRunnerError(
                "frozen manifest artifact is missing: %s" % listed
            )
        if sha256_file(artifact) != expected:
            raise ConfigurationRunnerError(
                "frozen artifact hash mismatch: %s" % listed
            )

    campaign_hash = _manifest_digest(entries, campaign_path)
    prediction_hash = _manifest_digest(entries, predictions_path)
    schedule_hash = _manifest_digest(entries, schedule_path)
    summary_hash = _manifest_digest(entries, summary_path)
    campaign = ConfigurationCampaign.load(campaign_path)

    predictions = EvidenceStore.read_jsonl(predictions_path).records
    expected_ids = {point.point_id for point in campaign.candidates}
    if len(predictions) != len(campaign.candidates):
        raise ConfigurationRunnerError(
            "frozen prediction count does not match the campaign"
        )
    if {record.candidate_id for record in predictions} != expected_ids:
        raise ConfigurationRunnerError(
            "frozen prediction ids do not match the campaign"
        )
    for record in predictions:
        if record.level != EvidenceLevel.L0 or record.status != EvidenceStatus.SUCCEEDED:
            raise ConfigurationRunnerError(
                "configuration campaign requires successful L0 predictions"
            )
        if record.provenance.get("campaign_sha256") != campaign_hash:
            raise ConfigurationRunnerError(
                "frozen prediction campaign hash is inconsistent"
            )

    try:
        schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationRunnerError(
            "cannot read frozen schedule or summary: %s" % exc
        ) from exc
    if not isinstance(schedule, Mapping) or not isinstance(summary, Mapping):
        raise ConfigurationRunnerError(
            "frozen schedule and summary must be JSON objects"
        )
    expected_schedule = configuration_measurement_schedule(
        campaign, campaign_hash
    )
    if schedule != expected_schedule:
        raise ConfigurationRunnerError(
            "frozen measurement schedule does not match the campaign"
        )
    if summary.get("campaign_sha256") != campaign_hash:
        raise ConfigurationRunnerError("frozen summary campaign hash differs")
    if int(summary.get("measurement_count", -1)) != len(schedule["actions"]):
        raise ConfigurationRunnerError("frozen summary measurement count differs")

    return FrozenConfigurationArtifacts(
        campaign_sha256=campaign_hash,
        prediction_sha256=prediction_hash,
        schedule_sha256=schedule_hash,
        summary_sha256=summary_hash,
        manifest_sha256=sha256_file(manifest_path),
    )


class DockerVLLMServerManager:
    """Recreate a pinned, network-isolated vLLM server for each block."""

    def __init__(
        self,
        campaign: ConfigurationCampaign,
        server_log_dir: Path,
        gpu_id: int = 0,
        use_sudo: bool = True,
    ) -> None:
        if gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        self.campaign = campaign
        self.server_log_dir = Path(server_log_dir)
        self.gpu_id = gpu_id
        self.privileged_prefix: Tuple[str, ...] = (
            ("sudo", "-n") if use_sudo else ()
        )
        self.current_candidate_id: Optional[str] = None
        self.current_log_path: Optional[Path] = None
        self.restart_count = 0

    @property
    def expected_server_image_id(self) -> str:
        digest = str(self.campaign.runtime["server_image"]).rsplit(
            "@sha256:", 1
        )[1]
        return "sha256:" + digest

    def verify_host(self) -> None:
        hostname = self._run_checked(["hostname"]).strip()
        if hostname != str(self.campaign.hardware["hostname"]):
            raise ConfigurationRunnerError(
                "host mismatch: expected %s, found %s"
                % (self.campaign.hardware["hostname"], hostname)
            )
        gpu_row = self._run_checked(
            [
                "nvidia-smi",
                "--id=%d" % self.gpu_id,
                "--query-gpu=name,uuid,driver_version,power.limit",
                "--format=csv,noheader,nounits",
            ]
        ).strip()
        columns = [item.strip() for item in gpu_row.split(",")]
        if len(columns) != 4:
            raise ConfigurationRunnerError("cannot parse live GPU identity")
        expected = self.campaign.hardware
        if columns[:3] != [
            str(expected["gpu_name"]),
            str(expected["gpu_uuid"]),
            str(expected["driver_version"]),
        ]:
            raise ConfigurationRunnerError(
                "live GPU identity differs from the frozen campaign"
            )
        try:
            power_limit_w = float(columns[3])
        except ValueError as exc:
            raise ConfigurationRunnerError(
                "cannot parse live GPU power limit"
            ) from exc
        if abs(power_limit_w - self.campaign.power_limit_w) > 1.0:
            raise ConfigurationRunnerError(
                "live GPU power limit differs from the frozen campaign"
            )

        client_id = self._docker_checked(
            [
                "image",
                "inspect",
                str(self.campaign.runtime["client_image"]),
                "--format",
                "{{.Id}}",
            ]
        ).strip()
        if client_id != str(self.campaign.runtime["client_image_id"]):
            raise ConfigurationRunnerError(
                "benchmark client image ID differs from the frozen campaign"
            )
        server_id = self._docker_checked(
            [
                "image",
                "inspect",
                str(self.campaign.runtime["server_image"]),
                "--format",
                "{{.Id}}",
            ]
        ).strip()
        if server_id != self.expected_server_image_id:
            raise ConfigurationRunnerError(
                "vLLM image ID differs from the frozen campaign"
            )
        internal = self._docker_checked(
            [
                "network",
                "inspect",
                "--format",
                "{{.Internal}}",
                str(self.campaign.runtime["network"]),
            ]
        ).strip()
        if internal != "true":
            raise ConfigurationRunnerError(
                "benchmark network must remain internal"
            )
        self._docker_checked(
            ["volume", "inspect", str(self.campaign.runtime["cache_volume"])]
        )

    def docker_run_command(
        self, point: ConfigurationPoint, block_index: int
    ) -> Sequence[str]:
        config = point.configuration
        chunk_flag = (
            "--enable-chunked-prefill"
            if config["chunked_prefill"]
            else "--no-enable-chunked-prefill"
        )
        return [
            *self.privileged_prefix,
            "docker",
            "run",
            "-d",
            "--name",
            str(self.campaign.runtime["server_container"]),
            "--label",
            "tokenpoweragent.campaign=%s" % self.campaign.campaign_id,
            "--label",
            "tokenpoweragent.candidate=%s" % point.point_id,
            "--label",
            "tokenpoweragent.block=%d" % block_index,
            "--gpus",
            "device=%d" % self.gpu_id,
            "--shm-size=16g",
            "--network",
            str(self.campaign.runtime["network"]),
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "HF_HOME=/root/.cache/huggingface",
            "--env",
            "VLLM_NO_USAGE_STATS=1",
            "--volume",
            "%s:/root/.cache/huggingface"
            % self.campaign.runtime["cache_volume"],
            str(self.campaign.runtime["server_image"]),
            self.campaign.model_id,
            "--revision",
            self.campaign.model_revision,
            "--tokenizer-revision",
            str(self.campaign.model["tokenizer_revision"]),
            "--served-model-name",
            str(self.campaign.runtime["served_model_name"]),
            "--dtype",
            str(config["precision"]),
            "--max-model-len",
            str(self.campaign.model["max_model_len"]),
            "--gpu-memory-utilization",
            "%g" % self.campaign.runtime["gpu_memory_utilization"],
            "--tensor-parallel-size",
            str(config["tensor_parallel"]),
            "--pipeline-parallel-size",
            str(config["pipeline_parallel"]),
            "--max-num-seqs",
            str(config["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(config["max_num_batched_tokens"]),
            chunk_flag,
            "--no-enable-prefix-caching",
            "--generation-config",
            str(self.campaign.runtime["generation_config"]),
            "--seed",
            str(self.campaign.runtime["seed"]),
        ]

    def restart(self, point: ConfigurationPoint, block_index: int) -> None:
        self.server_log_dir.mkdir(parents=True, exist_ok=True)
        if self._container_exists():
            if self.current_log_path is None:
                self.current_log_path = (
                    self.server_log_dir / "pre-campaign-server.log"
                )
            self.capture_current_logs()
            self._docker_checked(
                [
                    "stop",
                    "--time",
                    "30",
                    str(self.campaign.runtime["server_container"]),
                ]
            )
            self._docker_checked(
                ["rm", str(self.campaign.runtime["server_container"])]
            )

        self.current_log_path = self.server_log_dir / (
            "block-%03d-%s.log" % (block_index, point.point_id)
        )
        self._run_checked(self.docker_run_command(point, block_index))
        self.current_candidate_id = point.point_id
        self.restart_count += 1
        try:
            self._wait_until_ready()
            self._wait_for_cooldown()
        except Exception:
            self.capture_current_logs()
            raise

    def capture_current_logs(self) -> None:
        if self.current_log_path is None or not self._container_exists():
            return
        process = self._docker(
            [
                "logs",
                "--tail",
                "500",
                str(self.campaign.runtime["server_container"]),
            ],
            check=False,
        )
        self.current_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_log_path.write_text(
            (process.stdout or "") + (process.stderr or ""),
            encoding="utf-8",
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.campaign.server_ready_timeout_seconds
        health_script = (
            "import urllib.request; "
            "urllib.request.urlopen('http://127.0.0.1:8000/health', "
            "timeout=2).read()"
        )
        while time.monotonic() < deadline:
            status = self._docker(
                [
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    str(self.campaign.runtime["server_container"]),
                ],
                check=False,
            )
            if status.returncode != 0 or status.stdout.strip() != "true":
                raise ConfigurationRunnerError(
                    "vLLM server exited before becoming healthy"
                )
            health = self._docker(
                [
                    "exec",
                    str(self.campaign.runtime["server_container"]),
                    "python3",
                    "-c",
                    health_script,
                ],
                check=False,
            )
            if health.returncode == 0:
                return
            time.sleep(5)
        raise ConfigurationRunnerError(
            "vLLM health check exceeded %.0f seconds"
            % self.campaign.server_ready_timeout_seconds
        )

    def _wait_for_cooldown(self) -> None:
        deadline = time.monotonic() + self.campaign.server_ready_timeout_seconds
        while time.monotonic() < deadline:
            raw = self._run_checked(
                [
                    "nvidia-smi",
                    "--id=%d" % self.gpu_id,
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ).strip()
            try:
                temperature_c = float(raw)
            except ValueError as exc:
                raise ConfigurationRunnerError(
                    "cannot parse GPU temperature"
                ) from exc
            if temperature_c <= self.campaign.cooldown_temperature_c:
                return
            time.sleep(5)
        raise ConfigurationRunnerError(
            "GPU did not cool to %.1f C before measurement"
            % self.campaign.cooldown_temperature_c
        )

    def _container_exists(self) -> bool:
        process = self._docker(
            ["inspect", str(self.campaign.runtime["server_container"])],
            check=False,
        )
        return process.returncode == 0

    def _docker_checked(self, arguments: Sequence[str]) -> str:
        return self._run_checked(
            [*self.privileged_prefix, "docker", *arguments]
        )

    def _docker(
        self, arguments: Sequence[str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.privileged_prefix, "docker", *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _run_checked(command: Sequence[str]) -> str:
        try:
            return subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = ""
            if isinstance(exc, subprocess.CalledProcessError):
                detail = (exc.stderr or exc.stdout or "").strip()
            raise ConfigurationRunnerError(
                "command failed: %s%s"
                % (" ".join(command), (": " + detail) if detail else "")
            ) from exc


def run_configuration_campaign(
    campaign_path: Path,
    predictions_path: Path,
    schedule_path: Path,
    summary_path: Path,
    manifest_path: Path,
    output_path: Path,
    executor: ServingSandboxExecutor,
    server_manager: DockerVLLMServerManager,
    resume: bool = False,
    max_actions: int = 0,
) -> Mapping[str, Any]:
    campaign = ConfigurationCampaign.load(campaign_path)
    if max_actions < 0:
        raise ConfigurationRunnerError("max_actions must be non-negative")
    if executor.sample_ms != campaign.sample_ms:
        raise ConfigurationRunnerError(
            "executor sample interval does not match the campaign"
        )
    artifacts = verify_frozen_configuration_campaign(
        campaign_path,
        predictions_path,
        schedule_path,
        summary_path,
        manifest_path,
    )
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    actions = list(schedule["actions"])
    point_map = {point.point_id: point for point in campaign.candidates}

    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise ConfigurationRunnerError(
            "measurement output exists; pass resume=True to continue"
        )
    records = (
        EvidenceStore.read_jsonl(output_path)
        if output_path.exists()
        else EvidenceStore()
    )
    if len(records.records) > len(actions):
        raise ConfigurationRunnerError(
            "measurement output is longer than the frozen schedule"
        )
    for index, record in enumerate(records.records):
        action = actions[index]
        point = point_map[str(action["candidate_id"])]
        verify_configuration_measurement_record(
            campaign, point, action, record, artifacts
        )

    server_manager.verify_host()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    starting_index = len(records.records)
    pending_actions = actions[starting_index:]
    if max_actions:
        pending_actions = pending_actions[:max_actions]
    added = 0
    for action in pending_actions:
        action_index = int(action["action_index"])
        point = point_map[str(action["candidate_id"])]
        level = EvidenceLevel.parse(action["level"])
        recovery_restart = (
            action_index == starting_index and not bool(action["restart_server"])
        )
        if bool(action["restart_server"]) or recovery_restart:
            server_manager.restart(point, int(action["block_index"]))
            restart_reason = (
                "scheduled" if action["restart_server"] else "resume-recovery"
            )
        else:
            if server_manager.current_candidate_id != point.point_id:
                raise ConfigurationRunnerError(
                    "server candidate changed inside a frozen measurement block"
                )
            restart_reason = "none"

        candidate = _serving_candidate(campaign, point, action)
        environment = executor.inspect_environment(candidate)
        _assert_server_environment(
            campaign,
            point,
            environment,
            server_manager.expected_server_image_id,
        )
        try:
            record = executor.execute(
                candidate, level, seed=int(action["seed"])
            )
        except SandboxExecutionError:
            server_manager.capture_current_logs()
            raise
        _assert_measurement_record(
            campaign,
            point,
            action,
            record,
            server_manager.expected_server_image_id,
        )
        record = _annotate_measurement(
            campaign,
            point,
            action,
            record,
            artifacts,
            restart_reason,
            server_manager.current_log_path,
        )
        records.append(record)
        server_manager.capture_current_logs()
        _write_jsonl_atomic(records, output_path)
        added += 1
        if record.status != EvidenceStatus.SUCCEEDED:
            raise ConfigurationRunnerError(
                "campaign checkpointed a failed action at index %d; "
                "resume continues without selectively retrying it"
                % action_index
            )

    server_manager.capture_current_logs()
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": artifacts.campaign_sha256,
        "prediction_sha256": artifacts.prediction_sha256,
        "schedule_sha256": artifacts.schedule_sha256,
        "scheduled_measurements": len(actions),
        "existing_measurements": starting_index,
        "added_measurements": added,
        "completed_measurements": len(records.records),
        "remaining_measurements": len(actions) - len(records.records),
        "campaign_complete": len(records.records) == len(actions),
        "server_restarts_this_process": server_manager.restart_count,
        "output": str(output_path),
    }


def _serving_candidate(
    campaign: ConfigurationCampaign,
    point: ConfigurationPoint,
    action: Mapping[str, Any],
) -> Candidate:
    workload = campaign.target_workload
    rate = workload.request_rate_req_s
    return Candidate(
        candidate_id=point.point_id,
        required_gpus=1,
        target_nodes=1,
        config={
            **point.configuration,
            "image": campaign.runtime["client_image"],
            "server_container": campaign.runtime["server_container"],
            "network": campaign.runtime["network"],
            "cache_volume": campaign.runtime["cache_volume"],
            "base_url": campaign.runtime["base_url"],
            "model": campaign.model_id,
            "served_model_name": campaign.runtime["served_model_name"],
            "power_limit_w": campaign.power_limit_w,
            "campaign_index": int(action["action_index"]),
            "input_len": workload.input_tokens,
            "output_len": workload.output_tokens,
            "num_prompts": int(action["num_prompts"]),
            "num_warmups": campaign.num_warmups,
            "request_rate": "inf" if rate is None else "%g" % rate,
            "max_concurrency": workload.concurrency,
            "dataset_split": campaign.dataset_split,
        },
    )


def _assert_server_environment(
    campaign: ConfigurationCampaign,
    point: ConfigurationPoint,
    environment: ServerEnvironment,
    expected_image_id: str,
) -> None:
    if environment.image != campaign.runtime["server_image"]:
        raise ConfigurationRunnerError("live vLLM image reference drifted")
    if environment.image_id != expected_image_id:
        raise ConfigurationRunnerError("live vLLM image ID drifted")
    actual = environment.configuration
    expected = point.configuration
    for key in (
        "engine",
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "max_num_seqs",
        "max_num_batched_tokens",
        "chunked_prefill",
        "precision",
        "kv_cache_dtype",
        "prefix_caching",
        "model_revision",
        "tokenizer_revision",
    ):
        if actual.get(key) != expected.get(key):
            raise ConfigurationRunnerError(
                "server configuration differs at %s: expected %r, found %r"
                % (key, expected.get(key), actual.get(key))
            )


def _assert_measurement_record(
    campaign: ConfigurationCampaign,
    point: ConfigurationPoint,
    action: Mapping[str, Any],
    record: EvidenceRecord,
    expected_server_image_id: str,
) -> None:
    level = EvidenceLevel.parse(action["level"])
    if record.candidate_id != point.point_id or record.level != level:
        raise ConfigurationRunnerError("measurement candidate or level drifted")
    provenance = record.provenance
    expected_scalars = {
        "dataset_split": campaign.dataset_split,
        "model": campaign.model_id,
        "client_image_id": campaign.runtime["client_image_id"],
        "server_image": campaign.runtime["server_image"],
        "server_image_id": expected_server_image_id,
    }
    for key, expected in expected_scalars.items():
        if provenance.get(key) != expected:
            raise ConfigurationRunnerError(
                "measurement provenance differs at %s" % key
            )
    try:
        power_limit_w = float(provenance["power_limit_readback_w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationRunnerError(
            "measurement lacks a valid power limit"
        ) from exc
    if (
        not math.isfinite(power_limit_w)
        or abs(power_limit_w - campaign.power_limit_w) > 1.0
    ):
        raise ConfigurationRunnerError("measurement power limit drifted")
    configuration = provenance.get("server_configuration")
    if not isinstance(configuration, Mapping):
        raise ConfigurationRunnerError(
            "measurement lacks a server configuration"
        )
    _assert_server_environment(
        campaign,
        point,
        ServerEnvironment(
            image=str(provenance.get("server_image", "")),
            image_id=str(provenance.get("server_image_id", "")),
            command=tuple(provenance.get("server_command", ())),
            configuration=configuration,
        ),
        expected_server_image_id,
    )
    workload = campaign.target_workload
    expected_workload = {
        "input_len": workload.input_tokens,
        "output_len": workload.output_tokens,
        "num_prompts": int(action["num_prompts"]),
        "num_warmups": campaign.num_warmups,
        "request_rate": (
            "inf"
            if workload.request_rate_req_s is None
            else "%g" % workload.request_rate_req_s
        ),
        "max_concurrency": workload.concurrency,
        "temperature": 0,
        "ignore_eos": True,
    }
    if provenance.get("workload") != expected_workload:
        raise ConfigurationRunnerError("measurement workload drifted")
    if record.status == EvidenceStatus.SUCCEEDED:
        for metric in (
            "energy_j_per_1k_output_tokens",
            "throughput_tok_s",
            "ttft_ms",
            "tpot_ms",
        ):
            value = record.metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ConfigurationRunnerError(
                    "measurement lacks finite metric %s" % metric
                )


def _annotate_measurement(
    campaign: ConfigurationCampaign,
    point: ConfigurationPoint,
    action: Mapping[str, Any],
    record: EvidenceRecord,
    artifacts: FrozenConfigurationArtifacts,
    restart_reason: str,
    server_log_path: Optional[Path],
) -> EvidenceRecord:
    provenance = dict(record.provenance)
    provenance.update(
        {
            "campaign_id": campaign.campaign_id,
            "campaign_sha256": artifacts.campaign_sha256,
            "frozen_prediction_sha256": artifacts.prediction_sha256,
            "frozen_schedule_sha256": artifacts.schedule_sha256,
            "frozen_summary_sha256": artifacts.summary_sha256,
            "freeze_manifest_sha256": artifacts.manifest_sha256,
            "campaign_action_index": int(action["action_index"]),
            "campaign_block_index": int(action["block_index"]),
            "campaign_repeat": int(action["repeat"]),
            "candidate_role": point.role,
            "frozen_serving_configuration": dict(point.configuration),
            "server_restart_reason": restart_reason,
            "server_log_path": (
                str(server_log_path) if server_log_path is not None else None
            ),
            "evidence_level_semantics": (
                "measured-short-probe"
                if record.level == EvidenceLevel.L1
                else "target-workload-verified"
            ),
        }
    )
    return replace(record, provenance=provenance)


def verify_configuration_measurement_record(
    campaign: ConfigurationCampaign,
    point: ConfigurationPoint,
    action: Mapping[str, Any],
    record: EvidenceRecord,
    artifacts: FrozenConfigurationArtifacts,
) -> None:
    provenance = record.provenance
    expected = {
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": artifacts.campaign_sha256,
        "frozen_prediction_sha256": artifacts.prediction_sha256,
        "frozen_schedule_sha256": artifacts.schedule_sha256,
        "frozen_summary_sha256": artifacts.summary_sha256,
        "freeze_manifest_sha256": artifacts.manifest_sha256,
        "campaign_action_index": int(action["action_index"]),
        "campaign_block_index": int(action["block_index"]),
        "campaign_repeat": int(action["repeat"]),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ConfigurationRunnerError(
                "existing measurement differs at %s" % key
            )
    _assert_measurement_record(
        campaign,
        point,
        action,
        record,
        "sha256:"
        + str(campaign.runtime["server_image"]).rsplit("@sha256:", 1)[1],
    )


def _write_jsonl_atomic(records: EvidenceStore, output_path: Path) -> None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    records.write_jsonl(temporary)
    temporary.replace(output_path)


def _read_manifest(path: Path) -> Mapping[str, str]:
    entries: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise ConfigurationRunnerError("freeze manifest has an invalid line")
        listed = parts[1].lstrip("* ")
        if listed in entries:
            raise ConfigurationRunnerError("freeze manifest has duplicate paths")
        entries[listed] = parts[0].lower()
    if len(entries) != 6:
        raise ConfigurationRunnerError(
            "configuration freeze manifest must contain six artifacts"
        )
    return entries


def _manifest_digest(entries: Mapping[str, str], path: Path) -> str:
    matches = [
        digest
        for listed, digest in entries.items()
        if listed == str(path) or Path(listed).name == Path(path).name
    ]
    if len(matches) != 1:
        raise ConfigurationRunnerError(
            "freeze manifest must contain exactly one entry for %s" % path
        )
    actual = sha256_file(path)
    if matches[0] != actual:
        raise ConfigurationRunnerError("frozen artifact hash mismatch: %s" % path)
    return actual
