"""Persistent-vLLM serving probe with marker-aligned DCGM measurement."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord, EvidenceStatus
from tokenpoweragent.executors.sandbox import SandboxExecutionError, SandboxExecutor
from tokenpoweragent.schema import Candidate, EvidenceLevel


_RESULT_LABELS = {
    "Successful requests": "successful_requests",
    "Failed requests": "failed_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
    "Total input tokens": "input_tokens",
    "Total generated tokens": "output_tokens",
    "Request throughput (req/s)": "request_throughput_req_s",
    "Output token throughput (tok/s)": "output_throughput_tok_s",
    "Total token throughput (tok/s)": "total_throughput_tok_s",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "P50 TPOT (ms)": "p50_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "P50 ITL (ms)": "p50_itl_ms",
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
    "Mean E2EL (ms)": "mean_e2el_ms",
    "P50 E2EL (ms)": "p50_e2el_ms",
    "P95 E2EL (ms)": "p95_e2el_ms",
    "P99 E2EL (ms)": "p99_e2el_ms",
}

_REQUIRED_RESULT_KEYS = {
    "successful_requests",
    "failed_requests",
    "benchmark_duration_s",
    "input_tokens",
    "output_tokens",
    "request_throughput_req_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
}


def parse_vllm_benchmark_output(text: str) -> Mapping[str, float]:
    """Parse the stable human-readable result block from vLLM 0.23."""

    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, raw_value = line.rsplit(":", 1)
        key = _RESULT_LABELS.get(label.strip())
        if key is None:
            continue
        match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", raw_value)
        if match:
            metrics[key] = float(match.group(1))

    missing = sorted(_REQUIRED_RESULT_KEYS - metrics.keys())
    if missing:
        raise SandboxExecutionError(
            "vLLM benchmark output is missing metrics: %s" % ", ".join(missing)
        )
    if metrics["successful_requests"] <= 0:
        raise SandboxExecutionError("vLLM benchmark completed no requests")
    if metrics["failed_requests"] != 0:
        raise SandboxExecutionError(
            "vLLM benchmark reported %d failed requests"
            % int(metrics["failed_requests"])
        )
    if metrics["benchmark_duration_s"] <= 0 or metrics["output_tokens"] <= 0:
        raise SandboxExecutionError("vLLM benchmark returned non-positive work")
    return metrics


@dataclass(frozen=True)
class ServingContract:
    image: str
    server_container: str
    network: str
    cache_volume: str
    base_url: str
    model: str
    served_model_name: str
    power_limit_w: float
    input_len: int
    output_len: int
    num_prompts: int
    num_warmups: int
    request_rate: str
    max_concurrency: int


@dataclass(frozen=True)
class MarkedRun:
    returncode: int
    output: str
    energy_start_mj: int
    energy_end_mj: int
    window_seconds: float


@dataclass(frozen=True)
class ServerEnvironment:
    image: str
    image_id: str
    command: Tuple[str, ...]


class ServingSandboxExecutor(SandboxExecutor):
    """Benchmark a persistent vLLM server through an isolated client."""

    READY_EVENT = "READY"
    START_COMMAND = "GO"
    DONE_EVENT = "DONE"
    END_COMMAND = "ACK"

    def client_command(
        self, candidate: Candidate, seed: int, control_dir: Path
    ) -> Sequence[str]:
        contract = self._serving_contract(candidate)
        return [
            *self.privileged_prefix,
            "docker",
            "run",
            "--rm",
            "--network",
            contract.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "HF_HOME=/root/.cache/huggingface",
            "--env",
            "HOME=/tmp",
            "--env",
            "VLLM_CACHE_ROOT=/tmp/vllm-cache",
            "--env",
            "VLLM_NO_USAGE_STATS=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "TPA_CONTROL_DIR=/tpa-control",
            "--volume",
            "%s:/root/.cache/huggingface:ro" % contract.cache_volume,
            "--volume",
            "%s:/tpa-control:rw" % Path(control_dir).resolve(),
            "--entrypoint",
            "vllm",
            contract.image,
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            contract.base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            contract.model,
            "--served-model-name",
            contract.served_model_name,
            "--tokenizer",
            contract.model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(contract.input_len),
            "--random-output-len",
            str(contract.output_len),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(contract.num_prompts),
            "--num-warmups",
            str(contract.num_warmups),
            "--request-rate",
            contract.request_rate,
            "--max-concurrency",
            str(contract.max_concurrency),
            "--temperature",
            "0",
            "--ignore-eos",
            "--seed",
            str(seed),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--disable-tqdm",
        ]

    def execute(
        self, candidate: Candidate, level: EvidenceLevel, seed: int
    ) -> EvidenceRecord:
        if level == EvidenceLevel.L0:
            raise SandboxExecutionError("L0 belongs to replay or simulation")
        if candidate.required_gpus != 1 or candidate.target_nodes != 1:
            raise SandboxExecutionError(
                "ServingSandboxExecutor supports one GPU on one node"
            )

        contract = self._serving_contract(candidate)
        server_environment = self._verify_serving_environment(contract)
        image_id = self._image_id(contract.image)
        power_state = self._query_power_state()
        if not power_state.minimum_w <= contract.power_limit_w <= power_state.maximum_w:
            raise SandboxExecutionError(
                "power limit %.1f W is outside [%.1f, %.1f] W"
                % (
                    contract.power_limit_w,
                    power_state.minimum_w,
                    power_state.maximum_w,
                )
            )

        artifact_stem = self._artifact_stem(candidate.candidate_id, seed)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = self.telemetry_dir / (artifact_stem + ".dcgm.txt")
        client_path = self.telemetry_dir / (artifact_stem + ".client.txt")
        control_dir = self.telemetry_dir / (artifact_stem + ".control")
        events_fifo, commands_fifo = self._create_control_fifos(control_dir)

        telemetry_process = None
        telemetry_stream = None
        marked_run: Optional[MarkedRun] = None
        power_readback_w = power_state.current_w
        trial_seconds = 0.0
        try:
            self._set_power_limit(contract.power_limit_w)
            time.sleep(self.settle_seconds)
            power_readback_w = self._query_power_state().current_w
            if abs(power_readback_w - contract.power_limit_w) > 1.0:
                raise SandboxExecutionError(
                    "power-limit readback %.1f W does not match request %.1f W"
                    % (power_readback_w, contract.power_limit_w)
                )

            telemetry_process, telemetry_stream = self._start_telemetry(telemetry_path)
            time.sleep(max(0.2, 2 * self.sample_ms / 1000.0))
            trial_started_at = time.perf_counter()
            marked_run = self._run_gated(
                self.client_command(candidate, seed, control_dir),
                events_fifo,
                commands_fifo,
            )
            trial_seconds = time.perf_counter() - trial_started_at
            client_path.write_text(marked_run.output, encoding="utf-8")
        finally:
            try:
                if telemetry_process is not None and telemetry_stream is not None:
                    self._stop_telemetry(telemetry_process, telemetry_stream)
            finally:
                try:
                    self._restore_power_limit(power_state.current_w)
                finally:
                    self._remove_control_dir(control_dir)

        if marked_run is None:
            raise SandboxExecutionError("vLLM benchmark did not run")

        provenance = {
            "executor": "persistent-vllm-sandbox",
            "client_image": contract.image,
            "client_image_id": image_id,
            "client_command": list(self.client_command(candidate, seed, control_dir)),
            "server_container": contract.server_container,
            "server_image": server_environment.image,
            "server_image_id": server_environment.image_id,
            "server_command": list(server_environment.command),
            "docker_network": contract.network,
            "hf_cache_volume": contract.cache_volume,
            "base_url": contract.base_url,
            "model": contract.model,
            "served_model_name": contract.served_model_name,
            "gpu_id": self.gpu_id,
            "seed": seed,
            "power_limit_requested_w": contract.power_limit_w,
            "power_limit_readback_w": power_readback_w,
            "power_limit_original_w": power_state.current_w,
            "energy_counter": "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
            "energy_field_id": 156,
            "measurement_boundary": {
                "start": "client READY -> host energy read -> GO",
                "end": "client DONE -> host energy read -> ACK",
            },
            "sample_ms": self.sample_ms,
            "trial_wall_seconds": trial_seconds,
            "telemetry_path": str(telemetry_path),
            "client_output_path": str(client_path),
            "workload": {
                "input_len": contract.input_len,
                "output_len": contract.output_len,
                "num_prompts": contract.num_prompts,
                "num_warmups": contract.num_warmups,
                "request_rate": contract.request_rate,
                "max_concurrency": contract.max_concurrency,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
        if "campaign_index" in candidate.config:
            provenance["campaign_index"] = int(candidate.config["campaign_index"])

        gpu_hours = trial_seconds / 3600.0
        kind = (
            EvidenceKind.VERIFIED
            if level == EvidenceLevel.L4
            else EvidenceKind.MEASURED
        )
        if marked_run.returncode != 0:
            return EvidenceRecord(
                candidate_id=candidate.candidate_id,
                level=level,
                metrics={},
                gpu_hours=gpu_hours,
                kind=kind,
                status=EvidenceStatus.FAILED,
                provenance=provenance,
                failure_reason="benchmark client exited with status %d"
                % marked_run.returncode,
            )

        try:
            benchmark = dict(parse_vllm_benchmark_output(marked_run.output))
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

        energy_j = (marked_run.energy_end_mj - marked_run.energy_start_mj) / 1000.0
        if energy_j <= 0 or marked_run.window_seconds <= 0:
            raise SandboxExecutionError("non-positive measured energy or duration")

        total_tokens = benchmark["input_tokens"] + benchmark["output_tokens"]
        metrics = dict(benchmark)
        metrics.update(
            {
                "energy_j": energy_j,
                "avg_power_w": energy_j / marked_run.window_seconds,
                "measurement_window_ms": marked_run.window_seconds * 1000.0,
                "trial_elapsed_ms": trial_seconds * 1000.0,
                "j_per_output_token": energy_j / benchmark["output_tokens"],
                "j_per_total_token": energy_j / total_tokens,
                "ttft_ms": benchmark["p95_ttft_ms"],
                "tpot_ms": benchmark["p95_tpot_ms"],
                "throughput_tok_s": benchmark["output_throughput_tok_s"],
            }
        )
        provenance["energy_start_mj"] = marked_run.energy_start_mj
        provenance["energy_end_mj"] = marked_run.energy_end_mj
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=level,
            metrics=metrics,
            gpu_hours=gpu_hours,
            kind=kind,
            provenance=provenance,
        )

    def _run_gated(
        self,
        command: Sequence[str],
        events_fifo: Path,
        commands_fifo: Path,
    ) -> MarkedRun:
        event_fd = os.open(events_fifo, os.O_RDWR | os.O_NONBLOCK)
        command_fd = os.open(commands_fifo, os.O_RDWR | os.O_NONBLOCK)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            os.close(event_fd)
            os.close(command_fd)
            raise SandboxExecutionError("benchmark client failed to start: %s" % exc) from exc

        if process.stdout is None:
            process.kill()
            os.close(event_fd)
            os.close(command_fd)
            raise SandboxExecutionError("benchmark client has no output pipe")

        messages: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        stop_events = threading.Event()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    messages.put(("stdout", line))
            finally:
                messages.put(("stdout-eof", None))

        def read_events() -> None:
            buffered = b""
            while not stop_events.is_set():
                try:
                    chunk = os.read(event_fd, 4096)
                except BlockingIOError:
                    time.sleep(0.01)
                    continue
                except OSError:
                    return
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffered += chunk
                while b"\n" in buffered:
                    raw, buffered = buffered.split(b"\n", 1)
                    messages.put(("control", raw.decode("utf-8", errors="replace")))

        output_reader = threading.Thread(target=read_output, daemon=True)
        event_reader = threading.Thread(target=read_events, daemon=True)
        output_reader.start()
        event_reader.start()
        deadline = time.monotonic() + self.timeout_seconds
        output = []
        energy_start_mj: Optional[int] = None
        energy_end_mj: Optional[int] = None
        started_at: Optional[float] = None
        ended_at: Optional[float] = None

        try:
            output_done = False
            while not output_done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, self.timeout_seconds)
                try:
                    message_type, value = messages.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    continue
                if message_type == "stdout-eof":
                    output_done = True
                    continue
                if message_type == "stdout":
                    assert value is not None
                    output.append(value)
                    print(value, end="", flush=True)
                    continue
                if message_type != "control" or value is None:
                    continue
                if value == self.READY_EVENT:
                    if energy_start_mj is not None:
                        raise SandboxExecutionError("duplicate READY measurement event")
                    energy_start_mj = self._read_energy_mj()
                    started_at = time.perf_counter()
                    os.write(command_fd, (self.START_COMMAND + "\n").encode("ascii"))
                elif value == self.DONE_EVENT:
                    if energy_start_mj is None:
                        raise SandboxExecutionError("DONE event arrived before READY")
                    if energy_end_mj is not None:
                        raise SandboxExecutionError("duplicate DONE measurement event")
                    energy_end_mj = self._read_energy_mj()
                    ended_at = time.perf_counter()
                    os.write(command_fd, (self.END_COMMAND + "\n").encode("ascii"))
                else:
                    raise SandboxExecutionError(
                        "unknown measurement control event: %s" % value
                    )

            returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except (OSError, subprocess.SubprocessError, SandboxExecutionError) as exc:
            self._terminate_process(process)
            raise SandboxExecutionError("benchmark client failed: %s" % exc) from exc
        finally:
            stop_events.set()
            output_reader.join(timeout=1.0)
            event_reader.join(timeout=1.0)
            process.stdout.close()
            os.close(event_fd)
            os.close(command_fd)

        if energy_start_mj is None or energy_end_mj is None:
            raise SandboxExecutionError("benchmark output did not contain measurement markers")
        if started_at is None or ended_at is None or ended_at <= started_at:
            raise SandboxExecutionError("invalid benchmark measurement window")
        return MarkedRun(
            returncode=returncode,
            output="".join(output),
            energy_start_mj=energy_start_mj,
            energy_end_mj=energy_end_mj,
            window_seconds=ended_at - started_at,
        )

    @staticmethod
    def _create_control_fifos(control_dir: Path) -> Tuple[Path, Path]:
        """Create directional FIFOs that a different container UID can open."""

        control_dir.mkdir(mode=0o711)
        control_dir.chmod(0o711)
        events_fifo = control_dir / "events.fifo"
        commands_fifo = control_dir / "commands.fifo"
        os.mkfifo(events_fifo)
        os.mkfifo(commands_fifo)

        # Docker preserves host ownership on bind mounts. The benchmark image
        # runs as a different UID, so grant only the cross-boundary operations
        # needed by the protocol: container writes events and reads commands.
        events_fifo.chmod(0o622)
        commands_fifo.chmod(0o644)
        return events_fifo, commands_fifo

    @staticmethod
    def _remove_control_dir(control_dir: Path) -> None:
        for name in ("events.fifo", "commands.fifo"):
            try:
                (control_dir / name).unlink()
            except FileNotFoundError:
                pass
        try:
            control_dir.rmdir()
        except (FileNotFoundError, OSError):
            pass

    def _verify_serving_environment(
        self, contract: ServingContract
    ) -> ServerEnvironment:
        inspection_raw = self._run_checked(
            [
                *self.privileged_prefix,
                "docker",
                "inspect",
                contract.server_container,
            ]
        )
        try:
            inspection = json.loads(inspection_raw)
            container = inspection[0]
            running = container["State"]["Running"]
            networks = container["NetworkSettings"]["Networks"]
            server_image = str(container["Config"]["Image"])
            server_image_id = str(container["Image"])
            server_command = (
                str(container["Path"]),
                *(str(argument) for argument in container["Args"]),
            )
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SandboxExecutionError(
                "invalid Docker inspection for server container %s"
                % contract.server_container
            ) from exc

        if running is not True:
            raise SandboxExecutionError(
                "server container %s is not running" % contract.server_container
            )
        if "--no-enable-prefix-caching" not in server_command:
            raise SandboxExecutionError(
                "server container %s must explicitly disable prefix caching; "
                "matched repeated prompts otherwise bias cross-candidate comparisons"
                % contract.server_container
            )

        internal = self._run_checked(
            [
                *self.privileged_prefix,
                "docker",
                "network",
                "inspect",
                "--format",
                "{{.Internal}}",
                contract.network,
            ]
        ).strip()
        if internal != "true":
            raise SandboxExecutionError(
                "benchmark network %s must be internal" % contract.network
            )

        if contract.network not in networks:
            raise SandboxExecutionError(
                "server container %s is not attached to %s"
                % (contract.server_container, contract.network)
            )
        self._run_checked(
            [
                *self.privileged_prefix,
                "docker",
                "volume",
                "inspect",
                contract.cache_volume,
            ]
        )
        return ServerEnvironment(
            image=server_image,
            image_id=server_image_id,
            command=server_command,
        )

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    @staticmethod
    def _serving_contract(candidate: Candidate) -> ServingContract:
        def identifier(key: str) -> str:
            value = str(candidate.config.get(key, "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise SandboxExecutionError("candidate requires a valid %s" % key)
            return value

        image = str(candidate.config.get("image", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
            raise SandboxExecutionError("candidate requires a valid container image")
        server_container = identifier("server_container")
        network = identifier("network")
        cache_volume = identifier("cache_volume")
        base_url = str(candidate.config.get("base_url", "")).strip().rstrip("/")
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != server_container
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise SandboxExecutionError(
                "base_url must be an HTTP URL for the server container"
            )

        model = str(candidate.config.get("model", "")).strip()
        served_model_name = str(candidate.config.get("served_model_name", "")).strip()
        if not model or model.startswith("-") or not served_model_name:
            raise SandboxExecutionError("candidate requires model names")

        def positive_int(key: str, allow_zero: bool = False) -> int:
            try:
                value = int(candidate.config[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise SandboxExecutionError("candidate requires integer %s" % key) from exc
            minimum = 0 if allow_zero else 1
            if value < minimum:
                raise SandboxExecutionError("candidate has invalid %s" % key)
            return value

        try:
            power_limit_w = float(candidate.config["power_limit_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SandboxExecutionError(
                "candidate requires numeric power_limit_w"
            ) from exc
        if power_limit_w <= 0:
            raise SandboxExecutionError("candidate has invalid power_limit_w")

        raw_rate = str(candidate.config.get("request_rate", "inf")).strip().lower()
        if raw_rate != "inf":
            try:
                if float(raw_rate) <= 0:
                    raise ValueError
            except ValueError as exc:
                raise SandboxExecutionError("candidate has invalid request_rate") from exc

        return ServingContract(
            image=image,
            server_container=server_container,
            network=network,
            cache_volume=cache_volume,
            base_url=base_url,
            model=model,
            served_model_name=served_model_name,
            power_limit_w=power_limit_w,
            input_len=positive_int("input_len"),
            output_len=positive_int("output_len"),
            num_prompts=positive_int("num_prompts"),
            num_warmups=positive_int("num_warmups", allow_zero=True),
            request_rate=raw_rate,
            max_concurrency=positive_int("max_concurrency"),
        )
