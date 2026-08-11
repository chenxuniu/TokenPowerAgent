import json
import os
import stat
import sys

import pytest

from tokenpoweragent.executors.sandbox import PowerState, SandboxExecutionError
from tokenpoweragent.executors.serving import (
    MarkedRun,
    ServerEnvironment,
    ServingSandboxExecutor,
    parse_vllm_server_configuration,
    parse_vllm_benchmark_output,
)
from tokenpoweragent.schema import Candidate, EvidenceLevel


BENCHMARK_OUTPUT = """
============ Serving Benchmark Result ============
Successful requests:                     8
Failed requests:                         0
Benchmark duration (s):                  0.47
Total input tokens:                      2048
Total generated tokens:                  512
Request throughput (req/s):              17.13
Output token throughput (tok/s):         1096.33
Total token throughput (tok/s):          5481.66
---------------Time to First Token----------------
Mean TTFT (ms):                          63.40
P50 TTFT (ms):                           65.77
P95 TTFT (ms):                           66.60
P99 TTFT (ms):                           66.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.37
P50 TPOT (ms):                           6.35
P95 TPOT (ms):                           6.48
P99 TPOT (ms):                           6.53
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.37
P50 ITL (ms):                            6.34
P95 ITL (ms):                            6.68
P99 ITL (ms):                            7.16
----------------End-to-end Latency----------------
Mean E2EL (ms):                          464.68
P50 E2EL (ms):                           465.13
P95 E2EL (ms):                           466.45
P99 E2EL (ms):                           466.54
==================================================
"""


def serving_candidate(**overrides: object) -> Candidate:
    config = {
        "image": "tokenpower-vllm-client:v0.23.0",
        "server_container": "tpa-vllm-qwen7b",
        "network": "tpa-serving-bench",
        "cache_volume": "tpa-hf-cache",
        "base_url": "http://tpa-vllm-qwen7b:8000",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "served_model_name": "qwen2.5-7b",
        "power_limit_w": 700,
        "input_len": 512,
        "output_len": 128,
        "num_prompts": 64,
        "num_warmups": 2,
        "request_rate": "inf",
        "max_concurrency": 8,
        "dataset_split": "calibration",
    }
    config.update(overrides)
    return Candidate(candidate_id="qwen7b-serving-pl700", config=config)


def test_parse_vllm_benchmark_output() -> None:
    metrics = parse_vllm_benchmark_output(BENCHMARK_OUTPUT)

    assert metrics["successful_requests"] == 8
    assert metrics["output_tokens"] == 512
    assert metrics["output_throughput_tok_s"] == 1096.33
    assert metrics["p95_ttft_ms"] == 66.60
    assert metrics["p95_tpot_ms"] == 6.48
    assert metrics["p99_e2el_ms"] == 466.54


def test_parse_vllm_benchmark_rejects_failed_requests() -> None:
    output = BENCHMARK_OUTPUT.replace(
        "Failed requests:                         0",
        "Failed requests:                         1",
    )
    with pytest.raises(SandboxExecutionError, match="reported 1 failed"):
        parse_vllm_benchmark_output(output)


def test_serving_client_has_isolated_fixed_envelope(tmp_path) -> None:
    command = list(
        ServingSandboxExecutor(telemetry_dir=tmp_path).client_command(
            serving_candidate(), seed=3, control_dir=tmp_path / "control"
        )
    )

    assert command[:4] == ["sudo", "-n", "docker", "run"]
    assert command[command.index("--network") + 1] == "tpa-serving-bench"
    assert "host" not in command
    assert "--gpus" not in command
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in command
    assert "tpa-hf-cache:/root/.cache/huggingface:ro" in command
    assert "%s:/tpa-control:rw" % (tmp_path / "control").resolve() in command
    assert "TPA_CONTROL_DIR=/tpa-control" in command
    assert "HOME=/tmp" in command
    assert "VLLM_CACHE_ROOT=/tmp/vllm-cache" in command
    assert "VLLM_NO_USAGE_STATS=1" in command
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--seed") + 1] == "3"
    assert "--ignore-eos" in command
    assert (
        ServingSandboxExecutor._serving_contract(serving_candidate()).dataset_split
        == "calibration"
    )


def test_serving_contract_rejects_unknown_dataset_split(tmp_path) -> None:
    executor = ServingSandboxExecutor(telemetry_dir=tmp_path)
    with pytest.raises(SandboxExecutionError, match="dataset_split"):
        executor.client_command(
            serving_candidate(dataset_split="train-and-test"),
            seed=0,
            control_dir=tmp_path / "control",
        )


@pytest.mark.parametrize(
    "dataset_split",
    (
        "calibration",
        "validation",
        "holdout",
        "diagnostic",
        "configuration-search",
        "configuration-confirmation",
    ),
)
def test_serving_contract_accepts_registered_dataset_splits(
    dataset_split: str,
) -> None:
    contract = ServingSandboxExecutor._serving_contract(
        serving_candidate(dataset_split=dataset_split)
    )
    assert contract.dataset_split == dataset_split


def test_serving_client_rejects_non_server_url(tmp_path) -> None:
    executor = ServingSandboxExecutor(telemetry_dir=tmp_path)
    with pytest.raises(SandboxExecutionError, match="base_url"):
        executor.client_command(
            serving_candidate(base_url="http://example.com:8000"),
            seed=0,
            control_dir=tmp_path / "control",
        )


def test_serving_measurement_gate_brackets_workload(tmp_path) -> None:
    events = tmp_path / "events.fifo"
    commands = tmp_path / "commands.fifo"
    os.mkfifo(events)
    os.mkfifo(commands)

    script = r"""
import sys
import time

events = open(sys.argv[1], "w", buffering=1)
commands = open(sys.argv[2], "r", buffering=1)
events.write("READY\n")
assert commands.readline().strip() == "GO"
print("work-start", flush=True)
time.sleep(0.02)
events.write("DONE\n")
assert commands.readline().strip() == "ACK"
print("work-end", flush=True)
"""

    class FakeEnergyExecutor(ServingSandboxExecutor):
        def __init__(self) -> None:
            super().__init__(
                telemetry_dir=tmp_path,
                use_sudo=False,
                timeout_seconds=5,
            )
            self.values = iter((1000, 3500))

        def _read_energy_mj(self) -> int:
            return next(self.values)

    run = FakeEnergyExecutor()._run_gated(
        [sys.executable, "-c", script, str(events), str(commands)],
        events,
        commands,
    )

    assert run.returncode == 0
    assert run.energy_start_mj == 1000
    assert run.energy_end_mj == 3500
    assert run.window_seconds >= 0.02
    assert "work-start" in run.output
    assert "work-end" in run.output


def test_serving_control_fifos_allow_directional_container_access(tmp_path) -> None:
    control_dir = tmp_path / "control"

    events, commands = ServingSandboxExecutor._create_control_fifos(control_dir)

    assert stat.S_IMODE(control_dir.stat().st_mode) == 0o711
    assert stat.S_IMODE(events.stat().st_mode) == 0o622
    assert stat.S_IMODE(commands.stat().st_mode) == 0o644

    ServingSandboxExecutor._remove_control_dir(control_dir)
    assert not control_dir.exists()


def server_inspection(*extra_args: str) -> str:
    return json.dumps(
        [
            {
                "State": {"Running": True},
                "NetworkSettings": {
                    "Networks": {"tpa-serving-bench": {"IPAddress": "172.18.0.2"}}
                },
                "Config": {"Image": "vllm/vllm-openai:v0.23.0"},
                "Image": "sha256:server-image",
                "Path": "vllm",
                "Args": [
                    "serve",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--dtype",
                    "bfloat16",
                    "--max-num-seqs",
                    "256",
                    "--max-num-batched-tokens",
                    "8192",
                    "--enable-chunked-prefill",
                    *extra_args,
                ],
            }
        ]
    )


class FakeServingEnvironmentExecutor(ServingSandboxExecutor):
    def __init__(self, inspection: str, tmp_path) -> None:
        super().__init__(telemetry_dir=tmp_path, use_sudo=False)
        self.inspection = inspection

    def _run_checked(self, command):
        if command[:2] == ["docker", "inspect"]:
            return self.inspection
        if command[:3] == ["docker", "network", "inspect"]:
            return "true\n"
        if command[:3] == ["docker", "volume", "inspect"]:
            return "[]\n"
        raise AssertionError("unexpected command: %r" % (command,))


def test_serving_environment_records_pinned_server(tmp_path) -> None:
    executor = FakeServingEnvironmentExecutor(
        server_inspection("--no-enable-prefix-caching"), tmp_path
    )

    environment = executor._verify_serving_environment(
        executor._serving_contract(serving_candidate())
    )

    assert environment.image == "vllm/vllm-openai:v0.23.0"
    assert environment.image_id == "sha256:server-image"
    assert "--no-enable-prefix-caching" in environment.command
    assert environment.configuration["max_num_seqs"] == 256
    assert environment.configuration["max_num_batched_tokens"] == 8192
    assert environment.configuration["chunked_prefill"] is True


def test_parse_server_configuration_requires_explicit_batching_contract() -> None:
    with pytest.raises(SandboxExecutionError, match="max-num-seqs"):
        parse_vllm_server_configuration(
            ["vllm", "serve", "model", "--enable-chunked-prefill"]
        )


def test_serving_environment_rejects_implicit_prefix_cache(tmp_path) -> None:
    executor = FakeServingEnvironmentExecutor(server_inspection(), tmp_path)

    with pytest.raises(SandboxExecutionError, match="disable prefix caching"):
        executor._verify_serving_environment(
            executor._serving_contract(serving_candidate())
        )


def test_serving_record_exposes_agent_objective_energy_aliases(tmp_path) -> None:
    class FakeMeasuredExecutor(ServingSandboxExecutor):
        def __init__(self) -> None:
            super().__init__(
                telemetry_dir=tmp_path,
                use_sudo=False,
                settle_seconds=0,
            )

        def _verify_serving_environment(self, contract):
            return ServerEnvironment(
                image="vllm/vllm-openai:v0.23.0",
                image_id="sha256:server",
                command=("vllm", "serve"),
                configuration={"prefix_caching": False},
            )

        def _image_id(self, image):
            return "sha256:client"

        def _query_power_state(self):
            return PowerState(current_w=700, minimum_w=200, maximum_w=700)

        def _set_power_limit(self, power_limit_w):
            pass

        def _restore_power_limit(self, power_limit_w):
            pass

        def _start_telemetry(self, telemetry_path):
            return None, None

        def _run_gated(self, command, events_fifo, commands_fifo):
            return MarkedRun(
                returncode=0,
                output=BENCHMARK_OUTPUT,
                energy_start_mj=1000,
                energy_end_mj=257000,
                window_seconds=0.5,
            )

    record = FakeMeasuredExecutor().execute(
        serving_candidate(), EvidenceLevel.L1, seed=0
    )

    assert record.metrics["j_per_output_token"] == pytest.approx(0.5)
    assert record.metrics["energy_j_per_1k_tokens"] == pytest.approx(500)
    assert record.metrics["energy_j_per_1k_output_tokens"] == pytest.approx(500)
    assert record.metrics["energy_j_per_1k_total_tokens"] == pytest.approx(100)
