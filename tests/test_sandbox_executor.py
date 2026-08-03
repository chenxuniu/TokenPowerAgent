import pytest

from tokenpoweragent.executors.sandbox import (
    SandboxExecutionError,
    SandboxExecutor,
    parse_dcgmi_energy_mj,
    parse_power_state,
    parse_workload_result,
)
from tokenpoweragent.schema import Candidate


def test_sandbox_parses_host_measurement_outputs() -> None:
    power = parse_power_state("700.00, 200.00, 700.00\ntrailing notice\n")
    assert power.current_w == 700.0
    assert power.minimum_w == 200.0
    assert power.maximum_w == 700.0

    dmon = """#Entity  TOTEC
ID             mJ
GPU 0          81434600
"""
    assert parse_dcgmi_energy_mj(dmon) == 81434600


def test_sandbox_parses_workload_json_from_last_valid_line() -> None:
    output = """initializing CUDA
{"workload":"gemm","elapsed_ms":12.5,"operations":1000000,"tflops":80.0}
"""
    metrics = parse_workload_result(output)
    assert metrics["elapsed_ms"] == 12.5
    assert metrics["tflops"] == 80.0


def test_sandbox_docker_command_has_fixed_security_envelope(tmp_path) -> None:
    candidate = Candidate(
        candidate_id="gemm-pl500",
        config={
            "image": "tokenpower-sandbox:cuda12.8",
            "power_limit_w": 500,
            "command": ["--iterations", "20"],
        },
    )
    command = list(SandboxExecutor(telemetry_dir=tmp_path).docker_command(candidate))

    assert command[:4] == ["sudo", "-n", "docker", "run"]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in command
    assert "--privileged" not in command
    assert "tokenpower-sandbox:cuda12.8" in command


def test_sandbox_rejects_shell_command_strings(tmp_path) -> None:
    candidate = Candidate(
        candidate_id="unsafe",
        config={
            "image": "tokenpower-sandbox:cuda12.8",
            "power_limit_w": 500,
            "command": "sh -c 'do something'",
        },
    )
    executor = SandboxExecutor(telemetry_dir=tmp_path)
    with pytest.raises(SandboxExecutionError, match="sequence of arguments"):
        executor.docker_command(candidate)
