import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    ConfigurationCampaignError,
    freeze_configuration_campaign,
    sha256_file,
)
from tokenpoweragent.configuration_runner import (
    DockerVLLMServerManager,
    run_configuration_campaign,
    verify_frozen_configuration_campaign,
)
from tokenpoweragent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStore,
)
from tokenpoweragent.executors.serving import ServerEnvironment
from tokenpoweragent.schema import EvidenceLevel, Scenario


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "configs/campaigns/qwen2.5-7b-h100-config-search-v1.json"
)
PROFILE_PATH = (
    ROOT
    / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
)


def _outputs(tmp_path: Path):
    return {
        "scenario_path": tmp_path / "scenario.json",
        "predictions_path": tmp_path / "predictions.jsonl",
        "schedule_path": tmp_path / "schedule.json",
        "summary_path": tmp_path / "summary.json",
        "manifest_path": tmp_path / "freeze.sha256",
    }


def _freeze(tmp_path: Path):
    paths = _outputs(tmp_path)
    freeze_configuration_campaign(
        campaign_path=CAMPAIGN_PATH,
        calibration_path=PROFILE_PATH,
        **paths,
    )
    return paths


class FakeServerManager:
    def __init__(self, campaign: ConfigurationCampaign, tmp_path: Path) -> None:
        self.campaign = campaign
        self.current_candidate_id = None
        self.current_point = None
        self.current_log_path = None
        self.restart_count = 0
        self.restarts = []
        self.verify_calls = 0
        self.capture_calls = 0
        self.tmp_path = tmp_path

    @property
    def expected_server_image_id(self) -> str:
        return "sha256:" + self.campaign.runtime["server_image"].rsplit(
            "@sha256:", 1
        )[1]

    def verify_host(self) -> None:
        self.verify_calls += 1

    def restart(self, point, block_index) -> None:
        self.current_candidate_id = point.point_id
        self.current_point = point
        self.current_log_path = self.tmp_path / (
            "block-%03d-%s.log" % (block_index, point.point_id)
        )
        self.restart_count += 1
        self.restarts.append((point.point_id, block_index))

    def capture_current_logs(self) -> None:
        self.capture_calls += 1


class FakeConfigurationExecutor:
    def __init__(
        self,
        campaign: ConfigurationCampaign,
        manager: FakeServerManager,
    ) -> None:
        self.campaign = campaign
        self.manager = manager
        self.sample_ms = campaign.sample_ms
        self.calls = []

    def _configuration(self):
        assert self.manager.current_point is not None
        point = self.manager.current_point
        return {
            key: point.configuration[key]
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
            )
        }

    def inspect_environment(self, candidate):
        return ServerEnvironment(
            image=self.campaign.runtime["server_image"],
            image_id=self.manager.expected_server_image_id,
            command=("vllm", "serve"),
            configuration=self._configuration(),
        )

    def execute(self, candidate, level, seed):
        self.calls.append(
            (candidate.candidate_id, level, seed, dict(candidate.config))
        )
        energy = 400.0 + candidate.config["max_num_seqs"]
        kind = (
            EvidenceKind.VERIFIED
            if level == EvidenceLevel.L4
            else EvidenceKind.MEASURED
        )
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            level=level,
            metrics={
                "energy_j_per_1k_output_tokens": energy,
                "throughput_tok_s": 2000.0,
                "ttft_ms": 1000.0,
                "tpot_ms": 10.0,
            },
            gpu_hours=0.01,
            kind=kind,
            provenance={
                "dataset_split": candidate.config["dataset_split"],
                "model": self.campaign.model_id,
                "client_image_id": self.campaign.runtime["client_image_id"],
                "server_image": self.campaign.runtime["server_image"],
                "server_image_id": self.manager.expected_server_image_id,
                "server_command": ["vllm", "serve"],
                "server_configuration": self._configuration(),
                "power_limit_readback_w": self.campaign.power_limit_w,
                "workload": {
                    "input_len": candidate.config["input_len"],
                    "output_len": candidate.config["output_len"],
                    "num_prompts": candidate.config["num_prompts"],
                    "num_warmups": candidate.config["num_warmups"],
                    "request_rate": candidate.config["request_rate"],
                    "max_concurrency": candidate.config["max_concurrency"],
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )


def test_configuration_campaign_locks_controlled_search_space() -> None:
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)

    assert len(campaign.candidates) == 12
    assert campaign.target_workload.input_tokens == 2048
    assert campaign.target_workload.output_tokens == 128
    assert campaign.target_workload.concurrency == 32
    assert campaign.target_workload.num_requests == 256
    assert campaign.probe_requests == 64
    assert campaign.repeats == 3
    assert campaign.objectives == {
        "energy_j_per_1k_output_tokens": "min",
        "throughput_tok_s": "max",
    }
    assert campaign.slo == {"ttft_ms": 1600.0, "tpot_ms": 20.0}
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L0] == 0
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L1] == 0.01
    assert campaign.level_cost_gpu_hours[EvidenceLevel.L4] == 0.02
    assert campaign.runtime["server_image"].endswith(
        "6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
    )

    baselines = [point for point in campaign.candidates if point.role == "expert-baseline"]
    assert [point.point_id for point in baselines] == [
        "expert-seq256-bt8192-chunk"
    ]
    assert all(point.configuration["prefix_caching"] is False for point in campaign.candidates)
    assert all(point.configuration["tensor_parallel"] == 1 for point in campaign.candidates)
    assert all(point.configuration["pipeline_parallel"] == 1 for point in campaign.candidates)
    assert all(point.configuration["power_limit_w"] == 700 for point in campaign.candidates)


def test_freeze_writes_balanced_hash_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    paths = _outputs(tmp_path)

    summary = freeze_configuration_campaign(
        campaign_path=CAMPAIGN_PATH.relative_to(ROOT),
        calibration_path=PROFILE_PATH.relative_to(ROOT),
        **paths,
    )
    predictions = EvidenceStore.read_jsonl(paths["predictions_path"]).records
    scenario = Scenario.load(paths["scenario_path"])
    schedule = json.loads(paths["schedule_path"].read_text(encoding="utf-8"))

    assert summary["candidate_count"] == 12
    assert summary["l0_prediction_count"] == 12
    assert summary["measurement_count"] == 72
    assert len(predictions) == 12
    assert all(record.level == EvidenceLevel.L0 for record in predictions)
    assert len(scenario.candidates) == 12
    assert scenario.budget.gpu_hours == 0.12

    actions = schedule["actions"]
    assert [action["action_index"] for action in actions] == list(range(72))
    assert Counter(
        (action["candidate_id"], action["level"]) for action in actions
    ) == Counter(
        {
            (point.point_id, level): 3
            for point in ConfigurationCampaign.load(CAMPAIGN_PATH).candidates
            for level in ("L1", "L4")
        }
    )

    blocks = defaultdict(list)
    for action in actions:
        blocks[action["block_index"]].append(action)
    assert len(blocks) == 36
    for block in blocks.values():
        assert len(block) == 2
        assert block[0]["candidate_id"] == block[1]["candidate_id"]
        assert block[0]["repeat"] == block[1]["repeat"]
        assert {action["level"] for action in block} == {"L1", "L4"}
        assert [action["restart_server"] for action in block] == [True, False]
        assert {
            action["level"]: action["num_prompts"] for action in block
        } == {"L1": 64, "L4": 256}

    manifest_lines = paths["manifest_path"].read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 6
    for line in manifest_lines:
        expected, artifact = line.split("  ", 1)
        assert sha256_file(Path(artifact)) == expected

    with pytest.raises(ConfigurationCampaignError, match="refusing to overwrite"):
        freeze_configuration_campaign(
            campaign_path=CAMPAIGN_PATH.relative_to(ROOT),
            calibration_path=PROFILE_PATH.relative_to(ROOT),
            **paths,
        )


def test_configuration_campaign_rejects_prefix_cache_bias(tmp_path: Path) -> None:
    raw = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    raw["candidates"][0]["serving_configuration"]["prefix_caching"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationCampaignError, match="disable prefix caching"):
        ConfigurationCampaign.load(path)


def test_configuration_campaign_rejects_unpinned_server_image(tmp_path: Path) -> None:
    raw = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    raw["runtime"]["server_image"] = "vllm/vllm-openai:v0.23.0"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationCampaignError, match="immutable sha256"):
        ConfigurationCampaign.load(path)


def test_frozen_configuration_runner_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    paths = _freeze(tmp_path / "freeze")
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)
    manager = FakeServerManager(campaign, tmp_path)
    executor = FakeConfigurationExecutor(campaign, manager)
    output = tmp_path / "measurements.jsonl"

    artifacts = verify_frozen_configuration_campaign(
        CAMPAIGN_PATH,
        paths["predictions_path"],
        paths["schedule_path"],
        paths["summary_path"],
        paths["manifest_path"],
    )
    first = run_configuration_campaign(
        campaign_path=CAMPAIGN_PATH,
        predictions_path=paths["predictions_path"],
        schedule_path=paths["schedule_path"],
        summary_path=paths["summary_path"],
        manifest_path=paths["manifest_path"],
        output_path=output,
        executor=executor,
        server_manager=manager,
    )
    second = run_configuration_campaign(
        campaign_path=CAMPAIGN_PATH,
        predictions_path=paths["predictions_path"],
        schedule_path=paths["schedule_path"],
        summary_path=paths["summary_path"],
        manifest_path=paths["manifest_path"],
        output_path=output,
        executor=executor,
        server_manager=manager,
        resume=True,
    )
    records = EvidenceStore.read_jsonl(output).records

    assert len(artifacts.campaign_sha256) == 64
    assert first["added_measurements"] == 72
    assert first["completed_measurements"] == 72
    assert second["added_measurements"] == 0
    assert second["existing_measurements"] == 72
    assert len(records) == 72
    assert len(executor.calls) == 72
    assert manager.restart_count == 36
    assert [record.provenance["campaign_action_index"] for record in records] == list(
        range(72)
    )
    assert sum(
        record.provenance["server_restart_reason"] == "scheduled"
        for record in records
    ) == 36
    assert sum(record.level == EvidenceLevel.L1 for record in records) == 36
    assert sum(record.level == EvidenceLevel.L4 for record in records) == 36


def test_configuration_runner_resume_restarts_incomplete_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    paths = _freeze(tmp_path / "freeze")
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)
    original_manager = FakeServerManager(campaign, tmp_path)
    original_executor = FakeConfigurationExecutor(campaign, original_manager)
    full_output = tmp_path / "full.jsonl"
    run_configuration_campaign(
        CAMPAIGN_PATH,
        paths["predictions_path"],
        paths["schedule_path"],
        paths["summary_path"],
        paths["manifest_path"],
        full_output,
        original_executor,
        original_manager,
    )
    rows = full_output.read_text(encoding="utf-8").splitlines()
    partial_output = tmp_path / "partial.jsonl"
    partial_output.write_text("\n".join(rows[:3]) + "\n", encoding="utf-8")

    manager = FakeServerManager(campaign, tmp_path)
    executor = FakeConfigurationExecutor(campaign, manager)
    report = run_configuration_campaign(
        CAMPAIGN_PATH,
        paths["predictions_path"],
        paths["schedule_path"],
        paths["summary_path"],
        paths["manifest_path"],
        partial_output,
        executor,
        manager,
        resume=True,
    )
    records = EvidenceStore.read_jsonl(partial_output).records

    assert report["existing_measurements"] == 3
    assert report["added_measurements"] == 69
    assert records[3].provenance["server_restart_reason"] == "resume-recovery"
    assert manager.restarts[0][0] == records[3].candidate_id


def test_configuration_runner_can_stop_at_frozen_action_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    paths = _freeze(tmp_path / "freeze")
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)
    manager = FakeServerManager(campaign, tmp_path)
    executor = FakeConfigurationExecutor(campaign, manager)
    output = tmp_path / "measurements.jsonl"

    report = run_configuration_campaign(
        CAMPAIGN_PATH,
        paths["predictions_path"],
        paths["schedule_path"],
        paths["summary_path"],
        paths["manifest_path"],
        output,
        executor,
        manager,
        max_actions=2,
    )

    assert report["completed_measurements"] == 2
    assert report["remaining_measurements"] == 70
    assert report["campaign_complete"] is False
    assert manager.restart_count == 1
    assert len(EvidenceStore.read_jsonl(output).records) == 2


def test_server_manager_renders_isolated_pinned_candidate_command(
    tmp_path: Path,
) -> None:
    campaign = ConfigurationCampaign.load(CAMPAIGN_PATH)
    point = next(
        point
        for point in campaign.candidates
        if point.point_id == "seq8-bt4096-nochunk"
    )
    manager = DockerVLLMServerManager(
        campaign, tmp_path, gpu_id=0, use_sudo=False
    )
    command = list(manager.docker_run_command(point, block_index=7))

    assert command[:3] == ["docker", "run", "-d"]
    assert command[command.index("--network") + 1] == "tpa-serving-bench"
    assert "-p" not in command
    assert "HF_HUB_OFFLINE=1" in command
    assert campaign.runtime["server_image"] in command
    assert command[command.index("--max-num-seqs") + 1] == "8"
    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--no-enable-chunked-prefill" in command
    assert "--no-enable-prefix-caching" in command
