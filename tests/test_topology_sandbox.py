import json
from pathlib import Path

import pytest

from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.evidence import EvidenceKind, EvidenceRecord, EvidenceStatus
from tokenpoweragent.executors.base import Executor, RoutedExecutor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.executors.topology import TopologySandboxExecutor
from tokenpoweragent.search_space import CandidateGrid
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
    TopologyEnergyTwin,
    TopologyProjector,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    ROOT
    / "configs/calibration/qwen2.5-7b-h100-synthetic-example.json"
)


def reference_workload() -> InferenceWorkload:
    return InferenceWorkload(
        input_tokens=512,
        output_tokens=128,
        concurrency=8,
        num_requests=64,
    )


def candidate(
    candidate_id: str = "reference",
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    nodes: int = 1,
    required_gpus: int = 0,
    **overrides: object,
) -> Candidate:
    config = {
        "engine": "vllm",
        "precision": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tensor_parallel": tp,
        "pipeline_parallel": pp,
        "data_parallel": dp,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "chunked_prefill": True,
    }
    config.update(overrides)
    return Candidate(
        candidate_id=candidate_id,
        config=config,
        required_gpus=required_gpus or tp * pp * dp,
        target_nodes=nodes,
    )


def test_reference_projection_reproduces_single_gpu_anchor() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    estimate = TopologyProjector(profile).predict(
        candidate(), reference_workload(), EvidenceLevel.L2
    )

    assert estimate.feasible
    assert estimate.reference_point_id == "synthetic-c512-o128-c8"
    assert estimate.calibration_distance == pytest.approx(0)
    assert estimate.metrics["throughput_tok_s"] == pytest.approx(1800)
    assert estimate.metrics["ttft_ms"] == pytest.approx(120)
    assert estimate.metrics["tpot_ms"] == pytest.approx(8)
    assert estimate.metrics["avg_power_w"] == pytest.approx(460)
    assert estimate.metrics["relative_uncertainty"] == pytest.approx(0.22)


def test_l2_adds_tp_communication_that_l0_omits() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    projector = TopologyProjector(profile)
    target = candidate("tp4", tp=4)

    l0 = projector.predict(target, reference_workload(), EvidenceLevel.L0)
    l2 = projector.predict(target, reference_workload(), EvidenceLevel.L2)

    assert l0.metrics["throughput_tok_s"] > l2.metrics["throughput_tok_s"]
    assert l0.metrics["energy_j_per_1k_output_tokens_lower"] > 0
    assert l0.metrics["energy_j_per_1k_output_tokens_upper"] > l0.metrics[
        "energy_j_per_1k_output_tokens"
    ]
    assert l0.relative_uncertainty > l2.relative_uncertainty
    assert l0.decomposition["topology_terms_enabled"] is False
    assert l2.decomposition["topology_terms_enabled"] is True
    assert l2.decomposition["tp_communication_stage_ms"] > 0


def test_cross_node_projection_is_visible_and_more_uncertain() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    projector = TopologyProjector(profile)
    intra = projector.predict(
        candidate("tp8", tp=8), reference_workload(), EvidenceLevel.L2
    )
    cross = projector.predict(
        candidate("tp16", tp=16, nodes=2),
        reference_workload(),
        EvidenceLevel.L2,
    )

    assert cross.decomposition["tp_crosses_nodes"] is True
    assert cross.decomposition["tp_communication_stage_ms"] > intra.decomposition[
        "tp_communication_stage_ms"
    ]
    assert cross.relative_uncertainty > intra.relative_uncertainty


def test_executor_labels_simulation_and_extrapolation_distinctly() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    executor = TopologySandboxExecutor(
        profile,
        reference_workload(),
        expected_model="Qwen/Qwen2.5-7B-Instruct",
    )

    l0 = executor.execute(candidate(), EvidenceLevel.L0, seed=3)
    l2 = executor.execute(candidate(), EvidenceLevel.L2, seed=3)

    assert l0.kind == EvidenceKind.SIMULATED
    assert l2.kind == EvidenceKind.EXTRAPOLATED
    assert l2.provenance["validation_required"] is True
    assert l2.provenance["profile_publication_eligible"] is False
    assert "not a serving measurement" in l2.provenance["evidence_semantics"]


def test_invalid_geometry_and_memory_are_failed_evidence() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    executor = TopologySandboxExecutor(profile, reference_workload())
    geometry = executor.execute(
        candidate("bad-geometry", tp=4, required_gpus=2),
        EvidenceLevel.L2,
        seed=0,
    )
    huge_workload = InferenceWorkload(32768, 128, 4096, 4096)
    memory = TopologySandboxExecutor(profile, huge_workload).execute(
        candidate(
            "bad-memory",
            max_num_seqs=4096,
            max_num_batched_tokens=32768,
        ),
        EvidenceLevel.L0,
        seed=0,
    )

    assert geometry.status == EvidenceStatus.FAILED
    assert "TP*PP*DP" in (geometry.failure_reason or "")
    assert memory.status == EvidenceStatus.FAILED
    assert "memory" in (memory.failure_reason or "")


def test_profile_rejects_multigpu_l1_anchor() -> None:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw["calibration_points"][0]["configuration"]["tensor_parallel"] = 2

    with pytest.raises(CalibrationError, match="single-GPU"):
        CalibrationProfile.from_mapping(raw)


def test_candidate_grid_filters_geometry_and_memory_deterministically() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    grid = CandidateGrid.load(
        ROOT / "configs/search_spaces/qwen2.5-7b-core-grid.json"
    )
    workload = InferenceWorkload(1024, 128, 16, 128)

    first = grid.compile(profile=profile, workload=workload)
    second = grid.compile(profile=profile, workload=workload)

    assert first.candidates
    assert first.rejected
    assert [item.candidate_id for item in first.candidates] == [
        item.candidate_id for item in second.candidates
    ]
    assert len({item.candidate_id for item in first.candidates}) == len(
        first.candidates
    )
    for item in first.candidates:
        config = item.config
        assert item.required_gpus == (
            config["tensor_parallel"]
            * config["pipeline_parallel"]
            * config["data_parallel"]
        )


def test_topology_twin_starts_without_handwritten_prior_and_updates() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    target = candidate("no-prior", tp=2)
    twin = TopologyEnergyTwin(profile, reference_workload())

    prior = twin.predict(target)
    twin.update(
        EvidenceRecord(
            candidate_id=target.candidate_id,
            level=EvidenceLevel.L4,
            metrics={"throughput_tok_s": 2000, "energy_j_per_1k_tokens": 300},
            gpu_hours=1,
            kind=EvidenceKind.VERIFIED,
        )
    )
    verified = twin.predict(target)

    assert prior.source_level == int(EvidenceLevel.L0)
    assert verified.source_level == int(EvidenceLevel.L4)
    assert verified.metrics["throughput_tok_s"] == 2000
    assert verified.uncertainty < prior.uncertainty


def test_routed_executor_dispatches_by_fidelity() -> None:
    calls = []

    class RecordingExecutor(Executor):
        def __init__(self, name: str) -> None:
            self.name = name

        def execute(self, target, level, seed):
            calls.append((self.name, level, seed))
            return EvidenceRecord(
                candidate_id=target.candidate_id,
                level=level,
                metrics={"value": 1},
                gpu_hours=0,
                kind=(
                    EvidenceKind.SIMULATED
                    if level == EvidenceLevel.L0
                    else EvidenceKind.EXTRAPOLATED
                ),
            )

    router = RoutedExecutor(
        {
            EvidenceLevel.L0: RecordingExecutor("sandbox"),
            EvidenceLevel.L2: RecordingExecutor("topology"),
        }
    )
    router.execute(candidate(), EvidenceLevel.L0, seed=7)
    router.execute(candidate(), EvidenceLevel.L2, seed=8)

    assert calls == [
        ("sandbox", EvidenceLevel.L0, 7),
        ("topology", EvidenceLevel.L2, 8),
    ]


def test_agent_uses_sandbox_then_requires_replayed_l4_verification() -> None:
    profile = CalibrationProfile.load(PROFILE_PATH)
    scenario = Scenario.from_dict(
        {
            "schema_version": "1.0",
            "name": "sandbox-agent-test",
            "model": profile.model.model_id,
            "hardware": {"gpu": "H100"},
            "workload": reference_workload().to_dict(),
            "objectives": {
                "energy_j_per_1k_tokens": "min",
                "throughput_tok_s": "max",
            },
            "slo": {"ttft_ms": 1000, "tpot_ms": 100},
            "budget": {"gpu_hours": 1, "verify_top_k": 1},
            "available_levels": ["L0", "L2", "L4"],
            "level_cost_gpu_hours": {"L0": 0, "L2": 0, "L4": 1},
            "candidates": [
                {
                    "id": "agent-candidate",
                    "required_gpus": 2,
                    "target_nodes": 1,
                    "config": dict(candidate(tp=2).config),
                }
            ],
        }
    )
    sandbox = TopologySandboxExecutor(profile, reference_workload())
    verified = EvidenceRecord(
        candidate_id="agent-candidate",
        level=EvidenceLevel.L4,
        metrics={
            "energy_j_per_1k_tokens": 300,
            "throughput_tok_s": 2000,
            "ttft_ms": 100,
            "tpot_ms": 10,
        },
        gpu_hours=1,
        kind=EvidenceKind.VERIFIED,
    )
    router = RoutedExecutor(
        {
            EvidenceLevel.L0: sandbox,
            EvidenceLevel.L2: sandbox,
            EvidenceLevel.L4: ReplayExecutor([verified]),
        }
    )
    report = TokenPowerAgent(
        scenario,
        router,
        twin=TopologyEnergyTwin(profile, reference_workload()),
    ).run(max_steps=2)

    assert report.verified_pareto_ids == ["agent-candidate"]
    assert report.verification_gpu_hours == 1
    assert any(event.level == "L0" for event in report.events)
    assert report.events[-1].phase == "verification"
