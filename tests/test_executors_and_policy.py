from pathlib import Path

import pytest

from tokenpoweragent.executors.cluster import ClusterExecutor, LiveExecutionDisabled
from tokenpoweragent.policy.ipig import Action, CandidateBelief, IPIGPolicy
from tokenpoweragent.schema import EvidenceLevel, Scenario


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_executor_renders_target_resources_without_submitting() -> None:
    scenario = Scenario.load(ROOT / "configs/scenarios/replay_demo.json")
    candidate = scenario.candidate("cfg-fast")
    executor = ClusterExecutor(partition="batch", account="project")
    script = executor.render_slurm(candidate, EvidenceLevel.L4, seed=7)

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --account=project" in script
    assert "TOKENPOWERAGENT_SEED=7" in script
    with pytest.raises(LiveExecutionDisabled):
        executor.execute(candidate, EvidenceLevel.L4, seed=7)


def test_ipig_prefers_more_information_per_gpu_hour() -> None:
    policy = IPIGPolicy(gamma=1.0)
    beliefs = {
        "cheap": CandidateBelief(0.8, 0.8, 0.8, 0.0),
        "expensive": CandidateBelief(0.8, 0.8, 0.8, 0.0),
    }
    decision = policy.select(
        [
            Action("cheap", EvidenceLevel.L1, 0.1),
            Action("expensive", EvidenceLevel.L2, 1.0),
        ],
        beliefs,
        "resolve_slo",
    )
    assert decision.action.candidate_id == "cheap"
