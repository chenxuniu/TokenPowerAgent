"""Command-line entry points for replay, sandbox, and cluster execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.agent.budget_sweep import (
    BudgetSweepError,
    BudgetSweepProtocol,
    evaluate_budget_sweep,
)
from tokenpoweragent.agent.evaluation import (
    POLICY_NAMES,
    ReplayEvaluationError,
    build_policy,
    evaluate_replay_policies,
)
from tokenpoweragent.agent.llm import OpenAICompatibleCompletion
from tokenpoweragent.agent.planner import ConstrainedLLMPlanner, RuleBasedPlanner
from tokenpoweragent.agent.planner_evaluation import (
    PlannerBenchmarkProtocol,
    PlannerEvaluationError,
    evaluate_planner_benchmark,
)
from tokenpoweragent.calibration import (
    CalibrationBuildError,
    build_serving_calibration_profile,
)
from tokenpoweragent.configuration_campaign import (
    ConfigurationCampaign,
    ConfigurationCampaignError,
    freeze_configuration_campaign,
)
from tokenpoweragent.configuration_analysis import (
    ConfigurationAnalysisError,
    build_configuration_campaign_report,
)
from tokenpoweragent.configuration_confirmation import (
    ConfigurationConfirmationError,
    build_configuration_confirmation_report,
)
from tokenpoweragent.configuration_runner import (
    ConfigurationRunnerError,
    DockerVLLMServerManager,
    run_configuration_campaign,
)
from tokenpoweragent.evidence import EvidenceStore
from tokenpoweragent.executors.base import RoutedExecutor
from tokenpoweragent.executors.cluster import ClusterExecutor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.executors.sandbox import SandboxExecutionError, SandboxExecutor
from tokenpoweragent.executors.serving import ServingSandboxExecutor
from tokenpoweragent.executors.topology import (
    TopologySandboxError,
    TopologySandboxExecutor,
)
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.residual_calibration import (
    ResidualCalibrationError,
    build_workload_residual_profile,
)
from tokenpoweragent.search_space import CandidateGrid
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario
from tokenpoweragent.scope_analysis import (
    ScopeConfirmationAnalysisError,
    build_scope_confirmation_decision,
)
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
    ProjectionBackend,
    TopologyEnergyTwin,
)
from tokenpoweragent.validation import (
    HoldoutValidationError,
    WorkloadCampaignValidationError,
    build_holdout_validation_report,
    build_workload_campaign_validation_report,
)
from tokenpoweragent.workload_campaign import (
    WorkloadCampaign,
    WorkloadCampaignError,
    freeze_workload_campaign,
    run_serving_campaign,
)


def _power_limits(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("power limits must be comma-separated integers") from exc
    if not limits or any(limit <= 0 for limit in limits) or len(limits) != len(set(limits)):
        raise argparse.ArgumentTypeError("power limits must be unique positive integers")
    return limits


def _projection_backend(value: str) -> ProjectionBackend:
    try:
        return ProjectionBackend.parse(value)
    except CalibrationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolve_projection_backend(
    backend: Optional[ProjectionBackend],
    legacy_level: Optional[EvidenceLevel],
    default: ProjectionBackend,
) -> ProjectionBackend:
    if legacy_level is None:
        return backend or default
    if backend is not None:
        raise SystemExit("--backend and deprecated --level cannot be combined")
    mapping = {
        EvidenceLevel.L0: ProjectionBackend.L0_A,
        EvidenceLevel.L2: ProjectionBackend.L0_T,
    }
    try:
        resolved = mapping[legacy_level]
    except KeyError as exc:
        raise SystemExit(
            "deprecated --level accepts only L0 or L2; use --backend l0-a|l0-t"
        ) from exc
    print(
        "warning: --level %s is deprecated; using --backend %s and emitting L0"
        % (legacy_level.name, resolved.value),
        file=sys.stderr,
    )
    return resolved


def _campaign_schedule(
    power_limits: tuple[int, ...], repeats: int
) -> tuple[tuple[int, int], ...]:
    """Return (power limit, seed) pairs in a cyclically balanced order."""

    schedule = []
    for seed in range(repeats):
        offset = seed % len(power_limits)
        ordered_limits = power_limits[offset:] + power_limits[:offset]
        schedule.extend((power_limit, seed) for power_limit in ordered_limits)
    return tuple(schedule)


def _request_rate(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "inf":
        return normalized
    try:
        rate = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'") from exc
    if rate <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return "%g" % rate


def _policy_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("policies must be unique comma-separated names")
    unknown = [name for name in names if name not in POLICY_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown policies %s; choose from %s"
            % (", ".join(unknown), ", ".join(POLICY_NAMES))
        )
    return names


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-ipig-score", type=float, default=0.0)
    parser.add_argument("--frontier-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=POLICY_NAMES, default="ipig")
    parser.add_argument("--intent", help="override the scenario's natural-language intent")
    parser.add_argument("--planner", choices=("rule", "llm"), default="rule")
    parser.add_argument(
        "--planner-base-url",
        default=os.environ.get("TOKENPOWERAGENT_LLM_BASE_URL", ""),
        help="OpenAI-compatible base URL ending in /v1",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("TOKENPOWERAGENT_LLM_MODEL", ""),
    )
    parser.add_argument(
        "--planner-api-key-env",
        default="TOKENPOWERAGENT_LLM_API_KEY",
    )
    parser.add_argument("--planner-timeout-seconds", type=float, default=30.0)


def _planner_from_args(args: argparse.Namespace):
    if args.planner == "rule":
        return RuleBasedPlanner()
    if not args.planner_base_url or not args.planner_model:
        raise SystemExit(
            "--planner llm requires --planner-base-url and --planner-model"
        )
    completion = OpenAICompatibleCompletion(
        base_url=args.planner_base_url,
        model=args.planner_model,
        api_key=os.environ.get(args.planner_api_key_env),
        timeout_seconds=args.planner_timeout_seconds,
    )
    return ConstrainedLLMPlanner(
        completion,
        state_guard=RuleBasedPlanner(),
        prompt_version="state-table-v2",
    )


def _run_agent(
    args: argparse.Namespace,
    scenario: Scenario,
    executor,
    twin=None,
):
    if args.intent:
        scenario = replace(scenario, intent=args.intent.strip())
    return TokenPowerAgent(
        scenario=scenario,
        executor=executor,
        twin=twin,
        planner=_planner_from_args(args),
        policy=build_policy(args.policy, seed=args.seed),
    ).run(
        max_steps=args.max_steps,
        min_ipig_score=args.min_ipig_score,
        frontier_patience=args.frontier_patience,
        run_seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenpoweragent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser(
        "replay", help="run one bounded TokenPowerAgent replay episode"
    )
    replay.add_argument("--scenario", type=Path, required=True)
    replay.add_argument("--records", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    _add_agent_arguments(replay)

    agent_search = subparsers.add_parser(
        "agent-search",
        help="run an L0 CPU projection and replay measured L1-L4 evidence",
    )
    agent_search.add_argument("--scenario", type=Path, required=True)
    agent_search.add_argument("--calibration", type=Path, required=True)
    agent_search.add_argument(
        "--records",
        type=Path,
        required=True,
        help="sealed L1/L3/L4 evidence available to the routed executor",
    )
    agent_search.add_argument("--output", type=Path, required=True)
    agent_search.add_argument(
        "--sandbox-backend",
        type=_projection_backend,
        default=ProjectionBackend.L0_T,
        help="CPU L0 backend: l0-a (analytical) or l0-t (topology-aware)",
    )
    _add_agent_arguments(agent_search)

    benchmark = subparsers.add_parser(
        "benchmark-replay",
        help="compare acquisition policies against a sealed L4 replay oracle",
    )
    benchmark.add_argument("--scenario", type=Path, required=True)
    benchmark.add_argument("--records", type=Path, required=True)
    benchmark.add_argument(
        "--policies",
        type=_policy_names,
        default=POLICY_NAMES,
    )
    benchmark.add_argument("--episodes", type=int, default=20)
    benchmark.add_argument("--max-steps", type=int, default=20)
    benchmark.add_argument("--min-ipig-score", type=float, default=0.0)
    benchmark.add_argument("--frontier-patience", type=int, default=0)
    benchmark.add_argument("--output", type=Path, required=True)

    budget_sweep = subparsers.add_parser(
        "benchmark-budget-sweep",
        help="run a preregistered matched-budget policy replay sweep",
    )
    budget_sweep.add_argument("--scenario", type=Path, required=True)
    budget_sweep.add_argument("--records", type=Path, required=True)
    budget_sweep.add_argument("--protocol", type=Path, required=True)
    budget_sweep.add_argument("--output", type=Path, required=True)

    planner_benchmark = subparsers.add_parser(
        "benchmark-planner",
        help="measure bounded LLM planner conformance, fallback, and overhead",
    )
    planner_benchmark.add_argument("--protocol", type=Path, required=True)
    planner_benchmark.add_argument(
        "--planner-base-url",
        default=os.environ.get("TOKENPOWERAGENT_LLM_BASE_URL", ""),
        help="OpenAI-compatible base URL ending in /v1",
    )
    planner_benchmark.add_argument(
        "--planner-api-key-env",
        default="TOKENPOWERAGENT_LLM_API_KEY",
    )
    planner_benchmark.add_argument(
        "--planner-timeout-seconds", type=float, default=30.0
    )
    planner_benchmark.add_argument("--output", type=Path, required=True)

    render = subparsers.add_parser("render-slurm", help="render without submitting")
    render.add_argument("--scenario", type=Path, required=True)
    render.add_argument("--candidate", required=True)
    render.add_argument("--level", type=EvidenceLevel.parse, default=EvidenceLevel.L4)
    render.add_argument("--partition", default="gpu")
    render.add_argument("--account")
    render.add_argument("--seed", type=int, default=0)

    sandbox = subparsers.add_parser(
        "sandbox-smoke", help="run an L1 single-GPU power-cap campaign"
    )
    sandbox.add_argument("--image", default="tokenpower-sandbox:cuda12.8")
    sandbox.add_argument("--gpu-id", type=int, default=0)
    sandbox.add_argument("--power-limits", type=_power_limits, default=(350, 500, 700))
    sandbox.add_argument("--repeats", type=int, default=3)
    sandbox.add_argument("--matrix-size", type=int, default=16384)
    sandbox.add_argument("--warmup", type=int, default=20)
    sandbox.add_argument("--iterations", type=int, default=400)
    sandbox.add_argument("--sample-ms", type=int, default=100)
    sandbox.add_argument("--timeout-seconds", type=float, default=300.0)
    sandbox.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/sandbox-smoke.jsonl"),
    )
    sandbox.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("experiments/results/telemetry"),
    )
    sandbox.add_argument(
        "--no-sudo",
        action="store_true",
        help="run Docker and power-limit commands without sudo",
    )

    serving = subparsers.add_parser(
        "serving-smoke",
        help="measure a persistent single-GPU vLLM server at an active-window boundary",
    )
    serving.add_argument("--image", default="tokenpower-vllm-client:v0.23.0")
    serving.add_argument("--server-container", default="tpa-vllm-qwen7b")
    serving.add_argument("--network", default="tpa-serving-bench")
    serving.add_argument("--cache-volume", default="tpa-hf-cache")
    serving.add_argument(
        "--base-url", default="http://tpa-vllm-qwen7b:8000"
    )
    serving.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    serving.add_argument("--served-model-name", default="qwen2.5-7b")
    serving.add_argument("--gpu-id", type=int, default=0)
    serving.add_argument("--power-limits", type=_power_limits, default=(700,))
    serving.add_argument("--repeats", type=int, default=1)
    serving.add_argument("--input-len", type=int, default=512)
    serving.add_argument("--output-len", type=int, default=128)
    serving.add_argument("--num-prompts", type=int, default=64)
    serving.add_argument("--num-warmups", type=int, default=2)
    serving.add_argument("--request-rate", type=_request_rate, default="inf")
    serving.add_argument("--max-concurrency", type=int, default=8)
    serving.add_argument(
        "--dataset-split",
        choices=("calibration", "validation", "holdout", "diagnostic"),
        default="diagnostic",
        help="freeze the evidence role before running the workload",
    )
    serving.add_argument("--sample-ms", type=int, default=100)
    serving.add_argument("--timeout-seconds", type=float, default=600.0)
    serving.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/qwen7b-serving-smoke.jsonl"),
    )
    serving.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("experiments/results/serving-telemetry"),
    )
    serving.add_argument(
        "--no-sudo",
        action="store_true",
        help="run Docker and power-limit commands without sudo",
    )

    predict = subparsers.add_parser(
        "sandbox-predict",
        help="rank scenario candidates with a calibrated topology sandbox",
    )
    predict.add_argument("--scenario", type=Path, required=True)
    predict.add_argument("--calibration", type=Path, required=True)
    predict_backend = predict.add_mutually_exclusive_group()
    predict_backend.add_argument(
        "--backend",
        type=_projection_backend,
        help="CPU L0 backend: l0-a (analytical) or l0-t (topology-aware)",
    )
    predict_backend.add_argument(
        "--level",
        dest="legacy_projection_level",
        type=EvidenceLevel.parse,
        help=argparse.SUPPRESS,
    )
    predict.add_argument(
        "--candidate",
        action="append",
        help="predict only this candidate id; may be repeated",
    )
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/topology-sandbox.jsonl"),
    )
    predict.add_argument(
        "--summary",
        type=Path,
        help="defaults to OUTPUT with a .summary.json suffix",
    )

    expand = subparsers.add_parser(
        "expand-space",
        help="compile a deterministic TP/PP/batching candidate grid",
    )
    expand.add_argument("--space", type=Path, required=True)
    expand.add_argument("--scenario", type=Path)
    expand.add_argument("--calibration", type=Path)
    expand.add_argument("--output", type=Path, required=True)

    calibration = subparsers.add_parser(
        "build-calibration",
        help="aggregate measured L1 serving JSONL into a calibration profile",
    )
    calibration.add_argument("--records", type=Path, required=True)
    calibration.add_argument("--template", type=Path, required=True)
    calibration.add_argument("--profile-id", required=True)
    calibration.add_argument("--power-limit-w", type=float, required=True)
    calibration.add_argument("--min-repeats", type=int, default=3)
    calibration.add_argument("--publication-eligible", action="store_true")
    calibration.add_argument("--output", type=Path, required=True)

    residual = subparsers.add_parser(
        "fit-workload-residuals",
        help="fit a development-only workload correction from a sealed report",
    )
    residual.add_argument("--profile", type=Path, required=True)
    residual.add_argument("--campaign", type=Path, required=True)
    residual.add_argument("--report", type=Path, required=True)
    residual.add_argument("--profile-id", required=True)
    residual.add_argument("--output", type=Path, required=True)
    residual.add_argument("--diagnostics", type=Path, required=True)

    validation = subparsers.add_parser(
        "validate-holdout",
        help="compare a frozen sandbox prediction with blind serving evidence",
    )
    validation.add_argument("--predictions", type=Path, required=True)
    validation.add_argument("--measurements", type=Path, required=True)
    validation.add_argument("--freeze-manifest", type=Path, required=True)
    validation.add_argument("--min-repeats", type=int, default=3)
    validation.add_argument("--output", type=Path, required=True)

    campaign_validation = subparsers.add_parser(
        "validate-workload-campaign",
        help="analyze a frozen workload-transfer validation campaign",
    )
    campaign_validation.add_argument("--campaign", type=Path, required=True)
    campaign_validation.add_argument("--predictions", type=Path, required=True)
    campaign_validation.add_argument("--measurements", type=Path, required=True)
    campaign_validation.add_argument(
        "--freeze-manifest", type=Path, required=True
    )
    campaign_validation.add_argument("--artifact-manifest", type=Path)
    campaign_validation.add_argument("--output", type=Path, required=True)

    scope_analysis = subparsers.add_parser(
        "analyze-scope-confirmation",
        help="apply a campaign's pre-registered energy and latency scope rules",
    )
    scope_analysis.add_argument("--campaign", type=Path, required=True)
    scope_analysis.add_argument("--report", type=Path, required=True)
    scope_analysis.add_argument(
        "--artifact-manifest", type=Path, required=True
    )
    scope_analysis.add_argument("--output", type=Path, required=True)

    freeze_campaign = subparsers.add_parser(
        "freeze-workload-campaign",
        help="freeze all workload-transfer predictions before measurement",
    )
    freeze_campaign.add_argument("--campaign", type=Path, required=True)
    freeze_campaign.add_argument("--calibration", type=Path, required=True)
    freeze_backend = freeze_campaign.add_mutually_exclusive_group()
    freeze_backend.add_argument(
        "--backend",
        type=_projection_backend,
        help="CPU L0 backend: l0-a (analytical) or l0-t (topology-aware)",
    )
    freeze_backend.add_argument(
        "--level",
        dest="legacy_projection_level",
        type=EvidenceLevel.parse,
        help=argparse.SUPPRESS,
    )
    freeze_campaign.add_argument("--output", type=Path, required=True)
    freeze_campaign.add_argument(
        "--summary",
        type=Path,
        help="defaults to OUTPUT with a .summary.json suffix",
    )
    freeze_campaign.add_argument("--manifest", type=Path, required=True)

    freeze_config_campaign = subparsers.add_parser(
        "freeze-config-campaign",
        help="freeze a serving-configuration search before GPU measurement",
    )
    freeze_config_campaign.add_argument("--campaign", type=Path, required=True)
    freeze_config_campaign.add_argument(
        "--calibration", type=Path, required=True
    )
    freeze_config_campaign.add_argument("--scenario", type=Path, required=True)
    freeze_config_campaign.add_argument(
        "--predictions", type=Path, required=True
    )
    freeze_config_campaign.add_argument("--schedule", type=Path, required=True)
    freeze_config_campaign.add_argument("--summary", type=Path, required=True)
    freeze_config_campaign.add_argument("--manifest", type=Path, required=True)

    run_config_campaign = subparsers.add_parser(
        "run-config-campaign",
        help="run a frozen, resumable L1/L4 serving-configuration campaign",
    )
    run_config_campaign.add_argument("--campaign", type=Path, required=True)
    run_config_campaign.add_argument(
        "--predictions", type=Path, required=True
    )
    run_config_campaign.add_argument("--schedule", type=Path, required=True)
    run_config_campaign.add_argument("--summary", type=Path, required=True)
    run_config_campaign.add_argument(
        "--freeze-manifest", type=Path, required=True
    )
    run_config_campaign.add_argument("--output", type=Path, required=True)
    run_config_campaign.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("experiments/results/config-search-v1-telemetry"),
    )
    run_config_campaign.add_argument(
        "--server-log-dir",
        type=Path,
        default=Path("experiments/results/config-search-v1-server-logs"),
    )
    run_config_campaign.add_argument("--gpu-id", type=int, default=0)
    run_config_campaign.add_argument(
        "--timeout-seconds", type=float, default=900.0
    )
    run_config_campaign.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="run at most this many frozen actions; zero runs all remaining",
    )
    run_config_campaign.add_argument("--resume", action="store_true")
    run_config_campaign.add_argument(
        "--no-sudo",
        action="store_true",
        help="run Docker and power-limit commands without sudo",
    )

    validate_config_campaign = subparsers.add_parser(
        "validate-config-campaign",
        help="validate a completed configuration corpus and build its L4 oracle",
    )
    validate_config_campaign.add_argument("--campaign", type=Path, required=True)
    validate_config_campaign.add_argument(
        "--predictions", type=Path, required=True
    )
    validate_config_campaign.add_argument("--schedule", type=Path, required=True)
    validate_config_campaign.add_argument("--summary", type=Path, required=True)
    validate_config_campaign.add_argument(
        "--freeze-manifest", type=Path, required=True
    )
    validate_config_campaign.add_argument(
        "--measurements", type=Path, required=True
    )
    validate_config_campaign.add_argument("--output", type=Path, required=True)
    validate_config_campaign.add_argument("--corpus", type=Path, required=True)
    validate_config_campaign.add_argument("--artifact-list", type=Path)
    validate_config_campaign.add_argument("--artifact-manifest", type=Path)
    validate_config_campaign.add_argument(
        "--include-artifact",
        action="append",
        type=Path,
        default=[],
        help="include an additional environment or run-log file in raw hashes",
    )

    validate_config_confirmation = subparsers.add_parser(
        "validate-config-confirmation",
        help="validate an independently frozen paired L4 confirmation",
    )
    validate_config_confirmation.add_argument(
        "--campaign", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--predictions", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--schedule", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--summary", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--freeze-manifest", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--measurements", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--output", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--artifact-list", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--artifact-manifest", type=Path, required=True
    )
    validate_config_confirmation.add_argument(
        "--include-artifact",
        action="append",
        type=Path,
        default=[],
        help="include an additional environment or run-log file in raw hashes",
    )

    run_campaign = subparsers.add_parser(
        "run-serving-campaign",
        help="measure a pre-frozen workload-transfer campaign",
    )
    run_campaign.add_argument("--campaign", type=Path, required=True)
    run_campaign.add_argument("--predictions", type=Path, required=True)
    run_campaign.add_argument("--freeze-manifest", type=Path, required=True)
    run_campaign.add_argument("--image", default="tokenpower-vllm-client:v0.23.0")
    run_campaign.add_argument("--server-container", default="tpa-vllm-qwen7b")
    run_campaign.add_argument("--network", default="tpa-serving-bench")
    run_campaign.add_argument("--cache-volume", default="tpa-hf-cache")
    run_campaign.add_argument(
        "--base-url", default="http://tpa-vllm-qwen7b:8000"
    )
    run_campaign.add_argument("--served-model-name", default="qwen2.5-7b")
    run_campaign.add_argument("--gpu-id", type=int, default=0)
    run_campaign.add_argument("--timeout-seconds", type=float, default=900.0)
    run_campaign.add_argument("--output", type=Path, required=True)
    run_campaign.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("experiments/results/serving-campaign-telemetry"),
    )
    run_campaign.add_argument("--resume", action="store_true")
    run_campaign.add_argument(
        "--no-sudo",
        action="store_true",
        help="run Docker and power-limit commands without sudo",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark-planner":
        if not args.planner_base_url:
            raise SystemExit(
                "benchmark-planner requires --planner-base-url or "
                "TOKENPOWERAGENT_LLM_BASE_URL"
            )
        try:
            protocol = PlannerBenchmarkProtocol.load(args.protocol)
            completion = OpenAICompatibleCompletion(
                base_url=args.planner_base_url,
                model=protocol.planner_model,
                api_key=os.environ.get(args.planner_api_key_env),
                timeout_seconds=args.planner_timeout_seconds,
                temperature=protocol.temperature,
                max_tokens=protocol.max_tokens,
            )
            report = evaluate_planner_benchmark(
                protocol=protocol,
                completion=completion,
                protocol_sha256=_sha256_file(args.protocol),
                endpoint=args.planner_base_url,
            )
        except PlannerEvaluationError as exc:
            raise SystemExit("cannot benchmark planner: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print("wrote planner benchmark to %s" % args.output)
        return 0

    if args.command == "benchmark-budget-sweep":
        try:
            scenario = Scenario.load(args.scenario)
            records = EvidenceStore.read_jsonl(args.records).records
            protocol = BudgetSweepProtocol.load(args.protocol)
            report = evaluate_budget_sweep(
                scenario=scenario,
                records=records,
                protocol=protocol,
                source_sha256=_sha256_file(args.records),
                scenario_sha256=_sha256_file(args.scenario),
                protocol_sha256=_sha256_file(args.protocol),
            )
        except BudgetSweepError as exc:
            raise SystemExit("cannot benchmark budget sweep: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["budget_response"], indent=2, sort_keys=True))
        print("wrote budget sweep to %s" % args.output)
        return 0

    if args.command == "sandbox-smoke":
        if args.repeats < 1:
            raise SystemExit("--repeats must be positive")
        if args.matrix_size < 1 or args.warmup < 1 or args.iterations < 1:
            raise SystemExit("workload dimensions and iteration counts must be positive")
        executor = SandboxExecutor(
            telemetry_dir=args.telemetry_dir,
            gpu_id=args.gpu_id,
            sample_ms=args.sample_ms,
            use_sudo=not args.no_sudo,
            timeout_seconds=args.timeout_seconds,
        )
        records = EvidenceStore()
        for campaign_index, (power_limit_w, seed) in enumerate(
            _campaign_schedule(args.power_limits, args.repeats)
        ):
            candidate = Candidate(
                candidate_id="gemm-pl%d" % power_limit_w,
                config={
                    "image": args.image,
                    "power_limit_w": power_limit_w,
                    "campaign_index": campaign_index,
                    "command": [
                        "--matrix-size",
                        str(args.matrix_size),
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                        "--seed",
                        str(seed),
                    ],
                },
            )
            try:
                record = executor.execute(candidate, EvidenceLevel.L1, seed)
            except SandboxExecutionError as exc:
                print(
                    "sandbox failed for %s seed %d: %s"
                    % (candidate.candidate_id, seed, exc),
                    file=sys.stderr,
                )
                return 2
            records.append(record)
            print(json.dumps(record.to_dict(), sort_keys=True))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        records.write_jsonl(args.output)
        print("wrote %d evidence records to %s" % (len(records.records), args.output))
        return 0

    if args.command == "serving-smoke":
        if args.repeats < 1:
            raise SystemExit("--repeats must be positive")
        if (
            args.input_len < 1
            or args.output_len < 1
            or args.num_prompts < 1
            or args.num_warmups < 0
            or args.max_concurrency < 1
        ):
            raise SystemExit("serving workload dimensions must be positive")
        executor = ServingSandboxExecutor(
            telemetry_dir=args.telemetry_dir,
            gpu_id=args.gpu_id,
            sample_ms=args.sample_ms,
            use_sudo=not args.no_sudo,
            timeout_seconds=args.timeout_seconds,
        )
        records = EvidenceStore()
        for campaign_index, (power_limit_w, seed) in enumerate(
            _campaign_schedule(args.power_limits, args.repeats)
        ):
            candidate = Candidate(
                candidate_id="qwen7b-serving-pl%d" % power_limit_w,
                config={
                    "image": args.image,
                    "server_container": args.server_container,
                    "network": args.network,
                    "cache_volume": args.cache_volume,
                    "base_url": args.base_url,
                    "model": args.model,
                    "served_model_name": args.served_model_name,
                    "power_limit_w": power_limit_w,
                    "campaign_index": campaign_index,
                    "input_len": args.input_len,
                    "output_len": args.output_len,
                    "num_prompts": args.num_prompts,
                    "num_warmups": args.num_warmups,
                    "request_rate": args.request_rate,
                    "max_concurrency": args.max_concurrency,
                    "dataset_split": args.dataset_split,
                },
            )
            try:
                record = executor.execute(candidate, EvidenceLevel.L1, seed)
            except SandboxExecutionError as exc:
                print(
                    "serving sandbox failed for %s seed %d: %s"
                    % (candidate.candidate_id, seed, exc),
                    file=sys.stderr,
                )
                return 2
            records.append(record)
            print(json.dumps(record.to_dict(), sort_keys=True))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        records.write_jsonl(args.output)
        print("wrote %d evidence records to %s" % (len(records.records), args.output))
        return 0

    if args.command == "sandbox-predict":
        backend = _resolve_projection_backend(
            args.backend,
            args.legacy_projection_level,
            ProjectionBackend.L0_T,
        )
        try:
            scenario = Scenario.load(args.scenario)
            profile = CalibrationProfile.load(args.calibration)
            workload = InferenceWorkload.from_mapping(scenario.workload)
            executor = TopologySandboxExecutor(
                profile,
                workload,
                expected_model=scenario.model,
                profile_path=args.calibration,
                scenario_path=args.scenario,
                backend=backend,
            )
        except (CalibrationError, TopologySandboxError) as exc:
            raise SystemExit("invalid topology sandbox input: %s" % exc) from exc

        if args.candidate:
            requested = list(dict.fromkeys(args.candidate))
            try:
                candidates = [scenario.candidate(candidate_id) for candidate_id in requested]
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            candidates = list(scenario.candidates)

        records = EvidenceStore()
        for candidate in candidates:
            try:
                record = executor.execute(
                    candidate, EvidenceLevel.L0, args.seed
                )
            except TopologySandboxError as exc:
                raise SystemExit(
                    "topology sandbox failed for %s: %s"
                    % (candidate.candidate_id, exc)
                ) from exc
            records.append(record)
            print(json.dumps(record.to_dict(), sort_keys=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        records.write_jsonl(args.output)
        succeeded = sum(record.status.value == "succeeded" for record in records.records)
        slo_feasible = [
            record
            for record in records.records
            if record.status.value == "succeeded" and record.satisfies(scenario.slo)
        ]
        frontier = pareto_front(
            [(record.candidate_id, record.metrics) for record in slo_feasible],
            scenario.objectives,
        )
        robust_rows = []
        robust_slo_feasible_ids = []
        for record in records.records:
            if record.status.value != "succeeded":
                continue
            pessimistic = dict(record.metrics)
            for metric, direction in scenario.objectives.items():
                suffix = "_upper" if direction == "min" else "_lower"
                pessimistic[metric] = record.metrics.get(
                    metric + suffix, record.metrics[metric]
                )
            for metric in ("ttft_ms", "tpot_ms"):
                pessimistic[metric] = record.metrics.get(
                    metric + "_upper", record.metrics.get(metric, float("inf"))
                )
            if scenario.slo.accepts(pessimistic):
                robust_slo_feasible_ids.append(record.candidate_id)
                robust_rows.append((record.candidate_id, pessimistic))
        robust_frontier = pareto_front(robust_rows, scenario.objectives)
        summary_path = args.summary or args.output.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "scenario": scenario.name,
                    "profile_id": profile.profile_id,
                    "profile_publication_eligible": profile.publication_eligible,
                    "uncertainty_calibrated": profile.uncertainty_calibrated,
                    "level": EvidenceLevel.L0.name,
                    "sandbox_backend": backend.display_name,
                    "projection_backend": backend.value,
                    "candidate_count": len(records.records),
                    "feasible_count": succeeded,
                    "slo_feasible_ids": [
                        record.candidate_id for record in slo_feasible
                    ],
                    "predicted_pareto_ids": frontier,
                    "robust_slo_feasible_ids": robust_slo_feasible_ids,
                    "robust_predicted_pareto_ids": robust_frontier,
                    "robust_frontier_is_provisional": (
                        not profile.uncertainty_calibrated
                    ),
                    "scenario_sha256": records.records[0].provenance.get(
                        "scenario_sha256"
                    ),
                    "profile_sha256": records.records[0].provenance.get(
                        "profile_sha256"
                    ),
                    "validation_required": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "wrote %d predictions (%d feasible, %d predicted Pareto) to %s and %s"
            % (
                len(records.records),
                succeeded,
                len(frontier),
                args.output,
                summary_path,
            )
        )
        return 0

    if args.command == "expand-space":
        if (args.scenario is None) != (args.calibration is None):
            raise SystemExit(
                "--scenario and --calibration must be supplied together"
            )
        try:
            grid = CandidateGrid.load(args.space)
            profile = None
            workload = None
            if args.scenario is not None and args.calibration is not None:
                scenario = Scenario.load(args.scenario)
                profile = CalibrationProfile.load(args.calibration)
                if profile.model.model_id != scenario.model:
                    raise CalibrationError(
                        "profile model does not match scenario model"
                    )
                workload = InferenceWorkload.from_mapping(scenario.workload)
            compilation = grid.compile(profile=profile, workload=workload)
        except CalibrationError as exc:
            raise SystemExit("invalid search space: %s" % exc) from exc
        rendered = {
            "schema_version": grid.schema_version,
            "grid_id": grid.grid_id,
            "generated_count": len(compilation.candidates),
            "rejected_count": len(compilation.rejected),
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "required_gpus": candidate.required_gpus,
                    "target_nodes": candidate.target_nodes,
                    "config": dict(candidate.config),
                }
                for candidate in compilation.candidates
            ],
            "rejected": [
                {
                    "id": rejected.candidate_id,
                    "reason": rejected.reason,
                }
                for rejected in compilation.rejected
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rendered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "wrote %d candidates (%d rejected) to %s"
            % (
                len(compilation.candidates),
                len(compilation.rejected),
                args.output,
            )
        )
        return 0

    if args.command == "build-calibration":
        try:
            profile = build_serving_calibration_profile(
                records_path=args.records,
                template_path=args.template,
                profile_id=args.profile_id,
                power_limit_w=args.power_limit_w,
                min_repeats=args.min_repeats,
                publication_eligible=args.publication_eligible,
            )
        except CalibrationBuildError as exc:
            raise SystemExit("cannot build calibration profile: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "wrote %d calibration points from %d records to %s"
            % (
                len(profile["calibration_points"]),
                profile["metadata"]["matching_records"],
                args.output,
            )
        )
        return 0

    if args.command == "fit-workload-residuals":
        for path in (args.output, args.diagnostics):
            if path.exists():
                raise SystemExit("refusing to overwrite fitted artifact: %s" % path)
        try:
            profile, diagnostics = build_workload_residual_profile(
                profile_path=args.profile,
                campaign_path=args.campaign,
                report_path=args.report,
                profile_id=args.profile_id,
            )
        except (ResidualCalibrationError, OSError, KeyError) as exc:
            raise SystemExit("cannot fit workload residuals: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.diagnostics.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        energy = diagnostics["derived_energy_j_per_1k_output_tokens"]
        print(
            "fitted %d workload residual points: development LOO energy MAPE "
            "%.2f%% (max %.2f%%); wrote %s and %s"
            % (
                len(diagnostics["training_rows"]),
                energy["mean_absolute_percentage_error_pct"],
                energy["maximum_absolute_percentage_error_pct"],
                args.output,
                args.diagnostics,
            )
        )
        return 0

    if args.command == "validate-holdout":
        try:
            report = build_holdout_validation_report(
                predictions_path=args.predictions,
                measurements_path=args.measurements,
                freeze_manifest_path=args.freeze_manifest,
                min_repeats=args.min_repeats,
            )
        except (HoldoutValidationError, OSError, KeyError) as exc:
            raise SystemExit("cannot validate holdout: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = report["summary"]
        print(
            "validated %d blind repeats: MAPE %.2f%%, interval coverage %d/%d; wrote %s"
            % (
                report["protocol"]["repeat_count"],
                summary["mean_absolute_percentage_error_pct"],
                summary["interval_covered_metric_count"],
                summary["metric_count"],
                args.output,
            )
        )
        return 0

    if args.command == "validate-workload-campaign":
        try:
            report = build_workload_campaign_validation_report(
                campaign_path=args.campaign,
                predictions_path=args.predictions,
                measurements_path=args.measurements,
                freeze_manifest_path=args.freeze_manifest,
                artifact_manifest_path=args.artifact_manifest,
            )
        except (
            HoldoutValidationError,
            WorkloadCampaignValidationError,
            OSError,
            KeyError,
        ) as exc:
            raise SystemExit(
                "cannot validate workload campaign: %s" % exc
            ) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        energy = report["aggregate_metrics"][
            "energy_j_per_1k_output_tokens"
        ]
        print(
            "validated %d workloads and %d measurements: energy MAPE %.2f%%, "
            "Spearman %s, interval coverage %d/%d; wrote %s"
            % (
                report["protocol"]["workload_count"],
                report["protocol"]["measurement_count"],
                energy["mean_absolute_percentage_error_pct"],
                (
                    "N/A"
                    if energy["spearman_rank_correlation"] is None
                    else "%.3f" % energy["spearman_rank_correlation"]
                ),
                energy["interval_covered_workloads"],
                energy["interval_evaluated_workloads"],
                args.output,
            )
        )
        return 0

    if args.command == "analyze-scope-confirmation":
        try:
            decision = build_scope_confirmation_decision(
                campaign_path=args.campaign,
                report_path=args.report,
                artifact_manifest_path=args.artifact_manifest,
            )
        except (ScopeConfirmationAnalysisError, OSError, KeyError) as exc:
            raise SystemExit(
                "cannot analyze scope confirmation: %s" % exc
            ) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        primary = decision["primary_endpoint"]
        latency = decision["latency_scope"]
        print(
            "scope confirmation: energy MAPE %.2f%% (%s); "
            "supported TTFT MAPE %.2f%%, sparse TTFT MAPE %.2f%%; "
            "decision=%s; wrote %s"
            % (
                primary["observed_mape_pct"],
                "pass" if primary["passed"] else "fail",
                latency["supported_stratum"]["observed_mape_pct"],
                latency["sparse_stratum"]["observed_mape_pct"],
                latency["decision"],
                args.output,
            )
        )
        return 0

    if args.command == "freeze-workload-campaign":
        summary_path = args.summary or args.output.with_suffix(".summary.json")
        backend = _resolve_projection_backend(
            args.backend,
            args.legacy_projection_level,
            ProjectionBackend.L0_A,
        )
        try:
            summary = freeze_workload_campaign(
                campaign_path=args.campaign,
                calibration_path=args.calibration,
                output_path=args.output,
                summary_path=summary_path,
                manifest_path=args.manifest,
                backend=backend,
            )
        except (WorkloadCampaignError, OSError) as exc:
            raise SystemExit("cannot freeze workload campaign: %s" % exc) from exc
        print(
            "froze %d %s predictions for %s; wrote %s, %s, and %s"
            % (
                summary["prediction_count"],
                summary["level"],
                summary["campaign_id"],
                args.output,
                summary_path,
                args.manifest,
            )
        )
        return 0

    if args.command == "freeze-config-campaign":
        try:
            summary = freeze_configuration_campaign(
                campaign_path=args.campaign,
                calibration_path=args.calibration,
                scenario_path=args.scenario,
                predictions_path=args.predictions,
                schedule_path=args.schedule,
                summary_path=args.summary,
                manifest_path=args.manifest,
            )
        except (ConfigurationCampaignError, OSError) as exc:
            raise SystemExit(
                "cannot freeze configuration campaign: %s" % exc
            ) from exc
        print(
            "froze %d L0 predictions and %d frozen %s measurements "
            "for %s; manifest=%s"
            % (
                summary["l0_prediction_count"],
                summary["measurement_count"],
                "/".join(summary["levels"]),
                summary["campaign_id"],
                args.manifest,
            )
        )
        return 0

    if args.command == "run-config-campaign":
        try:
            campaign = ConfigurationCampaign.load(args.campaign)
            executor = ServingSandboxExecutor(
                telemetry_dir=args.telemetry_dir,
                gpu_id=args.gpu_id,
                sample_ms=campaign.sample_ms,
                use_sudo=not args.no_sudo,
                timeout_seconds=args.timeout_seconds,
            )
            server_manager = DockerVLLMServerManager(
                campaign=campaign,
                server_log_dir=args.server_log_dir,
                gpu_id=args.gpu_id,
                use_sudo=not args.no_sudo,
            )
            report = run_configuration_campaign(
                campaign_path=args.campaign,
                predictions_path=args.predictions,
                schedule_path=args.schedule,
                summary_path=args.summary,
                manifest_path=args.freeze_manifest,
                output_path=args.output,
                executor=executor,
                server_manager=server_manager,
                resume=args.resume,
                max_actions=args.max_actions,
            )
        except (
            ConfigurationCampaignError,
            ConfigurationRunnerError,
            SandboxExecutionError,
            OSError,
        ) as exc:
            raise SystemExit(
                "cannot run configuration campaign: %s" % exc
            ) from exc
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-config-campaign":
        try:
            report = build_configuration_campaign_report(
                campaign_path=args.campaign,
                predictions_path=args.predictions,
                schedule_path=args.schedule,
                summary_path=args.summary,
                freeze_manifest_path=args.freeze_manifest,
                measurements_path=args.measurements,
                report_path=args.output,
                corpus_path=args.corpus,
                artifact_list_path=args.artifact_list,
                artifact_manifest_path=args.artifact_manifest,
                include_artifacts=args.include_artifact,
            )
        except (
            ConfigurationAnalysisError,
            ConfigurationCampaignError,
            OSError,
        ) as exc:
            raise SystemExit(
                "cannot validate configuration campaign: %s" % exc
            ) from exc
        print(
            "validated %d frozen measurements: L4 Pareto=%s; "
            "publication_ready=%s; wrote %s and %s"
            % (
                report["protocol"]["measurement_count"],
                ",".join(report["oracle"]["pareto_ids"]),
                str(report["summary"]["publication_ready"]).lower(),
                args.output,
                args.corpus,
            )
        )
        return 0

    if args.command == "validate-config-confirmation":
        try:
            report = build_configuration_confirmation_report(
                campaign_path=args.campaign,
                predictions_path=args.predictions,
                schedule_path=args.schedule,
                summary_path=args.summary,
                freeze_manifest_path=args.freeze_manifest,
                measurements_path=args.measurements,
                report_path=args.output,
                artifact_list_path=args.artifact_list,
                artifact_manifest_path=args.artifact_manifest,
                include_artifacts=args.include_artifact,
            )
        except (
            ConfigurationConfirmationError,
            ConfigurationCampaignError,
            OSError,
        ) as exc:
            raise SystemExit(
                "cannot validate configuration confirmation: %s" % exc
            ) from exc
        print(
            "validated %d independent pairs: energy saving %.2f%% "
            "(95%% CI %.2f%% to %.2f%%), TTFT reduction %.2f%%; "
            "confirmation_passed=%s; wrote %s"
            % (
                report["protocol"]["pair_count"],
                report["summary"]["headline_energy_saving_pct"],
                report["summary"]["headline_energy_saving_ci95_pct"][0],
                report["summary"]["headline_energy_saving_ci95_pct"][1],
                report["summary"]["headline_ttft_reduction_pct"],
                str(report["summary"]["confirmation_passed"]).lower(),
                args.output,
            )
        )
        return 0

    if args.command == "run-serving-campaign":
        try:
            campaign = WorkloadCampaign.load(args.campaign)
            executor = ServingSandboxExecutor(
                telemetry_dir=args.telemetry_dir,
                gpu_id=args.gpu_id,
                sample_ms=campaign.sample_ms,
                use_sudo=not args.no_sudo,
                timeout_seconds=args.timeout_seconds,
            )
            report = run_serving_campaign(
                campaign_path=args.campaign,
                predictions_path=args.predictions,
                manifest_path=args.freeze_manifest,
                output_path=args.output,
                executor=executor,
                runtime={
                    "image": args.image,
                    "server_container": args.server_container,
                    "network": args.network,
                    "cache_volume": args.cache_volume,
                    "base_url": args.base_url,
                    "served_model_name": args.served_model_name,
                },
                resume=args.resume,
            )
        except (WorkloadCampaignError, SandboxExecutionError, OSError) as exc:
            raise SystemExit("cannot run serving campaign: %s" % exc) from exc
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    scenario = Scenario.load(args.scenario)
    if args.command == "benchmark-replay":
        records = EvidenceStore.read_jsonl(args.records).records
        try:
            report = evaluate_replay_policies(
                scenario=scenario,
                records=records,
                policy_names=args.policies,
                episodes=args.episodes,
                max_steps=args.max_steps,
                min_ipig_score=args.min_ipig_score,
                frontier_patience=args.frontier_patience,
                source_sha256=_sha256_file(args.records),
                scenario_sha256=_sha256_file(args.scenario),
            )
        except ReplayEvaluationError as exc:
            raise SystemExit("cannot benchmark replay: %s" % exc) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
        print("wrote replay benchmark to %s" % args.output)
        return 0

    if args.command == "agent-search":
        try:
            profile = CalibrationProfile.load(args.calibration)
            workload = InferenceWorkload.from_mapping(scenario.workload)
            sandbox_executor = TopologySandboxExecutor(
                profile=profile,
                workload=workload,
                expected_model=scenario.model,
                profile_path=args.calibration,
                scenario_path=args.scenario,
                backend=args.sandbox_backend,
            )
            replay_executor = ReplayExecutor.from_jsonl(args.records)
            routes = {
                level: (
                    sandbox_executor
                    if level == EvidenceLevel.L0
                    else replay_executor
                )
                for level in scenario.available_levels
            }
            result = _run_agent(
                args,
                scenario,
                RoutedExecutor(routes),
                twin=TopologyEnergyTwin(
                    profile, workload, prior_backend=args.sandbox_backend
                ),
            )
        except (CalibrationError, TopologySandboxError) as exc:
            raise SystemExit("cannot run agent search: %s" % exc) from exc
        rendered = result.to_dict()
        rendered["input_provenance"] = {
            "scenario_sha256": _sha256_file(args.scenario),
            "calibration_sha256": _sha256_file(args.calibration),
            "records_sha256": _sha256_file(args.records),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rendered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0

    if args.command == "replay":
        executor = ReplayExecutor.from_jsonl(args.records)
        report = _run_agent(args, scenario, executor)
        payload = report.to_dict()
        payload["input_provenance"] = {
            "scenario_sha256": _sha256_file(args.scenario),
            "records_sha256": _sha256_file(args.records),
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    candidate = scenario.candidate(args.candidate)
    executor = ClusterExecutor(partition=args.partition, account=args.account)
    print(executor.render_slurm(candidate, args.level, args.seed), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
