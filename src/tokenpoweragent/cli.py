"""Command-line entry points for replay, sandbox, and cluster execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.calibration import (
    CalibrationBuildError,
    build_serving_calibration_profile,
)
from tokenpoweragent.evidence import EvidenceStore
from tokenpoweragent.executors.cluster import ClusterExecutor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.executors.sandbox import SandboxExecutionError, SandboxExecutor
from tokenpoweragent.executors.serving import ServingSandboxExecutor
from tokenpoweragent.executors.topology import (
    TopologySandboxError,
    TopologySandboxExecutor,
)
from tokenpoweragent.pareto import pareto_front
from tokenpoweragent.search_space import CandidateGrid
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
)
from tokenpoweragent.validation import (
    HoldoutValidationError,
    build_holdout_validation_report,
)


def _power_limits(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("power limits must be comma-separated integers") from exc
    if not limits or any(limit <= 0 for limit in limits) or len(limits) != len(set(limits)):
        raise argparse.ArgumentTypeError("power limits must be unique positive integers")
    return limits


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenpoweragent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="run a deterministic replay episode")
    replay.add_argument("--scenario", type=Path, required=True)
    replay.add_argument("--records", type=Path, required=True)
    replay.add_argument("--max-steps", type=int, default=20)
    replay.add_argument("--output", type=Path)

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
    predict.add_argument(
        "--level", type=EvidenceLevel.parse, default=EvidenceLevel.L2
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

    validation = subparsers.add_parser(
        "validate-holdout",
        help="compare a frozen sandbox prediction with blind serving evidence",
    )
    validation.add_argument("--predictions", type=Path, required=True)
    validation.add_argument("--measurements", type=Path, required=True)
    validation.add_argument("--freeze-manifest", type=Path, required=True)
    validation.add_argument("--min-repeats", type=int, default=3)
    validation.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
        if args.level not in {EvidenceLevel.L0, EvidenceLevel.L2}:
            raise SystemExit("--level must be L0 or L2")
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
                record = executor.execute(candidate, args.level, args.seed)
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
                    "level": args.level.name,
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

    scenario = Scenario.load(args.scenario)
    if args.command == "replay":
        executor = ReplayExecutor.from_jsonl(args.records)
        report = TokenPowerAgent(scenario, executor).run(max_steps=args.max_steps)
        rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    candidate = scenario.candidate(args.candidate)
    executor = ClusterExecutor(partition=args.partition, account=args.account)
    print(executor.render_slurm(candidate, args.level, args.seed), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
