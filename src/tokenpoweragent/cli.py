"""Command-line entry points for replay, sandbox, and cluster execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.evidence import EvidenceStore
from tokenpoweragent.executors.cluster import ClusterExecutor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.executors.sandbox import SandboxExecutionError, SandboxExecutor
from tokenpoweragent.schema import Candidate, EvidenceLevel, Scenario


def _power_limits(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("power limits must be comma-separated integers") from exc
    if not limits or any(limit <= 0 for limit in limits) or len(limits) != len(set(limits)):
        raise argparse.ArgumentTypeError("power limits must be unique positive integers")
    return limits


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
        for power_limit_w in args.power_limits:
            candidate = Candidate(
                candidate_id="gemm-pl%d" % power_limit_w,
                config={
                    "image": args.image,
                    "power_limit_w": power_limit_w,
                    "command": [
                        "--matrix-size",
                        str(args.matrix_size),
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                    ],
                },
            )
            for seed in range(args.repeats):
                seeded_candidate = Candidate(
                    candidate_id=candidate.candidate_id,
                    config={
                        **candidate.config,
                        "command": [*candidate.config["command"], "--seed", str(seed)],
                    },
                )
                try:
                    record = executor.execute(seeded_candidate, EvidenceLevel.L1, seed)
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
