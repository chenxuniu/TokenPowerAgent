"""Command-line entry points for replay episodes and Slurm rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tokenpoweragent.agent.controller import TokenPowerAgent
from tokenpoweragent.executors.cluster import ClusterExecutor
from tokenpoweragent.executors.replay import ReplayExecutor
from tokenpoweragent.schema import EvidenceLevel, Scenario


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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
