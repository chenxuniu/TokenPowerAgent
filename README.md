# TokenPowerAgent

TokenPowerAgent is a research framework for **agentic multi-fidelity Pareto
search for energy-efficient LLM inference on GPU clusters**. It treats tuning
as an evidence-allocation problem: which serving configuration should be
examined next, at what fidelity, and when is target-scale verification worth
its GPU-hour cost?

The repository accompanies an IEEE-format submission to AgenticAI4HPC'26 at
SC26. The current code is an executable research scaffold, not a claim that all
paper experiments are complete.

## What Runs Today

- A versioned scenario and candidate schema.
- L0--L4 evidence records with provenance and budget accounting.
- A common executor interface with a deterministic replay backend.
- A Slurm renderer and injectable live-cluster execution boundary.
- A hybrid controller with semantic subgoals and a pluggable IPIG estimator.
- Pareto filtering, SLO checks, final L4-only recommendations, and tests.

The bundled IPIG estimator is a transparent posterior-uncertainty proxy. The
paper's full nested-posterior mutual-information estimator and the live
TokenPowerBench adapter are explicit next implementation milestones.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
tokenpoweragent replay \
  --scenario configs/scenarios/replay_demo.json \
  --records configs/replay/demo_records.jsonl \
  --max-steps 5
pytest
```

Render a target-scale Slurm job without submitting it:

```bash
tokenpoweragent render-slurm \
  --scenario configs/scenarios/replay_demo.json \
  --candidate cfg-efficient \
  --level L4
```

## Architecture

```text
Natural-language intent / Scenario
                 |
        Semantic planner
                 |
      Deterministic guard
                 |
   IPIG candidate-level policy
                 |
       Executor abstraction
       /                  \
ReplayExecutor       ClusterExecutor
       \                  /
       Evidence store + Energy Twin
                 |
       L4-verified Pareto set
```

The language-model planner is intentionally bounded: it may choose a semantic
subgoal and explain a decision, while typed code validates candidates, scores
actions, tracks budget, computes Pareto sets, and controls execution.

## Repository Layout

```text
src/tokenpoweragent/       Python package
configs/                   Example scenarios and replay evidence
tests/                     Unit and end-to-end replay tests
experiments/               Manifest and result layout for real campaigns
paper-draft/               IEEE conference manuscript source
docs/                      Detailed method and execution workflow
```

## Paper

The manuscript uses `IEEEtran` in conference mode and is organized as one file
per section, figure, and table under [`paper-draft`](paper-draft/). It currently
compiles to the workshop limit of 10 pages including references. All result
macros remain visibly marked until they are replaced by measured data.

AgenticAI4HPC'26 lists a submission deadline of **August 7, 2026 (AoE)** and
uses single-blind review. Recheck the official call immediately before
submission.

## Research Status

1. Replay evaluation and guard semantics: implemented.
2. TokenPowerBench import adapter: interface defined, integration pending.
3. Exact IPIG nested-posterior estimator: interface defined, implementation
   pending.
4. Live Slurm runner and telemetry verifier: boundary defined, cluster-specific
   integration pending.
5. H100/H200 multi-GPU and multi-node campaign: pending measured experiments.

See [`docs/TokenPowerAgent-Complete-Workflow.md`](docs/TokenPowerAgent-Complete-Workflow.md)
for the full method and [`experiments/README.md`](experiments/README.md) for the
measurement protocol.
