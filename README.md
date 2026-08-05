# TokenPowerAgent

TokenPowerAgent is a research framework for **agentic multi-fidelity Pareto
search for energy-efficient LLM inference on GPU clusters**. It treats tuning
as an evidence-allocation problem: which serving configuration should be
examined next, at what fidelity, and when is target-scale verification worth
its GPU-hour cost?

The current code is an executable research scaffold. Simulated and
extrapolated records are labeled as such and are never treated as completed
paper experiments.

## What Runs Today

- A versioned scenario and candidate schema.
- L0--L4 evidence records with provenance and budget accounting.
- A common executor interface with a deterministic replay backend.
- A calibrated topology projector with explicit TP/PP communication, memory,
  placement, energy, and uncertainty terms.
- A real-L1-to-calibration builder with source hashes and repeat aggregation.
- Pre-registered workload-transfer campaigns with frozen predictions,
  resumable balanced measurement, raw-artifact sealing, and ranking/error
  validation reports.
- A scope-gated workload residual fitter with leave-one-workload-out model
  selection, physically consistent energy reconstruction, and a disjoint final
  holdout protocol.
- Deterministic TP/PP/batching grid generation with geometry and memory guards.
- Fidelity routing and a topology-backed Energy Twin that can start without
  hand-written prior metrics.
- A Slurm renderer and injectable live-cluster execution boundary.
- A failure-aware plan-act-observe-reflect controller with semantic subgoals,
  frontier-stability stopping, conservative budget charging, and a pluggable
  IPIG estimator.
- A constrained OpenAI-compatible LLM planner. The model may select only a
  typed semantic subgoal; malformed output falls back to the deterministic
  planner and is recorded in the decision trace.
- Reproducible IPIG, random, cost-blind, and cheapest-first replay policies
  with Pareto recall, primary-objective regret, unnecessary escalation, and
  GPU-hour metrics. Policy benchmarks use action-specific empirical bootstrap
  sampling with common random numbers across policies; episode counts must not
  be interpreted as additional hardware measurements.
- Preregistered budget-response and bounded-planner benchmarks with frozen
  input hashes, per-call/per-episode records, control-plane latency, token
  usage, fallback, and matched-budget policy metrics.
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
  --policy ipig \
  --max-steps 5 \
  --output experiments/results/replay-agent-run.json

tokenpoweragent benchmark-replay \
  --scenario configs/scenarios/replay_demo.json \
  --records configs/replay/demo_records.jsonl \
  --policies ipig,random,cost-blind,cheapest-first \
  --episodes 20 \
  --max-steps 5 \
  --output experiments/results/replay-policy-benchmark.json
pytest
```

The bundled replay records are synthetic test fixtures. They verify control
flow and metric generation; they are not publication results.

Run the same bounded planner against an OpenAI-compatible endpoint:

```bash
export TOKENPOWERAGENT_LLM_BASE_URL=http://127.0.0.1:8001/v1
export TOKENPOWERAGENT_LLM_MODEL=qwen2.5-7b-agent

tokenpoweragent replay \
  --scenario configs/scenarios/replay_demo.json \
  --records configs/replay/demo_records.jsonl \
  --planner llm \
  --max-steps 5
```

The endpoint cannot bypass the action guard: candidate selection, fidelity
routing, budget accounting, SLO checks, Pareto computation, and L4-only final
recommendations remain deterministic.

Render a target-scale Slurm job without submitting it:

```bash
tokenpoweragent render-slurm \
  --scenario configs/scenarios/replay_demo.json \
  --candidate cfg-efficient \
  --level L4
```

After configuring Docker, NVIDIA Container Toolkit, and DCGM on a single-GPU
host, run the L1 TokenPowerSandbox smoke campaign:

```bash
sudo docker build --pull \
  -t tokenpower-sandbox:cuda12.8 \
  experiments/sandbox/cuda-gemm

tokenpoweragent sandbox-smoke \
  --power-limits 350,500,700 \
  --repeats 3 \
  --output experiments/results/h100-gemm-smoke.jsonl
```

See [`experiments/sandbox/README.md`](experiments/sandbox/README.md) for the
measurement boundary, safety controls, metrics, and pass criteria.

Build a non-publication calibration profile from repeated 700 W L1 serving
records, then run the topology sandbox:

```bash
tokenpoweragent build-calibration \
  --records experiments/results/qwen7b-serving-pl700.jsonl \
  --template configs/calibration/qwen2.5-7b-h100-profile-template.json \
  --profile-id qwen2.5-7b-h100-l1-v1 \
  --power-limit-w 700 \
  --min-repeats 3 \
  --output experiments/results/qwen2.5-7b-h100-l1-v1.json

tokenpoweragent sandbox-predict \
  --scenario configs/scenarios/topology_sandbox_demo.json \
  --calibration experiments/results/qwen2.5-7b-h100-l1-v1.json \
  --level L2 \
  --output experiments/results/qwen2.5-7b-topology-predictions.jsonl
```

The bundled topology descriptor contains placeholder link characteristics. It
cannot be marked publication eligible until those values and the uncertainty
intervals are calibrated against held-out real runs.

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
       Routed executor abstraction
       /          |             \
TopologySandbox  Replay     Serving/Cluster
       \          |             /
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

The manuscript is organized as one file per section, figure, and table under
[`paper-draft`](paper-draft/). All result macros remain visibly marked until
they are replaced by measured data.

## Research Status

1. Bounded agent runtime, replay baselines, and guard semantics: implemented.
2. L1 serving evidence to sandbox calibration: implemented.
3. Exact IPIG nested-posterior estimator: interface defined, implementation
   pending.
4. Live Slurm runner and telemetry verifier: boundary defined, cluster-specific
   integration pending.
5. Topology-aware L0/L2 projection and candidate compiler: implemented.
6. Single-H100 workload-transfer v2 and scope-confirmation v3: implemented
   with sealed final holdouts (51 blind measurements in total).
7. H100/H200/B200 multi-GPU and multi-node validation: pending measured experiments.

See [`docs/TokenPowerAgent-Complete-Workflow.md`](docs/TokenPowerAgent-Complete-Workflow.md)
for the full method,
[`docs/TokenPowerAgent-Agent-Experiments.md`](docs/TokenPowerAgent-Agent-Experiments.md)
for the runnable agent evaluation, and
[`experiments/README.md`](experiments/README.md) for the measurement protocol.
