# ServeCompass Agent Experiments

## Research Contract

ServeCompass receives a natural-language deployment intent, an SLO, a
finite GPU-hour budget, and a discrete serving-configuration space. At every
step it jointly chooses a candidate and an evidence fidelity. It may recommend
a configuration only after successful L4 verification.

The bounded control path is:

```text
intent + SLO + budget
        |
semantic planner: explore / resolve_slo / calibrate_scale / repair
        |
deterministic action guard and acquisition policy
        |
CPU Sandbox (L0) / short Probe (L1) / full Verify (L4)
        |
evidence store + energy twin + Pareto update
        |
continue / stop / abstain / L4 verify
```

The LLM planner cannot choose shell commands, bypass a budget, mark simulated
evidence as measured, compute the Pareto set, or issue a final recommendation.
Invalid planner output is recorded and falls back to the rule-based planner.
The current Workshop path is single-GPU; `verify` is reserved for the
deterministic release gate and is rejected if returned by the language model.

## Implemented Experiment Modes

### Single episode

```bash
servecompass replay \
  --scenario configs/scenarios/replay_demo.json \
  --records configs/replay/demo_records.jsonl \
  --policy ipig \
  --max-steps 5 \
  --output experiments/results/replay-agent-run.json
```

Every event records the planner, subgoal, candidate, fidelity, expected and
actual cost, information score, uncertainty before and after observation,
Pareto frontier before and after observation, remaining budget, observed
metrics, and any planner fallback.

### Policy benchmark

```bash
servecompass benchmark-replay \
  --scenario configs/scenarios/replay_demo.json \
  --records configs/replay/demo_records.jsonl \
  --policies ipig,random,cost-blind,cheapest-first \
  --episodes 50 \
  --max-steps 10 \
  --output experiments/results/agent-policy-benchmark.json
```

The benchmark constructs its oracle from the median successful L4 metrics for
every candidate. It reports Pareto recall and precision, primary-objective
regret, GPU-hours to the first oracle hit, unnecessary L4 verifications,
failed actions, and total GPU-hours. For each episode and candidate--fidelity
key, a stable hash selects one empirical repeat; all policies share that mapping
as common random numbers. These are bootstrap sensitivity episodes over the
available repeats, not independent hardware runs.

### Budget-response benchmark

```bash
servecompass benchmark-budget-sweep \
  --scenario experiments/results/config-search-v1-scenario.json \
  --records experiments/results/config-search-v1-replay-corpus.jsonl \
  --protocol configs/benchmarks/qwen2.5-7b-h100-budget-sweep-v1.json \
  --output experiments/results/config-search-v1-budget-sweep.json
```

The preregistered protocol evaluates 0.04, 0.06, 0.08, 0.10, and 0.12
GPU-hour budgets with 500 matched-seed episodes per policy and budget. It
checks the scenario and corpus SHA-256 before running, retains every episode,
and reports success/recall/cost/regret at each budget plus normalized
budget-response AUC. A curve need not be monotonic: additional noisy evidence
can alter which candidate receives final verification.

### Bounded planner conformance and overhead

```bash
servecompass benchmark-planner \
  --protocol configs/benchmarks/qwen2.5-7b-planner-conformance-v2-holdout.json \
  --planner-base-url http://127.0.0.1:8000/v1 \
  --output experiments/results/planner-conformance-v2-holdout.json
```

The V1 protocol is an immutable prompt-only development diagnostic. The V2
holdout freezes 30 new natural-language cases over cold start, SLO-boundary,
failure-repair, topology calibration, steady exploration, and guard-challenge
states. Three repeats produce 90 planner calls. The report separates
schema-valid raw proposals, state-guard interventions, admitted subgoals,
fallback and endpoint errors, forbidden proposals and acceptances, P50/P95
control-plane latency, token usage, model identity, and every raw response.
Expected labels are frozen against the deterministic state-priority policy
before any V2 model call; this measures bounded conformance and overhead, not
an energy benefit caused by the language model.

### Routed Sandbox search extension

```bash
servecompass agent-search \
  --scenario configs/scenarios/topology_sandbox_demo.json \
  --calibration experiments/results/qwen2.5-7b-h100-workload-v2.json \
  --records experiments/results/qwen2.5-7b-config-search-evidence.jsonl \
  --policy ipig \
  --max-steps 12 \
  --output experiments/results/qwen2.5-7b-agent-search.json
```

This command exercises the broader topology extension retained for the MLSys
study. It is not part of the single-GPU Workshop claim. The Workshop evaluation
uses only L0 Sandbox predictions, L1 measured Probes, and L4 measured Verify
records from the sealed corpus.

## Sealed H100 Dataset

The 51 blind H100 measurements validate workload transfer and scope gating
under one fixed vLLM configuration. A separate sealed configuration corpus now
establishes evidence fidelity and replayed search efficiency. It uses this
primary H100/Qwen2.5-7B-Instruct target workload:

| Dimension | Workshop value |
|---|---|
| input/output tokens | 2048/128 |
| concurrency | 32 |
| target prompts | 256 |
| L1 probe prompts | 64 |
| max number of sequences | 8, 16, 32 |
| max batched tokens | 2048, 4096, 8192 |
| chunked prefill | off, on |
| prefix caching | off for the first controlled campaign |
| precision | BF16 |
| TP/PP | 1/1 on the single-H100 campaign |

The completed corpus contains 12 feasible configurations, one frozen L0
prediction per candidate, and three cyclically balanced repeats at both L1 and
L4, for 72 measured runs. Within one replay episode, the episode seed selects
the same replicate for a candidate--stage pair under every policy, independent
of when that policy requests it. A second workload, for example 512/128 at
concurrency eight, remains a useful optional Workshop extension. H200, B200,
TP, PP, and multi-node experiments belong to the MLSys expansion.

## Claim Boundary

Synthetic replay fixtures test software behavior only. The existing workload
holdouts support Sandbox accuracy and scope-gating claims. Agent search claims
require the new frozen configuration corpus and comparisons against all
declared baselines. CPU predictions, replayed measurements, live measurements,
and target-verified measurements must remain separately labeled in every
table and figure.
