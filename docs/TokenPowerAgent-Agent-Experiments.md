# TokenPowerAgent Agent Experiments

## Research Contract

TokenPowerAgent receives a natural-language deployment intent, an SLO, a
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
L0/L2 topology sandbox or L1/L3/L4 measured/replayed executor
        |
evidence store + energy twin + Pareto update
        |
continue / stop / abstain / L4 verify
```

The LLM planner cannot choose shell commands, bypass a budget, mark simulated
evidence as measured, compute the Pareto set, or issue a final recommendation.
Invalid planner output is recorded and falls back to the rule-based planner.

## Implemented Experiment Modes

### Single episode

```bash
tokenpoweragent replay \
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
tokenpoweragent benchmark-replay \
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
failed actions, and total GPU-hours.

### Routed Sandbox search

```bash
tokenpoweragent agent-search \
  --scenario configs/scenarios/topology_sandbox_demo.json \
  --calibration experiments/results/qwen2.5-7b-h100-workload-v2.json \
  --records experiments/results/qwen2.5-7b-config-search-evidence.jsonl \
  --policy ipig \
  --max-steps 12 \
  --output experiments/results/qwen2.5-7b-agent-search.json
```

L0 and L2 actions are routed to the CPU topology sandbox. Available L1, L3,
and L4 actions are routed to the sealed evidence corpus. Replacing replay with
a live executor does not change the controller contract.

## Next H100 Dataset

The existing 51 blind H100 measurements validate workload transfer and scope
gating under one fixed vLLM configuration. They establish the Sandbox as an
evidence source but do not yet establish configuration-search efficiency.

The next dataset must vary serving configurations while holding the workload
fixed. Use one primary H100/Qwen2.5-7B-Instruct target workload for the
Workshop corpus:

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

Apply memory and geometry guards before measurement. Select 12 feasible,
balanced configurations rather than measuring the full factorial. Collect a
short L1 probe and a full L4 target run for every candidate, with three repeats
per level in cyclically randomized candidate order. Freeze all L0 predictions,
the candidate order, and the replay budget before the first GPU measurement.
Within one replay episode, the episode seed selects the same replicate for a
given candidate--fidelity pair under every policy, independent of when that
policy requests the pair.

The publication benchmark then treats the completed corpus as an oracle and
runs at least 50 matched-seed replay episodes per policy. A smaller live run
should independently confirm the configurations selected by the agent. A
second workload (for example 512/128 at concurrency 8) is useful only after the
primary corpus is sealed. H200, B200, TP, PP, and multi-node
experiments belong to the MLSys expansion after this single-H100 protocol is
stable.

## Claim Boundary

Synthetic replay fixtures test software behavior only. The existing workload
holdouts support Sandbox accuracy and scope-gating claims. Agent search claims
require the new frozen configuration corpus and comparisons against all
declared baselines. CPU predictions, replayed measurements, live measurements,
and target-verified measurements must remain separately labeled in every
table and figure.
