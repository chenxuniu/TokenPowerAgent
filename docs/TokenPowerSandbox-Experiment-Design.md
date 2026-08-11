# TokenPowerSandbox Experiment Design

## Core claim

TokenPowerSandbox tests whether a calibrated, uncertainty-aware,
multi-fidelity model can rank the energy-performance trade-offs of LLM serving
configurations across context length, batching, TP, PP, and cluster topology,
so that ServeCompass sends only promising or uncertain candidates to
expensive multi-GPU validation.

It does **not** claim that one H100 reproduces a multi-node run. A single-GPU
probe anchors compute, measured collective characteristics supply topology
terms, and held-out multi-GPU serving runs determine whether the resulting
ranking is useful.

## Factor contract

Keep these roles separate in every manifest and plot.

| Role | Variables |
|---|---|
| Scenario/workload | model, input tokens, output tokens, concurrency, arrival rate, SLO |
| Agent-controlled configuration | TP, PP, DP, `max_num_seqs`, `max_num_batched_tokens`, chunked prefill, KV-cache dtype |
| Experimental controls | engine/container revision, model revision, GPU clocks/power limit, prefix caching, seed, warmup, request trace |
| Outcomes | output throughput, p95 TTFT, p95 TPOT, energy/output token, total energy, feasibility |

Power cap remains fixed at the platform default in the main study. The existing
490 W/700 W campaign is a measurement-boundary diagnostic and may become a
secondary appendix experiment; it is not the paper's independent variable.

## Fidelity semantics

| Label | Acquired evidence | Output label | Permitted use |
|---|---|---|---|
| L0-A | calibrated analytical CPU projection without communication terms | `simulated`, `level=L0` | cheapest feasibility and configuration screening |
| L0-T | topology-aware CPU projection with analytical communication terms | `extrapolated`, `level=L0` | multi-GPU what-if ranking; never a measurement |
| L1 | real single-GPU vLLM + DCGM serving run | `measured` | phase/workload calibration |
| L2 | real intra-node serving probe on 2--8 GPUs | `measured` | TP/PP, collective, and memory calibration |
| L3 | sparse real multi-node serving run | `measured` | fabric, placement, and scale correction |
| L4 | independent rerun on the exact target deployment | `verified` | reported recommendation and headline savings |

`sandbox-predict --backend l0-t` enables topology-aware CPU extrapolation, but
the resulting evidence remains L0. Evidence levels describe how a result was
acquired, not how sophisticated its predictor is. L1--L4 therefore require
real hardware execution.

## RQ1: Does the sandbox preserve real rankings?

### Calibration set

On each GPU type, collect L1 points with prefix caching disabled and the power
limit fixed at its default:

- Input tokens: 128, 512, 2,048, 8,192.
- Output tokens: 32, 128, 512.
- Concurrency: 1, 8, 32, 128, subject to memory feasibility.
- `max_num_seqs`: 8, 32, 128, 256.
- `max_num_batched_tokens`: 2,048, 8,192, 32,768.
- Chunked prefill: on/off where both modes are valid.

Do not take the full Cartesian product. Use a space-filling design of 36 points
per model/GPU pair, repeated three times in a cyclically balanced order. Freeze
24 points as calibration and 12 as a held-out single-GPU test before fitting.

### Multi-GPU blind test

Choose 24 additional feasible points spanning TP 1/2/4/8, PP 1/2, short/long
contexts, and low/high concurrency. The sandbox may not see their serving
metrics before prediction files are hashed. Run each point three times in a
randomized block, then compare predictions with medians of the real runs.

Report:

- MAPE and median absolute percentage error for energy, throughput, TTFT, TPOT.
- Spearman and Kendall rank correlation.
- Top-k recall and pairwise ordering accuracy.
- Predicted interval coverage and interval width.
- Pareto precision/recall and hypervolume error.

Primary gate: Spearman >= 0.80 for energy and throughput, Pareto recall >= 0.80,
and empirical interval coverage within 10 percentage points of its declared
target. If this fails, narrow the claim to the hardware/topology regimes that
pass and expose the failed regimes.

## RQ2: Which fidelity should the agent buy next?

Construct a replay oracle from the held-out real measurements. Start each
episode with the same calibration profile and reveal an observation only when a
policy buys it. Compare:

- Random search.
- Uniform grid order.
- Hand-tuned heuristic (largest batch that fits, then largest TP).
- Single-fidelity Bayesian optimization.
- Multi-objective qNEHVI.
- ServeCompass with information-per-GPU-hour selection.
- Ablations without uncertainty, topology terms, L1, L2, and the L4 gate.

Run at least 20 deterministic policy seeds. Plot Pareto recall, hypervolume
regret, and recommendation regret against cumulative GPU-hours. Calibration
cost is charged once per episode; sandbox queries cost zero incremental
GPU-hours.

Primary gate: at 20% of exhaustive-search GPU-hours, ServeCompass should
recover at least 80% of the true Pareto set and have no more than 5% median
energy regret at matched SLO. Report confidence intervals over seeds.

## RQ3: Does calibration transfer?

Use leave-one-domain-out tests rather than mixing all measurements:

- Context transfer: fit on <=2K tokens, test on 8K/16K.
- Scale transfer: fit on TP <=2, test on TP 4/8 and PP 2.
- Topology transfer: fit intra-node, test cross-node.
- GPU transfer: fit H100, test H200 or B200 with and without three adaptation
  points.
- Model transfer: fit Qwen2.5-7B, test a larger Qwen model with and without
  architecture-aware adaptation.

The important result is not only mean error. Show whether uncertainty grows in
the held-out domain and whether the agent reacts by requesting higher-fidelity
evidence.

## RQ4: Does the final recommendation save energy under an SLO?

For each target scenario, independently rerun the selected configuration and
the following baselines at L4:

- vLLM default.
- Expert/naive fixed configuration.
- Best configuration found by equal-budget random search.
- Best qNEHVI recommendation at the same total GPU-hour budget.
- Exhaustive oracle when affordable; otherwise a clearly labeled dense-sweep
  upper bound.

Headline quantities are energy saved at matched SLO, performance loss relative
to the fastest feasible configuration, search GPU-hours, and end-to-end energy
including search. A recommendation is publication eligible only after three
independent L4 repetitions with zero failed requests.

## Execution order

1. Convert the current Qwen2.5-7B/H100 L1 runs into a real calibration profile;
   do not reuse the synthetic example.
2. Complete the 36-point single-GPU design on H100 and freeze the 24/12 split.
3. Measure NCCL all-reduce and point-to-point bandwidth/latency for every real
   topology and replace the example topology numbers.
4. Produce and hash paired L0-A/L0-T predictions for the blind multi-GPU points.
5. Run the multi-GPU points, score RQ1, and inspect failure regimes.
6. Only after RQ1 passes, run the 20-seed replay comparison for RQ2.
7. Add H200/B200 adaptation points and perform RQ3 transfer tests.
8. Verify all reported recommendations independently at L4 for RQ4.

Every raw run must retain the Git commit, scenario/calibration hashes,
container digest, model/tokenizer revision, driver/CUDA/NCCL versions, GPU UUIDs,
topology, power-limit readback, randomization block, telemetry, and exact command.
