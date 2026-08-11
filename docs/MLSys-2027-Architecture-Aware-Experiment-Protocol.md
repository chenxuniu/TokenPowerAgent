# ServeCompass MLSys 2027 Architecture-Aware Experiment Protocol

**Status:** design draft; do not use as a frozen measurement manifest  
**Purpose:** upgrade the validated single-H100 prototype into a cross-model,
cross-hardware, and topology-aware MLSys evaluation.

Hardware, privilege, storage, compute-budget, and calendar details are defined
in the companion
[MLSys 2027 complete execution plan](MLSys-2027-Complete-Execution-Plan.md).

## 1. Research Claim

The MLSys paper should not depend on a large energy reduction in one serving
scenario. Its primary claim is:

> ServeCompass jointly selects a serving configuration and the fidelity of
> the evidence used to evaluate it, recovering a verified SLO-constrained
> energy-performance Pareto frontier with fewer real GPU-hours across model
> architectures, GPU generations, and parallel topologies.

Energy savings over a default or expert configuration are a downstream result.
The main system result is frontier quality per real GPU-hour.

## 2. Model Panel

The panel is organized by architecture rather than model popularity. Every
measurement campaign must pin the model, tokenizer, and inference-engine
revisions before predictions are generated.

| Tier | Model | Architecture role | Evaluation scope |
|---|---|---|---|
| Existing anchor | Qwen2.5-7B-Instruct | Small dense calibration case | Preserve the frozen single-H100 evidence; do not rerun the full matrix |
| Core | Qwen3-32B | Dense decoder | Full H100/H200/B200 matrix where memory permits |
| Core | Qwen3-30B-A3B-Instruct-2507 | Small-active MoE | Full H100/H200/B200 matrix and expert-parallelism ablation |
| Core | Kimi-Linear-48B-A3B-Instruct | Hybrid/linear-attention MoE | Four- and eight-GPU topology plus long-context workloads |
| Scale-out | Kimi-K2.5 | Frontier-scale MoE | Selected H200/B200 deployment study, not a full factorial sweep |
| Reference | Llama 70B instruct checkpoint | Widely used dense serving reference | Selected matched scenarios for comparison with prior work |

Official deployment references:

- Qwen3: <https://github.com/QwenLM/Qwen3>
- Kimi-Linear vLLM recipe:
  <https://github.com/vllm-project/recipes/blob/main/moonshotai/Kimi-Linear.md>
- Kimi-K2.5 vLLM recipe:
  <https://github.com/vllm-project/recipes/blob/main/moonshotai/Kimi-K2.5.md>

Kimi-K2.5 is admitted only after the resource inventory confirms a supported
deployment. A Blackwell NVFP4 run and an H200 FP8 run are deployment-option
comparisons, not pure hardware comparisons, because precision also changes.

## 3. Experimental Unit

One scenario is the immutable tuple

```text
(model revision, engine image, precision, GPU type, node topology,
 workload distribution, SLO, search budget)
```

The agent chooses an action

```text
(serving configuration, evidence fidelity)
```

from the scenario's valid action space. Model identity, hardware identity, and
topology are scenario context; they are not silently changed within an episode.

## 4. Search Space

The first full campaign should expose the following dimensions:

| Family | Knobs |
|---|---|
| Scheduler and batching | `max_num_seqs`, `max_num_batched_tokens`, chunked prefill, prefix caching |
| Memory | KV-cache dtype, GPU-memory utilization, maximum model length |
| Parallel topology | TP, PP, DP/replicas, and EP for supported MoE engines |
| Hardware control | GPU power cap; application clocks only in a separately declared ablation |
| Workload | prompt/output-length distribution, arrival process, concurrency, request rate |

Quantization is held fixed in the primary cross-hardware comparison. A separate
deployment-mode study may compare BF16, FP8, and NVFP4, but it must report model
quality checks and must not attribute the entire difference to GPU architecture.

Invalid actions are rejected before execution using memory, divisibility,
engine-support, and topology constraints. The LLM planner never bypasses these
guards.

## 5. L0-L4 Evidence Ladder

Use both the numeric level and semantic name consistently in the paper, schema,
figures, and logs.

| Fidelity | Semantic name | Resource | Evidence | Promotion rule |
|---|---|---|---|---|
| L0 | Estimate | CPU | Analytical/topology prediction with calibrated uncertainty | Candidate is plausibly feasible and informative |
| L1 | Phase Probe | One target GPU | Warmed short serving or phase run with real telemetry | Observed SLO and energy interval justify topology measurement |
| L2 | Node Probe | Two to eight GPUs in one target node | Reduced but valid TP/PP/EP serving run | Intra-node residual and frontier uncertainty justify cluster measurement |
| L3 | Cluster Probe | Two or more target nodes | Sparse real multi-node serving run | Cross-node residual remains competitive and target verification is affordable |
| L4 | Target Verify | Exact target deployment | Complete workload, exact topology, independent repeats | Required for every final deployment claim |

L0 prediction must never be presented as measurement. L1-L3 evidence must never
be presented as target-scale verification unless it exactly matches the frozen
L4 deployment and workload. Communication, MoE routing, placement, and idle-node
energy require their own residuals and uncertainty terms.

## 6. Acquisition Policy

At step `t`, the policy selects configuration `x` and fidelity `f` using

```text
score(x, f) = expected Pareto information gain(x, f)
              / expected real GPU-hours(x, f)
```

subject to:

1. SLO-feasibility probability above the campaign threshold;
2. a reserved budget for independent full-deployment verification;
3. diversity across model, hardware, and topology uncertainty regions;
4. deterministic safety and support guards;
5. abstention when the calibrated scope gate is not satisfied.

The LLM planner translates operator intent, diagnoses failures, and proposes a
bounded subgoal. Numerical acquisition values, Pareto computation, resource
checks, and deployment promotion remain deterministic.

## 7. Workload Strata

Each core model should cover at least three preregistered strata:

| Stratum | Representative purpose |
|---|---|
| Interactive | Low request rate, concurrency 1-8, strict P95 TTFT |
| Throughput | Saturating request rate, concurrency 32-128, goodput objective |
| Long context | 8K-32K prompts, controlled output length, KV-memory pressure |
| Bursty holdout | Poisson or trace-derived arrivals unseen during calibration |

Exact token lengths and arrival parameters are frozen only after a pilot confirms
that every baseline can complete the scenario. The bursty stratum is reserved as
a holdout and is not used to fit transfer residuals.

## 8. Evaluation Matrix

The core factorial matrix is:

```text
3 core architectures x 3 GPU generations x 3 workload strata
```

Cells that are physically invalid are recorded as unsupported rather than
replaced with a different precision or model. TP/PP/EP topology is varied inside
each valid cell. Kimi-K2.5 and Llama 70B are focused scale-out/reference studies,
not additional full factorial axes.

Each scenario reports:

- exhaustive-oracle results when the bounded space is small enough;
- otherwise, a high-budget pooled oracle constructed before policy replay;
- at least three independent search seeds;
- at least three independent Verify repeats for promoted configurations;
- aggregate results and per-scenario distributions, not only the best case.

## 9. Baselines and Ablations

Required baselines:

1. inference-engine default;
2. documented human/expert configuration;
3. random search with the same GPU-hour budget;
4. cost-aware multi-objective Bayesian optimization;
5. exhaustive or pooled oracle;
6. cheapest-first and cost-blind acquisition controls.

Required ablations:

1. no Estimate evidence;
2. no cross-hardware transfer;
3. no adaptive fidelity selection;
4. no LLM planner, using the deterministic acquisition policy alone;
5. no scope guard;
6. no verification reserve.

The agentic contribution is established only if the guarded LLM planner improves
success, recovery, or search cost beyond the deterministic optimizer. Planner
latency alone is not evidence of utility.

## 10. Primary Metrics

### Frontier quality

- normalized dominated hypervolume;
- oracle Pareto recall and precision;
- best feasible energy regret at matched SLO;
- rank correlation between lower-fidelity and Verify evidence.

### Search cost

- real GPU-hours to first verified Pareto point;
- real GPU-hours to 90% and 95% oracle hypervolume;
- number of unnecessary Verify actions;
- CPU simulation time and planner token/latency overhead reported separately.

### Deployment outcome

- GPU and, where available, node energy per 1K output tokens;
- P95/P99 TTFT and TPOT;
- output-token throughput and SLO-attaining goodput;
- average and peak power;
- failure, OOM, timeout, and abstention rates.

## 11. Measurement Boundary

For one GPU, DCGM energy is measured over the warmed benchmark window. For
multiple GPUs, the primary GPU-energy value is the sum of synchronized per-GPU
energy-counter deltas. Report skew between first and last marker.

Node-level power is a separate metric when rack/PDU/BMC telemetry is available.
Do not add CPU or network energy estimates to measured GPU energy. Report both
boundaries explicitly when node telemetry is available.

## 12. Decision Gates

The full campaign starts only after all gates pass:

- exact model and tokenizer revisions are cached;
- immutable engine and client images are pinned;
- GPU count, memory, interconnect, and node network are inventoried;
- one warm and one measured request complete at every intended topology;
- energy counters are monotonic on every GPU;
- marker skew and telemetry sample loss are within frozen thresholds;
- baseline configurations satisfy the scenario's feasibility requirements;
- predictions and schedules are hashed before holdout measurements begin.

## 13. Role of the Existing Single-H100 Evidence

The current Qwen2.5-7B/H100 study remains useful as:

- a calibration and artifact-integrity case;
- evidence that frozen workload predictions can rank energy outcomes;
- a scope-gating and abstention example;
- a unit test for the serving executor and planner guard.

It is not the MLSys headline, and its 1.39% energy reduction is not generalized
to other models, workloads, GPUs, or topologies.

## 14. Immediate Execution Order

1. Collect one inventory artifact per node type.
2. Decide valid TP/PP/EP ranges from the observed topology and memory.
3. Smoke-test Qwen3-32B on one H100/H200/B200 node.
4. Add Qwen3-30B-A3B and validate MoE/EP telemetry.
5. Validate Kimi-Linear on four GPUs before enabling eight-GPU campaigns.
6. Freeze the core model-hardware-workload matrix.
7. Run the H100 development split, then blind H200/B200 transfer campaigns.
8. Run Kimi-K2.5 only after the core methodology and telemetry are stable.
