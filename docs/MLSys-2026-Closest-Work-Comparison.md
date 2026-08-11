# ServeCompass versus the Closest MLSys 2026 Systems

**Verified:** 2026-07-29  
**Sources:** Published MLSys 2026 papers or author full-text versions  
**Comparison target:** The current ServeCompass manuscript and repository, not an assumed future implementation

## 1. Short Answer

The six papers do not make ServeCompass redundant, but they remove several broad novelty claims.

The cleanest distinction is:

> Existing systems optimize **which serving configuration to deploy** or **how to control it at runtime**. ServeCompass optimizes **which evidence to purchase next**, jointly choosing a serving configuration and a measurement fidelity under a real GPU-hour budget, and requires target-scale measured verification before releasing a Pareto recommendation.

That distinction is meaningful only if the final system implements and evaluates:

1. a real configuration-and-fidelity acquisition algorithm;
2. calibrated uncertainty across L0--L4;
3. measured energy as a first-class objective;
4. H100/H200/B200 transfer;
5. an actual LLM planner whose contribution is isolated from the numerical optimizer;
6. live target-scale verification.

At present, these are partly manuscript designs and partly executable interfaces. They are not yet measured MLSys-level results.

## 2. Method-Level Comparison

| Work | Primary problem | Decisions | Core method | Objective and constraints | Evidence | What it does not address |
|---|---|---|---|---|---|---|
| [BEAM](https://proceedings.mlsys.org/paper_files/paper/2026/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html) | Fine-grained runtime energy control | GPU frequency, prefill chunk size, microbatch count | Event-driven, phase-aware controller using an offline performance-energy lookup table | Minimize measured GPU energy under per-request TTFT/TBT SLOs | Real vLLM runs on 8x and 4x A100 nodes; NVML energy | No LLM agent, active profiling policy, cross-hardware transfer, or configuration-fidelity co-selection |
| [BOute](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ed1d3d4c64dc1b95332a8cde3f2a0bdf-Abstract-Conference.html) | Joint heterogeneous model routing and deployment | Routing thresholds, GPU allocation, DP/TP/PP | Offline simulator/performance database plus constrained qNEHVI | Latency-quality Pareto frontier under cost, quality, latency, and availability constraints | H100, RTX PRO 6000, RTX 5090, RTX 4090; simulator validated against real runs | No energy objective, LLM agent, adaptive fidelity, or profiling-cost objective |
| [PROMPTS](https://proceedings.mlsys.org/paper_files/paper/2026/hash/48253da5351effdfea994fd7bbff7005-Abstract-Conference.html) | Expert-like distributed configuration recommendation | TPU ICI-mesh sharding | Coordinator, Analyzer, and Proposal agents; profiler tools and RAG | Improve step time/throughput | Eight production cases, 2 to 2,048 TPU chips; one agent invocation | No energy, SLO Pareto search, sequential acquisition, or implemented verifier loop |
| [Meta deployment study](https://proceedings.mlsys.org/paper_files/paper/2026/hash/97dc07f1253ab33ee514f395a82fa7cc-Abstract-Conference.html) | Large-scale deployment design-space exploration | Hardware, TP/PP/DP/EP/CP, runtime, batching, KV policy | 100K+ operator measurements per hardware, interpolation, pruning, systematic simulation | Maximize throughput under TTFT/TTIT SLOs; TCO and power-cap support | H100/H200/MI300X-like production platforms; up to 256 accelerators; typically +/-5% simulator error | No agent, active measurement allocation, calibrated acquisition uncertainty, or measured energy frontier |
| [Charon](https://arxiv.org/abs/2605.17164) | Unified training/inference simulation and design search | GPU count, parallelism, batch and serving configurations | Compiler-style graph simulator with profiling, prediction, and analytical backends; rule-based pruning | Performance/cost search and throughput-latency frontier | A100/H800/H20/L20-like systems and large training clusters; overall error below 5.35% | No LLM agent, measured energy objective, or information-per-GPU-hour fidelity acquisition |
| [OptiKIT](https://arxiv.org/abs/2601.20408) | Production automation for quantization, benchmarking, and serving tuning | Quantization, TP, `max_num_seqs`, `max_num_batched_tokens` | Ray workflow, exponential load search, Optuna/TPE over 30 full trials | Maximize SLO-compliant throughput per GPU; quality is a preceding gate | Real H100 experiments on Qwen-7B, Mistral-24B, and Llama-70B | No LLM agent, energy objective, Pareto output, cross-hardware transfer, or adaptive scale fidelity |
| **ServeCompass draft** | Measurement-efficient energy/SLO Pareto identification | Serving configuration **and evidence level** L0--L4 | Bounded semantic planner plus deterministic IPIG acquisition and Energy Twin | Recover verified energy-latency-throughput Pareto set while minimizing real GPU-hours | Planned replay, phase probes, intra-node, sparse multi-node, and target-trace runs on H100/H200/B200 | Exact IPIG, calibrated Twin, LLM planner, live executor, and hardware results remain incomplete |

## 3. Capability Matrix

Legend:

- **Yes:** experimentally demonstrated in the published work.
- **Partial:** indirect, fixed-stage, simulator-only, or not a primary contribution.
- **No:** absent or outside the paper's scope.
- **Design:** present in the ServeCompass manuscript design, but MLSys-level evidence is pending.

| Work | LLM-agent reasoning | Measured energy objective | SLO constraint | Pareto output | Active configuration + fidelity selection | Real-GPU measurement-cost objective | Target-hardware evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEAM | No | Yes | Yes | Partial | No | No | Yes |
| BOute | No | No | Yes | Yes | No | No | Partial |
| PROMPTS | Yes | No | No | No | No | Partial | Partial |
| Meta deployment study | No | Partial | Yes | Partial | No | No | Partial |
| Charon | No | No | Yes | Yes | No | Partial | Partial |
| OptiKIT | No | No | Yes | No | Partial | Partial | Yes |
| **ServeCompass draft** | **Design** | **Design** | **Design** | **Design** | **Design** | **Design** | **Design** |

No published row has the ServeCompass combination. That is evidence of a potential gap, not proof that the draft has already filled it.

## 4. Pairwise Analysis

### 4.1 ServeCompass versus BEAM

**Strong overlap**

- Both treat energy and TTFT/TBT as first-class.
- Both include phase-sensitive serving knobs.
- Both can vary power/frequency policy, chunking, and batching behavior.
- Both can show energy-latency Pareto behavior.

**Actual distinction**

| BEAM | ServeCompass |
|---|---|
| Millisecond-scale online controller | Minutes-to-hours experiment campaign and deployment tuner |
| Adjusts a small set of fine-grained knobs after offline profiling | Searches broader deployment/runtime configurations and decides what new evidence to collect |
| Finds the energy-minimizing action under the current request state | Recovers a set of SLO-feasible Pareto configurations under a measurement budget |
| Offline lookup table is assumed available | Profiling cost and fidelity selection are part of the optimization problem |
| One hardware family in evaluation, A100 | Planned transfer and calibration across H100, H200, and B200 |
| Every decision occurs on the live system | Cheap evidence is allowed, but every released recommendation must pass target-scale verification |

**Threat level:** Very high for the energy/SLO motivation, low for the multi-fidelity acquisition algorithm.

**Required paper response**

Define the time-scale boundary explicitly:

> BEAM controls a deployed service at request-event timescales; ServeCompass plans the measurement campaign that selects and verifies a deployment configuration.

BEAM can also be treated as a candidate runtime policy inside the ServeCompass search space. The final comparison should include BEAM or a faithful BEAM-style controller for energy-at-matched-SLO.

### 4.2 ServeCompass versus BOute

**Strong overlap**

- Both formulate LLM serving as constrained multi-objective optimization.
- Both use Pareto/hypervolume concepts.
- Both search deployment and parallelism configurations on heterogeneous GPUs.
- BOute already uses constrained qNEHVI.

**Actual distinction**

| BOute | ServeCompass |
|---|---|
| Objectives are latency and response quality; cost is primarily a resource/budget constraint | Objectives are measured energy, latency, and throughput under SLOs |
| Optimizer chooses routing and deployment configuration | Optimizer chooses configuration and evidence fidelity |
| Offline simulation database is prepared before MOBO | Historical, simulated, short real, reduced-scale, and target-scale evidence are acquired sequentially |
| Every MOBO evaluation queries the same offline database | Each action has a different real GPU-hour cost and uncertainty relationship to L4 truth |
| No LLM agent | Bounded LLM planner selects semantic subgoals, while deterministic code performs numerical acquisition |
| Simulator accuracy is validated, but no mandatory per-recommendation live gate | A recommendation is not released without target-scale evidence |

**Threat level:** Highest for the optimization algorithm.

**Required paper response**

Constrained qNEHVI must be a baseline, not part of the novelty claim. If ServeCompass is only “BOute with energy replacing quality,” the contribution is incremental. The result must show that adaptive fidelity selection reaches a comparable or better measured frontier with fewer real GPU-hours than qNEHVI operating at full fidelity.

### 4.3 ServeCompass versus PROMPTS

**Strong overlap**

- Both use tool-grounded language-model reasoning.
- Both separate language reasoning from system data/tools.
- Both reason about parallelism, hardware, and communication bottlenecks.
- Both aim to reduce expensive experimental trials.

**Actual distinction**

| PROMPTS | ServeCompass |
|---|---|
| One-shot top-three sharding proposal | Sequential plan-act-observe-update loop |
| Analyzer and Proposal agents directly synthesize configurations | LLM chooses typed semantic subgoals; deterministic IPIG selects the numerical action |
| Optimizes ICI-mesh sharding and performance | Optimizes energy-latency-throughput over serving/runtime configurations |
| Compares against expert-validated configurations | Compares frontier recovery against a dense measured oracle |
| Search effort is number of evaluated configurations | Search cost is cumulative real GPU-hours |
| Average proposal compilability is 69%; evaluator-verifier loop is future work | Deterministic guard and mandatory verification are core design requirements |
| Eight production TPU cases, 2 to 2,048 chips | Planned NVIDIA GPU serving campaign across H100/H200/B200 |

**Threat level:** Highest for the agentic contribution.

**Required paper response**

The current repository defaults to `RuleBasedPlanner`; an actual LLM planner is not yet implemented. To claim an agent contribution, the final evaluation must compare:

- full hybrid agent;
- optimizer-only state machine;
- LLM-only tool user;
- PROMPTS-style one-shot proposal;
- rule-based planner.

The paper must report agent tokens, latency, invalid actions, recovery, and whether semantic planning changes GPU-hour efficiency.

### 4.4 ServeCompass versus Meta's Deployment Study

**Strong overlap**

- Both span hardware, workload, parallelism, runtime, batching, and KV policy.
- Both use measured profiles to simulate unmeasured configurations.
- Both enforce latency SLOs and identify efficient deployment choices.
- Both explicitly model multi-node communication and phase-specific behavior.

**Actual distinction**

| Meta deployment study | ServeCompass |
|---|---|
| Broad, benchmark-driven performance simulator | Evidence-management and search layer that may use such a simulator |
| Systematically searches a pruned space of millions of configurations | Actively chooses a small number of evidence acquisitions |
| 100K+ operator measurements per hardware already populate the database | Initial evidence may be sparse; new measurements are selected under budget |
| Primarily maximizes throughput under TTFT/TTIT SLOs | Identifies measured energy-latency-throughput Pareto configurations |
| Power-capped hardware and TCO are supported indirectly | Energy telemetry and joules are direct objectives |
| Main case-study results are simulator outputs | Final recommendation must be measured on target hardware |
| Typical prediction error is around +/-5%, mainly interpolation | Uncertainty and topology distance determine whether escalation is required |

**Threat level:** Highest for search-space breadth and simulator credibility.

**Required paper response**

Do not try to beat Meta on simulator breadth. Position Meta/Charon/Vidur as possible L0 providers. ServeCompass's contribution is deciding when the simulator is sufficient, when it needs calibration, and when real execution is unavoidable.

### 4.5 ServeCompass versus Charon

**Strong overlap**

- Both support inference design-space exploration.
- Both model computation, communication, overlap, parallelism, and runtime effects.
- Both produce throughput-latency frontiers.
- Both are motivated by avoiding large profiling cost.

**Critical terminology correction**

Charon is a **hybrid multi-engine simulator**, not an active multi-fidelity optimizer in the ServeCompass sense.

Charon can execute an operator through:

- a profiling backend;
- a random-forest prediction backend;
- an analytical backend;
- a prioritized fused fallback.

That choice is based on backend availability and operator support. It does not sequentially select a configuration and evidence fidelity according to expected Pareto information per GPU-hour.

**Actual distinction**

| Charon | ServeCompass |
|---|---|
| Builds a high-fidelity operator/graph simulator | Orchestrates evidence across simulator and real cluster |
| Rule-pruned design search | Uncertainty- and cost-aware sequential acquisition |
| Performance, memory, and cost/throughput focus | Measured energy-latency-throughput focus |
| Searches a simulated frontier in about two minutes | Minimizes the real measurement cost of recovering a verified frontier |
| Simulator errors under 5.35% are a primary result | Calibration coverage and decision error at the Pareto/SLO boundary must be primary |
| Production validation is reported for selected cases | Verification is a mandatory release condition for every returned recommendation |

**Threat level:** Very high if ServeCompass is presented as a new simulator; moderate if it is presented as an evidence-allocation layer above simulators.

### 4.6 ServeCompass versus OptiKIT

**Strong overlap**

- Both automate vLLM tuning.
- Both tune tensor parallelism, concurrency, and token-batch limits.
- Both enforce TTFT/TPOT or latency SLOs.
- Both execute real benchmarking campaigns and report GPU-hours.

**Actual distinction**

| OptiKIT | ServeCompass |
|---|---|
| Enterprise orchestration system for quantization plus serving tuning | Research system for measurement-efficient energy Pareto identification |
| TPE over 30 complete real benchmark trials | Adaptive L0--L4 evidence selection |
| Scalar objective: SLO-compliant throughput per GPU | Energy-latency-throughput Pareto set under SLOs |
| Fixed low-cost infeasibility gate, followed by open-loop certification | Fidelity is part of the action chosen for each candidate |
| Real H100 evaluation | Planned H100/H200/B200 transfer and held-out hardware tests |
| No LLM agent | Bounded semantic planner with agent ablation |
| Reports GPU-hour cost | Optimizes frontier information per GPU-hour |

**Threat level:** Highest for the practical automated-tuning system story.

**Required paper response**

Use Optuna/TPE as a baseline with the same configuration space. Compare:

- hypervolume versus GPU-hours;
- energy at matched SLO;
- number of full target-scale runs;
- wall-clock tuning time;
- failure and infeasibility handling.

## 5. Draft Claims versus Repository Status

The current manuscript is more complete than the current executable method. This distinction must remain visible until experiments are finished.

| Capability | Current manuscript | Repository today | Required before MLSys submission |
|---|---|---|---|
| LLM semantic planner | Bounded language-model planner chooses typed subgoals | Planner interface and `RuleBasedPlanner`; no concrete LLM implementation | Implement at least one reproducible LLM planner and log prompts/tool calls |
| IPIG acquisition | Nested posterior mutual information about L4 feasible Pareto membership per GPU-hour | `PosteriorUncertaintyProxy` with fixed weighted heuristic | Implement the stated estimator or revise the paper to match the actual algorithm |
| Energy Twin | Phase- and communication-aware probabilistic model with conformal calibration | Highest-fidelity empirical lookup plus fixed uncertainty tiers | Implement energy/performance surrogate, calibration, and transfer uncertainty |
| Multi-fidelity execution | Replay, phase, intra-node, multi-node, and target-trace evidence | Replay backend and Slurm renderer; live runner is injectable but disabled | Integrate TokenPowerBench, telemetry, Slurm execution, and failure ingestion |
| Configuration-fidelity loop | Full L0--L4 closed loop | Runnable with replay and proxy scores | Demonstrate against fixed-ladder, simulator-only, TPE, qNEHVI, and multi-fidelity BO |
| Target verification | Mandatory L4 gate | Structurally enforced for replay; no live target run | Verify recommendations on H100/H200/B200 |
| GPU-hour accounting | Primary optimization axis | Implemented budget checks and event accounting | Use measured costs and plot hypervolume/regret versus cumulative GPU-hours |
| Pareto/SLO logic | Verified feasible frontier | Implemented deterministic SLO and Pareto filters | Validate frontier precision/recall against a dense measured oracle |
| Results | Reserved result macros and planned plots | No hardware campaign | Populate all headline results without simulated placeholders |

This table identifies the true critical path. The primary MLSys risk is not that the idea lacks differentiation; it is that the current implementation does not yet substantiate the differentiating claims.

## 6. What the Paper Can Safely Claim

### 6.1 Claims to avoid

- First automated LLM serving tuner.
- First LLM agent for system configuration.
- First energy-aware LLM serving controller.
- First Pareto or Bayesian optimizer for LLM serving.
- First simulator-driven LLM deployment optimizer.
- First framework to reduce configuration-search trials.
- First hybrid analytical/profiled/predicted LLM simulator.

### 6.2 Defensible contribution statement

Subject to a broader non-MLSys literature review and successful experiments:

> ServeCompass formulates energy-efficient LLM serving tuning as a budgeted multi-fidelity Pareto-identification problem in which each action jointly selects a serving configuration and an evidence level. Its hybrid agent separates semantic experiment planning from deterministic information-gain acquisition, calibrates cross-hardware uncertainty, and requires target-scale measured verification before returning a deployment recommendation.

### 6.3 Recommended one-paragraph positioning

> BEAM optimizes fine-grained runtime energy control, BOute applies constrained multi-objective Bayesian optimization to routing and deployment, PROMPTS uses multi-agent reasoning to propose distributed sharding, OptiKIT automates full-fidelity vLLM tuning, and Meta and Charon use calibrated simulation to explore large deployment spaces. ServeCompass addresses a different decision boundary: given heterogeneous prior, simulated, reduced-scale, and target-scale evidence, it jointly selects the next configuration and measurement fidelity to recover a measured energy-latency-throughput Pareto set under a real GPU-hour budget. The final verification gate prevents a simulator-only prediction from becoming a deployment recommendation.

## 7. Baseline Mapping

| Closest work | Baseline to implement | Comparison metric |
|---|---|---|
| BEAM | BEAM implementation or faithful frequency/chunk/microbatch controller | Energy and SLO attainment under matched deployment |
| BOute | Constrained qNEHVI with all evaluations at L4; optionally simulator-only qNEHVI | Hypervolume regret versus GPU-hours |
| PROMPTS | One-shot LLM proposal using the same tools/history | Top-k frontier recall, invalid proposals, trials and tokens |
| Meta deployment study | Pruned simulator-only enumeration | Simulated frontier quality and false recommendation rate |
| Charon | Best available simulator-only search; Charon adapter if feasible | Simulator error, calibration, and need for escalation |
| OptiKIT | Optuna/TPE with 30 full benchmark trials and the same search space | GPU-hours, wall time, energy-at-SLO, final frontier quality |

## 8. Final Differentiation Verdict

| Dimension | Differentiation strength | Condition |
|---|---|---|
| Energy + SLO | Weak alone | BEAM already establishes this problem |
| Pareto/MOBO | Weak alone | BOute and Charon already establish it |
| Agentic recommendation | Weak alone | PROMPTS already establishes it |
| Automated vLLM tuning | Weak alone | OptiKIT already establishes it |
| Simulator-based search | Weak alone | Meta and Charon are substantially stronger simulators |
| Configuration + fidelity co-selection | Strong | Must be a real algorithm, not a fixed ladder |
| Pareto information per real GPU-hour | Strong | Must outperform full-fidelity and fixed-fidelity baselines |
| Calibrated H100/H200/B200 transfer | Potentially strong | Requires held-out hardware experiments and negative-transfer analysis |
| Mandatory target-scale verification | Moderately strong | Must be enforced and evaluated under simulator bias/failure |
| Hybrid agent with deterministic numerical core | Moderately strong | Must show measurable agent value over optimizer-only control |
| Entire combination | Strongest current position | Must be implemented and measured end to end |

The best MLSys framing is therefore not “an agent tunes vLLM.” It is:

> **A measurement-budgeted agentic experimentation system for verified energy Pareto identification.**

