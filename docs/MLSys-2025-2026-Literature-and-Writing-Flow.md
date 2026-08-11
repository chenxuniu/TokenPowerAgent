# MLSys 2025-2026 Literature Study and Writing Flow for ServeCompass

Last updated: 2026-07-30

## 1. Executive Summary

This study audits the official MLSys 2025 and MLSys 2026 proceedings, identifies the papers closest to ServeCompass, and extracts the narrative and evaluation patterns that MLSys accepted papers use.

The central conclusion is:

> ServeCompass should not be framed as merely an LLM agent that tunes LLM serving. Its strongest defensible contribution is an agentic evidence-acquisition system that jointly chooses a serving configuration and a measurement fidelity under a real GPU-hour budget, then requires target-scale measured verification before releasing an energy-SLO Pareto recommendation.

The literature evolved quickly:

- In 2025, the relevant capabilities were separated across different papers: SLO scheduling, heterogeneous deployment search, simulation, calibrated uncertainty, and agent evaluation.
- In 2026, MLSys accepted direct work on energy-aware serving, multi-objective Bayesian optimization, agentic configuration recommendation, large-scale simulation, automated SLO-driven tuning, and LLM-generated optimizers.
- Therefore, energy, SLOs, Pareto optimization, simulation, automated tuning, and agents are each weak novelty claims by themselves.
- The strongest remaining research boundary is to optimize not only **which configuration to deploy**, but also **which evidence to acquire next**, at which scale, and at what real experimental cost.

## 2. Corpus Size and Counting Method

### 2.1 Official paper counts

| Year | Archival papers in official proceedings | Track composition | Notes |
|---|---:|---:|---|
| MLSys 2025 | **61** | 61 research papers | One research-paper track |
| MLSys 2026 | **135** | 108 research + 27 industry | The Industry Track was introduced in 2026 |

Primary sources:

- [MLSys 2025 official proceedings: 61 papers](https://proceedings.mlsys.org/paper_files/paper/2025)
- [MLSys 2026 official proceedings: 135 papers](https://proceedings.mlsys.org/paper_files/paper/2026)
- [MLSys 2026 official program](https://mlsys.org/virtual/2026/calendar)
- [MLSys 2026 Research Track CFP](https://mlsys.org/Conferences/2026/CallForResearchPapers)
- [MLSys 2026 Industry Track CFP](https://mlsys.org/Conferences/2026/CallForIndustrialTrackPapers)

The 108/27 split was reconstructed from unique paper titles in the official research-track and industry-track oral sessions. The combined set matches the 135 proceedings entries.

The count excludes:

- Workshops, competitions, tutorials, and the Young Professional Symposium.
- Duplicate oral and poster appearances of the same paper.
- Non-archival program events.

No official submission denominator was found during this audit. The frequently quoted acceptance-rate calculations should therefore be treated as third-party estimates rather than official MLSys statistics.

### 2.2 Title-level trend

The following is a mechanical, overlapping keyword count over official paper titles. It measures the growth of visible topic labels, not semantic relevance.

| Title keyword group | MLSys 2025 | MLSys 2026 |
|---|---:|---:|
| LLM serving or inference | 12 | 20 |
| Energy, power, or DVFS | 1 | 3 |
| Autotuning, simulation, or simulator | 1 | 3 |
| Agent, agents, agentic, or multi-agent | 3 | 13 |

The qualitative change is more important than the raw count. By 2026, the proceedings include direct papers on all major components that ServeCompass initially combined.

## 3. MLSys 2025: Relevant Work

### 3.1 Closest papers

| Paper | Core problem and method | Evidence | Relevance to ServeCompass | Remaining difference |
|---|---|---|---|---|
| [ThunderServe](https://proceedings.mlsys.org/paper_files/paper/2025/file/c2a0e26dd9ee7d57e92bb1c24b39659a-Paper-Conference.pdf) | Searches heterogeneous cloud GPU groupings, prefill/decode placement, TP/PP, and replica orchestration using hierarchical optimization and a simulator | 32 heterogeneous A6000/A5000/A40/3090 Ti GPUs compared with 8 A100 GPUs; up to 2.1x throughput and 2.5x lower latency deadline under a matched price budget | LLM serving configuration search, SLOs, simulator, heterogeneous multi-GPU deployment | No measured energy objective, Pareto output, LLM agent, adaptive fidelity, or next-experiment policy |
| [DiffServe](https://proceedings.mlsys.org/paper_files/paper/2025/file/414fd191b3246a19a55741b938380136-Paper-Conference.pdf) | Jointly selects a diffusion-model cascade, threshold, batch size, and model placement using performance models and optimization | 16 A100 GPUs plus event simulation and real-system validation; quality and SLO-violation trade-offs | Automatic knob selection, SLO constraints, model-based search, multiple competing objectives | Diffusion rather than LLM serving; no energy, agent, active measurement, or fidelity selection |
| [Rubick](https://proceedings.mlsys.org/paper_files/paper/2025/file/270339c997293ca2988c62f4308e389f-Paper-Conference.pdf) | Jointly optimizes execution plans and resource allocation using a small profiling set and a predictive performance model | 64 A800 GPUs; seven profiling points per job; reported prediction error below roughly 10% | Sparse profiling, model-guided configuration selection, profiling-cost awareness | Training cluster scheduling rather than inference; no energy Pareto set, LLM agent, or measurement-fidelity decision |
| [SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf) | Uses request and system state to adjust iteration-level request order and workload, balancing TTFT and TPOT | Llama and Qwen models on A100 GPUs; SLO attainment improves from 45.5% to 99.4% | SLO-aware LLM serving, runtime feedback, explicit TTFT/TPOT trade-off | Runtime scheduling policy rather than an experiment-planning agent; no energy, Pareto search, or simulator |
| [AIOpsLab](https://proceedings.mlsys.org/paper_files/paper/2025/file/d1f9e4a9f109b6e8b75ed362736f22ec-Paper-Conference.pdf) | Builds an interactive cloud sandbox that deploys services, generates workloads, injects faults, exports telemetry, and exposes controlled tools to agents | Kubernetes-based environments, 100 benchmark problems, four LLM agents, trajectory and token-cost analysis | Strong precedent for sandboxed plan-act-observe agent evaluation, controlled tools, and failure analysis | Incident detection and mitigation rather than numerical serving optimization; no energy or Pareto objective |
| [FlexInfer](https://proceedings.mlsys.org/paper_files/paper/2025/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf) | Uses a performance estimator to select CPU/GPU execution policies for prefill and decode from hardware and workload features | Xeon plus A100/H100 systems and 30B-70B models | Workload- and hardware-dependent policy selection across inference phases | Narrow CPU-offload decision; no energy objective, SLO Pareto set, agent, or active evidence acquisition |
| [Lumos](https://proceedings.mlsys.org/paper_files/paper/2025/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf) | Builds a trace-driven execution graph and replays modified model and parallelism configurations | GPT-3 15B-175B on up to 512 H100 GPUs; 3.3% average replay error | Strong model for an L0 simulator and topology-aware what-if evidence source | Training-only performance model; no measured energy, uncertainty-aware acquisition, or agent |
| [Interference-aware Runtime Prediction](https://proceedings.mlsys.org/paper_files/paper/2025/file/40b8fb4f90004405e14b1ede6ab42373-Paper-Conference.pdf) | Predicts runtime from sparse heterogeneous observations using matrix factorization and conformal uncertainty | 24 devices, 249 benchmarks, 410,970 observations; 5.2% prediction error | Sparse profiling, uncertainty calibration, and cross-hardware prediction are directly relevant to the Energy Twin | Edge runtime rather than LLM GPU energy; no configuration-fidelity search |
| [Seesaw](https://proceedings.mlsys.org/paper_files/paper/2025/file/cbc4ab80cd77aa0eb87da062fbcddb46-Paper-Conference.pdf) | Uses phase-specific parallelism and runtime model re-sharding for LLM inference | A10, L4, and A100 systems with 15B-70B models | Demonstrates that prefill/decode and scale-dependent configuration choices matter | Proposes one serving mechanism; exhaustive selection remains external |

### 3.2 Adjacent but not direct competitors

| Paper | Why it looks related | Why it is not a direct competitor |
|---|---|---|
| [ProtoRAIL](https://proceedings.mlsys.org/paper_files/paper/2025/file/42e2b24104bc92d724ce45c0c2f91e1d-Paper-Conference.pdf) | Active knowledge-in-the-loop and risk-aware resource policy | The agent is an imitation-learning controller for cloud vCPU oversubscription, not an LLM agent |
| [LAVA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/9de62e421d58234dbf773abf43268630-Abstract-Conference.html) | Prediction, replanning, production deployment, and indirect energy reduction | VM lifetime allocation rather than GPU inference configuration |
| [MEADOW](https://proceedings.mlsys.org/paper_files/paper/2025/hash/259a5df46308d60f8454bd4adcc3b462-Abstract-Conference.html) | Low-power LLM inference and phase-level latency | Fixed FPGA dataflow rather than a serving configuration search |
| [AI Metropolis](https://proceedings.mlsys.org/paper_files/paper/2025/hash/4f31327e046913c7238d5b671f5d820e-Abstract-Conference.html) | LLMs, multiple agents, and simulation | The agents are the simulated workload; the paper optimizes execution of the simulation |
| [The Hidden Bloat](https://proceedings.mlsys.org/paper_files/paper/2025/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html) | GPU resource efficiency | Framework debloating rather than serving configuration, energy, or SLO search |

Serving mechanism papers such as FlashInfer, QServe, NEO, LServe, LeanAttention, and KV-cache studies are important sources of candidate knobs. They should not be presented as the closest optimization competitors.

### 3.3 2025 conclusion

No MLSys 2025 paper combines:

- LLM inference serving;
- measured GPU energy;
- TTFT/TPOT SLOs;
- a Pareto objective;
- an LLM agent;
- adaptive multi-fidelity evidence acquisition; and
- target-scale measured verification.

The components existed, but in separate systems. This is useful historical context, but novelty must be judged against the denser 2026 landscape.

## 4. MLSys 2026: Direct Competitors

### 4.1 Six papers that must be discussed directly

| Paper | Track | Established contribution | What ServeCompass cannot claim | Defensible distinction |
|---|---|---|---|---|
| [BEAM](https://proceedings.mlsys.org/paper_files/paper/2026/file/eb3c42ddfa16d8421fdba13528107cc1-Paper-Conference.pdf) | Research | Joint frequency, prefill-chunk, and microbatch control for measured energy reduction under per-request SLOs | First energy-aware SLO controller or first joint serving-knob optimization | BEAM controls a deployed service at request-event timescales; ServeCompass plans a measurement campaign and chooses configuration plus evidence fidelity |
| [BOute](https://proceedings.mlsys.org/paper_files/paper/2026/file/ed1d3d4c64dc1b95332a8cde3f2a0bdf-Paper-Conference.pdf) | Research | Constrained qNEHVI for latency-quality routing and heterogeneous deployment Pareto search | First multi-objective Bayesian optimizer for LLM serving | Adaptive fidelity, measured energy, information per real GPU-hour, and mandatory target verification |
| [PROMPTS](https://proceedings.mlsys.org/paper_files/paper/2026/file/48253da5351effdfea994fd7bbff7005-Paper-Conference.pdf) | Industry | Multi-agent profile analysis, retrieval, and sharding recommendation across production TPU cases | First LLM agent for distributed configuration recommendation | Sequential evidence acquisition, a deterministic numerical optimizer, measured energy/SLO frontier, and an implemented verifier |
| [Optimizing Deployment Configurations for LLM Inference](https://proceedings.mlsys.org/paper_files/paper/2026/file/97dc07f1253ab33ee514f395a82fa7cc-Paper-Conference.pdf) | Industry | Search over millions of parallelism, runtime, batching, KV, hardware, and topology configurations using 100K+ operator measurements per hardware type | First large-scale configuration simulator or broad deployment search | Treat the simulator as one evidence level and decide when calibration or real execution is necessary |
| [Charon](https://proceedings.mlsys.org/paper_files/paper/2026/file/dbc8ce0fdfcd55172d73fb05dbae07fc-Paper-Conference.pdf) | Research | Unified compiler-style simulator with profiled, predicted, analytical, and fused operator backends | First multi-backend LLM simulator or simulator-guided Pareto exploration | Charon chooses modeling backends by operator support; ServeCompass should choose evidence fidelity by expected decision information per experimental cost |
| [OptiKIT](https://proceedings.mlsys.org/paper_files/paper/2026/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html) | Industry | Automated enterprise workflow for quantization, quality gates, SLO benchmarking, and Optuna/TPE vLLM tuning | First automated SLO-driven vLLM tuner | Measured energy Pareto identification, adaptive scale fidelity, cross-hardware transfer, and agentic recovery |

### 4.2 Additional methodological threats

These papers are not direct substitutes for ServeCompass, but they affect what can be claimed about the agent, optimizer, sandbox, and uncertainty components.

| Paper | Why it matters |
|---|---|
| [AccelOpt](https://proceedings.mlsys.org/paper_files/paper/2026/hash/0f8426558905746fc38da5e335700aec-Abstract-Conference.html) | Establishes a self-improving iterative LLM agent for accelerator optimization, with optimization memory, a dedicated benchmark, cost analysis, and trajectory improvement |
| [FlashInfer-Bench](https://proceedings.mlsys.org/paper_files/paper/2026/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html) | Establishes a correctness- and performance-aware closed loop connecting agent-generated kernels, real traces, benchmarking, and safe deployment |
| [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2511.16964) | Systematically studies explore/exploit strategies, error-fixing agents, trajectory granularity, LLM query budgets, and GPU inference speedup |
| [Automated Algorithm Design for Auto-Tuning Optimizers](https://proceedings.mlsys.org/paper_files/paper/2026/hash/4f31327e046913c7238d5b671f5d820e-Abstract-Conference.html) | Uses LLMs to synthesize, test, and iteratively refine optimizers for real autotuning tasks across hardware platforms |
| [When Machine Learning Isn't Sure](https://proceedings.mlsys.org/paper_files/paper/2026/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html) | Makes uncertainty calibration, rejection, and safe fallback explicit in ML-for-systems deployments |
| [CORE](https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html) | Shows that coordinated energy control across resources and inference phases is stronger than independent governors |
| [CATWILD](https://proceedings.mlsys.org/paper_files/paper/2026/hash/2093ed77c549eda95bd6f7212b735b43-Abstract-Conference.html) | Industrial-scale compiler autotuning; raises the bar for production search methodology and deployment evidence |
| [DriftBench](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ea0b5818ae9255ee1fb1e3b4442d2ffe-Abstract-Conference.html) | Makes infrastructure drift in LLM serving a benchmark concern; relevant to cross-run and cross-hardware calibration |

### 4.3 Capability matrix

`Yes` means the capability is experimentally demonstrated. `Partial` means indirect, system-specific, or not a primary output.

| System | LLM-agent reasoning | Measured energy | SLO | Pareto output | Config-fidelity co-selection | Real-GPU cost objective | Target-scale verification |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEAM | No | Yes | Yes | Partial | No | No | Yes |
| BOute | No | No | Yes | Yes | No | No | Partial |
| PROMPTS | Yes | No | No | No | No | Partial | Partial |
| Meta deployment study | No | Partial | Yes | Partial | No | No | Partial |
| Charon | No | No | Yes | Yes | No | Partial | Partial |
| OptiKIT | No | No | Yes | No | Partial | Partial | Yes |
| ServeCompass target | Yes | Yes | Yes | Yes | **Yes** | **Yes** | **Yes** |

The ServeCompass row describes the intended paper contribution, not the repository's current completed evidence.

## 5. How the Accepted Papers Build Their Story

### 5.1 BEAM: coupling to causal mechanisms

Narrative:

1. LLM inference energy is growing.
2. SLOs create latency slack.
3. Batching and DVFS consume the same slack and interact.
4. Optimizing them independently is suboptimal.
5. A joint event-driven, phase-aware controller is required.

Method flow:

1. Empirical knob characterization.
2. Challenges derived from the characterization.
3. Design principles mapped to those challenges.
4. Offline performance-energy profiling.
5. Prefill and decode control mechanisms.
6. Runtime implementation.

Evaluation flow:

1. End-to-end energy and SLO result.
2. Behavior under different slack and request loads.
3. Event-response analysis.
4. Component ablation.
5. Prediction accuracy and overhead.

Lesson for ServeCompass:

> Every claimed design principle should have a dedicated experiment that shows why it is necessary.

### 5.2 BOute: coupled decisions to formal optimization

Narrative:

1. Low-cost LLM serving requires both model routing and GPU deployment.
2. Routing and deployment affect each other.
3. Sequential or isolated optimization misses the joint frontier.
4. The problem is naturally constrained and multi-objective.
5. Constrained MOBO searches it efficiently.

Method flow:

1. Characterize routing-only, homogeneous, and heterogeneous cases.
2. Formalize objectives and constraints.
3. Construct offline performance data and a simulator.
4. Prune candidates and encode structured configurations.
5. Apply constrained qNEHVI.
6. Convert solutions into deployable configurations.

Evaluation flow:

1. End-to-end comparison with isolated baselines.
2. Show actual configurations selected by the optimizer.
3. Plot the Pareto frontier.
4. Compare at matched quality, latency, and cost constraints.
5. Report search overhead and ablations.

Lesson for ServeCompass:

> The paper must show a frontier and a matched-constraint comparison, not only a scalar score.

### 5.3 PROMPTS and OptiKIT: expert bottleneck to workflow evidence

PROMPTS narrative:

1. Distributed configuration requires expert knowledge.
2. Manual profiling and reasoning loops are slow.
3. Black-box search is expensive and transfers poorly.
4. A structured multi-agent workflow can diagnose, retrieve knowledge, and propose configurations.

OptiKIT narrative:

1. Enterprise inference optimization is fragmented across quality, benchmark, and deployment tools.
2. Manual workflows depend on a small number of experts.
3. A production pipeline must include quality gates, SLO certification, tuning, and archival.
4. The system is evaluated by both performance benefit and engineering/GPU-hour cost.

Shared evaluation pattern:

1. Cover a regime matrix rather than one benchmark.
2. Evaluate output validity and compilability.
3. Compare top-k recommendation quality.
4. Include failure cases.
5. Report LLM, engineering, or GPU-hour cost.

Lesson for ServeCompass:

> The agent needs an independent evaluation axis: valid actions, recovery, provenance, cost, and contribution over an optimizer-only state machine.

### 5.4 Meta and Charon: validate the instrument before using it

Narrative:

1. The deployment space is too large and costly for exhaustive hardware measurement.
2. A simulator can make exploration tractable.
3. Simulator conclusions are credible only after error is validated on real systems.
4. Once validated, the simulator can support large-scale case studies and design insights.

Evaluation order:

1. Validate operator and end-to-end prediction error.
2. Evaluate memory, communication, overlap, and scale behavior.
3. Only then use the simulator for configuration search.
4. Verify selected findings or mechanisms on hardware.
5. State where conclusions remain simulated.

Lesson for ServeCompass:

> The Energy Twin cannot be evaluated only by the quality of configurations it recommends. It first needs held-out calibration, coverage, rank-correlation, and failure-under-shift results.

### 5.5 2025 patterns that remain useful

SOLA:

- Opens with a concrete failure distribution for TTFT and TPOT.
- Converts the observed bias and variance into two explicit trade-offs.
- Maps each trade-off to a state-aware mechanism.
- Uses end-to-end goodput plus per-mechanism analysis.

ThunderServe:

- Starts from heterogeneous cloud and network constraints.
- Defines a two-level optimization corresponding to the actual decision hierarchy.
- Uses matched price budgets to make the comparison fair.
- Reports the selected deployment, search time, simulator behavior, and online reconfiguration.

AIOpsLab:

- Defines the sandbox, agent-cloud interface, task taxonomy, and oracle before evaluating agents.
- Measures success, steps, latency, tokens, and failure causes.
- Treats interaction trajectories as first-class experimental evidence.

Lumos:

- Validates replay accuracy before estimating unseen configurations.
- Separates same-configuration reconstruction from out-of-configuration extrapolation.

## 6. Reusable MLSys Narrative Patterns

### Pattern A: empirical coupling

`Coupled knobs -> measured failure -> design principles -> mechanism-specific ablation`

Best examples: BEAM and SOLA.

Use this to justify why configuration and fidelity must be chosen jointly.

### Pattern B: formal decision problem

`Coupled decisions -> objectives and constraints -> optimizer -> frontier -> matched-constraint comparison`

Best example: BOute.

Use this for IPIG and real-GPU-hour-budgeted Pareto identification.

### Pattern C: agentic workflow

`Expert bottleneck -> typed agent workflow -> regime matrix -> validity/cost/failure analysis`

Best examples: PROMPTS, AIOpsLab, AccelOpt, and PyTorch multi-agent optimization.

Use this to prove the LLM agent is more than a wrapper around Bayesian optimization.

### Pattern D: trustworthy simulator

`Validate instrument -> characterize uncertainty -> search with it -> verify selected conclusions`

Best examples: Charon, Meta, Lumos, and uncertainty-aware systems work.

Use this for the Energy Twin, sandbox evidence, and mandatory L4 verification.

## 7. Recommended ServeCompass Paper Flow

MLSys 2026 used a ten-page, two-column main-paper limit excluding references. MLSys 2027 rules must be checked after its official CFP is released. Under a ten-page body budget, the recommended structure is:

| Section | Suggested body budget | Purpose |
|---|---:|---|
| Abstract | 0.25 page | Problem, gap, key idea, implementation, two headline results |
| 1. Introduction | 1.0 page | Expensive evidence problem, three observations, system insight, contributions |
| 2. Empirical Motivation | 1.0 page | Demonstrate rank inversion, simulator bias, and fixed-fidelity waste |
| 3. Problem Formulation | 0.75 page | Define scenario, configuration, fidelity, SLO, Pareto target, and GPU-hour budget |
| 4. System Overview | 0.75 page | Closed-loop architecture and responsibility boundaries |
| 5. Multi-Fidelity Evidence Model | 1.25 pages | L0-L4 contract, Energy Twin, uncertainty, transfer, provenance |
| 6. Agentic Pareto Search | 1.25 pages | Typed planner, IPIG, safe tools, verifier, stopping rule |
| 7. Implementation | 0.5 page | vLLM, TokenPowerBench, Slurm, telemetry, failure handling |
| 8. Evaluation | 2.5 pages | Calibration, GPU-hour efficiency, agent value, hardware results, ablations |
| 9. Related Work | 0.5 page | Direct competitors and precise boundary |
| 10. Limitations and Conclusion | 0.25 page | Scope, risks, and final claim |

### 7.1 Introduction flow

Paragraph 1: Operational problem

- LLM inference configurations interact across batching, KV cache, parallelism, runtime, placement, and power policy.
- The best configuration changes with model, workload, hardware, and topology.
- Full multi-node sweeps consume expensive GPU-hours.

Paragraph 2: Why current solutions are insufficient

- Simulator-only search can be biased.
- Full-fidelity BO buys reliable evidence at excessive cost.
- Existing agents recommend configurations but do not explicitly optimize the cost and fidelity of the next experiment.

Paragraph 3: Three empirical observations

1. Cheap evidence is useful for pruning and ranking some candidates.
2. Rankings can invert across GPU type, node scale, topology, and workload.
3. A fixed fidelity ladder spends target-scale resources on candidates whose uncertainty is already irrelevant.

Paragraph 4: Core insight

- Treat evidence fidelity as part of the action.
- Select `(configuration, fidelity)` by expected information about the verified Pareto set per real GPU-hour.

Paragraph 5: System

- The agent converts intent into a typed scenario and semantic subgoal.
- A deterministic optimizer performs numerical candidate and fidelity selection.
- The Energy Twin maintains calibrated beliefs.
- The verifier prevents simulator-only recommendations.

Paragraph 6: Contributions

1. Problem formulation and evidence contracts.
2. IPIG configuration-fidelity acquisition.
3. Calibrated Energy Twin and cross-hardware transfer.
4. Agentic closed loop with guarded execution and verification.
5. H100/H200/B200 evaluation and GPU-hour savings.

### 7.2 Motivation section

The motivation should be experimental rather than descriptive.

Recommended observations:

1. **Configuration rank inversion:** show that an L0/L1 winner is not always an L4 winner.
2. **Scale-dependent communication energy:** show divergence between one GPU, one node, and multiple nodes.
3. **Hardware transfer asymmetry:** show that H100-to-H200 transfer differs from H200-to-B200 transfer.
4. **Fixed-ladder waste:** show how many real GPU-hours are spent escalating candidates that cannot change the final frontier.

The first figure should show the problem, not the agent architecture.

### 7.3 Method section

Recommended order:

1. Evidence-level definitions and contracts.
2. Energy Twin observations, features, and uncertainty.
3. Pareto target and SLO feasibility.
4. Information-per-cost acquisition.
5. Typed agent subgoals and tool interface.
6. Verification and stopping.
7. Failure recovery and provenance.

The LLM should not directly choose unrestricted numerical configurations. Its responsibility should remain semantic and auditable.

### 7.4 Evaluation flow

Organize evaluation by research questions rather than implementation components.

| RQ | Question | Required evidence |
|---|---|---|
| RQ1 | Is low-fidelity evidence calibrated and decision-useful? | Held-out prediction error, interval coverage, rank correlation, SLO classification, and Pareto recall |
| RQ2 | Does adaptive fidelity reduce real experimental cost? | Hypervolume regret and Pareto recall versus cumulative real GPU-hours |
| RQ3 | Does the final recommendation save energy at a matched SLO? | J/token and J/request under matched TTFT/TPOT and throughput constraints |
| RQ4 | Does transfer across hardware and scale help safely? | Leave-one-hardware, leave-one-node-count, and negative-transfer experiments |
| RQ5 | Does the agent add value beyond the optimizer? | Full agent versus optimizer-only state machine versus LLM-only planner |
| RQ6 | Is target verification necessary? | Simulator-only false recommendation rate, topology shift, and L4 correction |
| RQ7 | What are the practical costs and failures? | Wall time, GPU-hours, LLM tokens, invalid actions, OOM, preemption, and telemetry loss |

### 7.5 Baselines

Mandatory baselines:

| Literature threat | Baseline |
|---|---|
| BOute | Full-fidelity constrained qNEHVI over the same candidate space |
| OptiKIT | Optuna/TPE with 30 full benchmark trials |
| Meta and Charon | Simulator-only or Energy-Twin-only search |
| Multi-fidelity claim | Fixed L0-L1-L2-L3-L4 ladder and a standard multi-fidelity BO method |
| PROMPTS | One-shot or top-k LLM recommendation using the same tools and history |
| Agent claim | Deterministic optimizer-only state machine |
| Search efficiency | Random search and exhaustive-grid oracle on a bounded subset |
| Energy controller | BEAM or a faithful frequency/chunk/microbatch controller where implementable |

All methods must share:

- The same feasible candidate pool.
- The same total real GPU-hour budget.
- The same initial observations.
- The same target-scale verification policy, or an explicitly measured lack of verification.
- The same SLO and Pareto reference point.

### 7.6 Ablations

1. Remove the LLM planner.
2. Replace IPIG with posterior uncertainty only.
3. Use a fixed fidelity ladder.
4. Remove cross-GPU transfer.
5. Remove uncertainty calibration.
6. Remove topology features.
7. Allow recommendations without L4 verification.
8. Remove failure-recovery memory.

## 8. Figure and Table Flow

### Main-body figures

1. **Figure 1: Empirical motivation.** Configuration rank inversion and simulator-to-target mismatch across H100/H200/B200 or scale.
2. **Figure 2: System overview.** Intent compiler, agent planner, IPIG, executor/sandbox, Energy Twin, evidence store, and verifier.
3. **Figure 3: Evidence fidelity and cost.** What L0-L4 observe, their costs, biases, and escalation paths.
4. **Figure 4: Headline result.** Hypervolume regret or verified Pareto recall versus cumulative real GPU-hours.
5. **Figure 5: Energy at matched SLO.** J/token or J/request under equal TTFT/TPOT requirements.
6. **Figure 6: Transfer and verification.** Cross-hardware calibration, false recommendations, and L4 correction.

### Main-body tables

1. Closest-work capability matrix.
2. Experimental matrix: models, GPUs, node counts, workloads, SLOs, and search spaces.
3. End-to-end result summary.
4. Agent validity, recovery, and cost.

The architecture figure should not be Figure 1 unless no real motivation data are available. A systems paper is stronger when the first visual proves the problem.

## 9. Claim-to-Evidence Contract

| Intended claim | Minimum acceptable evidence |
|---|---|
| The sandbox reduces real experimentation | Held-out L4 rank correlation, uncertainty coverage, Pareto recall, and a measured reduction in GPU-hours |
| Adaptive multi-fidelity search is superior | Equal-budget comparison against full-fidelity qNEHVI, TPE, fixed ladder, simulator-only search, and a standard multi-fidelity optimizer |
| The agent contributes independently | Full agent versus optimizer-only and LLM-only variants, including injected failures |
| Recommendations are energy-efficient | Repeated target-scale H100/H200/B200 measurements at matched TTFT/TPOT SLOs |
| Cross-hardware transfer works | Leave-one-hardware-out tests and explicit negative-transfer cases |
| Verification makes deployment safer | Number and severity of recommendations changed or rejected by L4 validation |
| The system is practical | Wall time, real GPU-hours, LLM cost, scheduler failures, and provenance completeness |

## 10. Claims to Avoid

Do not claim:

- The first automated LLM serving tuner.
- The first energy-aware LLM inference controller.
- The first Pareto or MOBO serving optimizer.
- The first LLM agent for systems optimization.
- The first simulator for LLM deployment search.
- The first multi-agent configuration recommender.

Potentially defensible claims, after implementation and evaluation:

- The first system to jointly acquire serving configurations and measurement fidelities for verified energy-SLO Pareto identification under a real GPU-hour budget.
- A calibrated multi-fidelity evidence model spanning replay, simulation, reduced scale, intra-node, multi-node, and target-trace execution.
- A hybrid agentic architecture in which an LLM performs typed semantic planning while deterministic tools perform numerical optimization, constraint checking, and verification.
- A target-scale verification contract that quantifies and prevents false simulator-only recommendations.

## 11. Recommended One-Sentence Positioning

> Existing systems optimize which serving configuration to deploy or how to control it at runtime; ServeCompass optimizes which evidence to purchase next, jointly choosing a serving configuration and a measurement fidelity under a real GPU-hour budget, and requires target-scale measured verification before releasing an energy-SLO Pareto recommendation.

## 12. Practical Next Steps

1. Produce a small H100/H200/B200 motivation dataset before expanding the paper text.
2. Demonstrate at least one real configuration rank inversion across fidelity, hardware, or node scale.
3. Implement a reproducible full-fidelity qNEHVI baseline and Optuna/TPE baseline.
4. Replace the current uncertainty proxy with the algorithm that the paper will claim, or narrow the claim.
5. Implement and evaluate the LLM planner against an optimizer-only state machine.
6. Enable a live cluster executor with complete telemetry and provenance.
7. Make `hypervolume/regret versus cumulative real GPU-hours` the headline result.
8. Use energy at matched TTFT/TPOT SLO as the primary deployment result.

## 13. Related Repository Documents

- [Detailed MLSys 2026 closest-work comparison](./MLSys-2026-Closest-Work-Comparison.md)
- [MLSys 2027 transition and literature review](./MLSys-2027-Transition-and-Literature-Review.md)
- [ServeCompass complete workflow](./ServeCompass-Complete-Workflow.md)
