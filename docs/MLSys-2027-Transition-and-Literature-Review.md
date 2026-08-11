# ServeCompass: MLSys 2027 Format, Literature, and Repositioning Review

**Last verified:** 2026-07-28  
**Target:** MLSys 2027 Research Track  
**Scope of this audit:** Official MLSys formatting rules and all accepted MLSys 2025/2026 papers, with emphasis on energy-efficient LLM inference, automated serving optimization, agentic systems, multi-objective search, simulation, and multi-fidelity experimentation.

## 1. Executive Decision

ServeCompass is a strong topical fit for MLSys. The current idea, however, is not yet differentiated enough if it is described only as:

> An LLM agent uses a sandbox to try serving configurations and recommends an energy-efficient Pareto point.

By MLSys 2026, separate papers already cover nearly every component of that sentence:

| Component | Closest MLSys 2026 work |
|---|---|
| Energy-aware serving control under SLOs | BEAM |
| Multi-objective Bayesian optimization for LLM serving | BOute |
| LLM agents proposing distributed-system configurations | PROMPTS |
| Automated SLO-driven serving tuning | OptiKIT |
| Simulator-driven exploration of deployment configurations | Meta deployment optimizer and Charon |
| Sandboxed, closed-loop evaluation of system agents | FlashInfer-Bench; AIOpsLab in 2025 |

The defensible MLSys thesis is narrower and technically stronger:

> **ServeCompass is a measurement-budgeted, cross-hardware, multi-fidelity agent that jointly chooses the next LLM serving configuration and the evidence fidelity needed to evaluate it, in order to recover a verified energy-latency-throughput Pareto frontier under SLO constraints.**

The words that must carry real algorithmic and experimental content are:

- **measurement-budgeted:** optimize useful information per real GPU-hour, not merely the number of trials;
- **multi-fidelity:** choose among simulation, short real measurements, reduced-scale runs, and target-scale validation;
- **cross-hardware:** quantify transfer and uncertainty across H100, H200, and B200 rather than training an unrelated model for every GPU;
- **verified:** never promote a deployment recommendation using simulated energy alone;
- **agentic:** demonstrate a measurable benefit from the LLM agent beyond a conventional optimizer.

This is viable for MLSys 2027, but a replay-only prototype or an LLM wrapper around random/qNEHVI search would not meet the likely main-conference novelty bar.

## 2. What Is Officially Known About MLSys 2027

As of 2026-07-28, MLSys has not published an official MLSys 2027 CFP, submission deadline, venue, or `mlsys2027` style package. The current official materials remain those for MLSys 2026.

The safe planning baseline is MLSys 2026 because its CFP explicitly reused the MLSys 2025 format, and the 2025/2026 style files are materially stable.

| Rule | Verified MLSys 2025/2026 policy | MLSys 2027 status |
|---|---|---|
| Research-paper length | 10 pages, references excluded | Not announced |
| Layout | Two columns, official MLSys LaTeX style | Not announced |
| Review | Double blind | Not announced |
| Appendix | Unlimited; separately uploaded at initial submission; reviewers need not read it | Not announced |
| References | Every entry must list all authors | Not announced |
| Artifact evaluation | Voluntary after acceptance; does not affect acceptance | Not announced |
| 2026 deadline | 2025-10-30, 20:00 UTC | No 2027 deadline announced |

Planning assumption only: if MLSys 2027 follows the recent cadence, the deadline may fall in autumn 2026. This is not an official date. We should treat early October 2026 as the internal deadline for a complete paper rather than waiting for the CFP.

Official sources:

- [MLSys 2025 Call for Papers](https://mlsys.org/Conferences/2025/CallForPapers)
- [MLSys 2026 Call for Research Papers](https://mlsys.org/Conferences/2026/CallForResearchPapers)
- [MLSys 2026 Artifact Evaluation](https://mlsys.org/Conferences/2026/CallForAEs)
- [MLSys 2025 proceedings, 61 papers](https://proceedings.mlsys.org/paper_files/paper/2025)
- [MLSys 2026 accepted-paper program, 135 papers](https://mlsys.org/virtual/2026/papers.html)

## 3. Format Migration: Current SC Draft to MLSys

The current paper uses IEEE conference format in `paper-draft/main.tex`. It should not be overwritten. When MLSys 2027 releases its official package, create a separate `paper-mlsys27/` tree and port the content selectively.

### 3.1 Mechanical changes

| Current SC/IEEE paper | MLSys review paper |
|---|---|
| `\documentclass[conference]{IEEEtran}` | `\documentclass{article}` and official `\usepackage{mlsys2027}` |
| IEEE title and author blocks | `\mlsystitle`, `mlsysauthorlist`, and MLSys affiliation macros |
| Visible authors/TTU affiliation | No authors, affiliation, acknowledgment, or identifying metadata |
| Numeric citations through `cite` | Author-year citations through MLSys `natbib` |
| `IEEEtran.bst` | Official MLSys `.bst` |
| `IEEEkeywords` | `\mlsyskeywords`, normally PDF metadata rather than a visible IEEE block |
| SC artifact language | MLSys artifact-availability and post-acceptance AE language |
| IEEE bibliography controls | Remove `@IEEEtranBSTCTL` and IEEE-only commands |

The official style uses US Letter, 10 pt Times text with approximately 11 pt leading, figure captions below figures, table captions above tables, and Type 1 embedded fonts.

### 3.2 Blind-review requirements

For the Research Track:

1. Use the review style without the `[accepted]` option.
2. Remove author names, affiliations, email addresses, acknowledgments, grants, and identifying PDF metadata.
3. Refer to TokenPowerBench in the third person, as to any other prior work.
4. Use an anonymous artifact URL and remove identifying repository/account details.
5. Do not write phrases such as “our prior TokenPowerBench paper” in the review version.
6. Keep full author lists in every bibliography entry even though the manuscript itself is anonymous.

The downloadable 2026 example contains a dangerous inconsistency: its “blind” example activates `[accepted]`. The CFP and style behavior make clear that `[accepted]` must not be used for initial review.

### 3.3 Content and page-budget changes

The current IEEE PDF is about 10 pages total, with references beginning on page 9. Under MLSys rules, references do not count toward the 10-page body limit. Reflow into the narrower MLSys text area will consume some of that space, but the paper should still gain roughly 1–2 body pages.

Those pages should be spent on:

- the actual multi-fidelity acquisition algorithm;
- uncertainty calibration and the target-scale verification gate;
- comparison against the 2026 closest work;
- real H100/H200/B200 results and ablations.

They should not be spent on additional generic LLM-energy motivation.

The current abstract is too long and too procedural for the MLSys style. The final abstract should be one paragraph of approximately 5–6 sentences:

1. problem and measurement-cost gap;
2. central insight;
3. system and algorithm;
4. hardware/workload evaluation;
5. one or two headline numerical results;
6. implication.

The implementation currently uses `algpseudocode`; the 2026 MLSys package bundles `algorithm`/`algorithmic`. This needs a compile-time migration rather than loading two conflicting algorithm packages.

### 3.4 Appendix strategy

Keep every claim needed for acceptance in the 10-page paper. Put the following in the separately submitted appendix:

- complete search spaces and invalid-configuration rules;
- all workload traces and SLO definitions;
- energy-meter calibration;
- prompt templates and tool schemas;
- extra Pareto fronts and per-model results;
- statistical details and additional seeds;
- failure cases and complete agent traces;
- reproduction commands.

The 2025/2026 CFP says to upload the appendix separately for review, even though some template prose discusses an appendix after the references. Follow the final 2027 CFP and submission portal if they differ.

## 4. MLSys 2025 Literature Audit

The official MLSys 2025 proceedings contain 61 papers. No accepted paper combines an LLM serving configuration agent, measured energy, Pareto optimization, multi-fidelity evidence, and explicit GPU-hour minimization. The following papers nevertheless establish important pieces of the problem.

### 4.1 Highest-overlap papers

| Paper | What it contributes | Overlap with ServeCompass | Remaining distinction |
|---|---|---|---|
| [ThunderServe](https://proceedings.mlsys.org/paper_files/paper/2025/hash/c2a0e26dd9ee7d57e92bb1c24b39659a-Abstract-Conference.html) | Tabu-search-based deployment and rescheduling across heterogeneous cloud GPUs; optimizes prefill/decode placement and parallelism | Heterogeneous GPU configuration, phase-aware serving, SLO-aware optimization | Price/performance rather than measured energy; no agent or multi-fidelity acquisition |
| [FlexInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | Performance estimator dynamically selects CPU/GPU execution policy for prefill and decode | Predictive twin plus configuration selection | Fixed policy family, latency objective, single-server setting; no uncertainty or measurement budget |
| [Seesaw](https://proceedings.mlsys.org/paper_files/paper/2025/hash/cbc4ab80cd77aa0eb87da062fbcddb46-Abstract-Conference.html) | Dynamic model resharding between prefill and decode with KV-cache support | Phase-specific TP/PP choices and runtime adaptation | Specialized mechanism, not general black-box energy search |
| [SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html) | State-aware request scheduling for TTFT/TPOT SLO attainment | SLO formulation and serving-state observations | Does not explore deployment configurations or optimize energy |
| [DiffServe](https://proceedings.mlsys.org/paper_files/paper/2025/hash/414fd191b3246a19a55741b938380136-Abstract-Conference.html) | Simulator-calibrated MILP for model cascade, batch, threshold, and placement | Sandbox-to-live validation and multi-dimensional serving optimization | Diffusion serving and FID/SLO objectives rather than LLM token energy |

### 4.2 Simulation, uncertainty, and agent methodology

| Paper | Relevance to our method |
|---|---|
| [Lumos](https://proceedings.mlsys.org/paper_files/paper/2025/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) | Trace-driven execution-graph simulation reaches 3.3% average replay error on experiments up to 512 H100 GPUs. It is a strong precedent for a calibrated digital twin, although it models training time rather than inference energy. |
| [Interference-aware Edge Runtime Prediction with Conformal Matrix Completion](https://proceedings.mlsys.org/paper_files/paper/2025/hash/40b8fb4f90004405e14b1ede6ab42373-Abstract-Conference.html) | Demonstrates sparse cross-device prediction with conformal uncertainty. It motivates calibrated intervals and coverage reporting for H100/H200/B200 transfer. |
| [AIOpsLab](https://proceedings.mlsys.org/paper_files/paper/2025/hash/d1f9e4a9f109b6e8b75ed362736f22ec-Abstract-Conference.html) | Defines a controlled agent-cloud interface, fault injection, telemetry, and a 100-task benchmark. It is the clearest 2025 precedent for evaluating system agents through a sandbox and measuring steps, time, tokens, and cost. |
| [ProtoRAIL](https://proceedings.mlsys.org/paper_files/paper/2025/hash/42e2b24104bc92d724ce45c0c2f91e1d-Abstract-Conference.html) | Uses risk-aware active imitation learning for cloud resource control. It supports a safety-aware acquisition and deployment-gating argument. |
| [LAVA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/9de62e421d58234dbf773abf43268630-Abstract-Conference.html) | Calibrates and stress-tests a learned resource policy in trace replay before a production A/B test. It is a useful methodological precedent for sandbox-to-live validation. |
| [Rubick](https://proceedings.mlsys.org/paper_files/paper/2025/hash/270339c997293ca2988c62f4308e389f-Abstract-Conference.html) | Uses performance models to jointly select execution plans and cluster resources, showing how model-guided reconfiguration should be evaluated at cluster scale. |

### 4.3 Serving mechanisms that define the search space

| Paper | Role in ServeCompass |
|---|---|
| [NEO](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html) | CPU/GPU KV and attention offloading can be represented as a discrete policy family or specialized baseline. |
| [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html) | Supplies attention/KV implementation choices that affect latency and energy but is not itself an automated tuner. |
| [Context Parallelism for Scalable Million-Token Inference](https://proceedings.mlsys.org/paper_files/paper/2025/hash/78834433edc3291f4c6cbbd2759324db-Abstract-Conference.html) | Demonstrates that multi-node inference behavior up to 128 H100 GPUs cannot be inferred from a single-GPU sandbox without communication-aware calibration. |
| [MEADOW](https://proceedings.mlsys.org/paper_files/paper/2025/hash/259a5df46308d60f8454bd4adcc3b462-Abstract-Conference.html) | Background for low-power LLM inference, but on an FPGA and without cluster-scale configuration search. |

### 4.4 Meaning of the 2025 landscape

The 2025 literature supports the need for all three layers:

1. ThunderServe, FlexInfer, Seesaw, and SOLA show that serving configuration is workload-, phase-, and hardware-dependent.
2. Lumos and related predictive systems show that calibrated replay/simulation can reduce real profiling.
3. AIOpsLab and ProtoRAIL show how autonomous system agents should be constrained and evaluated.

This is useful prior art, but “we combine these ideas” is not by itself an MLSys contribution. The paper must identify and solve a new decision problem.

## 5. MLSys 2026 Literature Audit

The official MLSys 2026 program contains 135 papers and is substantially more crowded. No paper was found that simultaneously performs LLM-driven sequential serving search, fidelity selection, measured energy Pareto optimization, and explicit real-GPU-budget minimization. However, six papers are close enough that the paper must compare against them directly.

### 5.1 Direct competitors

| Paper | Method and evidence | What it already claims | Exact opening for ServeCompass |
|---|---|---|---|
| [BEAM](https://proceedings.mlsys.org/paper_files/paper/2026/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html) | Event-driven vLLM controller jointly tunes GPU frequency, chunk size, and microbatch count; A100 experiments with Llama-3.3-70B and Qwen2.5-32B | Up to 51% GPU-energy reduction under TTFT/TBT SLOs | Offline/online controller rather than agentic experimentation; no fidelity selection, cross-hardware transfer, or GPU-hour objective |
| [BOute](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ed1d3d4c64dc1b95332a8cde3f2a0bdf-Abstract-Conference.html) | Constrained qNEHVI over model routing, GPU allocation, and DP/TP/PP on heterogeneous GPUs | Multi-objective LLM serving optimization over latency, quality, and cost; 38% average cost reduction | Must beat or complement qNEHVI on measured-energy frontiers and show the benefit of jointly selecting fidelity |
| [PROMPTS](https://proceedings.mlsys.org/paper_files/paper/2026/hash/48253da5351effdfea994fd7bbff7005-Abstract-Conference.html) | Coordinator, analyzer, and proposal agents use profiles, compiler data, roofline reasoning, and RAG to recommend sharding | Agentic system configuration over 8 workloads and 2–512 TPU chips; expert configuration appears in top three | Sequential evidence acquisition, energy/Pareto objectives, uncertainty, and target-scale verification |
| [Optimizing Deployment Configurations for LLM Inference](https://proceedings.mlsys.org/paper_files/paper/2026/hash/97dc07f1253ab33ee514f395a82fa7cc-Abstract-Conference.html) | Lightweight benchmark-driven simulator searches millions of hardware, parallelism, batch, KV, and runtime configurations; covers H100/H200/MI300X and up to 256 accelerators | Deployment exploration with roughly +/-5% simulator error in many cases | Adaptive measurements, explicit energy uncertainty, agent intervention, and verification under a real profiling budget |
| [Charon](https://mlsys.org/virtual/2026/poster/3638) | Unified operator-level simulator with analytical, profiled, and learned backends | Training/inference simulation with at most 5.35% reported error and orders-of-magnitude cheaper exploration | Energy objective, active fidelity selection, LLM-agent role, and measured target-scale frontier |
| [OptiKIT](https://mlsys.org/media/mlsys-2026/Slides/3752.pdf) | Ray plus TPE/Optuna tunes TP and vLLM batching knobs in about 30 trials | Automated SLO-driven tuning on A100/H100/H200, up to 2.8x throughput per GPU | Multi-objective energy frontier, multi-fidelity acquisition, calibrated transfer, and agent reliability |

### 5.2 Important adjacent papers

| Paper | Why it matters |
|---|---|
| [From Tokens to Layers](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html) | Reports up to 22% lower per-token energy for an MoE serving mechanism and studies TTFT/TBT tradeoffs. It must be cited when motivating energy-aware scheduling. |
| [MorphServe](https://proceedings.mlsys.org/paper_files/paper/2026/hash/8144a9d62e506af0fcdeac0e456b2710-Abstract-Conference.html) | Dynamically selects quantization and KV-cache policies under runtime pressure, exposing an accuracy-efficiency frontier. |
| [FlashInfer-Bench](https://proceedings.mlsys.org/paper_files/paper/2026/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html) | Evaluates kernel-generating agents using isolated execution, correctness gates, real serving workloads, iterative feedback, and deployment integration. It raises the expected standard for sandboxed agent evaluation. |
| [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://proceedings.mlsys.org/paper_files/paper/2026/hash/bd49b53516ce9ea248fb73522d71a508-Abstract-Conference.html) | Shows multi-agent code optimization on H100 and analyzes explore/exploit and repair behavior. It is evidence that “using multiple agents” alone is not novel. |
| [AccelOpt](https://ppl.stanford.edu/accelopt.html) | Uses beam search and optimization memory in a self-improving kernel agent. It motivates evaluating memory, retries, and accumulated experience. |
| [CORE](https://mlsys.org/media/mlsys-2026/Slides/3814_0uuYG7Z.pdf) | Jointly controls CPU/GPU/memory DVFS for mobile LLM energy-latency optimization; adjacent to the power-control dimension, though not a GPU cluster. |
| [CRAFT](https://mlsys.org/virtual/2026/poster/3508) | Optimizes MoE expert replication under memory constraints, representing another specialized serving policy that a generic tuner must not claim to replace without evidence. |
| [CATWILD](https://mlsys.org/virtual/2026/poster/3551) | Production compiler autotuning demonstrates the value and operational constraints of long-running automatic optimization. |
| [Agentic Operator Generation for ML ASICs](https://proceedings.mlsys.org/paper_files/paper/2026/hash/8c54e9bfed4119c873f575d1d1e2f0a0-Abstract-Conference.html) | Uses linting, JIT, and large correctness suites to gate agent-generated operators. It is relevant to tool safety and invalid-action handling. |
| [MLCommons Chakra](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53fe824f289060ce705ed7c01dae59d2-Abstract-Conference.html) | Standardized traces and replay strengthen the case for publishing a reusable ServeCompass benchmark rather than only a one-off optimizer. |

## 6. Novelty Claims: Unsafe and Defensible

### 6.1 Claims we should not make

The following statements are contradicted by the 2025/2026 record:

- “the first agent for LLM/system optimization”;
- “the first automated tuner for LLM serving”;
- “the first energy-aware controller for LLM inference”;
- “the first Pareto or Bayesian optimizer for LLM serving”;
- “the first simulator-based optimizer of LLM deployment configurations”;
- “the first sandbox for evaluating systems agents.”

We also should not claim that simulated multi-node energy is equivalent to real multi-node energy.

### 6.2 Claims that may be defensible after experiments

Use “to our knowledge” only after a broader, non-MLSys literature audit. The venue-specific evidence currently supports investigating:

1. **Configuration-and-fidelity co-selection.** The policy chooses both \(x_t\), the next serving configuration, and \(f_t\), the evidence fidelity.
2. **Pareto information gain per real GPU-hour.** The acquisition objective values frontier improvement and uncertainty reduction against the actual cost of evidence.
3. **Cross-hardware calibrated transfer.** H100 measurements inform H200/B200 search while preserving architecture-specific uncertainty.
4. **A target-scale verification gate.** No recommended point is called deployment-ready until it has passed a real target-hardware measurement.
5. **A benchmark for agentic energy tuning.** Fixed tasks, hidden replay records, tool contracts, failures, budgets, and live validation make agent methods comparable.

The fifth contribution could be especially valuable. MLSys often rewards a reusable benchmark plus a strong reference system when the benchmark exposes a real systems bottleneck.

## 7. Recommended MLSys System Design

### 7.1 Separate language reasoning from numerical optimization

The LLM should not be trusted to invent numerical acquisition values. A stronger architecture is:

| Component | Responsibility |
|---|---|
| Intent compiler | Converts operator language into objectives, SLO constraints, allowed hardware, budget, and risk tolerance |
| LLM planner | Chooses experiments at a semantic level, narrows or expands the search space, diagnoses failures, and decides when new evidence is needed |
| Multi-fidelity optimizer | Computes constrained Pareto acquisition values and proposes configuration/fidelity candidates |
| Energy Twin | Predicts latency, throughput, and energy with calibrated uncertainty and cross-hardware features |
| Sandbox executor | Replays measured records or runs a simulator without consuming target GPUs |
| Live executor | Launches vLLM/TokenPowerBench trials and collects synchronized latency, throughput, power, and topology data |
| Safety and verification policy | Rejects invalid actions and requires full target-scale validation before release |
| Evidence ledger | Records every observation, cost, provenance, model version, and agent decision |

This hybrid design makes the agent scientifically testable: the numerical optimizer supplies a strong systems baseline, while the agent must demonstrate value in intent translation, search-space adaptation, failure recovery, and evidence planning.

### 7.2 Define fidelity as evidence quality, not GPU generation

A possible hierarchy is:

| Fidelity | Execution | Main purpose | Typical cost |
|---|---|---|---|
| F0 | Analytical model or learned Energy Twin | Eliminate implausible regions and estimate uncertainty | Near-zero target GPU-hours |
| F1 | Short microbenchmark, reduced requests, or isolated phase | Calibrate compute, memory, and token-length effects | Seconds to minutes |
| F2 | Real reduced-scale serving run on one GPU/node | Measure runtime interactions and power behavior | Minutes |
| F3 | Full target hardware, duration, parallelism, and workload | Verify deployment recommendation | Highest cost |

H100, H200, and B200 are hardware domains, not fidelity levels. A B200 simulation is still low-fidelity evidence for a B200 deployment.

### 7.3 What the sandbox can and cannot claim

The sandbox can:

- replay previously measured TokenPowerBench records;
- simulate unmeasured configurations with uncertainty;
- emulate multi-node communication and scheduling effects after calibration;
- inject tool failures, OOMs, noisy power samples, and simulator bias;
- evaluate search policies repeatedly without repeatedly consuming the cluster.

The sandbox cannot by itself prove:

- exact H100/H200/B200 energy;
- NCCL/topology effects on a new cluster;
- node-level power outside the chosen metering boundary;
- thermal drift, contention, or power-capping behavior;
- that a simulated Pareto point satisfies a real deployment SLO.

The paper should therefore present the sandbox as an inexpensive evidence source, not as a replacement for the cluster.

## 8. Experiments Required for MLSys

### 8.1 Hardware and workload matrix

Use H100 as the primary development domain and hold out H200 and B200 for transfer tests.

| Axis | Minimum credible coverage |
|---|---|
| GPU | H100, H200, B200 |
| Scale | 1 GPU, one multi-GPU node, and at least one multi-node or otherwise communication-sensitive case if available |
| Model | One small model that supports broad sweeps, such as Llama-3.1-8B; one TP-required model, such as Qwen2.5-32B or a 70B model |
| Serving engine | One pinned vLLM version; a second engine is optional rather than mandatory |
| Prompt classes | Short/short, long/short, short/long, long/long input-output regimes |
| Arrival process | Steady Poisson-like load and at least one bursty or trace-driven load |
| SLO | TTFT, TPOT/TBT, and goodput or request-completion target |
| Energy boundary | Clearly state GPU-only NVML/DCGM energy versus node/rack energy |

Search knobs should include a meaningful but controllable subset of:

- tensor parallelism and replica count;
- `max_num_seqs`;
- `max_num_batched_tokens`;
- chunked-prefill enablement and chunk size;
- KV-cache utilization/policy;
- power cap or supported frequency control;
- scheduling policy;
- prefill/decode placement when disaggregation is available.

### 8.2 Baseline families

Search algorithms:

| Baseline | Purpose |
|---|---|
| Default vLLM/configuration | Operational reference |
| Random search | Lower-bound search policy |
| TPE/Optuna | OptiKIT-style automatic tuning |
| Constrained qNEHVI | BOute-style multi-objective optimizer |
| Multi-fidelity BO, such as MF-JESMOC where practical | Specialized algorithmic competitor |
| Simulator-only search | Charon/Meta-style cheap exploration without active live measurements |
| Fixed fidelity ladder | Tests whether adaptive fidelity selection matters |
| Exhaustive measured grid or dense offline oracle | Computes frontier regret for tractable subspaces |

Agent ablations:

| Variant | Question answered |
|---|---|
| Optimizer only | Is the LLM necessary? |
| LLM planner only | Does free-form agent reasoning outperform established search? |
| LLM plus optimizer, no fidelity choice | Does the hybrid architecture help? |
| Full ServeCompass | Does joint configuration/fidelity selection help? |
| Full system without cross-hardware prior | Does transfer save measurements? |
| Full system without verification gate | What reliability does the gate add? |

BEAM is a runtime controller rather than the same experimental-search problem. Compare against a BEAM-inspired or available implementation in a dedicated energy-at-SLO experiment, and avoid presenting it as an identical search baseline.

### 8.3 Headline metrics

The primary x-axis should be **cumulative real GPU-hours**, not iteration count.

Report:

- dominated hypervolume versus cumulative real GPU-hours;
- Pareto regret or epsilon-coverage versus the dense measured oracle;
- energy saved at matched TTFT/TPOT SLO and matched goodput;
- joules per output token and total workload energy;
- number of F3 target-scale trials;
- wall-clock tuning time and total experiment cost;
- simulator/twin error, interval coverage, and calibration by fidelity;
- H100-to-H200/B200 transfer gain and negative-transfer rate;
- invalid action, failed trial, retry, and recovery rates;
- agent inference tokens, dollar cost, and decision latency.

At least 10 search seeds are desirable for cheap replay experiments. Live runs can use fewer seeds if the paper reports confidence intervals and separates expensive hardware variance from search-policy variance.

### 8.4 Five central research questions

**RQ1: Search efficiency.**  
How quickly does ServeCompass recover a high-quality energy-latency-throughput frontier under a real GPU-hour budget?

**RQ2: Fidelity choice.**  
Does adaptively choosing the evidence fidelity outperform a fixed simulator-then-live ladder?

**RQ3: Hardware transfer.**  
How much do H100 observations reduce the H200/B200 measurement budget, and when does transfer fail?

**RQ4: Agent value.**  
What does the LLM planner add beyond constrained qNEHVI/TPE in intent translation, search-space revision, and failure recovery?

**RQ5: Reliability.**  
Does uncertainty calibration plus the target-scale verification gate prevent false Pareto recommendations under simulator bias and tool failures?

## 9. Recommended Paper Structure

The MLSys version should be organized around one claim and its evidence:

| Section | Target pages | Purpose |
|---|---:|---|
| Abstract | 0.25 | Problem, insight, system, setup, numerical result |
| 1. Introduction | 0.8 | Measurement-cost gap, thesis, contributions |
| 2. Background and Motivation | 0.8 | TokenPowerBench evidence; why single-fidelity sweeps fail |
| 3. Related Work | 1.0 | BEAM, BOute, PROMPTS, OptiKIT, Meta/Charon, 2025 precedents |
| 4. Problem Formulation | 0.7 | Constrained multi-objective, multi-fidelity, cost-aware objective |
| 5. ServeCompass Design | 1.5 | Agent/optimizer split, evidence ledger, safety and verification |
| 6. Multi-Fidelity Algorithm | 1.2 | Acquisition, uncertainty, transfer model, stopping rule |
| 7. Implementation | 0.6 | vLLM, TokenPowerBench, metering, sandbox/live executors |
| 8. Evaluation | 2.5 | Main results, ablations, calibration, transfer, reliability |
| 9. Discussion/Limitations | 0.35 | Meter boundary, nonstationarity, generality |
| 10. Conclusion | 0.2 | Main result and implication |

The existing architecture figures can be reused after redrawing them around:

1. the configuration-and-fidelity decision loop;
2. the evidence ladder and verification gate;
3. a budget-versus-frontier-quality result;
4. cross-hardware transfer and uncertainty.

## 10. Publication Strategy: SC Workshop Versus MLSys 2027

MLSys 2026 allowed an expanded version of a proceedings workshop paper only with program-chair approval, disclosure of the workshop version, and substantial added novelty. MLSys 2027 may or may not retain that exact rule.

If the SC workshop paper is archival, the safest choices are:

| Strategy | Consequence |
|---|---|
| Prioritize MLSys and do not publish the overlapping full idea at SC | Cleanest novelty and dual-submission position |
| Publish a narrow SC position/prototype paper | Reserve the formal multi-fidelity algorithm, benchmark, cross-hardware study, and live system for MLSys; prepare a precise delta table |
| Publish essentially the current full system at SC and add only more experiments | High risk; more experiments alone may not constitute sufficient added novelty |

Because MLSys 2026 explicitly lists autonomous/agentic systems, LLM inference, and LLM-based system optimization, the topical fit is already excellent. The decision should be driven by novelty preservation, not topic fit.

## 11. Concrete Go/No-Go Criteria

Proceed with an MLSys 2027 Research Track submission if, by the paper freeze, all of the following are true:

- the policy mathematically and operationally chooses both configuration and fidelity;
- at least one optimizer-only baseline is strong and correctly implemented;
- H100/H200/B200 results show either cross-hardware savings or an honest negative result with a useful calibration mechanism;
- every reported deployment recommendation is verified on real target hardware;
- the agent adds measurable value beyond qNEHVI/TPE;
- the paper reports total profiling GPU-hours and agent overhead;
- results include uncertainty across workloads and search seeds;
- related work directly discusses BEAM, BOute, PROMPTS, OptiKIT, Meta, and Charon.

Do not submit the current framing to MLSys if:

- “multi-fidelity” is only a fixed sequence of replay followed by one live run;
- the Energy Twin lacks calibrated error bars;
- the agent simply rewrites an optimizer recommendation in natural language;
- only simulated energy is reported;
- the comparison is limited to default/random/grid;
- the result is a single best configuration rather than a frontier and budget-quality curve.

## 12. Immediate Next Actions

1. Freeze the MLSys problem statement and the exact fidelity ladder.
2. Implement constrained qNEHVI/TPE before adding more LLM logic.
3. Define the agent-versus-optimizer interface and log schema.
4. Build the H100 measured corpus and reserve H200/B200 as transfer domains.
5. Add uncertainty calibration and a mandatory F3 verification gate.
6. Reproduce the strongest comparable settings from BEAM, BOute, and OptiKIT where feasible.
7. Build the replay benchmark so all search policies can run over identical hidden records.
8. Start the MLSys related-work rewrite using the papers in this audit.
9. Recheck the MLSys site weekly for the 2027 CFP and official style package.
10. Create `paper-mlsys27/` only when the official template is available, or use an explicitly labeled 2026-format planning branch that cannot be mistaken for the final template.

## 13. Bottom Line

The project remains publishable and is unusually well matched to MLSys, especially because H100, H200, and B200 are available. The 2026 literature changes the center of gravity: the novelty is no longer that an agent can tune LLM inference or that energy has a Pareto tradeoff. The contribution must be a rigorous decision system that spends real measurements intelligently, transfers evidence across hardware with calibrated uncertainty, and verifies every recommendation on the target cluster.

That is a stronger paper than the current SC formulation, but it is also a much clearer one.
