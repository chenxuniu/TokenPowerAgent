# ServeCompass: MLSys 2027 Revision Notes

## Core Thesis

ServeCompass is not another energy predictor, generic MOBO tuner, or LLM
configuration recommender. Its defensible claim is:

> Existing systems optimize which serving configuration to deploy or how to
> control it at runtime. ServeCompass optimizes which evidence to purchase
> next, jointly choosing a configuration and measurement fidelity under a real
> GPU-hour budget, and requires target-scale measurement before releasing
> SLO-constrained Eco, MaxGoodput, LowLatency, or Balanced deployment profiles.

Everything in the paper should support or test that sentence.

## Style Learned from MLSys 2025 and 2026

The closest successful papers use a compact claim-evidence flow:

1. **Concrete system tension.** BEAM starts from latency slack and coupled
   batching/DVFS; BOute starts from coupled routing/deployment; SOLA starts from
   measurable TTFT/TPOT imbalance.
2. **Two or three observations.** The introduction and motivation quantify why
   a simpler policy fails before presenting architecture.
3. **One mechanism per observation.** Every named component has an explicit
   reason to exist and an ablation.
4. **Validation before optimization.** Charon, Lumos, and Meta's deployment
   study validate model/simulator fidelity before using it for search.
5. **Matched-resource comparisons.** Search papers compare under identical
   budgets or constraints, not only against defaults.
6. **Agents as systems.** PROMPTS and AIOpsLab evaluate proposal quality,
   validity, failures, and operational outcomes rather than showing anecdotal
   conversations.
7. **Result-first abstracts.** The current abstract reports the blind pilot;
   the final version must replace that WIP result with scale, matched baselines,
   and verified Pareto/deployment outcomes.

The full literature and writing-flow study is in
`../docs/MLSys-2025-2026-Literature-and-Writing-Flow.md`.

## Completed in This Polish Round

- Migrated the paper from IEEE/SC workshop format to the official MLSys 2026
  double-blind style as a provisional MLSys 2027 format.
- Rewrote the abstract around the method and the first measured pilot without
  implying that the full campaign is complete.
- Reframed the introduction around configuration--fidelity co-selection and
  explicitly separated ServeCompass from BEAM, BOute, PROMPTS, Charon,
  OptiKIT, and Meta's deployment study.
- Reorganized motivation into three observations and four design requirements.
- Corrected the formulation: the returned measured frontier is an estimate of
  the unknown true frontier, not an unjustified subset of it.
- Added a budgeted terminal decision-loss objective and quality constraints for
  precision-changing configurations.
- Rebuilt evaluation around five research questions, strong operational and
  algorithmic baselines, oracle and deployment spaces, cross-hardware transfer,
  agent ablations, and matched-SLO L4 verification.
- Added the protocol-frozen H100/Qwen2.5-7B blind pilot to the abstract,
  evaluation, results, limitations, and conclusion from one shared metric file.
- Added a measured-versus-predicted table with medians, MAD, APE, and diagnostic
  interval containment for energy, throughput, P95 TTFT, and P95 TPOT.
- Aligned the implementation section with the code: persistent vLLM sandbox,
  FIFO measurement boundaries, calibration builder, campaign freeze, and blind
  validator are implemented; IPIG, the production planner, and live L2--L4
  execution remain explicit WIP.
- Compressed the body to exactly 10 pages; references begin on page 11.
- Added verified MLSys 2025 references for ThunderServe, SOLA, Lumos, and
  AIOpsLab, and moved Related Work after Results to preserve the systems flow.

## Current Submission Blockers

### 1. RQ1 still has only one workload-transfer point

The pilot proves protocol feasibility but not statistical coverage. Run the
already-frozen six-workload validation campaign (18 L1 measurements), calibrate
interval widths, and reserve a new untouched holdout. A competitive empirical
motivation figure still needs configuration-rank changes and evidence error as
workload, hardware, and topology shift.

### 2. The implementation does not yet match the paper method

- `IPIGPolicy` currently uses an auditable uncertainty proxy, not the nested
  posterior Pareto-information estimator in the paper.
- The default planner is `RuleBasedPlanner`; no production LLM planner is wired
  into the controller.
- `SimpleEnergyTwin` now supports calibrated L0/L1 workload transfer but is not
  yet the phase-residual, conformal cross-fidelity model in the design.
- `ClusterExecutor` renders Slurm jobs, but live submission, monitoring,
  TokenPowerBench ingestion, and retry handling still require integration.
- The scenario schema accepts compiled candidate lists; automatic compilation
  from a high-level knob space is not complete.
- The persistent single-GPU vLLM executor, immutable calibration profile,
  campaign freezer, and blind validator are implemented and measured.
- A serving-simulator adapter, L2/L3 topology calibration, and cross-hardware
  transfer pipeline are still missing.

These are paper-blocking engineering tasks, not artifact polish.

### 3. The strongest baselines are not optional

Naive defaults alone will not support an MLSys claim. At minimum run random-L4,
TPE-L4, constrained qNEHVI-L4, one generic multi-fidelity MOBO method, a fixed
fidelity ladder, simulator-only ranking, optimizer-only, and LLM-only. The vLLM
default and expert configuration remain deployment baselines, not the main
algorithmic comparison.

### 4. The H100/H200/B200 story must isolate transfer

Using three GPU generations is valuable only if the paper explains what moves
between them. Report cold start versus warm start, same-hardware scale transfer,
and cross-hardware transfer separately. Never imply that H100 power or
interconnect coefficients directly predict B200 without calibration.

### 5. The agent contribution may be negative

The optimizer-only controller could match or beat the LLM planner on ordinary
episodes. That result is acceptable if the LLM measurably improves intent
translation, failure repair, or human escalation. Do not claim that "agentic"
itself improves Pareto search; test exactly where semantics adds value.

### 6. The final result narrative is still open

The paper now has a real pilot result: 25.0% four-metric MAPE, 38.4% energy APE,
0.34--3.80% observed CV, and 4/4 diagnostic interval containment. These are not
the final MLSys headlines. Search curves, calibrated coverage, Pareto recall,
GPU-hour savings, agent ablations, and independent L4 deployment results remain
missing.

## Recommended Evidence Order

1. Run the frozen six-workload H100 validation campaign and calibrate intervals.
2. Reserve and run a final untouched workload holdout.
3. Run one dense H100 7B/8B configuration grid to build an oracle frontier and
   debug random, TPE, qNEHVI, and fixed-ladder baselines.
4. Implement nested-posterior IPIG, the LLM planner, and live L2--L4 execution.
5. Add multi-GPU H100 topology probes, then H200/B200 transfer campaigns.
6. Run agent/failure ablations in replay and representative live episodes.
7. Freeze independent L4 recommendations before replacing the pilot abstract
   sentence with Pareto/GPU-hour and matched-SLO deployment results.

## Current Readiness

- **Problem framing and literature:** strong draft.
- **Method specification:** strong but ahead of the code.
- **Format and paper organization:** provisional MLSys-ready.
- **Implementation:** robust L0/L1 path; adaptive agent and L2--L4 path pending.
- **Experiments and results:** one blind pilot complete; final claims pending.

The main rejection risk is no longer an unclear idea. It is evidence: reviewers
must see that configuration--fidelity co-selection beats strong multi-fidelity
optimization per real GPU-hour and that the LLM planner contributes something
measurable beyond the optimizer.
