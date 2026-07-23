# TokenPowerAgent Revision Notes

## Editorial Position

The paper is framed around one defensible thesis: agentic multi-fidelity Pareto
search for energy-efficient LLM inference is an evidence-allocation problem,
not merely a prediction problem. TokenPowerAgent therefore decides which
configuration to inspect, which fidelity level is sufficient, and when
expensive cluster measurements are justified.

The revised manuscript follows the visual and rhetorical style of MicroEvo:
an early motivation figure, a compact capability table, a system overview,
mechanism-focused figures, explicit research questions, and a restrained
semantic color palette shared across figures.

## Completed Review

- Replaced the broad agent narrative with a measurement-budgeted search
  formulation and the intent-conditioned Pareto information gain (IPIG)
  objective.
- Aligned the title, abstract, formulation, algorithm, evaluation, and
  conclusion around agentic multi-fidelity Pareto search.
- Made acquisition benefits fidelity-dependent and reserved L4 verification
  budget explicitly in the search algorithm.
- Defined a phase- and communication-aware Energy Twin with uncertainty that
  grows across topology and scale shifts.
- Defined the L0-L4 sandbox ladder from historical replay to target-scale
  verification; the paper explicitly states that a sandbox is not a substitute
  for a real multi-node system.
- Added a typed tool boundary, deterministic feasibility checks, provenance,
  failure semantics, and human approval for expensive jobs.
- Expanded related work across power measurement, inference prediction,
  serving simulation, energy-aware control, and agentic HPC.
- Added an evaluation matrix, baselines, splits, metrics, ablations, and SC26
  artifact/reproducibility requirements.
- Added replay and cluster executors that share one typed evidence interface,
  plus a bounded semantic planner whose output is scored and guarded by code.
- Verified that all cited keys resolve and that the manuscript compiles in ten
  pages without missing references or layout overflows.

## Required Before Submission

- Run the evaluation and replace all macros in `results/metrics.tex`.
- Add parity, calibration, best-so-far, and ablation plots to the Results
  section.
- Report the exact GPU, node, fabric, power-sampling, software, and workload
  versions used by the experiments.
- Freeze the paper's strongest quantitative claim only after target-scale
  verification; then revise the abstract and conclusion consistently.
- Confirm author list, affiliations, funding, acknowledgments, and institutional
  publication clearance.
- Recheck the workshop CFP and IEEE template immediately before submission.

## High-Value Experimental Story

The strongest submission does not need exhaustive 405B coverage. A convincing
story is: broad L0 replay, L1/L2 calibration on accessible hardware, sparse L3
multi-node probes selected by the agent, and L4 verification only for the final
Pareto set. The key comparison is energy/SLO regret versus cumulative GPU-hours,
because that directly tests whether the agent spends measurement budget better
than random search, single- and multi-fidelity multi-objective Bayesian
optimization, and simulator-only ranking.
