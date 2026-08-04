# AgenticAI4HPC Revision Notes

## Central Claim

TokenPowerAgent does not use an LLM to predict energy or directly choose a
deployment. It uses a bounded semantic planner to organize an auditable loop in
which deterministic tools jointly select a serving configuration and the next
evidence fidelity under a GPU-hour budget. A recommendation is released only
after target-scale verification.

## Implemented System

- A typed scenario combines natural-language intent, candidate configurations,
  SLOs, available evidence levels, and a real GPU-hour budget.
- A constrained LLM planner emits one JSON subgoal from a fixed vocabulary and
  falls back to a deterministic planner on malformed output.
- IPIG selects a candidate--fidelity action using an auditable
  uncertainty-per-cost proxy conditioned on SLO, frontier, and topology terms.
- L0 uses replay or the CPU-resident Energy Twin; L1--L4 retain explicit
  measured-fidelity semantics and immutable provenance.
- Failed executions become charged evidence records and drive a REPAIR state.
- A reserved verification budget and independent verifier prevent unmeasured
  recommendations.
- Sealed replay compares IPIG with random, cost-blind information gain, and a
  cheapest-first ladder under matched budgets.

## Measured Foundation

The paper reports 51 blind H100 runs across two disjoint holdouts. Energy MAPE
is 6.23% and 7.35%, with rank correlations of 0.976 and 0.933. A preregistered
scope gate correctly separates supported TTFT transfer at concurrency four
from an unsupported sparse-concurrency region. These results validate the
agent's cheap-evidence substrate and abstention mechanism, not search
superiority.

## Remaining Workshop Experiment

RQ3 needs one frozen single-H100 configuration corpus. Hold workload fixed and
vary serving knobs that can be changed on the available node: maximum sequences,
maximum batched tokens, and chunked prefill. Measure each feasible candidate
three times in randomized order, then replay every acquisition policy on the
same sealed rows. The headline result is Pareto recall per GPU-hour, supported
by primary-objective regret, time to first oracle hit, and unnecessary L4
actions.

This experiment is intentionally narrower than the eventual MLSys study. TP,
PP, placement, H200/B200 transfer, multi-node fidelity, nested-posterior IPIG,
and live scheduler integration remain MLSys extensions.

## Claim Guardrails

- Do not call the Docker container itself a simulator; it is the isolated
  execution substrate.
- Do not call L0 output measured or present a single H100 as a cluster model.
- Do not claim IPIG wins until the frozen corpus produces that result.
- Do not infer multi-node energy, communication, or scheduler behavior from
  the current single-GPU measurements.
- Do not attribute causal benefit to the LLM without a planner ablation.
