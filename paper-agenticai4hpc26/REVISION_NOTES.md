# AgenticAI4HPC Revision Notes

## Central Claim

TokenPowerAgent does not use an LLM to predict energy or directly choose a
deployment. It uses a bounded semantic planner to organize an auditable loop in
which deterministic tools jointly select a serving configuration and the next
evidence fidelity under a GPU-hour budget. A recommendation is released only
after full target-workload verification.

## Implemented System

- A typed scenario combines natural-language intent, candidate configurations,
  SLOs, available evidence levels, and a real GPU-hour budget.
- A constrained LLM planner emits one JSON subgoal from a fixed vocabulary and
  falls back to a deterministic planner on malformed output.
- IPIG selects a candidate--stage action using an auditable
  uncertainty-per-cost proxy conditioned on SLO and frontier terms.
- The operational path is CPU Sandbox, short H100 Probe, and full H100 Verify;
  stable artifact identifiers are L0, L1, and L4.
- Failed executions become charged evidence records and drive a REPAIR state.
- A reserved verification budget and independent verifier prevent unmeasured
  recommendations.
- Sealed replay compares IPIG with random, cost-blind information gain, and a
  cheapest-first ladder under matched budgets.

## Measured Foundation

The paper reports 51 blind H100 workload-transfer runs across two disjoint
holdouts. Energy MAPE is 6.23% and 7.35%, with rank correlations of 0.976 and
0.933. A preregistered scope gate separates supported TTFT transfer at
concurrency four from an unsupported sparse-concurrency region.

A separate 12-candidate corpus contains 72 successful Probe/Verify runs. The
Sandbox preserves
energy rank (0.884 Spearman) but reverses TTFT rank (-0.968) and has zero L4
Pareto recall. L1 reaches 1.15% energy MAPE and recovers the true frontier with
100% recall and 33.3% precision. L4 verifies `seq32-bt2048-chunk` as the unique
Pareto point; relative to the expert default it uses 1.04% less energy, delivers
3.06% higher throughput, and reduces TTFT by 21.59%.

## Sealed Policy Result

The frozen 84-row replay corpus remains unchanged. A post-run audit found that
the first runner globally coupled every action to the same repeat index, leaving
only three distinct evidence worlds; that summary is retained for provenance
and excluded from the paper. Commit `c41d8a6` implements action-specific
empirical bootstrap sampling with common random numbers across policies.

Across 500 corrected 20-step episodes per policy, IPIG reaches the oracle in
63.6% of episodes versus 46.2% for random while using 12.7% fewer GPU-hours. It
matches cost-blind information gain's hit rate at 26.6% lower cost. A static
cheapest-first ladder matches IPIG's verified decisions at 4.2% lower cost, so
the paper explicitly declines a universal IPIG-superiority claim. The 500
episodes resample three hardware repeats per candidate--level key; they are not
500 new hardware experiments.

This experiment is intentionally narrower than the eventual MLSys study. TP,
PP, placement, H200/B200 transfer, multi-node fidelity, posterior Pareto
information, and live scheduler integration remain MLSys extensions.

## Claim Guardrails

- Do not call the Docker container itself a simulator; it is the isolated
  execution substrate.
- Do not call Sandbox/L0 output measured or present a single H100 as a cluster
  model.
- Report the mixed policy result: IPIG improves on random and cost-blind but
  does not beat cheapest-first in the measured search space.
- Keep multi-GPU and multi-node mechanisms out of the Workshop paper's central
  design, algorithm, figures, and contributions.
- Do not attribute causal benefit to the LLM without a planner ablation.
