# TokenPowerAgent: AgenticAI4HPC 2026 Paper

This directory contains the IEEE-format Workshop paper:

> **TokenPowerAgent: Evidence-Gated Multi-Fidelity Tuning for LLM Serving on
> Single-GPU HPC Nodes**

The target is the [AgenticAI4HPC 2026
Workshop](https://ornl.github.io/events/agenticai4hpc2026/), co-located with
SC26. The deadline is August 7, 2026 (AoE). Full papers may use up to 10 pages,
including references, in IEEE conference format.

This paper is intentionally distinct from `../paper-draft/`, the longer-term
MLSys manuscript. The Workshop paper contributes the implemented bounded agent
runtime, its three-stage single-GPU evidence path, and a sealed H100
configuration corpus. It does not claim multi-GPU, multi-node, or cross-hardware
accuracy.

## Build

Set `main.tex` as the Overleaf main document. Locally:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build main.tex
```

The verified PDF is 10 US-letter pages, including references. Build products
under `build/` and `output/` are ignored by Git.

## Evidence Status

The current draft contains two completed, preregistered H100/Qwen2.5-7B
workload-transfer holdouts:

- 8 workloads and 24 blind measurements: 6.23% energy MAPE and 0.976
  Spearman rank correlation;
- 9 workloads and 27 blind measurements: 7.35% energy MAPE and 0.933
  Spearman rank correlation;
- a preregistered latency release gate that supports TTFT at concurrency 4
  (9.27% MAPE) and abstains below concurrency 4 (64.80% MAPE).

These 51 runs validate the CPU-first workload model and release gate. A separate
12-candidate configuration corpus contains 72 balanced Probe/Verify measurements and
190 hash-verified raw artifacts. L0 preserves energy rank (0.884 Spearman) but
misses the L4 frontier; L1 obtains 1.15% energy MAPE and 100% frontier recall;
L4 verifies `seq32-bt2048-chunk` as the unique SLO-feasible Pareto point. The
locked winner is then evaluated in five new paired, alternating Verify repeats
without refitting or reselection. It wins all five energy pairs, saves 1.39%
energy on average (95% CI [1.19%, 1.59%]), and lowers median paired TTFT by
21.43%; the confirmation report is publication-ready with no warnings. The
corrected 500-episode-per-policy empirical replay reaches that oracle in 63.6%
of IPIG episodes versus 46.2% for random while using 12.7% fewer GPU-hours.
Cost-blind information gain matches IPIG's hit rate at 26.6% greater cost;
cheapest-first also matches its decisions and is 4.2% cheaper. The replay
episodes bootstrap three measured repeats per candidate--level key and are not
additional hardware measurements.

A five-budget response study contributes 10,000 matched replay episodes. IPIG's
normalized success AUC is 39.8% versus 20.4% for random; cheapest-first reaches
42.9% and is explicitly reported as the strongest policy in this small grid.
Two planner protocols add 90 calls each. On the disjoint V2 holdout, raw
subgoal accuracy is 73.3%, below the frozen 90% gate, while a deterministic
state guard intervenes on 26.7% of calls and admits the expected subgoal on all
calls. This is evidence for guarded autonomy, not standalone LLM reliability.

## Directory Layout

```text
paper-agenticai4hpc26/
  main.tex                 IEEE conference entry point
  references.bib           verified bibliography
  sections/                one source file per section
  figures/                 TikZ system figures and vector result panels
  tables/                  scope and exact-value result tables
  results/metrics.tex      single source for measured values
  results/*summary.json    compact sealed controller evidence
  results/PROVENANCE.md    source-archive hashes and cross-checks
  SUBMISSION_CHECKLIST.md  final scientific and format gates
  REVISION_NOTES.md        claim boundary and experiment priorities
```

## Result-Figure Policy

The 10-page paper uses plots for trends and decision geometry, while retaining
tables where exact values or release thresholds matter:

- `fig4_parity.pdf` replaces the former evidence-summary table and exposes all
  17 blind workload predictions, uncertainty intervals, and the frozen gate;
- `fig6a_rank_inversion.pdf` and `fig6b_objective_space.pdf` form the central
  configuration-fidelity figure, paired with the compact verified-value table;
- `fig7a_budget_response.pdf` and `fig8a_planner_accuracy.pdf` report the two
  non-scalar agent results in the main paper;
- `fig7b_gpu_hours_spent.pdf` and `fig8b_planner_latency.pdf` remain available
  as supplementary panels, since their exact scalar results are clearer in the
  main text and adding them would exceed the page limit.

The scope-gate table is retained because it gives exact preregistered strata
and avoids the annotation collision in the alternative scope plot.  The GEMM
power-cap plot is not a main-paper result: it validates instrumentation but does
not answer the configuration-search research questions.

## Editing Rules

1. Preserve the evidence boundary: Sandbox/L0 is simulated, Probe/L1 is
   measured but provisional, and recommendations require successful full
   Verify/L4 evidence.
2. Populate agent-search macros only from the sealed H100 corpus and replay
   report. Never use the synthetic three-candidate demo as a paper result.
3. Keep all policies on the same candidate corpus, seed set, and GPU-hour
   budget.
4. Report failures and charged GPU-hours; failed actions cannot disappear from
   accounting.
5. Report raw planner proposals separately from guard-admitted subgoals; never
   attribute deterministic guard correctness to the LLM.
6. Replace `Anonymous Authors` before submission. The Workshop page does not
   request anonymous review.
