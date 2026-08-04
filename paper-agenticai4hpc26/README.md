# TokenPowerAgent: AgenticAI4HPC 2026 Paper

This directory contains the IEEE-format Workshop paper:

> **TokenPowerAgent: Evidence-Gated Experiment Planning for Energy-Efficient
> LLM Inference on HPC Clusters**

The target is the [AgenticAI4HPC 2026
Workshop](https://ornl.github.io/events/agenticai4hpc2026/), co-located with
SC26. The deadline is August 7, 2026 (AoE). Full papers may use up to 10 pages,
including references, in IEEE conference format.

This paper is intentionally distinct from `../paper-draft/`, the longer-term
MLSys manuscript. The Workshop paper contributes the implemented bounded agent
runtime and its evidence-gating protocol. It does not claim multi-node accuracy
or search superiority before those experiments exist.

## Build

Set `main.tex` as the Overleaf main document. Locally:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf main.tex
```

The verified PDF is 10 US-letter pages, including references. Build products
under `build/` and `output/` are ignored by Git.

## Evidence Status

The current draft contains two completed, preregistered H100/Qwen2.5-7B
holdouts:

- 8 workloads and 24 blind measurements: 6.23% energy MAPE and 0.976
  Spearman rank correlation;
- 9 workloads and 27 blind measurements: 7.35% energy MAPE and 0.933
  Spearman rank correlation;
- a preregistered latency release gate that supports TTFT at concurrency 4
  (9.27% MAPE) and abstains below concurrency 4 (64.80% MAPE).

These 51 runs validate the CPU-first evidence substrate and release gate. They
do not compare acquisition policies. The only unresolved paper values are the
RQ3 configuration-search results in `results/metrics.tex` and
`tables/agent-results.tex`.

## Directory Layout

```text
paper-agenticai4hpc26/
  main.tex                 IEEE conference entry point
  references.bib           verified bibliography
  sections/                one source file per section
  figures/                 standalone TikZ system figures
  tables/                  evidence, scope, and search tables
  results/metrics.tex      single source for measured values
  results/PROVENANCE.md    source-archive hashes and cross-checks
  SUBMISSION_CHECKLIST.md  final scientific and format gates
  REVISION_NOTES.md        claim boundary and experiment priorities
```

## Editing Rules

1. Preserve the evidence boundary: L0 is simulated, L1--L4 are measured, and
   recommendations require successful L4 evidence.
2. Populate agent-search macros only from the sealed H100 corpus and replay
   report. Never use the synthetic three-candidate demo as a paper result.
3. Keep all policies on the same candidate corpus, seed set, and GPU-hour
   budget.
4. Report failures and charged GPU-hours; failed actions cannot disappear from
   accounting.
5. Replace `Anonymous Authors` before submission. The Workshop page does not
   request anonymous review.
