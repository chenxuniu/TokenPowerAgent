# TokenPowerAgent Paper

Overleaf-ready IEEE conference manuscript for AgenticAI4HPC'26 at SC26.

**Title:** *TokenPowerAgent: Agentic Multi-Fidelity Pareto Search for
Energy-Efficient LLM Inference on GPU Clusters*

## Build

Set `main.tex` as the Overleaf main document. For a local build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The manuscript is designed for the workshop's 10-page limit, including
references. Generated files belong in `build/`; `build/main.pdf` is the local
review copy and is intentionally excluded from Git.

## Project Layout

```text
paper-draft/
  main.tex                    IEEE preamble, title, author, and section order
  references.bib              verified bibliography used by the manuscript
  sections/                   one source file per section
    00-abstract-keywords.tex
    01-introduction.tex
    02-motivation.tex
    03-related-work.tex
    04-problem-formulation.tex
    05-design.tex
    06-implementation.tex
    07-evaluation.tex
    08-results.tex
    09-artifact.tex
    10-limitations.tex
    11-conclusion.tex
    99-acknowledgment.tex
  figures/                    standalone TikZ figure sources
    motivation-gap.tex
    system-architecture.tex
    calibration-ladder.tex
    agent-policy.tex
    decision-trace.tex
  tables/                     standalone table sources
    related-work-gap.tex
    evaluation-matrix.tex
    results-template.tex
  results/
    metrics.tex               central location for measured result macros
  REVISION_NOTES.md           review summary and completion checklist
  SUBMISSION_CHECKLIST.md     venue-specific final checks
```

## Editing Workflow

1. Replace the neutral `TBD` macros in `results/metrics.tex` after experiments.
2. Add generated result plots under `figures/` and include them from
   `sections/08-results.tex`.
3. Keep every measured claim traceable to a result macro, table, or plot.
4. Update the abstract and conclusion only after the final experiment table is
   frozen.
5. Confirm the author list, funding, and acknowledgments before submission.
6. Run every item in `SUBMISSION_CHECKLIST.md`, including the 10-page and
   single-blind checks.

The manuscript intentionally contains no invented measurements. Its current
results section is an explicit, publication-ready shell for real data.
