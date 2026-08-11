# ServeCompass Paper

Overleaf-ready research-paper draft targeting MLSys 2027.

**Title:** *ServeCompass: Agentic Multi-Fidelity Discovery of Verified LLM
Deployment Profiles*

MLSys 2027 had not published its call or style files as of August 11, 2026. This
repository therefore uses the official MLSys 2026 double-blind style as a
provisional format. Replace `mlsys2026.sty` and `mlsys2026.bst` when the 2027
package is released, then recheck every venue requirement.

## Build

Set `main.tex` as the Overleaf main document. For a local build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The provisional 2026 research-paper limit is 10 body pages, excluding
references and appendices. The review style is double blind and suppresses the
author block. Generated files belong in `build/`; `build/main.pdf` is excluded
from Git.

The current build uses exactly 10 body pages; references begin on page 11.

## Current Evidence Snapshot

The manuscript now reports one real, protocol-frozen H100/Qwen2.5-7B pilot:

- three L1 calibration repeats at 512 input tokens, 128 output tokens, and
  concurrency 8;
- one L0 prediction frozen before three blind L1 repeats at 1,024 input tokens,
  128 output tokens, and concurrency 16;
- 25.0% MAPE across GPU energy, output throughput, P95 TTFT, and P95 TPOT;
- 0.34--3.80% observed CV and 4/4 diagnostic interval containment.

This is evidence for protocol feasibility only. It is not yet evidence for
multi-node fidelity, Pareto-search efficiency, agent advantage, or deployment
energy savings. A six-workload, 18-run validation campaign is frozen and is the
next measurement step.

## Project Layout

```text
paper-draft/
  main.tex                    MLSys preamble, title, and section order
  mlsys2026.sty               official provisional review style
  mlsys2026.bst               official provisional bibliography style
  references.bib              bibliography used by the manuscript
  sections/                   one source file per section
  figures/                    standalone TikZ figure sources
  tables/                     standalone table sources
    pilot-validation.tex      current blind H100 pilot result
  results/
    metrics.tex               pilot values plus unresolved campaign values
  REVISION_NOTES.md           scientific review and remaining gaps
  SUBMISSION_CHECKLIST.md     MLSys-specific preflight
```

## Editing Workflow

1. Keep the central claim fixed: ServeCompass co-selects configuration and
   evidence fidelity to maximize decision information per real GPU-hour.
2. Replace every unresolved `\draftvalue` macro only from validated experiment
   output; the `Pilot*` macros already map to the frozen validation report.
3. Add the motivation, calibration, GPU-hour search, deployment, and ablation
   plots under `figures/` and include them from the appropriate section.
4. Keep every quantitative abstract and conclusion claim traceable to one
   figure, table, and raw-result artifact.
5. Preserve double-blind wording, third-person self-citations, and anonymous
   artifact links for review.
6. Re-run `SUBMISSION_CHECKLIST.md` after the official MLSys 2027 call appears.

The manuscript contains no invented measurements. Pilot values are rendered in
the paper; unresolved full-campaign macros remain centralized but unused until
the corresponding H100/H200/B200 result is frozen.
