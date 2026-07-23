# AgenticAI4HPC'26 Submission Checklist

Last verified: July 22, 2026. Recheck the [official workshop call](https://ornl.github.io/events/agenticai4hpc2026/)
before submission because venue instructions can change.

## Fixed Venue Requirements

- [x] Target: AgenticAI4HPC'26, co-located with SC26.
- [x] Deadline recorded as August 7, 2026, Anywhere on Earth.
- [x] Full-paper limit recorded as 10 pages **including references**.
- [x] IEEE conference template, two columns, US letter, conference mode.
- [x] Single-blind review: author names and affiliations are present.
- [x] Reproducibility and artifact-evaluation plans are discussed.

## Must Complete Before Upload

- [ ] Replace every `TBD` macro in `results/metrics.tex` with measured results.
- [ ] Add parity/calibration, quality-versus-GPU-hour, and ablation plots.
- [ ] Re-run all L4 recommendations independently and report variance.
- [ ] Freeze hardware, topology, CUDA, NCCL, serving-engine, model, and
      TokenPowerBench versions.
- [ ] Check that every quantitative abstract/conclusion claim points to a table
      or figure.
- [ ] Confirm the exact spelling and order of all authors.
- [ ] Confirm all author email addresses; the current addresses are placeholders
      following the TTU `firstname.lastname` pattern until verified.
- [ ] Add grant numbers and acknowledgments after internal clearance.
- [ ] Run the IEEE PDF checker required by the submission site, if provided.
- [ ] Confirm embedded fonts, searchable text, US-letter page size, and no page
      numbers or headers added outside `IEEEtran`.

## Mechanical Preflight

Run from `paper-draft/`:

```bash
latexmk -C -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdfinfo build/main.pdf | rg '^Pages:|^Page size:'
rg -n 'TBD|draftvalue' results sections main.tex
rg -n 'undefined|Overfull|LaTeX Warning|Package .* Warning' build/main.log
```

The final PDF must report 10 pages or fewer. A clean warning search should
produce no unresolved citations, references, or overfull boxes.
