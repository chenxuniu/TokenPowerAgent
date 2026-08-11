# MLSys 2027 Submission Checklist

Last checked: August 3, 2026. MLSys 2027 requirements are not yet public. The
items marked provisional come from the official [MLSys 2026 research-paper
call](https://mlsys.org/Conferences/2026/CallForResearchPapers) and style
package; replace them when the 2027 call is released.

## Venue and Format

- [ ] Confirm the MLSys 2027 deadline, research track, page limit, and template.
- [x] Provisional format uses the official MLSys 2026 two-column style.
- [x] Review mode uses `\usepackage{mlsys2026}` without `[accepted]`.
- [x] Author and affiliation output is suppressed for double-blind review.
- [x] Acknowledgments are excluded from the review PDF.
- [x] Abstract is one paragraph with a quantified, scope-bounded pilot result.
- [x] Keep the body within 10 pages; pages after the limit contain only
      references or permitted appendices under the final 2027 policy.
- [ ] List all authors in every reference and use complete APA-style metadata.
- [x] Cite prior TokenPowerBench work in the third person.
- [x] Remove identifying repository, institution, grant, and cluster names.

## Scientific Completion Gates

- [ ] Replace every `\draftvalue` in `results/metrics.tex` with measured output.
- [x] Complete a three-repeat calibration and three-repeat blind H100 holdout;
      keep the prediction freeze and validation report hashes.
- [x] Freeze the six-workload H100 validation campaign before measurement.
- [ ] Run all 18 measurements in the frozen validation campaign, calibrate
      intervals, and reserve a new untouched holdout.
- [ ] Add a pilot motivation figure showing configuration-rank changes and
      cross-fidelity error; the current motivation figure is schematic.
- [ ] Implement and test nested-posterior IPIG rather than the current
      uncertainty proxy.
- [ ] Implement the real LLM planner and retain the rule-based planner as a
      baseline.
- [ ] Calibrate the Energy Twin with uncertainty coverage on held-out L4 runs.
- [ ] Complete dense oracle subspaces and replay episodes for every search
      baseline.
- [ ] Run H100, H200, and B200 campaigns, including sparse multi-node L3/L4
      verification where available.
- [ ] Run optimizer-only, LLM-only, failure-recovery, uncertainty, IPIG, and
      verification-gate ablations.
- [ ] Re-run every released recommendation independently at L4 and retain all
      SLO failures.
- [ ] Report model-quality constraints for every precision-changing result.

## Reproducibility and Claims

- [x] Freeze the pilot GPU, driver/CUDA, vLLM, model, container, serving knobs,
      telemetry boundary, split, and source hashes.
- [ ] Freeze GPU/node/fabric, NCCL, TokenPowerBench, and all remaining full-
      campaign versions.
- [ ] Report repetitions, randomized run order, bootstrap intervals, failed
      jobs, and paired tests.
- [ ] Distinguish GPU energy, node energy, and facility energy in every claim.
- [ ] Map every headline number to a table/figure and raw artifact record.
- [ ] Prepare an anonymous artifact archive with replay instructions and result
      regeneration scripts.
- [x] Add the frozen pilot result to the abstract without promoting it to a
      target-scale or Pareto claim.
- [ ] Replace the pilot result with final matched-budget Pareto and deployment
      results after L4 freeze.

## Mechanical Preflight

Run from `paper-draft/`:

```bash
latexmk -C -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdfinfo build/main.pdf | rg '^Pages:|^Page size:'
rg -n 'draftvalue|Anonymous Institution|Anonymous Author' results sections main.tex
rg -n 'undefined|Overfull|LaTeX Warning|Package .* Warning' build/main.log
pdffonts build/main.pdf
```

The final warning search must contain no unresolved citations, references, or
overfull boxes. Confirm that PDF metadata is anonymous and that all fonts are
embedded before upload.

Current preflight: 10 body pages plus 3 reference pages, no overfull boxes or
undefined citations/references, and all PDF fonts embedded.
