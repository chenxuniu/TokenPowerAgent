# AgenticAI4HPC 2026 Submission Checklist

Last checked against the official Workshop page: August 4, 2026.

## Venue and Format

- [x] Target: AgenticAI4HPC 2026, co-located with SC26.
- [x] Deadline: August 7, 2026 (AoE).
- [x] Standard IEEE conference format, two columns, US letter.
- [x] At most 10 pages including references; current PDF is exactly 10 pages.
- [x] No unresolved citations, references, overfull boxes, or clipped figures.
- [ ] Replace `Anonymous Authors` with the final author and affiliation block.
- [ ] Confirm title, author order, funding text, and conflicts in the submission
      system.
- [ ] Decide whether to include an optional SC-style AD/AE appendix; it must not
      push the main manuscript beyond the permitted format.

## Final Experiment Before Submission

- [x] Freeze a single-H100 serving-configuration corpus with 12 feasible
      candidates and three randomized repeats per candidate.
- [x] Hold workload, model revision, vLLM image, power limit, prefix caching,
      precision, TP, and PP fixed while varying serving knobs.
- [x] Preserve every raw measurement, failed run, telemetry path, seed, and
      artifact hash.
- [ ] Run IPIG, random, cost-blind information gain, and cheapest-first replay
      with identical priors, candidate corpus, seeds, and GPU-hour budgets.
- [ ] Populate policy metrics only from the sealed `benchmark-replay` report.
- [x] Replace every visible result placeholder and re-read the abstract, RQ3 results,
      limitations, and conclusion for claim consistency.

The 72-run corpus now establishes the L0/L1/L4 fidelity result and the verified
frontier. The running CPU replay is the remaining policy-comparison step.

## Reproducibility Gates

- [x] H100 evidence uses frozen predictions, preregistered thresholds, three
      repeats, raw-artifact manifests, and nested SHA-256 provenance.
- [x] The runtime records planner source, selected candidate and fidelity,
      expected and realized cost, uncertainty/frontier snapshots, failure
      reason, and fallback behavior.
- [x] L4 verification is deterministic and cannot be bypassed by the LLM.
- [x] All 69 automated tests pass.
- [ ] Archive the final configuration corpus and benchmark report locally and
      download a checksum-matched copy from the cluster.
- [ ] Pin the final Git commit and container/model digests in the artifact text.
- [ ] Prepare an optional artifact archive with replay commands and no private
      cluster credentials or identifying paths.

## Mechanical Preflight

Run from `paper-agenticai4hpc26/`:

```bash
latexmk -C -outdir=output/pdf main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf main.tex
pdfinfo output/pdf/main.pdf | rg '^Pages:|^Page size:'
pdftotext output/pdf/main.pdf - | rg -n 'TBD|Anonymous Authors'
rg -n 'Overfull|Undefined control|LaTeX Warning: (Citation|Reference)' \
  output/pdf/main.log
pdffonts output/pdf/main.pdf
```

The final placeholder scan must be empty. Confirm all fonts are embedded and
perform one last visual inspection of all 10 rendered pages.
