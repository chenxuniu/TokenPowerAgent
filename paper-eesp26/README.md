# TokenPowerSandbox: SC EESP 2026 WIP Paper

This directory contains the self-contained IEEE source for a four-page body
plus references. It is intentionally narrower than the MLSys-oriented
TokenPowerAgent paper.

Current title: **TokenPowerSandbox: Evidence-Gated Workload Transfer for
Energy-Efficient LLM Inference (Work in Progress)**.

## Core Claim

TokenPowerSandbox is a multi-fidelity evidence workflow, not a Docker image or
a virtual GPU cluster. It combines:

1. A restricted real-H100 measurement executor.
2. An analytical workload projector with a sparse measured residual.
3. A scope and provenance gate that requires independent validation.

The current paper tests one bounded claim: whether sparse H100 measurements can
screen unseen context-length/concurrency workloads under one fixed serving
configuration.

## Evidence in This Draft

The sealed study uses Qwen2.5-7B-Instruct, vLLM 0.23.0, BF16, and one H100 80GB
at 700 W:

- 3 anchor measurements at 512 input / 128 output tokens, concurrency 8.
- 6 development workloads x 3 repeats to fit a four-feature log residual.
- 8 disjoint holdout workloads frozen before measurement.
- 24 blind H100 measurements, all successful, with 62 manifest-verified raw
  and configuration artifacts.
- Energy MAPE: 63.01% analytical base to 6.23% corrected.
- Energy ranking: Spearman rho 0.976 and 27/28 correctly ordered pairs.
- Corrected throughput/TTFT/TPOT MAPE: 10.53% / 32.67% / 7.99%.
- Median energy repeat CV: 0.78%.

The paper does not claim configuration ranking, calibrated interval coverage,
TP/PP fidelity, H100/H200/B200 transfer, or deployment energy savings.

The final result hashes and chronology are stored in
`results/holdout_v2_provenance.json`. The sealed local archive has SHA-256
`2d74179245392d44f2efd2fbc5601b4721e278f206e45970c5953481e12ecafb`.
Do not use the final holdout to refit the reported model.

## Build and Check

```bash
make pdf
make check
```

The PDF is written to `build/main.pdf`. The expected layout is four body pages
and a fifth references page.

## Layout

```text
sections/                         one LaTeX file per section
figures/                          workflow and holdout-error TikZ figures
tables/                           aggregate, point-level, and claim-ladder tables
results/metrics.tex               centralized headline values
results/holdout_v2_provenance.json final hashes, chronology, and metrics
analysis/check_submission.py      source/PDF consistency gate
```

## Scope Boundary

Intent parsing, adaptive multi-fidelity Pareto search, agent ablations, serving
configuration search, and multi-GPU cluster optimization belong to subsequent
TokenPowerAgent stages. The WIP establishes the evidence substrate and its
same-H100 workload-transfer boundary.
