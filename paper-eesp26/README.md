# TokenPowerSandbox: SC EESP 2026 WIP Paper

This directory contains the self-contained IEEE source for a four-page body
plus references. It is intentionally narrower than the MLSys-oriented
TokenPowerAgent paper.

Current title: **TokenPowerSandbox: Evidence-Gated Multi-Fidelity Screening for
Energy-Efficient LLM Inference on HPC Clusters**.

## Core Claim

TokenPowerSandbox is a multi-fidelity evidence workflow, not a Docker image or
a virtual GPU cluster. It combines:

1. A CPU-resident workload projector with a sparse measured residual.
2. A restricted real-H100 measurement executor.
3. A scope and provenance gate that requires independent validation and can
   abstain when a preregistered accuracy rule fails.

The current paper tests one bounded claim: whether sparse H100 measurements can
screen unseen context-length/concurrency workloads under one fixed serving
configuration.

## Evidence in This Draft

The sealed study uses Qwen2.5-7B-Instruct, vLLM 0.23.0, BF16, and one H100 80GB
at 700 W:

- 3 anchor measurements at 512 input / 128 output tokens, concurrency 8.
- 6 development workloads x 3 repeats to fit a four-feature log residual.
- 8 disjoint holdout workloads frozen before 24 real H100 measurements.
- A second 9-workload confirmation frozen without refitting before 27 more
  H100 measurements.
- All 51 blind runs succeeded; the two raw manifests verify 62 and 61 files.
- Energy MAPE: 63.01% analytical base to 6.23% corrected.
- Energy ranking: Spearman rho 0.976 and 27/28 correctly ordered pairs.
- Corrected throughput/TTFT/TPOT MAPE: 10.53% / 32.67% / 7.99%.
- No-refit confirmation energy MAPE: 7.35%, passing the preregistered 15%
  endpoint with Spearman rho 0.933 and 33/36 ordered pairs.
- Confirmation TTFT MAPE: 9.27% at concurrency 4, but 64.80% at concurrency 1
  or 2; the preregistered decision releases TTFT at 4 and abstains below 4.

The paper does not claim configuration ranking, calibrated interval coverage,
TP/PP fidelity, H100/H200/B200 transfer, or deployment energy savings.

Final hashes and chronology are stored in `results/holdout_v2_provenance.json`
and `results/scope_v3_provenance.json`. Readable reports and both complete
sealed archives are versioned under `../experiments/evidence/eesp26/`. Do not
use either holdout to refit the reported model.

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
figures/                          workflow and scope-gate TikZ figures
tables/                           holdout, confirmation, and claim-ladder tables
results/metrics.tex               centralized headline values
results/holdout_v2_provenance.json final hashes, chronology, and metrics
results/scope_v3_provenance.json   confirmation decision and hashes
analysis/check_submission.py      source/PDF consistency gate
```

## Scope Boundary

Intent parsing, adaptive multi-fidelity Pareto search, agent ablations, serving
configuration search, and multi-GPU cluster optimization belong to subsequent
TokenPowerAgent stages. The WIP establishes the evidence substrate and its
same-H100 workload-transfer boundary.
