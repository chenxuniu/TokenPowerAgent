# EESP 2026 H100 Evidence

This directory publishes the sealed evidence used by the TokenPowerSandbox
work-in-progress paper. The experiment uses one NVIDIA H100 80GB HBM3 GPU,
Qwen2.5-7B-Instruct, vLLM 0.23.0, BF16, and a fixed 700 W power limit.

## Evidence Sets

| Set | Frozen workloads | Real H100 runs | Model update after freeze | Energy MAPE | Purpose |
| --- | ---: | ---: | --- | ---: | --- |
| Holdout v2 | 8 | 24 | None | 6.23% | Final blind workload-transfer holdout |
| Scope confirmation v3 | 9 | 27 | None | 7.35% | Preregistered replication and latency scope decision |

The v3 decision rule was frozen before measurement. It accepts energy
prediction when all-workload MAPE is at most 15%, accepts TTFT prediction at
concurrency 4 when its stratum MAPE is at most 20%, and requires abstention
below concurrency 4 when the sparse-stratum TTFT MAPE exceeds 20%. The observed
TTFT MAPE was 9.27% at concurrency 4 and 64.80% at concurrency 1 or 2, yielding
`support_latency_at_concurrency_ge_4_abstain_below_4`.

## Layout

- `holdout-v2/` contains the directly readable v2 validation report,
  prediction summary, manifests, and analysis provenance.
- `scope-v3/` contains the directly readable v3 report, preregistered decision,
  prediction summary, manifests, and analysis provenance.
- `archives/` contains the complete sealed packages, including frozen
  predictions, measurements, client outputs, DCGM telemetry, run logs, and
  internal SHA-256 manifests.
- `SHA256SUMS` authenticates every file published in this directory.

## Verification

From the repository root:

```bash
shasum -a 256 -c experiments/evidence/eesp26/SHA256SUMS
```

To verify every raw v3 artifact while preserving its original relative paths:

```bash
mkdir -p /tmp/tpa-scope-v3
tar -xzf experiments/evidence/eesp26/archives/tpa-qwen7b-scope-v3-final-20260804.tar.gz \
  -C /tmp/tpa-scope-v3
cd /tmp/tpa-scope-v3
sha256sum -c experiments/results/scope-confirmation-v3-artifacts.sha256
```

Holdout data are immutable evaluation evidence and must not be used to refit
the reported model. Any model change requires a newly frozen disjoint holdout.
