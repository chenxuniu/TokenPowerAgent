# Measured Result Provenance

The Workshop metrics in `metrics.tex` were checked against the following local
archives on August 4, 2026. Raw telemetry is intentionally kept out of Git;
these digests identify the immutable source bundles.

| Evidence bundle | SHA-256 |
|---|---|
| `tpa-qwen7b-holdout-v2-final-20260804.tar.gz` | `2d74179245392d44f2efd2fbc5601b4721e278f206e45970c5953481e12ecafb` |
| `tpa-qwen7b-scope-v3-final-20260804.tar.gz` | `9a4dbfa5be8c703aea6104a8cb4ef8a890876603ee05a624780d6f43d22cd3fd` |

## Cross-Checked Headline Values

| Study | Workloads | Measurements | Energy MAPE | Spearman |
|---|---:|---:|---:|---:|
| Holdout v2 | 8 | 24 | 6.227225% | 0.976190 |
| Scope confirmation v3 | 9 | 27 | 7.348367% | 0.933333 |

The preregistered scope decision is
`support_latency_at_concurrency_ge_4_abstain_below_4`. Its supported TTFT MAPE
is 9.273277% at concurrency four; the combined sparse-concurrency MAPE is
64.804732%.

The source archives contain the frozen predictions, measurement JSONL, raw
100-ms DCGM telemetry, benchmark output, campaign and artifact manifests,
validation reports, decision report, Git commit identifiers, and nested
analysis checksums.
