# Measured Result Provenance

The Workshop metrics in `metrics.tex` were checked against the following sealed
archives on August 4--5, 2026. Raw telemetry is intentionally kept out of Git;
these digests identify the immutable source bundles.

| Evidence bundle | SHA-256 |
|---|---|
| `tpa-qwen7b-holdout-v2-final-20260804.tar.gz` | `2d74179245392d44f2efd2fbc5601b4721e278f206e45970c5953481e12ecafb` |
| `tpa-qwen7b-scope-v3-final-20260804.tar.gz` | `9a4dbfa5be8c703aea6104a8cb4ef8a890876603ee05a624780d6f43d22cd3fd` |
| `tpa-qwen7b-config-search-v1-final-20260805.tar.gz` | `d157300787af4cf16379bd316e7138da00279463c381149bc99605d5af851b33` |

## Cross-Checked Headline Values

| Study | Workloads | Measurements | Energy MAPE | Spearman |
|---|---:|---:|---:|---:|
| Holdout v2 | 8 | 24 | 6.227225% | 0.976190 |
| Scope confirmation v3 | 9 | 27 | 7.348367% | 0.933333 |

## Configuration Corpus

The preregistered grid contains 12 candidates and 72 successful measurements
(36 L1 and 36 L4). L0 energy MAPE/rank correlation are 19.448074%/0.884060;
L1 values are 1.153024%/0.874126. L0 has 0% L4-Pareto recall because TTFT rank
correlation is -0.968427. L1 has 100% recall and 33.333% precision. Median L4
measurements identify `seq32-bt2048-chunk` as the unique Pareto point.

The bundle contains 196 entries and verifies 190 raw artifacts. Its internal
measurement, report, and replay-corpus SHA-256 values are:

- measurement: `4a6bac3b82117cd5265d0d66e57b412b966f071dcdf79e16f1504a60bea096f0`;
- validation report: `b6aa51dc7e9c3912622e278718f7b1ea55fee9b0b387a5776fcb0af678dd447f`;
- replay corpus: `2811f77938f349b95ec679804dcb0d262377fd0d7277d3740a73dd1815c8a471`.

The preregistered scope decision is
`support_latency_at_concurrency_ge_4_abstain_below_4`. Its supported TTFT MAPE
is 9.273277% at concurrency four; the combined sparse-concurrency MAPE is
64.804732%.

The source archives contain the frozen predictions, measurement JSONL, raw
100-ms DCGM telemetry, benchmark output, campaign and artifact manifests,
validation reports, decision report, Git commit identifiers, and nested
analysis checksums.
