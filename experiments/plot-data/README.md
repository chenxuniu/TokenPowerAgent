# TokenPowerAgent Plot Data

This directory is a self-contained, analysis-ready export of the evidence
available on the local workstation on 2026-08-05. CSV files distinguish real
H100 measurements, L0 predictions, empirical replay episodes, and planner
calls. The `raw/` directory retains the source JSON/JSONL used to build them.

## Quick Inventory

| Data layer | Rows |
|---|---:|
| GEMM H100 trials | 9 |
| Real Qwen serving trials | 79 |
| Frozen serving predictions | 23 |
| Independent winner-confirmation pairs | 5 |
| Configuration-search evidence (L0/L1/L4) | 84 |
| Numeric evidence values in long format | 6263 |
| Validation workload-metric rows | 138 |
| Policy replay episodes | 14000 |
| Planner calls | 180 |
| DCGM telemetry samples | 19840 |

## Headline Validation Results

| Campaign | Energy MAPE (%) | Spearman | Pairwise order accuracy |
|---|---:|---:|---:|
| workload_transfer_v1 | 60.03 | 0.886 | 0.867 |
| workload_holdout_v2 | 6.23 | 0.976 | 0.964 |
| scope_confirmation_v3 | 7.35 | 0.933 | 0.917 |

## Independent Winner Confirmation

Five locked, paired L4 repeats confirm the selected configuration without
refitting or reselection. Mean energy saving is
1.39% (95% CI
[1.19%, 1.59%); all five energy pairs favor the
winner, and median paired TTFT falls
21.43%.

## Planner Results

| Benchmark | Calls | Raw accuracy | Guarded accuracy | P50 latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|---:|---:|
| planner_conformance_v1_development | 90 | 0.833 | 0.867 | 319.4 | 407.0 |
| planner_conformance_v2_holdout | 90 | 0.733 | 1.000 | 247.3 | 515.3 |

## Recommended Plot Inputs

- `workload_validation.csv`: predicted-versus-observed scatter plots, error by
  context/concurrency, and interval coverage.
- `serving_trials.csv`: repeat-level energy, throughput, TTFT, TPOT, and power.
- `winner_confirmation_pairs.csv`: the five seed-matched baseline-versus-winner
  comparisons and per-metric improvements.
- `winner_confirmation_summary.csv`: confidence intervals, win rates, and exact
  sign-test results for the independent confirmation.
- `config_candidate_summary.csv`: L0/L1/L4 comparison for each serving knob
  configuration and the observed oracle Pareto label.
- `policy_summary.csv` and `policy_episodes.csv`: success/cost/regret curves and
  confidence intervals over replay seeds.
- `planner_calls.csv`: latency, token usage, raw decisions, guard interventions,
  and state-dependent accuracy.
- `dcgm_telemetry.csv`: trace-level power, clocks, utilization, temperature, and
  cumulative energy. Filter `in_measurement_window == true` for the benchmark
  interval.

## Provenance and Known Gaps

- `source_manifest.csv` hashes every discovered local archive and explicitly
  marks missing artifacts.
- `archive_members.csv` hashes every member of each selected evidence archive.
- `raw_file_manifest.csv` hashes every copied or regenerated raw JSON/JSONL.
- `dataset_manifest.csv` hashes every generated table.
- The complete configuration-search archive was not downloaded to this
  workstation. Its sealed replay corpus contains all 12 L0 predictions and all
  72 L1/L4 measurement records, so candidate-level plots are complete; only its
  original DCGM/client time-series files are unavailable here.
- The original three calibration JSONL rows are not local. Their aggregate,
  MAD values, source hash, and fitted residual model are preserved in
  `calibration_points.csv` and `residual_model_coefficients.csv`.
- `raw/config-search-v1-budget-sweep-regenerated.json` was deterministically
  regenerated from the sealed replay corpus and frozen protocol. Its 10,000
  episodes reproduce every compact paper summary value exactly.

## Pandas Example

```python
import pandas as pd

df = pd.read_csv("workload_validation.csv")
energy = df[df.metric == "energy_j_per_1k_output_tokens"]
ax = energy.plot.scatter(x="observed_median", y="prediction")
```
