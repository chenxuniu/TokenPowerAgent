# Data Dictionary

## Evidence Semantics

| Value | Meaning |
|---|---|
| `L0` | CPU-side sandbox prediction. No target GPU trial was run for that row. |
| `L1` | Low-cost real H100 calibration or proxy-workload measurement. |
| `L4` | Real H100 verification on the frozen target workload and configuration. |
| `measured` | Metric comes from a hardware measurement record. |
| `predicted` / simulated kind | Metric comes from the topology/workload model. |

Do not average L0, L1, and L4 rows together. Use `level` as a facet or compare
the aligned columns in `config_candidate_summary.csv`.

## Common Identity Columns

| Column | Meaning |
|---|---|
| `experiment_id` | Stable export identifier for the campaign or benchmark. |
| `candidate_id` | Serving configuration or GEMM power-cap identifier. |
| `workload_id` | Frozen workload identifier encoding context/output/concurrency. |
| `repeat` | Campaign repeat index, normally 0, 1, or 2. |
| `seed` | Random seed recorded by the runner or replay. |
| `source_archive` | Selected evidence archive identifier. |
| `source_member` | Original path inside the source tar.gz. |
| `dataset_split` | Development, validation, holdout, or configuration-search split. |

## Serving Metrics

| Column | Unit | Meaning |
|---|---:|---|
| `energy_j` | J | GPU energy in the synchronized measurement window. |
| `energy_j_per_1k_output_tokens` | J/1k output tokens | Primary energy objective. |
| `j_per_output_token` | J/token | Same energy normalized per output token. |
| `throughput_tok_s` | output tokens/s | Output-token serving throughput. |
| `ttft_ms` | ms | P95 time to first token used by the evaluation. |
| `tpot_ms` | ms | P95 time per output token used by the evaluation. |
| `avg_power_w` | W | DCGM average GPU power during the active window. |
| `benchmark_duration_s` | s | vLLM main benchmark duration. |
| `gpu_hours` | GPU-h | Evidence acquisition cost assigned to the trial. |

The energy scope is GPU-only. Host CPU, storage, and network energy are not
included.

## Validation Tables

`workload_validation.csv` is one row per campaign, workload, and metric.

| Column | Meaning |
|---|---|
| `prediction` | Frozen prediction produced before holdout measurements. |
| `observed_median` | Median across the three hardware repeats. |
| `observations` | JSON array containing every repeat value. |
| `signed_prediction_error_pct` | `(prediction-observed)/abs(observed)*100`. |
| `absolute_percentage_error_pct` | Absolute value of signed error. |
| `prediction_interval_lower/upper` | Frozen prediction interval. |
| `interval_covers_observed_median` | Whether the observed median lies in that interval. |
| `observed_cv_pct` | Repeat coefficient of variation in percent. |

`validation_aggregate.csv` contains campaign-level MAPE, Spearman rank
correlation, pairwise ordering accuracy, repeat CV, and interval coverage.

## Configuration Search

`config_search_evidence.csv` retains every L0/L1/L4 record. The compact
`config_candidate_summary.csv` aligns median values by candidate:

- `l0_*`, `l1_*`, `l4_*`: prediction or median measurement at each level.
- `l0_vs_l4_*_error_pct`: signed L0 prediction error relative to L4.
- `oracle_pareto`: membership in the Pareto set built from median L4 evidence.
- `max_num_seqs`, `max_num_batched_tokens`, `chunked_prefill`: serving knobs.

## Independent Winner Confirmation

`winner_confirmation_pairs.csv` contains one row per seed-matched L4 pair.
The `baseline_*` and `winner_*` columns are measured values; corresponding
`improvement_*_pct` columns are positive when the locked winner improves the
metric. `winner_confirmation_summary.csv` records the paired mean and median
improvements, 95% confidence interval, win rate, and one-sided exact sign-test
p-value. `winner_confirmation_candidates.csv` contains aggregate statistics
for the two locked configurations.

## Policy Replay

`policy_episodes.csv` contains one row per replay seed, policy, and budget.
These are empirical bootstrap/replay episodes over measured evidence, not new
GPU executions.

| Column | Meaning |
|---|---|
| `success_rate` | Aggregate fraction of episodes that verify the oracle Pareto candidate. |
| `oracle_pareto_recall/precision` | Episode-level frontier recovery. |
| `spent_gpu_hours` | Exploration plus final L4 verification cost. |
| `first_oracle_hit_gpu_hours` | Cost when an oracle candidate was first acquired. |
| `best_primary_regret_pct` | Energy regret of the best verified feasible candidate. |
| `unnecessary_l4_verifications` | L4 checks not belonging to the oracle frontier. |

The five-budget benchmark uses common random numbers: the same seed selects the
same candidate-level empirical repeat for every policy and budget.

## Planner Calls

`planner_calls.csv` retains all 180 calls, including raw responses.

| Column | Meaning |
|---|---|
| `proposed_subgoal` | Parsed LLM proposal before the state guard. |
| `selected_subgoal` | Subgoal admitted after deterministic guarding. |
| `raw_expected_subgoal` | Whether the raw proposal matches the frozen expected set. |
| `guarded_expected_subgoal` | Whether the admitted subgoal matches the expected set. |
| `semantic_guard_intervened` | Whether the guard replaced the LLM proposal. |
| `planner_latency_ms` | Full planner call latency. |
| `completion_latency_ms` | Inference endpoint latency. |
| `rule_latency_ms` | Deterministic state-rule latency. |
| `state_*` | Frozen agent state shown to the planner. |

## DCGM Telemetry

`dcgm_telemetry.csv` is one row per DCGM sample and trace.

| Column | Unit | Meaning |
|---|---:|---|
| `sample_index` | sample | Order within one trace. |
| `sample_elapsed_ms` | ms | Index multiplied by the recorded sampling period. |
| `in_measurement_window` | boolean | Sample lies between synchronized energy start/end counters. |
| `power_w`, `instant_power_w` | W | DCGM power fields. |
| `total_energy_mj` | mJ | Monotonic cumulative GPU energy counter. |
| `energy_since_measurement_start_j` | J | Counter delta from the benchmark boundary. |
| `sm_clock_mhz`, `memory_clock_mhz` | MHz | GPU clocks. |
| `gpu_util_pct`, `memory_util_pct` | percent | DCGM utilization fields. |
| `temperature_c` | C | GPU temperature. |

Two early GEMM pilot traces use a shorter DCGM schema and are labeled
`gemm_pilot_unsealed`. Missing fields are empty; their available samples are
still retained.

## Null Values

An empty CSV cell means the source did not define that field. It does not mean
zero. Examples include unavailable prediction intervals for diagnostics,
missing first-hit cost in unsuccessful policy episodes, and unavailable DCGM
fields in the early pilot schema.
