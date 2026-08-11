# Workload Campaigns

Campaign files pre-register a model revision, one serving configuration, the
measurement protocol, and a set of workload points. They separate workload
transfer from configuration transfer: TP, PP, batching limits, precision,
power limit, and cache behavior remain fixed while input length and concurrency
change.

`qwen2.5-7b-h100-workload-transfer-validation-v1.json` contains six validation
points selected across the context-length and concurrency space. Every point
uses 128 output tokens and eight request waves. These records may later tune an
uncertainty model, but they must not be relabeled as calibration or final
holdout evidence.

`qwen2.5-7b-h100-workload-transfer-holdout-v2.json` contains eight disjoint
context-by-concurrency combinations inside the measured validation envelope.
It is reserved for the final evaluation of a profile fitted on validation-v1.
Do not run it until the v2 profile, all eight predictions, and their manifest
have been frozen and archived.

`qwen2.5-7b-h100-scope-confirmation-v3.json` is a post-holdout, independently
pre-registered confirmation campaign. It does not refit the v2 model. Its nine
new workloads cross three interpolated input lengths with concurrency 1, 2,
and 4 to test the low-occupancy latency boundary observed in holdout-v2. The
embedded decision rule is part of the campaign hash and must not be edited
after predictions are frozen.

`qwen2.5-7b-h100-config-search-v1.json` is the first controlled
serving-configuration corpus for ServeCompass. It fixes one H100, one
Qwen2.5-7B revision, one 2048/128-token workload at concurrency 32, BF16,
TP=PP=DP=1, prefix caching off, and a 700 W limit. Twelve candidates vary
`max_num_seqs`, `max_num_batched_tokens`, and chunked prefill; the current
`256/8192/chunked` server is retained as the expert baseline. Each candidate
has one CPU-side L0 prediction, three 64-request L1 probes, and three
256-request L4 validations. The 72 GPU measurements use a frozen cyclic order
with one server restart per candidate-repeat block.

`qwen2.5-7b-h100-winner-confirmation-v1.json` is the independent
post-selection confirmation. It locks `seq32-bt2048-chunk` as the winner and
`expert-seq256-bt8192-chunk` as the baseline using hashes from config-search-v1;
the winner may not be refitted or reselected after this file is frozen. Five
new seeds produce five L4 pairs (10 target-workload measurements). Pair order
alternates between repeats, and the server restarts before every action. The
preregistered pass rule requires five complete pairs, an energy win in every
pair, a positive lower endpoint of the exact paired-bootstrap 95% interval for
mean energy saving, a positive median TTFT reduction, and SLO compliance in
every run.

The required order is:

1. Freeze every campaign prediction and its SHA-256 manifest.
2. Review and archive those artifacts before any measurement.
3. Run the campaign against the unchanged persistent server.
4. Use a separate, untouched campaign for final holdout evaluation.

Freeze commands refuse to overwrite an existing artifact. Measurement commands
verify the frozen hashes and server contract before running, checkpoint after
every repeat, and require `--resume` to continue a partial campaign.

Freeze the configuration campaign before changing or restarting the server:

```bash
servecompass freeze-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-config-search-v1.json \
  --calibration experiments/results/qwen2.5-7b-h100-workload-v2.json \
  --scenario experiments/results/config-search-v1-scenario.json \
  --predictions experiments/results/config-search-v1-l0-predictions.jsonl \
  --schedule experiments/results/config-search-v1-schedule.json \
  --summary experiments/results/config-search-v1-freeze-summary.json \
  --manifest experiments/results/config-search-v1-freeze.sha256
```

After independently archiving that freeze, run the first L1/L4 pair as a
foreground launch check. `--max-actions` partitions the immutable schedule; it
does not regenerate or reorder it:

```bash
servecompass run-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-config-search-v1.json \
  --predictions experiments/results/config-search-v1-l0-predictions.jsonl \
  --schedule experiments/results/config-search-v1-schedule.json \
  --summary experiments/results/config-search-v1-freeze-summary.json \
  --freeze-manifest experiments/results/config-search-v1-freeze.sha256 \
  --output experiments/results/config-search-v1-measurements.jsonl \
  --max-actions 2
```

The runner recreates vLLM on the internal Docker network for every frozen
candidate-repeat block, verifies the live image and serving knobs before each
measurement, and atomically checkpoints every action. Continue after a clean
launch check or an SSH interruption by repeating the command with `--resume`;
omit `--max-actions` to execute every remaining action.

After all 72 actions complete, validate the schedule, chronology, evidence
labels, serving contract, repeats, and raw artifacts before policy replay:

```bash
servecompass validate-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-config-search-v1.json \
  --predictions experiments/results/config-search-v1-l0-predictions.jsonl \
  --schedule experiments/results/config-search-v1-schedule.json \
  --summary experiments/results/config-search-v1-freeze-summary.json \
  --freeze-manifest experiments/results/config-search-v1-freeze.sha256 \
  --measurements experiments/results/config-search-v1-measurements.jsonl \
  --output experiments/results/config-search-v1-validation.json \
  --corpus experiments/results/config-search-v1-replay-corpus.jsonl \
  --artifact-list experiments/results/config-search-v1-artifacts.list \
  --artifact-manifest experiments/results/config-search-v1-artifacts.sha256 \
  --include-artifact experiments/results/config-search-v1-run.log \
  --include-artifact experiments/results/environment/h100-config-search-preflight-20260804.txt
```

The resulting report uses the median of three successful L4 repeats to define
the SLO-feasible energy--throughput oracle. It reports L0/L1 error, rank
correlation, pairwise ordering, and Pareto precision/recall against that oracle.
The replay corpus keeps L0 predictions and measured L1/L4 records distinctly
labeled; the artifact manifest covers the freeze, measurements, DCGM traces,
client outputs, server logs, run log, and environment snapshot.

Freeze the winner confirmation before starting any new measurement:

```bash
servecompass freeze-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-winner-confirmation-v1.json \
  --calibration experiments/results/qwen2.5-7b-h100-workload-v2.json \
  --scenario experiments/results/winner-confirmation-v1-scenario.json \
  --predictions experiments/results/winner-confirmation-v1-l0-predictions.jsonl \
  --schedule experiments/results/winner-confirmation-v1-schedule.json \
  --summary experiments/results/winner-confirmation-v1-freeze-summary.json \
  --manifest experiments/results/winner-confirmation-v1-freeze.sha256
```

After checking and archiving the freeze, run all 10 L4 actions. The same command
with `--resume` safely continues after an SSH interruption:

```bash
servecompass run-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-winner-confirmation-v1.json \
  --predictions experiments/results/winner-confirmation-v1-l0-predictions.jsonl \
  --schedule experiments/results/winner-confirmation-v1-schedule.json \
  --summary experiments/results/winner-confirmation-v1-freeze-summary.json \
  --freeze-manifest experiments/results/winner-confirmation-v1-freeze.sha256 \
  --output experiments/results/winner-confirmation-v1-measurements.jsonl \
  --telemetry-dir experiments/results/winner-confirmation-v1-telemetry \
  --server-log-dir experiments/results/winner-confirmation-v1-server-logs
```

Validate the paired result and hash every raw artifact without changing the
locked selection:

```bash
servecompass validate-config-confirmation \
  --campaign configs/campaigns/qwen2.5-7b-h100-winner-confirmation-v1.json \
  --predictions experiments/results/winner-confirmation-v1-l0-predictions.jsonl \
  --schedule experiments/results/winner-confirmation-v1-schedule.json \
  --summary experiments/results/winner-confirmation-v1-freeze-summary.json \
  --freeze-manifest experiments/results/winner-confirmation-v1-freeze.sha256 \
  --measurements experiments/results/winner-confirmation-v1-measurements.jsonl \
  --output experiments/results/winner-confirmation-v1-report.json \
  --artifact-list experiments/results/winner-confirmation-v1-artifacts.list \
  --artifact-manifest experiments/results/winner-confirmation-v1-artifacts.sha256 \
  --include-artifact experiments/results/winner-confirmation-v1-run.log
```
