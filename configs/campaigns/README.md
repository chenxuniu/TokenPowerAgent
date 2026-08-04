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
serving-configuration corpus for TokenPowerAgent. It fixes one H100, one
Qwen2.5-7B revision, one 2048/128-token workload at concurrency 32, BF16,
TP=PP=DP=1, prefix caching off, and a 700 W limit. Twelve candidates vary
`max_num_seqs`, `max_num_batched_tokens`, and chunked prefill; the current
`256/8192/chunked` server is retained as the expert baseline. Each candidate
has one CPU-side L0 prediction, three 64-request L1 probes, and three
256-request L4 validations. The 72 GPU measurements use a frozen cyclic order
with one server restart per candidate-repeat block.

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
tokenpoweragent freeze-config-campaign \
  --campaign configs/campaigns/qwen2.5-7b-h100-config-search-v1.json \
  --calibration experiments/results/qwen2.5-7b-h100-workload-v2.json \
  --scenario experiments/results/config-search-v1-scenario.json \
  --predictions experiments/results/config-search-v1-l0-predictions.jsonl \
  --schedule experiments/results/config-search-v1-schedule.json \
  --summary experiments/results/config-search-v1-freeze-summary.json \
  --manifest experiments/results/config-search-v1-freeze.sha256
```
