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

The required order is:

1. Freeze every campaign prediction and its SHA-256 manifest.
2. Review and archive those artifacts before any measurement.
3. Run the campaign against the unchanged persistent server.
4. Use a separate, untouched campaign for final holdout evaluation.

Freeze commands refuse to overwrite an existing artifact. Measurement commands
verify the frozen hashes and server contract before running, checkpoint after
every repeat, and require `--resume` to continue a partial campaign.
