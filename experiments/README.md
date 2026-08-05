# Experiment Layout

Real runs should be immutable, attributable, and separable from paper plotting.
Use one manifest per campaign and one directory per scenario/seed.

```text
experiments/
  manifests/        frozen campaign descriptions committed before execution
  results/          generated evidence, telemetry, logs, and summaries
```

Each manifest should record the git commit, scenario hash, TokenPowerBench
commit, model and tokenizer revisions, container digest, engine version, GPU
and interconnect model, driver/CUDA/NCCL versions, power sampling interval,
warmup policy, seeds, and declared per-level GPU-hour caps.

The minimum paper campaign should produce four evidence groups:

1. A held-out dense sweep used only as a replay oracle.
2. Cross-fidelity calibration pairs for L0/L1/L2 versus L4.
3. Repeated agent episodes for budget-to-frontier curves and ablations.
4. Independent L4 reruns of every reported recommendation.

Do not commit raw result directories by default. Promote only anonymized,
publication-ready summaries through the artifact release process.

The first real-hardware bring-up is the L1 CUDA GEMM campaign documented in
[`sandbox/README.md`](sandbox/README.md). It validates the sandbox and energy
measurement contract before adding vLLM, model weights, or request traces. It
is a diagnostic, not the paper's primary experiment.

The current H100 evidence has been exported to
[`plot-data/`](plot-data/README.md) as analysis-ready CSV plus self-contained
raw JSON/JSONL. Column semantics are documented in
[`plot-data/DATA_DICTIONARY.md`](plot-data/DATA_DICTIONARY.md). Rebuild the
tables after adding an archive with:

```bash
python experiments/analysis/export_plot_data.py \
  --budget-report /path/to/config-search-v1-budget-sweep.json
```

The frozen research-question manifests are:

- `rq1-sandbox-validity.json`: held-out prediction, ranking, and Pareto validity.
- `rq2-agent-search-efficiency.json`: information gained per GPU-hour.
- `rq3-transfer-and-adaptation.json`: context, scale, topology, GPU, and model transfer.
- `rq4-verified-deployment-benefit.json`: independent target-scale benefit.

See [`../docs/TokenPowerSandbox-Experiment-Design.md`](../docs/TokenPowerSandbox-Experiment-Design.md)
for the factor contract, fidelity semantics, claim gates, and execution order.
