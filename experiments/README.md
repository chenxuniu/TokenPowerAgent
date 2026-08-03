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
measurement contract before adding vLLM, model weights, or request traces.
