# Agent Benchmarks

These files freeze CPU/replay experiments that strengthen the single-GPU
Workshop evaluation without creating new hardware ground truth.

`qwen2.5-7b-planner-conformance-v1.json` defines 30 natural-language cases
over four controller states plus six guard challenges. The expected subgoal for
every case must contain the deterministic safety planner's decision. Three
repeats produce 90 language-model calls. The benchmark reports accepted typed
output, raw and guarded subgoal agreement, fallback, forbidden-subgoal
acceptance, control-plane latency, and token usage. `verify` is reserved for the
deterministic release gate and cannot be accepted from the language model.

`qwen2.5-7b-h100-budget-sweep-v1.json` binds the five declared GPU-hour budgets
to the sealed 12-candidate H100 scenario and 84-row replay corpus by SHA-256.
Every policy receives the same candidate-level empirical bootstrap world for a
given seed, and seed ids are matched across budgets. The resulting episodes are
sensitivity analyses over three measured repeats, not additional GPU runs.

Run both experiments from the repository root:

```bash
tokenpoweragent benchmark-planner \
  --protocol configs/benchmarks/qwen2.5-7b-planner-conformance-v1.json \
  --planner-base-url http://127.0.0.1:8000/v1 \
  --output experiments/results/planner-conformance-v1.json

tokenpoweragent benchmark-budget-sweep \
  --scenario experiments/results/config-search-v1-scenario.json \
  --records experiments/results/config-search-v1-replay-corpus.jsonl \
  --protocol configs/benchmarks/qwen2.5-7b-h100-budget-sweep-v1.json \
  --output experiments/results/config-search-v1-budget-sweep.json
```

Do not edit a protocol after its first result has been generated. Archive its
SHA-256 with the output and the exact Git commit.
