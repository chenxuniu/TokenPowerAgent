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

V1 is the frozen prompt-only development diagnostic. Its H100/Qwen2.5-7B run
found that schema and `verify` guards do not reject a legal but
state-inconsistent subgoal. Do not alter or overwrite that protocol or result.

`qwen2.5-7b-planner-conformance-v2-holdout.json` freezes 30 disjoint intents
across cold-start, SLO-boundary, repair, topology-calibration, steady-search,
and adversarial states. It pins the explicit state-table prompt and the
`state-priority-v1` admission guard. V2 reports the raw LLM proposal separately
from any guard intervention and the finally admitted subgoal. Its development
provenance pins the V1 protocol, report, and diagnostics hashes.

`qwen2.5-7b-h100-budget-sweep-v1.json` binds the five declared GPU-hour budgets
to the sealed 12-candidate H100 scenario and 84-row replay corpus by SHA-256.
Every policy receives the same candidate-level empirical bootstrap world for a
given seed, and seed ids are matched across budgets. The resulting episodes are
sensitivity analyses over three measured repeats, not additional GPU runs.

Run both experiments from the repository root:

```bash
servecompass benchmark-planner \
  --protocol configs/benchmarks/qwen2.5-7b-planner-conformance-v2-holdout.json \
  --planner-base-url http://127.0.0.1:8000/v1 \
  --output experiments/results/planner-conformance-v2-holdout.json

servecompass benchmark-budget-sweep \
  --scenario experiments/results/config-search-v1-scenario.json \
  --records experiments/results/config-search-v1-replay-corpus.jsonl \
  --protocol configs/benchmarks/qwen2.5-7b-h100-budget-sweep-v1.json \
  --output experiments/results/config-search-v1-budget-sweep.json
```

Do not edit a protocol after its first result has been generated. Archive its
SHA-256 with the output and the exact Git commit.
