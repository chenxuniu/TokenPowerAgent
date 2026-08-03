# TokenPowerSandbox L1 Smoke Experiment

This experiment validates the smallest real-hardware TokenPowerAgent loop:

```text
typed candidate -> restricted Docker container -> CUDA GEMM
       |                                           |
host power-cap controller                    H100 execution
       |                                           |
       +-------------- DCGM measurement -----------+
```

It is an L1 single-GPU phase probe. It validates container isolation, power-cap
control, CUDA execution, DCGM energy accounting, telemetry capture, provenance,
and conversion into an `EvidenceRecord`. It does not claim that GEMM predicts
LLM serving energy or that a single H100 predicts multi-GPU scaling.

## Build

From the repository root on the H100 host:

```bash
sudo docker build --pull \
  -t tokenpower-sandbox:cuda12.8 \
  experiments/sandbox/cuda-gemm
```

The multi-stage image compiles a fixed FP16 cuBLAS GEMM workload with CUDA
12.8.1 and copies only the executable into the runtime image.

## Run

Install the repository in an isolated Python environment, then run three power
limits with three repetitions each:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

tokenpoweragent sandbox-smoke \
  --power-limits 350,500,700 \
  --repeats 3 \
  --matrix-size 16384 \
  --warmup 20 \
  --iterations 400 \
  --sample-ms 100 \
  --output experiments/results/h100-gemm-smoke.jsonl
```

The executor restores the original power limit after every run, including most
failure paths. After an interrupted campaign, verify it explicitly:

```bash
nvidia-smi --query-gpu=power.limit --format=csv,noheader
sudo nvidia-smi -pl 700
```

## Outputs

Each successful evidence record contains:

- `energy_j`: DCGM field 156 start/end difference in joules.
- `elapsed_ms`: host-observed container execution time.
- `kernel_elapsed_ms`: CUDA-event time for measured GEMM iterations.
- `throughput_tflops`: measured FP16 GEMM throughput.
- `avg_power_w`: energy divided by the host measurement interval.
- `gross_j_per_tflop`: gross interval energy per trillion measured operations.

Raw 100 ms DCGM traces and container stdout/stderr are stored beneath
`experiments/results/telemetry/`. Container startup and warmup are included in
`energy_j`; the later serving experiment will keep vLLM alive and use explicit
request-window markers instead.

## Pass Criteria

The smoke experiment passes when all nine records succeed, field 156 is
monotonic, every power-cap readback matches its request, power returns to 700 W,
and at least one lower cap reduces energy or average power without producing an
invalid CUDA result. The expected performance ordering is measured rather than
assumed.
