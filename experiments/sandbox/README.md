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
12.8.1 and copies only the executable into the runtime image. A deterministic
hash kernel initializes nonzero FP16 inputs for every seed, and the workload
copies one output value to the host so the computation has a checked result.

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

Repeats use a cyclically balanced power-cap order. For three limits and three
repeats, the rounds are `350,500,700`, `500,700,350`, and `700,350,500`, so
temperature and run-order drift are not assigned to one configuration alone.

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
request-window markers instead. Provenance includes the immutable local Docker
image ID, campaign position, requested/read-back power cap, and artifact paths.

## Pass Criteria

The smoke experiment passes when all nine records succeed, field 156 is
monotonic, every power-cap readback matches its request, power returns to 700 W,
and at least one lower cap reduces energy or average power without producing an
invalid CUDA result. The expected performance ordering is measured rather than
assumed.

## Persistent vLLM Serving Probe

The next L1 probe keeps a pinned vLLM server alive so model loading, tokenizer
initialization, and CUDA graph capture are outside the measured request window.
Its benchmark client has no GPU and joins an internal Docker network that can
reach only the serving container. Build the marker-enabled client from the
immutable vLLM 0.23.0 image. Repeated candidates intentionally reuse the same
seed for paired comparison, so the server must explicitly disable automatic
prefix caching; otherwise whichever candidate runs later can reuse the first
candidate's prompt KV cache. Launch the fixed server envelope as follows:

```bash
sudo docker volume create tpa-hf-cache

sudo docker run -d \
  --name tpa-vllm-qwen7b \
  --gpus all \
  --shm-size=16g \
  -p 127.0.0.1:8000:8000 \
  -v tpa-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f \
  Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name qwen2.5-7b \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --generation-config vllm \
  --seed 0
```

Then build the client and create and attach its internal network once:

```bash
sudo docker build \
  -t tokenpower-vllm-client:v0.23.0 \
  experiments/sandbox/vllm-client

sudo docker network create --internal tpa-serving-bench
sudo docker network connect tpa-serving-bench tpa-vllm-qwen7b
```

Run a long-enough 700 W pilot after the server health check passes:

```bash
tokenpoweragent serving-smoke \
  --power-limits 700 \
  --repeats 1 \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 64 \
  --num-warmups 2 \
  --request-rate inf \
  --max-concurrency 8 \
  --sample-ms 100 \
  --output experiments/results/qwen7b-serving-pl700.jsonl
```

The pinned client inserts two blocking FIFO handshakes around vLLM's own
`benchmark_start_time` and `benchmark_duration` boundaries. At `READY`, the
host reads DCGM field 156 before replying `GO`; after all requests finish, the
client sends `DONE` and waits for the host's ending energy read and `ACK`.
Container startup, tokenizer loading, readiness checks, warmups, and result
formatting are therefore outside the energy interval. The record contains
P50/P95/P99 TTFT, TPOT, ITL, and end-to-end latency, output and request
throughput, joules, average power, and joules per token. Temperature is fixed at
zero and EOS is ignored so every request performs the declared output-token
work.
