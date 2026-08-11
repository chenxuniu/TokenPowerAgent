# ServeCompass MLSys 2027 Complete Execution Plan

**Status:** working execution plan  
**Last verified:** 2026-08-11  
**Target:** MLSys 2027 Research Track  
**Companion method protocol:**
[MLSys-2027-Architecture-Aware-Experiment-Protocol.md](MLSys-2027-Architecture-Aware-Experiment-Protocol.md)

## 1. Executive Decision

ServeCompass should now be developed as a full MLSys systems paper, not as
an expanded workshop paper. The paper's primary claim should be:

> ServeCompass uses calibrated multi-fidelity evidence to recover an
> SLO-constrained energy-performance Pareto frontier while spending fewer real
> GPU-hours than single-fidelity search across model architectures, GPU
> generations, and parallel topologies.

The current Qwen2.5-7B/H100 result remains a validated anchor and engineering
artifact. Its 1.39% confirmed energy saving is not the MLSys headline. The
headline must instead be verified frontier quality per GPU-hour, with energy
savings at matched SLO reported honestly as a deployment outcome.

The project should use:

- **vLLM as the primary inference engine**;
- **SGLang as a targeted replication engine**, especially for Kimi models;
- **TensorRT-LLM as an optional optimized-system baseline** on a frozen subset;
- **CPU execution for L0 Estimate and policy replay**;
- **real GPUs for L1-L4 evidence and all final claims**;
- **one same-architecture multi-node allocation** for a defensible L3 result;
- **H100, H200, and B200 access** for cross-generation transfer.

Kimi-K2.5 is a stretch case. It must not block the core paper.

## 2. Venue Assumptions

No official MLSys 2027 CFP or deadline was located as of 2026-08-11. Until the
2027 CFP appears, use the official MLSys 2026 research-track rules as the
planning envelope:

- double-blind review;
- two-column format;
- up to 10 body pages, excluding references;
- a separately uploaded appendix;
- optional post-acceptance artifact evaluation.

The official 2026 paper deadline was 2025-10-30, and the 2025 deadline was also
in late October. Therefore, **2026-10-16 is our internal data and paper freeze**
and **2026-10-23 is our internal submission-ready deadline**. These are project
deadlines, not announced MLSys 2027 dates.

Official planning references:

- [MLSys 2026 Research CFP](https://mlsys.org/Conferences/2026/CallForResearchPapers)
- [MLSys 2026 dates](https://mlsys.org/Conferences/current/Dates)

## 3. Scope and Research Questions

### 3.1 In scope

1. Selecting serving configuration and evidence fidelity jointly.
2. Dense, MoE, and hybrid/linear-attention model architectures.
3. H100, H200, and B200 transfer.
4. Single GPU, intra-node multi-GPU, and selected multi-node topologies.
5. Energy, TTFT, TPOT, throughput/goodput, and GPU-hour search cost.
6. Guarded LLM planning plus deterministic feasibility, acquisition, and
   verification logic.
7. Blind held-out workloads and independent confirmation runs.

### 3.2 Out of scope for the core deadline

1. Training or fine-tuning models.
2. Exhaustive search over every model, hardware, workload, and topology tuple.
3. Full rack or datacenter carbon accounting.
4. Treating hosted API billing as physical energy measurement.
5. A complete Kimi-K2.5 factorial sweep.
6. Comparing every inference engine on every scenario.

### 3.3 Research questions

| ID | Question | Primary evidence |
|---|---|---|
| RQ1 | How accurately does the Energy Twin rank energy and SLO outcomes? | Blind parity, calibration, rank correlation |
| RQ2 | Does adaptive fidelity reduce GPU-hours at matched frontier quality? | Hypervolume/recall versus aggregate GPU-hours |
| RQ3 | Does the method transfer across model architectures and GPU generations? | Leave-one-model and leave-one-GPU holdouts |
| RQ4 | When do single-GPU or intra-node probes fail to predict target topology? | L1/L2 to L3/L4 rank inversion and residuals |
| RQ5 | Does the guarded LLM planner add value beyond deterministic acquisition? | Planner ablation, recovery, invalid-action rate, time/tokens |
| RQ6 | Are final recommendations better than engine defaults and expert settings? | Matched-SLO energy, TTFT, TPOT, and goodput |

## 4. Hardware Plan

Hardware below means scheduled access, not hardware ownership. Aggregate
GPU-hours are `number of GPUs x wall-clock hours`; node-hours and GPU-hours must
not be mixed.

### 4.1 Three resource tiers

| Tier | GPU access | CPU access | What can be claimed |
|---|---|---|---|
| Minimum publishable | Two nodes with 4-8 identical GPUs for one main architecture; 1-4 GPUs each on the other two generations | One 32-64 core, 128 GB RAM CPU node | Multi-fidelity search, one real multi-node case, limited cross-generation transfer |
| Recommended | Two 8-GPU H200 or B200 nodes; one 8-GPU H100 node; one 8-GPU node of the remaining generation | Four 32-64 core, 128-256 GB RAM CPU nodes | Strong topology study, H100/H200/B200 transfer, dense/MoE/hybrid model panel |
| Ideal | Two 8-GPU nodes each for H100, H200, and B200 | 4-8 CPU nodes plus a dedicated orchestration node | Balanced cross-generation multi-node matrix and stronger external validity |

**Recommended request:** obtain one exclusive two-node, 16-GPU allocation on
H200 or B200 for L3/L4, plus repeated access to one 8-GPU H100, H200, and B200
node. The nodes do not have to be reserved continuously; campaigns can be
scheduled in blocks.

### 4.2 Why these GPU types matter

| GPU | Memory relevant to this study | Role |
|---|---:|---|
| H100 SXM | 80 GB HBM3 | Constrained-memory baseline and continuity with existing evidence |
| H200 | 141 GB HBM3e | Tests whether more memory changes optimal topology and batching |
| B200 | 180 GB HBM3e | Tests Blackwell transfer, larger KV capacity, and high-power operation |

The exact server form factor must be recorded. An isolated PCIe GPU is not
equivalent to an HGX/DGX node with NVSwitch. Official NVIDIA specifications list
H100 SXM at 80 GB and up to 700 W, H200 at 141 GB, and B200 at up to 180 GB:

- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 specifications](https://www.nvidia.com/es-la/data-center/h200/)
- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/Blackwell_Tuning_Guide.pdf)

### 4.3 Required topology properties

For every admitted GPU node, inventory and preserve:

- GPU model, UUID, memory, VBIOS, driver, and CUDA compatibility;
- PCIe tree, NUMA placement, NVLink/NVSwitch topology, and MIG state;
- CPU sockets, cores, RAM, local NVMe, and filesystem mount type;
- NIC type, link rate, InfiniBand/RoCE status, and GPUDirect RDMA support;
- Fabric Manager status on HGX/DGX NVSwitch systems;
- DCGM, NCCL, container runtime, and scheduler versions;
- default/min/max power limit and supported clocks;
- time synchronization and per-node clock skew.

Use the existing read-only collector:

```bash
bash experiments/mlsys27/collect_cluster_inventory.sh \
  > experiments/results/environment/<node>-inventory.txt
```

### 4.4 Allocation mode

Measured campaigns require exclusive access to the selected GPUs. Prefer:

1. exclusive Slurm allocation;
2. fixed node list and GPU UUIDs;
3. no MIG unless MIG is explicitly the experimental condition;
4. no colocated GPU workload;
5. a 5-10 minute thermal stabilization period before a balanced campaign;
6. randomized or cyclically balanced run order;
7. model server kept warm across compatible repetitions.

## 5. Root and Administrator Requirements

### 5.1 Short answer

**The agent and normal experiments should not run as root.** Root or cluster
administrator access is required once for host setup and for a small set of
hardware controls. A scheduler, wrapper, or prolog/epilog should expose only the
required operations.

### 5.2 Permission matrix

| Task | Root/admin needed? | Recommended owner |
|---|---|---|
| Install/upgrade NVIDIA driver | Yes | Cluster admin |
| Install Fabric Manager on NVSwitch systems | Yes | Cluster admin |
| Install and enable DCGM host engine | Yes | Cluster admin |
| Install/configure NVIDIA Container Toolkit or CDI | Yes, once | Cluster admin |
| Configure Docker/containerd/Podman/Enroot/Apptainer | Yes, once | Cluster admin |
| Mount shared model storage and set quotas | Yes | Storage/admin team |
| Download pinned model files into a user cache | No | Research user |
| Start vLLM/SGLang inside an allocated job | No | Research user |
| Read permitted DCGM/NVML telemetry | Normally no after setup | Research user |
| Change power caps or application clocks | Usually yes | Restricted wrapper or Slurm prolog |
| Reset a failed GPU or restart host services | Yes | Cluster admin |
| Run CPU L0 simulation and policy replay | No | Research user |

DCGM exposes `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` as field 156 in
millijoules. The preflight must verify that ordinary jobs can read this counter
and that it is monotonic:

- [NVIDIA DCGM field identifiers](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)

### 5.3 Preferred rootless execution path

Choose the runtime already supported by the cluster:

1. **Slurm + Pyxis/Enroot** for DGX/HGX clusters;
2. **Apptainer** where Docker daemons are prohibited;
3. **Podman with NVIDIA CDI** for rootless user containers;
4. Docker only when the site explicitly supports it.

NVIDIA documents rootless Docker and recommends CDI for Podman GPU access. The
admin still performs the initial toolkit/CDI setup:

- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NVIDIA CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

Do not run the complete agent with `sudo`. If power cap is an experiment axis,
provide a narrow command such as:

```text
tpa-set-power-limit <allocated-gpu-uuid> <approved-watts>
```

The wrapper must verify allocation ownership, apply an approved range, log the
change, and restore the default in job epilog. If the site cannot provide this,
hold power at its default and remove power cap from the primary search space.

### 5.4 Administrator request checklist

Request the following in one ticket:

- compatible production driver on H100/H200/B200 nodes;
- Fabric Manager only where the node contains NVSwitch/NVLink fabric requiring it;
- active DCGM service and read access to fields 150, 155, 156, 157, and 160;
- approved container runtime with pinned-image support;
- exclusive-node Slurm mode and stable GPU UUID assignment;
- read access to NCCL topology and NIC counters;
- optional constrained power-limit wrapper;
- 2 TB shared persistent storage and 500 GB local NVMe scratch per active node;
- outbound model-registry access from a staging host, or an approved offline import path;
- clarification whether raw BMC/PDU node-power telemetry is available.

## 6. Model and Storage Plan

### 6.1 Core model panel

| Model | Role | Current upstream repository size | Minimum practical deployment |
|---|---|---:|---|
| Qwen2.5-7B-Instruct | Existing dense anchor | 15.2 GB | 1 GPU |
| Qwen3-32B | Dense core model | about 65.5 GB | 1 H200/B200 or 2 H100 recommended |
| Qwen3-30B-A3B | MoE core model | about 61.1 GB | 1 H200/B200 or 2 H100 recommended |
| Kimi-Linear-48B-A3B-Instruct | Hybrid/linear-attention MoE | 98.3 GB | 4 GPUs recommended for long-context serving |
| Llama 70B instruct checkpoint | Conventional dense reference | revision dependent; budget about 140-160 GB | 2-4 GPUs |
| Kimi-K2.5 | Optional scale-out case | 595 GB | 8 H200/B200 GPUs; preflight before admission |

The first four model repositories total about **240.1 GB** before cache,
container, result, and scratch overhead. Verify exact byte counts after pinning
the revisions:

- [Qwen2.5-7B files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [Qwen3-32B files](https://huggingface.co/Qwen/Qwen3-32B/tree/main)
- [Qwen3-30B-A3B files](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/main)
- [Kimi-Linear files](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/main)
- [Kimi-K2.5 files](https://huggingface.co/moonshotai/Kimi-K2.5/tree/main)

Repository size is not sufficient GPU memory. Runtime memory also includes KV
cache, CUDA graphs, allocator reserve, communication buffers, and framework
overhead. Every topology therefore needs an actual load-and-request preflight.

### 6.2 Storage tiers

| Storage tier | Capacity | Intended use |
|---|---:|---|
| Bare minimum | 1 TB shared + 250 GB local per node | Four core models, one engine image family, compressed results |
| Recommended | 2 TB shared + 500 GB local per node | Core models, Llama reference, vLLM/SGLang images, staging, raw telemetry |
| With Kimi-K2.5/TRT-LLM variants | 4 TB shared + 1 TB local on scale-out nodes | 595 GB checkpoint, alternate precision, containers, temporary conversion/build artifacts |

Recommended 2 TB shared allocation:

| Component | Reserved space |
|---|---:|
| Four core model snapshots | 241 GB |
| Llama 70B reference | 160 GB |
| Hugging Face/Xet cache and download staging | 200 GB |
| vLLM and SGLang images | 150 GB |
| Optional TensorRT-LLM image and artifacts | 250 GB |
| Raw telemetry, benchmark JSONL, logs, and manifests | 300 GB |
| Temporary copies and failed-run reserve | 300 GB |
| Free-space safety margin | 399 GB |
| **Total** | **2 TB** |

Kimi-K2.5 adds at least 595 GB of weights. Reserve another 1-2 TB for its
staging, alternate precision, and runtime artifacts instead of squeezing it
into the core allocation.

### 6.3 Download and cache procedure

Do not use an unpinned `main` branch in a measurement campaign. For each model:

1. record the model license and access requirements;
2. select a commit SHA;
3. download once to shared persistent storage;
4. generate a file manifest and SHA-256 list;
5. mount the snapshot read-only into every serving container;
6. stage it to local NVMe before the timed campaign when shared-I/O variability
   affects startup;
7. exclude download and model-load time from steady-state serving energy;
8. report load time separately if deployment startup is studied.

Example staging pattern:

```bash
hf download Qwen/Qwen3-32B \
  --revision <commit-sha> \
  --local-dir /shared/tpa/models/qwen3-32b/<commit-sha>

find /shared/tpa/models/qwen3-32b/<commit-sha> -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /shared/tpa/manifests/qwen3-32b-<commit-sha>.sha256
```

For multi-node vLLM, every worker should see the same model path and software
environment. A common read-only mount or deterministic node-local staging
provides that contract.

## 7. Inference Strategy

### 7.1 Primary engine: vLLM

Use one immutable vLLM image digest for the main matrix. vLLM supports tensor
parallelism, pipeline parallelism, and Ray-based multi-node execution:

- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)

Primary reasons:

- continuity with existing Qwen2.5-7B measurements;
- broad serving-knob and distributed-parallel support;
- OpenAI-compatible client and mature benchmark tooling;
- direct comparison to relevant MLSys serving work.

### 7.2 Secondary engine: SGLang

Run SGLang only on a selected replication subset:

- one dense model;
- one MoE or Kimi model;
- one Hopper and one Blackwell node;
- default and agent-selected configurations.

This determines whether the agent's ranking is tied to one engine. It is not a
second full factorial matrix. Kimi-Linear's official model page provides both
vLLM and SGLang launch paths.

### 7.3 Optional optimized baseline: TensorRT-LLM

Use a prebuilt release container or pip package before considering a source
build. Run it on one frozen subset only. Current official documentation orders
prebuilt container/pip installation ahead of source build:

- [TensorRT-LLM installation](https://nvidia.github.io/TensorRT-LLM/latest/installation/index.html)

TensorRT-LLM is useful as a strong optimized endpoint, but full coverage would
consume disproportionate engineering time and confound engine optimization with
the evidence-selection contribution.

### 7.4 Other inference options

| Option | Allowed role | Why it is not the primary energy experiment |
|---|---|---|
| Hugging Face Transformers | Correctness and one-request smoke test | Not a production serving baseline |
| Hosted API | LLM planner endpoint only | Hardware placement and physical telemetry are hidden |
| NIM/private endpoint | Valid only with exclusive known GPUs and DCGM/PDU access | Otherwise energy cannot be attributed |
| Quantized checkpoint | Separate precision/deployment ablation | Changes quality, memory, and numerical behavior |
| CPU offload | Explicit heterogeneous-serving study only | Must include CPU/node energy boundary |
| Synthetic GEMM/model | Infrastructure smoke test | Cannot support LLM-serving conclusions |
| CPU replay/Energy Twin | L0 search and policy evaluation | Prediction, not final deployment evidence |

Cloud or hosted inference may accelerate planner development, but it cannot
replace real H100/H200/B200 serving for the paper's energy claims.

## 8. Evidence Ladder and Runtime Resources

| Level | Name | Hardware | Typical action | Publication contract |
|---|---|---|---|---|
| L0 | Estimate | CPU only | Predict all valid configurations and uncertainty | Never a final recommendation |
| L1 | Phase Probe | 1 target GPU | Short warmed serving/phase measurement | Calibrates compute, memory, and workload residuals |
| L2 | Node Probe | 2-8 GPUs in one node | Reduced but valid TP/PP/EP serving run | Calibrates intra-node communication and placement |
| L3 | Cluster Probe | 2 target nodes | Sparse multi-node serving measurement | Calibrates network, placement, and scale residuals |
| L4 | Target Verify | Exact deployment | Full workload with independent repeats | Required for every final deployment claim |

L0 can run on ordinary CPU partitions. It reduces scarce GPU demand but does
not simulate arbitrary multi-node behavior without calibration. L2 and L3 are
real measurements, not CPU simulations.

## 9. Experiment Matrix

### 9.1 Core breadth matrix

Use three core architectures:

1. Qwen3-32B dense;
2. Qwen3-30B-A3B MoE;
3. Kimi-Linear-48B-A3B hybrid MoE.

Evaluate them across H100, H200, and B200 and three primary workload strata:

| Workload | Initial setting | Purpose |
|---|---|---|
| Interactive | input 256-1K, output 128, concurrency 1-8 | TTFT-sensitive serving |
| Throughput | input 2K, output 128, concurrency 32-128 | Saturation and goodput |
| Long context | input 8K-32K, output 128-256 | KV/memory and prefill pressure |
| Bursty holdout | frozen Poisson or trace-derived arrivals | Unseen transfer and scope-gate validation |

The initial core breadth is therefore `3 models x 3 GPU generations x 3
development workloads = 27 scenario cells`. Physically invalid cells are marked
unsupported; they are not silently replaced with another precision.

Qwen2.5-7B is a regression/calibration anchor, not a fourth full-matrix model.
Llama 70B is a selected literature-facing reference. Kimi-K2.5 is admitted only
after the 27-cell core methodology works.

### 9.2 Configuration dimensions

| Family | Knobs |
|---|---|
| Scheduler | `max_num_seqs`, `max_num_batched_tokens`, scheduling policy |
| Cache | prefix caching, chunked prefill, KV dtype, cache capacity |
| Memory | GPU memory utilization, maximum model length |
| Parallelism | TP, PP, DP/replicas, EP for supported MoE paths |
| Placement | rank-to-GPU, NUMA/NIC affinity, node count |
| Hardware | default power and optional approved power cap |
| Precision | BF16 primary; FP8/NVFP4 only in a separate ablation |

### 9.3 Adaptive measurement counts

For each valid development scenario:

| Stage | Maximum candidates | Repeats | Selection |
|---|---:|---:|---|
| L0 | 64-128 | deterministic or 20 posterior samples | All valid configurations |
| L1 | 6-12 | 3 | High information per GPU-hour plus diversity |
| L2 | 3-6 | 3 | Candidates whose topology can change the frontier |
| L3 | 2-4 in selected scenarios | 3 | Highest unresolved cross-node uncertainty |
| L4 | 2-3 finalists plus defaults | 5 | Independent target confirmation |

Do not run L3 in all 27 cells. Select approximately 12 representative cells
covering dense/MoE/hybrid, Hopper/Blackwell, and interactive/throughput/long
context conditions.

### 9.4 Topology study

The focused topology matrix should include:

- Qwen3-32B and Qwen3-30B-A3B;
- H200 or B200 as the main topology platform;
- 1, 2, 4, and 8 GPUs within one node when valid;
- 2 nodes with 4 or 8 GPUs per node;
- TP, PP, and EP/DP combinations supported by the engine;
- one interactive and one throughput/long-context workload.

Kimi-Linear receives 4-GPU and 8-GPU runs plus one two-node probe if software
support is stable. Do not promise a 1M-token sweep; start at 32K and use 128K as
the stretch long-context point.

### 9.5 Baselines

Required search baselines under the same real GPU-hour budget:

1. engine default;
2. documented expert configuration;
3. random search;
4. cheapest-first;
5. cost-blind Bayesian optimization;
6. cost-aware constrained multi-objective BO/qNEHVI;
7. exhaustive oracle on bounded spaces or a frozen pooled oracle;
8. deterministic acquisition without the LLM planner.

Build a frozen real-measurement corpus first, then run 500-1,000 policy replay
episodes per policy. In addition, run at least three genuinely online episodes
for the strongest competing policies on six representative scenarios. Replay
alone is insufficient for the strongest systems claim.

### 9.6 Required ablations

1. no L0 Estimate;
2. L0 only;
3. fixed ladder rather than adaptive fidelity;
4. no L2 Node Probe;
5. no L3 Cluster Probe;
6. no cross-model transfer;
7. no cross-GPU transfer;
8. no uncertainty calibration;
9. no scope guard/abstention;
10. no LLM planner;
11. no reserved L4 verification budget.

## 10. Metrics and Statistical Protocol

### 10.1 Deployment metrics

- GPU joules per 1K output tokens;
- GPU joules per 1K total tokens;
- optional node joules per 1K tokens when BMC/PDU data are available;
- output-token throughput and SLO-attaining goodput;
- P50/P95/P99 TTFT, TPOT, and end-to-end latency;
- average/peak power, temperature, clocks, and utilization;
- OOM, timeout, crash, retry, and SLO violation rates.

### 10.2 Search metrics

- normalized dominated hypervolume;
- verified Pareto recall and precision;
- best feasible energy regret at matched SLO;
- aggregate GPU-hours to first verified Pareto point;
- GPU-hours to 90% and 95% oracle hypervolume;
- unnecessary L3/L4 actions;
- wall-clock time to recommendation;
- CPU simulation time and planner time reported separately.

### 10.3 Agent metrics

- planner P50/P95 latency;
- prompt/completion/total tokens;
- typed-output and endpoint success rates;
- raw proposal acceptance and guard intervention rates;
- invalid/forbidden proposal acceptance count;
- recovery success after OOM, unsupported topology, or telemetry failure;
- utility over deterministic acquisition at matched search budget.

### 10.4 Repetition and inference

- development measurements: at least 3 independent repetitions;
- L4 and headline comparisons: at least 5 independent paired repetitions;
- policy runs: at least 3 live seeds and 500 replay episodes per policy;
- report median and dispersion for serving metrics;
- use paired differences for default-versus-agent confirmation;
- include bootstrap confidence intervals and nonparametric tests where suitable;
- report all preregistered failures and unsupported cases.

## 11. Measurement Boundary

The primary GPU-energy boundary is the sum of synchronized DCGM energy-counter
deltas over the warmed request window on every allocated GPU:

```text
E_gpu = sum_g (counter_end[g] - counter_start[g])
```

Record:

- start/end counter for each GPU UUID;
- marker skew across nodes;
- telemetry sample interval and loss;
- whether server startup, warmup, graph capture, and cooldown are excluded;
- client-side start/end time;
- server logs and benchmark request IDs;
- node energy separately when available.

Never add estimated CPU/network energy to measured GPU energy and call the sum a
measurement. Report GPU-only and node-level boundaries as separate results.

## 12. Estimated Compute Budget

These values are planning estimates and must be revised after the 32B and Kimi
smoke tests establish real load and run times.

| Work package | Minimum GPU-hours | Recommended GPU-hours |
|---|---:|---:|
| Driver/runtime/topology smoke tests | 50 | 100 |
| Core 27-cell L1/L2 corpus | 300 | 600 |
| Cross-generation blind transfer | 150 | 300 |
| Focused multi-GPU/multi-node L3 study | 250 | 600 |
| L4 defaults/finalists and independent confirmations | 250 | 500 |
| Live search baselines and ablations | 150 | 400 |
| SGLang/TRT-LLM selected replication | 50 | 150 |
| Failure recovery and rerun reserve | 150 | 350 |
| **Total** | **1,350** | **3,000** |

Request **3,000 aggregate GPU-hours** across H100/H200/B200, with permission to
reallocate between types. If resource pressure is high, protect in this order:

1. L4 independent confirmation;
2. strongest baseline comparisons;
3. one real multi-node L3 study;
4. cross-generation holdouts;
5. SGLang replication;
6. TensorRT-LLM and Kimi-K2.5 stretch cases.

CPU request:

- 20,000-40,000 CPU-core-hours for L0 sweeps, posterior sampling, BO replay,
  bootstrap analysis, and figure generation;
- one always-available 8-16 core orchestration/login environment;
- four 32-64 core workers preferred for parallel policy replay.

## 13. Execution Schedule

### Phase 0: Resource freeze, 2026-08-05 to 2026-08-09

- submit admin/storage/GPU requests;
- inventory all available H100/H200/B200 nodes;
- choose the main two-node architecture;
- freeze core models and licenses;
- replace every floating image/model version with a digest/SHA.

**Exit gate:** at least one valid 8-GPU node and a credible two-node path are
identified; 2 TB storage is allocated.

### Phase 1: Runtime and model preflight, 2026-08-10 to 2026-08-21

- install/validate rootless cluster runtime;
- validate DCGM on every GPU UUID;
- stage Qwen3-32B, Qwen3-30B-A3B, and Kimi-Linear;
- run NCCL all-reduce/all-to-all and one-request serving smoke tests;
- measure model load time and peak memory;
- freeze valid TP/PP/EP ranges.

**Exit gate:** monotonic energy counters, stable request completion, pinned
containers, and no unresolved topology/fabric failure.

### Phase 2: Core single-node corpus, 2026-08-22 to 2026-09-04

- run L0 for all valid configurations;
- run balanced L1/L2 campaigns on the main development split;
- fit workload, hardware, and topology residuals;
- freeze blind H200/B200 or leave-one-model predictions before measurements.

**Exit gate:** prediction intervals are calibrated enough to support promotion
decisions; low-fidelity rankings are not trivially random.

### Phase 3: Transfer and topology, 2026-09-05 to 2026-09-18

- execute blind cross-generation campaigns;
- run 1/2/4/8-GPU intra-node measurements;
- run selected two-node L3 probes;
- analyze rank inversions and scope-gate failures;
- add Kimi-Linear 32K and optional 128K context cases.

**Exit gate:** at least one valid multi-node result and one held-out hardware or
model transfer result are complete.

### Phase 4: Agent and baseline study, 2026-09-19 to 2026-10-02

- freeze replay corpus and pooled oracle;
- run random, cheapest-first, cost-blind, and cost-aware BO;
- run ServeCompass and deterministic no-LLM ablation;
- run live episodes on six representative scenarios;
- measure planner latency, token use, guard interventions, and recovery.

**Exit gate:** the main algorithmic result is reproducible under equal
GPU-hour budgets.

### Phase 5: Independent confirmation, 2026-10-03 to 2026-10-09

- freeze finalists before observing confirmation data;
- run five paired repetitions of default, expert, and agent recommendation;
- rerun failed/ambiguous cells according to the preregistered policy;
- complete SGLang and optional TensorRT-LLM subset;
- stop adding new model families.

**Exit gate:** all headline claims have independent L4 evidence and confidence
intervals.

### Phase 6: Paper and artifact freeze, 2026-10-10 to 2026-10-16

- lock raw artifacts read-only;
- generate all tables/figures from checked-in analysis scripts;
- complete the 10-page anonymous paper and separate appendix;
- run artifact reproduction from a clean environment;
- conduct one internal systems review and one statistics review.

### Phase 7: Submission buffer, 2026-10-17 to 2026-10-23

- respond only to correctness, clarity, anonymity, and formatting issues;
- do not add an unvalidated headline experiment;
- update dates and formatting immediately when the official MLSys 2027 CFP is
  released.

## 14. Go/No-Go Gates

### G0: Resource feasibility

Pass if:

- H100/H200/B200 access is confirmed;
- at least one same-architecture two-node path exists;
- model storage and an approved GPU container runtime exist;
- DCGM energy is readable without running the agent as root.

If G0 fails, remove multi-node claims or change the paper scope before running a
large corpus.

### G1: Model/runtime feasibility

Pass if every core model loads, completes a warm request, and provides stable
telemetry on at least one intended topology. Unsupported cells remain explicit.

### G2: Twin usefulness

Target criteria, frozen before holdout:

- energy rank correlation at least 0.8 on development data;
- useful interval coverage near the nominal level;
- clear scope abstention where TTFT or topology transfer is unreliable.

Failure means improve the residual model or narrow the claim, not widen the
interval until every point is covered.

### G3: Search contribution

Target criteria:

- at least 90% verified Pareto recall;
- at least 95% oracle hypervolume on bounded scenarios;
- at least 25% fewer real GPU-hours than the strongest single-fidelity baseline;
- no final recommendation without L4 evidence.

These are success targets, not results to claim before measurement.

### G4: Agentic contribution

The LLM planner must improve recovery, valid subgoal selection, or search cost
over deterministic acquisition. If it does not, honestly present the planner as
an interface/recovery layer and make the multi-fidelity optimizer the primary
algorithmic contribution.

## 15. Data and Reproducibility Contract

Every run must record:

- git commit and dirty-state flag;
- model/tokenizer IDs, commit SHAs, and file-manifest SHA;
- container reference and immutable image digest;
- GPU UUIDs, topology inventory SHA, driver, DCGM, and NCCL versions;
- exact command, environment variables, seed, and scheduler job ID;
- scenario/candidate/fidelity IDs and frozen manifest SHA;
- raw client output, server log, DCGM trace, and energy counters;
- status, failure reason, and retry relationship;
- measurement-window markers and timestamps.

Recommended result layout:

```text
experiments/mlsys27/
  inventory/
  manifests/
  campaigns/
  raw/
    <campaign>/<scenario>/<run-id>/
  processed/
  reports/
  figures/
  archives/
```

Raw data are append-only. Derived CSVs, plots, and paper macros are regenerated
from scripts. Hashes do not improve an algorithm's accuracy; they prove that the
reported analysis corresponds to the preregistered predictions and preserved
measurements.

## 16. Paper Plan

Assuming the 2026 ten-page body format remains applicable:

| Section | Pages | Evidence required |
|---|---:|---|
| Introduction | 1.0 | Problem, insight, contributions, headline results |
| Motivation and related work | 1.25 | Rank inversion, gap versus BEAM/BOute/Charon/PROMPTS |
| Problem formulation | 0.75 | Joint configuration-fidelity objective and budgets |
| System design | 2.0 | L0-L4 ladder, Twin, planner/guards, executor |
| Algorithm | 1.0 | Acquisition, uncertainty, promotion, stopping |
| Implementation | 0.5 | vLLM, Ray, DCGM, Slurm/container runtime |
| Evaluation setup | 1.0 | Models, GPUs, workloads, baselines, statistics |
| Results | 2.0 | Frontier/GPU-hours, transfer, topology, deployment outcome |
| Limitations/conclusion | 0.5 | Scope, energy boundary, future work |

The paper should lead with one end-to-end summary table comparing engine default,
expert setting, strongest BO baseline, and ServeCompass on:

- energy per 1K output tokens;
- P95 TTFT and TPOT;
- goodput;
- verified frontier recall/hypervolume;
- real GPU-hours consumed;
- wall-clock recommendation time.

## 17. Principal Risks and Fallbacks

| Risk | Early signal | Fallback |
|---|---|---|
| No two-node same-GPU allocation | G0 inventory fails | Remove L3 generality; retain single-node multi-GPU paper only if claims are rewritten |
| Kimi software instability | Load/request smoke fails | Keep Qwen dense+MoE core; move Kimi to appendix or omit |
| Kimi-K2.5 storage/compute explosion | 4 TB or 8 H200/B200 unavailable | Omit without weakening the core paper |
| L0 accuracy poor out of distribution | Low rank correlation/coverage | Scope gate, model-specific calibration, more L1 diversity |
| Agent does not beat deterministic policy | Planner ablation ties/wins | Reframe planner as guarded interface; center fidelity acquisition |
| Energy gain over default remains small | L4 confirms small difference | Lead with GPU-hour/frontier result; report energy honestly |
| Power cap permission denied | Admin wrapper unavailable | Fix default power and remove cap from core matrix |
| BMC/PDU power unavailable | No node telemetry | Report GPU energy only with an explicit boundary |
| Storage I/O distorts startup | Variable load time | Pre-stage local NVMe and exclude load from serving window |
| Framework drift | Container/image changes | Pin digest; treat upgrades as a new campaign |

## 18. Immediate Action List

### Within 24 hours

1. Send the administrator request from Section 5.4.
2. Run the inventory collector on every candidate H100/H200/B200 node.
3. Confirm whether any allocation provides two same-architecture nodes.
4. Reserve 2 TB shared storage and 500 GB local scratch per active node.
5. Create the model-license/revision registry.

### Within 72 hours

1. Pin vLLM, SGLang, client, DCGM, and NCCL versions.
2. Download Qwen3-32B and Qwen3-30B-A3B once to shared storage.
3. Run one load, one warm request, one measured request, and one restart test on
   each GPU generation.
4. Run NCCL all-reduce and all-to-all on 2/4/8 GPUs.
5. Record real load time, peak memory, and serving duration to replace the
   provisional GPU-hour budget.

### First experiment to freeze

Use Qwen3-32B with one interactive and one throughput workload on H100, H200,
and B200. Search only `max_num_seqs`, `max_num_batched_tokens`, chunked prefill,
prefix caching, and valid TP. This campaign validates the cross-generation
pipeline before MoE, Kimi, EP, PP, or multi-node complexity is introduced.

## 19. Final Resource Request Summary

Request the following now:

```text
GPU compute:
  3,000 aggregate GPU-hours, reallocatable across H100/H200/B200
  Preferred topology: two 8-GPU H200 or B200 nodes
  Additional access: one 8-GPU node for each remaining generation
  Exclusive GPUs during measured windows

CPU compute:
  20,000-40,000 CPU-core-hours
  Four 32-64 core, 128-256 GB workers preferred

Storage:
  2 TB shared persistent storage for the core study
  500 GB local NVMe scratch per active node
  4 TB shared only if Kimi-K2.5/TRT-LLM variants are admitted

Privileges:
  No root for the agent or normal experiments
  Admin setup for driver, Fabric Manager where required, DCGM, runtime/CDI
  Optional restricted power-cap wrapper with automatic restore

Network:
  Same-fabric multi-node allocation with InfiniBand or approved RoCE
  Model-registry egress from staging host or offline snapshot import
```

This resource level is sufficient for a strong MLSys study if the adaptive
fidelity mechanism prevents a full factorial GPU sweep. The paper's value is
precisely that most candidates are eliminated on CPU or with short probes, while
real target deployments are reserved for uncertainty reduction and final
verification.
