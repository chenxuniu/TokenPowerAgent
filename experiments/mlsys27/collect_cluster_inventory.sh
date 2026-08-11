#!/usr/bin/env bash

# Read-only inventory for deciding whether a node can enter an MLSys campaign.
# The script deliberately avoids sudo and does not change GPU or service state.

set -u

section() {
  printf '\n===== %s =====\n' "$1"
}

run_if_available() {
  local executable="$1"
  shift
  if command -v "$executable" >/dev/null 2>&1; then
    "$executable" "$@" 2>&1 || true
  else
    printf '%s: not found\n' "$executable"
  fi
}

section "Capture"
if ! date -Is 2>/dev/null; then
  date -u '+%Y-%m-%dT%H:%M:%SZ'
fi
run_if_available hostname
run_if_available uname -a
run_if_available uptime

section "CPU and memory"
run_if_available lscpu
run_if_available free -h
run_if_available df -h

section "GPU identity"
run_if_available nvidia-smi -L
run_if_available nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,persistence_mode,ecc.mode.current,mig.mode.current,power.default_limit,power.min_limit,power.max_limit \
  --format=csv

section "GPU topology"
run_if_available nvidia-smi topo -m
run_if_available nvidia-smi nvlink -s

section "GPU health"
run_if_available nvidia-smi
run_if_available dcgmi --version
run_if_available dcgmi discovery -l

section "PCI devices"
if command -v lspci >/dev/null 2>&1; then
  lspci -nn 2>&1 | grep -Ei 'NVIDIA|Ethernet|InfiniBand|Network' || true
else
  printf 'lspci: not found\n'
fi

section "Network and fabric"
run_if_available ip -brief link
run_if_available ibv_devices
run_if_available ibv_devinfo -l
run_if_available ibstat

section "Runtime"
run_if_available docker --version
run_if_available nvidia-ctk --version
run_if_available python3 --version

section "Services"
if command -v systemctl >/dev/null 2>&1; then
  for service in docker nvidia-dcgm nvidia-persistenced nvidia-fabricmanager; do
    printf '%s: ' "$service"
    systemctl is-active "$service" 2>&1 || true
  done
else
  printf 'systemctl: not found\n'
fi

section "Scheduler context"
for variable in \
  SLURM_JOB_ID \
  SLURM_JOB_NUM_NODES \
  SLURM_NNODES \
  SLURM_GPUS_ON_NODE \
  SLURM_JOB_NODELIST; do
  printf '%s=%s\n' "$variable" "${!variable-}"
done

section "Repository"
if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
  git rev-parse --show-toplevel 2>&1 || true
  git rev-parse HEAD 2>&1 || true
  git status --short --branch 2>&1 || true
else
  printf 'not inside a Git repository\n'
fi
