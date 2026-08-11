"""Patch the pinned vLLM benchmark with TokenPower measurement gates."""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path


spec = importlib.util.find_spec("vllm.benchmarks.serve")
if spec is None or spec.origin is None:
    raise RuntimeError("cannot locate vllm.benchmarks.serve")

path = Path(spec.origin)
source = path.read_text(encoding="utf-8")

start = "    benchmark_start_time = time.perf_counter()\n"
start_replacement = (
    "    from tokenpower_marker import measurement_gate as _tpa_measurement_gate\n"
    "    _tpa_measurement_gate.start()\n"
    + start
)
end = "    benchmark_duration = time.perf_counter() - benchmark_start_time\n"
end_replacement = end + "    _tpa_measurement_gate.end()\n"

if source.count(start) != 1 or source.count(end) != 1:
    raise RuntimeError("pinned vLLM benchmark no longer matches the marker patch")

path.write_text(
    source.replace(start, start_replacement).replace(end, end_replacement),
    encoding="utf-8",
)
py_compile.compile(str(path), doraise=True)
