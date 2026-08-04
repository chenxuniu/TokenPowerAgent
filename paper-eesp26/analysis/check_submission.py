#!/usr/bin/env python3
"""Check the TokenPowerSandbox short-paper source and rendered PDF."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "build/main.pdf"
PROVENANCE = ROOT / "results/holdout_v2_provenance.json"
SCOPE_PROVENANCE = ROOT / "results/scope_v3_provenance.json"
EVIDENCE_ROOT = ROOT.parent / "experiments/evidence/eesp26"


def command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    errors: list[str] = []

    required = [
        "main.tex",
        "references.bib",
        "results/metrics.tex",
        "results/holdout_v2_provenance.json",
        "results/scope_v3_provenance.json",
        "figures/sandbox-ladder.tex",
        "figures/scope-gate.tex",
        "tables/holdout-validation.tex",
        "tables/scope-confirmation.tex",
        "tables/roadmap.tex",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"missing required source: {relative}")

    tex_files = [ROOT / "main.tex"]
    tex_files.extend(sorted((ROOT / "sections").glob("*.tex")))
    tex_files.extend(sorted((ROOT / "figures").glob("*.tex")))
    tex_files.extend(sorted((ROOT / "tables").glob("*.tex")))
    tex_files.append(ROOT / "results/metrics.tex")
    source = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)

    blockers = {
        "FILL_ME": "unresolved site marker",
        "CALIBRATE_TO": "unresolved calibration marker",
        "\\draftvalue": "unmeasured result placeholder",
        "\\pending": "pending result placeholder",
        "Single-H100 Blind Pilot": "obsolete pilot section",
        "25.0\\%": "obsolete one-point MAPE",
        "38.4\\%": "obsolete one-point energy error",
        "Qwen2.5-32B": "obsolete eight-GPU experiment",
        "L0/L2": "ambiguous combined prediction-level label",
    }
    for marker, description in blockers.items():
        if marker in source:
            errors.append(f"{description}: {marker}")

    main_tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    if "Anonymous Authors" not in main_tex:
        errors.append("anonymous author block is missing")
    if "Evidence-Gated Multi-Fidelity Screening" not in main_tex:
        errors.append("HPC multi-fidelity title is missing")
    if "TokenPowerAgent:" in source:
        errors.append("TokenPowerAgent appears as the paper title/system")

    if PROVENANCE.exists():
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        for label, digest in provenance.get("artifacts", {}).items():
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                errors.append(f"invalid {label}: expected a 64-character SHA-256")
        chronology = provenance.get("chronology", {})
        for field in (
            "prediction_precedes_measurements",
            "prediction_manifest_verified",
            "raw_artifact_manifest_verified",
        ):
            if chronology.get(field) is not True:
                errors.append(f"provenance chronology check is false: {field}")
        protocol = provenance.get("protocol", {})
        expected_counts = {
            "development_workloads": 6,
            "development_measurements": 18,
            "holdout_workloads": 8,
            "holdout_measurements": 24,
            "raw_artifact_manifest_entries": 62,
        }
        for field, expected in expected_counts.items():
            if protocol.get(field) != expected:
                errors.append(
                    f"provenance {field} is {protocol.get(field)!r}, expected {expected}"
                )
        energy = provenance.get("blind_holdout", {}).get(
            "energy_j_per_1k_output_tokens", {}
        )
        if not (6.22 < energy.get("corrected_mape_pct", -1) < 6.24):
            errors.append("headline energy MAPE is inconsistent with the sealed report")
        if energy.get("pairwise_concordant_pairs") != 27:
            errors.append("headline energy pairwise count is inconsistent")

    if SCOPE_PROVENANCE.exists():
        provenance = json.loads(SCOPE_PROVENANCE.read_text(encoding="utf-8"))
        for label, digest in provenance.get("artifacts", {}).items():
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                errors.append(f"invalid scope-v3 {label}: expected SHA-256")
        chronology = provenance.get("chronology", {})
        for field in (
            "prediction_precedes_measurements",
            "prediction_manifest_verified",
            "raw_artifact_manifest_verified",
        ):
            if chronology.get(field) is not True:
                errors.append(f"scope-v3 chronology check is false: {field}")
        protocol = provenance.get("protocol", {})
        expected_counts = {
            "workloads": 9,
            "repeats_per_workload": 3,
            "measurements": 27,
            "raw_artifact_manifest_entries": 61,
        }
        for field, expected in expected_counts.items():
            if protocol.get(field) != expected:
                errors.append(
                    f"scope-v3 {field} is {protocol.get(field)!r}, expected {expected}"
                )
        primary = provenance.get("primary_endpoint", {})
        if not (7.34 < primary.get("observed_mape_pct", -1) < 7.36):
            errors.append("scope-v3 energy MAPE is inconsistent")
        if primary.get("passed") is not True:
            errors.append("scope-v3 primary endpoint did not pass")
        latency = provenance.get("latency_scope", {})
        if latency.get("decision") != (
            "support_latency_at_concurrency_ge_4_abstain_below_4"
        ):
            errors.append("scope-v3 latency decision is inconsistent")
        if not (9.26 < latency.get("supported_mape_pct", -1) < 9.28):
            errors.append("scope-v3 supported TTFT MAPE is inconsistent")
        if not (64.79 < latency.get("sparse_mape_pct", -1) < 64.81):
            errors.append("scope-v3 sparse TTFT MAPE is inconsistent")

    if EVIDENCE_ROOT.exists():
        evidence_sums = EVIDENCE_ROOT / "SHA256SUMS"
        if evidence_sums.exists():
            verification = subprocess.run(
                ["shasum", "-a", "256", "-c", str(evidence_sums)],
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
            )
            if verification.returncode != 0:
                errors.append(
                    "published EESP evidence SHA256SUMS verification failed"
                )
        else:
            errors.append("published EESP evidence SHA256SUMS is missing")

    if not PDF.exists():
        errors.append("build/main.pdf is missing")
    else:
        info = command("pdfinfo", str(PDF))
        pages = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
        if pages is None or int(pages.group(1)) != 5:
            errors.append("expected four body pages plus one references page")
        if not re.search(r"^Author:\s+Anonymous Authors$", info, re.MULTILINE):
            errors.append("PDF metadata is not anonymous")

        page_four = command(
            "pdftotext", "-f", "4", "-l", "4", "-layout", str(PDF), "-"
        )
        page_five = command(
            "pdftotext", "-f", "5", "-l", "5", "-layout", str(PDF), "-"
        )
        compact_page_four = re.sub(r"\s+", "", page_four).upper()
        compact_page_five = re.sub(r"\s+", "", page_five).upper()
        if "REFERENCES" in compact_page_four:
            errors.append("references begin inside the four-page body")
        if "ASSISTANCEDISCLOSURE" not in compact_page_four:
            errors.append("AI-assistance disclosure is not inside the body limit")
        if "REFERENCES" not in compact_page_five:
            errors.append("references heading is not on page five")

        fonts = command("pdffonts", str(PDF)).splitlines()[2:]
        for row in fonts:
            match = re.search(
                r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row
            )
            if match is None:
                errors.append(f"cannot parse pdffonts row: {row}")
            elif match.group(1) != "yes":
                errors.append(f"font is not embedded: {row.split()[0]}")

    log_path = ROOT / "build/main.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in (
            "Overfull \\hbox",
            "Overfull \\vbox",
            "undefined references",
            "undefined citations",
        ):
            if marker in log:
                errors.append(f"LaTeX log contains: {marker}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Submission source, provenance, and PDF gates passed.")


if __name__ == "__main__":
    main()
