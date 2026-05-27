#!/usr/bin/env python3
"""Summarize CUDA v1a per-layer payload ratio gaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def top(rows: List[Dict[str, str]], field: str, n: int = 20) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: float(r.get(field, 0) or 0), reverse=True)[:n]


def summarize(model: str, logs: Path) -> None:
    rows = read_rows(logs / f"{model}_layer_records.csv")
    out_dir = logs
    fields = [
        "model",
        "threshold",
        "round_index",
        "round_type",
        "codec_mode",
        "layer_name",
        "shape",
        "numel",
        "backend",
        "cpu_payload_bytes",
        "cuda_v0_payload_bytes",
        "cuda_payload_bytes",
        "payload_growth_factor",
        "lost_compression_bytes",
        "full_residual_bytes",
        "compact_intermediate_d2h_bytes",
        "compact_vs_full_residual_pct",
        "cuda_kernel_ms",
        "cuda_payload_d2h_ms",
        "cpu_payload_assembly_ms",
        "cpu_reconstruct_input_ms",
        "cpu_encoder_ms",
        "cuda_state_commit_ms",
        "cpu_fallback_ms",
        "decode_status",
        "codec_failure_reason",
        "state_committed",
    ]
    write_rows(out_dir / f"payload_ratio_gap_{model}.csv", rows, fields)
    write_rows(out_dir / f"payload_ratio_gap_{model}_top_payload_bytes.csv", top(rows, "cuda_payload_bytes"), fields)
    write_rows(out_dir / f"payload_ratio_gap_{model}_top_growth_factor.csv", top(rows, "payload_growth_factor"), fields)
    write_rows(out_dir / f"payload_ratio_gap_{model}_top_lost_bytes.csv", top(rows, "lost_compression_bytes"), fields)
    write_rows(out_dir / f"payload_ratio_gap_{model}_top_cuda_time.csv", top(rows, "cuda_kernel_ms"), fields)


def write_summary(logs: Path, models: List[str]) -> None:
    summary = logs / "payload_ratio_gap_summary.md"
    with summary.open("w") as f:
        f.write("# CUDA v1a Payload Ratio Gap Summary\n\n")
        f.write("This report is generated from per-layer CUDA v1a benchmark records. `cpu_codec_full_residual_probe` is diagnostic-only; `hybrid_compact_v1a` is the compact candidate.\n\n")
        f.write("| model | codec | rows | total CPU payload MB | total CUDA payload MB | total lost MB | median compact % |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for model in models:
            rows = read_rows(logs / f"{model}_layer_records.csv")
            for codec in sorted({r.get("codec_mode", "") for r in rows}):
                subset = [r for r in rows if r.get("codec_mode") == codec]
                if not subset:
                    continue
                cpu_mb = sum(float(r.get("cpu_payload_bytes", 0) or 0) for r in subset) / (1024.0 * 1024.0)
                cuda_mb = sum(float(r.get("cuda_payload_bytes", 0) or 0) for r in subset) / (1024.0 * 1024.0)
                lost_mb = sum(float(r.get("lost_compression_bytes", 0) or 0) for r in subset) / (1024.0 * 1024.0)
                compact_values = sorted(float(r.get("compact_vs_full_residual_pct", 0) or 0) for r in subset if float(r.get("full_residual_bytes", 0) or 0) > 0)
                median_compact = compact_values[len(compact_values) // 2] if compact_values else 0.0
                f.write(f"| {model} | {codec} | {len(subset)} | {cpu_mb:.3f} | {cuda_mb:.3f} | {lost_mb:.3f} | {median_compact:.2f} |\n")
        f.write("\nTop-offender CSVs are written next to this summary for payload bytes, growth factor, and lost compression bytes.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--logs", type=Path, default=PROJECT_ROOT / "logs" / "cuda_v1a")
    args = parser.parse_args()
    for model in args.models:
        summarize(model, args.logs)
    write_summary(args.logs, args.models)
    print(args.logs / "payload_ratio_gap_summary.md")


if __name__ == "__main__":
    main()
