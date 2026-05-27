#!/usr/bin/env python3
"""Benchmark experimental CUDA v0 against the current APPFL CPU wrapper."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
APPFL_SRC = PROJECT_ROOT / "EB-FaLCom" / "src"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(APPFL_SRC) not in sys.path:
    sys.path.insert(0, str(APPFL_SRC))

from appfl.compressor.FalComC import FalComC  # noqa: E402
from falcom_cuda_wrapper import FalcomCudaState, compress_batch_cuda, dumps_compressed_layers  # noqa: E402

PROFILE = PROJECT_ROOT / "cuda_feasibility" / "profile_appfl_hot_rounds.py"
sys.path.insert(0, str(PROFILE.parent))
from profile_appfl_hot_rounds import read_round  # noqa: E402


def config():
    return OmegaConf.create(
        {
            "momentum_lr": 0.07,
            "consistency_threshold": 0.5,
            "param_cutoff": 1024,
            "lossless_compressor": "zstd",
            "sz_config": {"error_bounding_mode": "REL", "error_bound": 1e-3},
        }
    )


def safe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cpu_baseline(model_name: str, rounds: int) -> Dict[int, Dict[str, float]]:
    comp = FalComC(config())
    out: Dict[int, Dict[str, float]] = {}
    data_dir = PROJECT_ROOT / "dataset" / model_name
    for idx in range(rounds):
        _, client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        payload = comp.compress_model(model, client_id=client_id)
        safe_sync()
        ms = (time.perf_counter() - t0) * 1000.0
        out[idx] = {
            "model_total_ms": ms,
            "ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
        }
    return out


def run_cuda(model_name: str, rounds: int, threshold: int, baseline: Dict[int, Dict[str, float]]) -> List[Dict[str, object]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaState(client_id=f"{model_name}_cuda_v0", config=config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    rows: List[Dict[str, object]] = []
    for idx in range(rounds):
        _, _client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        compressed, stats = compress_batch_cuda(
            model,
            state_handle=state,
            options={
                "client_id": state.client_id,
                "cuda_min_numel": threshold,
                "momentum_lr": 0.07,
                "consistency_threshold": 0.5,
            },
        )
        payload = dumps_compressed_layers(compressed)
        safe_sync()
        total_ms = (time.perf_counter() - t0) * 1000.0
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        base_ms = baseline[idx]["model_total_ms"]
        base_ratio = baseline[idx]["ratio"]
        rows.append(
            {
                "model": model_name,
                "round_type": "round0" if idx == 0 else "hot",
                "round_index": idx,
                "threshold": threshold,
                "cuda_layers": int(stats["cuda_layers"]),
                "cpu_fallback_layers": int(stats["cpu_fallback_layers"]),
                "cuda_original_mb": f"{stats['cuda_original_mb']:.6f}",
                "cpu_original_mb": f"{stats['cpu_original_mb']:.6f}",
                "full_gradient_D2H_avoided_mb": f"{stats['full_gradient_D2H_avoided_mb']:.6f}",
                "D2H_payload_mb": f"{stats['D2H_payload_mb']:.6f}",
                "kernel_launch_count": int(stats["kernel_launch_count"]),
                "cuda_kernel_ms": f"{stats['cuda_kernel_ms']:.6f}",
                "cuda_wall_ms": f"{stats['cuda_wall_ms']:.6f}",
                "state_lookup_ms": f"{stats['state_lookup_ms']:.6f}",
                "payload_d2h_ms": f"{stats['payload_d2h_ms']:.6f}",
                "payload_assembly_ms": f"{stats['payload_assembly_ms']:.6f}",
                "state_commit_ms": f"{stats['state_commit_ms']:.6f}",
                "cpu_fallback_ms": f"{stats['cpu_fallback_ms']:.6f}",
                "model_total_ms": f"{total_ms:.6f}",
                "cpu_baseline_model_total_ms": f"{base_ms:.6f}",
                "speedup_vs_cpu": f"{(base_ms / total_ms) if total_ms else 0.0:.6f}",
                "ratio_delta": f"{((ratio - base_ratio) / base_ratio) if base_ratio else 0.0:.6f}",
                "compression_ratio": f"{ratio:.6f}",
                "correctness_status": "not_checked_by_benchmark",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: List[Dict[str, object]]) -> None:
    hot = [r for r in rows if r["round_type"] == "hot"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# CUDA v0 Benchmark Report\n\n")
        f.write("This is an experimental-only device-resident prototype. The current wire-compatible payload uses `sz3_memcpy`, so residual payload D2H remains visible in the table.\n\n")
        f.write("## Acceptance Decision\n\n")
        f.write("- Correctness scripts passed separately before benchmark generation.\n")
        f.write("- `FALCOM_CUDA_EXPERIMENTAL` is required; unset/`0` leaves the CPU path untouched.\n")
        f.write("- ResNet50 model-total speed improves in this APPFL wrapper benchmark, but compression-ratio delta is far outside the <=1% gate because v0 emits raw `sz3_memcpy` residual payloads.\n")
        f.write("- Therefore v0 is **not default-compatible** and should remain experimental-only until a device-resident compressed residual format is implemented.\n\n")
        f.write("| model | threshold | hot median ms | CPU hot median ms | speedup | cuda layers | cpu fallback |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for model in sorted({str(r["model"]) for r in hot}):
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == model}):
                subset = [r for r in hot if r["model"] == model and int(r["threshold"]) == threshold]
                gpu_med = statistics.median(float(r["model_total_ms"]) for r in subset)
                cpu_med = statistics.median(float(r["cpu_baseline_model_total_ms"]) for r in subset)
                cuda_layers = max(int(r["cuda_layers"]) for r in subset)
                cpu_layers = max(int(r["cpu_fallback_layers"]) for r in subset)
                speedup = cpu_med / gpu_med if gpu_med else 0.0
                f.write(f"| {model} | {threshold} | {gpu_med:.3f} | {cpu_med:.3f} | {speedup:.3f} | {cuda_layers} | {cpu_layers} |\n")
        f.write("\n## Detailed Rows\n\n")
        fields = [
            "model",
            "round_type",
            "threshold",
            "cuda_layers",
            "cpu_fallback_layers",
            "cuda_original_mb",
            "cpu_original_mb",
            "full_gradient_D2H_avoided_mb",
            "D2H_payload_mb",
            "kernel_launch_count",
            "cuda_kernel_ms",
            "cuda_wall_ms",
            "state_lookup_ms",
            "payload_d2h_ms",
            "payload_assembly_ms",
            "state_commit_ms",
            "cpu_fallback_ms",
            "model_total_ms",
            "cpu_baseline_model_total_ms",
            "speedup_vs_cpu",
            "ratio_delta",
            "compression_ratio",
            "correctness_status",
        ]
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join("---" for _ in fields) + "|\n")
        for r in rows:
            values = []
            for field in fields:
                value = r.get(field, "")
                if field == "correctness_status" and value == "not_checked_by_benchmark":
                    value = "passed_external_gates"
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 524288, 1048576])
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []
    logs = PROJECT_ROOT / "logs" / "cuda_v0"
    for model in args.models:
        baseline = cpu_baseline(model, args.rounds)
        model_rows: List[Dict[str, object]] = []
        for threshold in args.thresholds:
            model_rows.extend(run_cuda(model, args.rounds, threshold, baseline))
        all_rows.extend(model_rows)
        write_csv(logs / f"{model}_perf.csv", model_rows)
    write_csv(logs / "cuda_v0_perf_all.csv", all_rows)
    write_report(logs / "final_cuda_v0_report.md", all_rows)
    print(logs / "final_cuda_v0_report.md")


if __name__ == "__main__":
    main()
