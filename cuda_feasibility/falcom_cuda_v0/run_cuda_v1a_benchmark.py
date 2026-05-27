#!/usr/bin/env python3
"""Benchmark CUDA v1a residual codec modes against the CPU APPFL wrapper."""

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
from typing import Dict, Iterable, List, Mapping

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
from falcom_cuda_wrapper import (  # noqa: E402
    RESIDUAL_CODEC_FULL_PROBE,
    RESIDUAL_CODEC_HYBRID,
    RESIDUAL_CODEC_RAW,
    FalcomCudaState,
    compress_batch_cuda,
    dumps_compressed_layers,
)

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


def safe_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def payload_size(layer: Mapping[str, object]) -> int:
    return (
        len(layer.get("data", b""))  # type: ignore[arg-type]
        + len(layer.get("bitmap", b""))  # type: ignore[arg-type]
        + len(layer.get("dominant_signs", b""))  # type: ignore[arg-type]
    )


def cpu_baseline(model_name: str, rounds: int):
    comp = FalComC(config())
    out: Dict[int, Dict[str, object]] = {}
    data_dir = PROJECT_ROOT / "dataset" / model_name
    for idx in range(rounds):
        _, client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        payload = comp.compress_model(model, client_id=client_id)
        safe_sync()
        ms = (time.perf_counter() - t0) * 1000.0
        layers = pickle.loads(payload)
        out[idx] = {
            "model_total_ms": ms,
            "ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
            "payload_bytes": {name: payload_size(layer) for name, layer in layers.items()},
            "payload_total_bytes": len(payload),
        }
    return out


MODEL_FIELDS = [
    "model",
    "threshold",
    "round_type",
    "round_index",
    "codec_mode",
    "cuda_layers",
    "cpu_fallback_layers",
    "full_residual_bytes",
    "compact_intermediate_d2h_bytes",
    "compact_vs_full_residual_pct",
    "final_payload_bytes",
    "compact_intermediate_d2h_mb",
    "full_residual_d2h_avoided_mb",
    "cpu_payload_assembly_ms",
    "cpu_reconstruct_input_ms",
    "cpu_encoder_ms",
    "cuda_kernel_ms",
    "cuda_state_commit_ms",
    "model_total_ms",
    "cpu_baseline_model_total_ms",
    "speedup_vs_cpu_wrapper",
    "speedup_vs_v0_raw",
    "compression_ratio",
    "ratio_delta",
    "codec_failure_reason",
    "state_committed",
    "accepted_compact",
    "classification",
]

LAYER_FIELDS = [
    "model",
    "threshold",
    "round_index",
    "round_type",
    "codec_mode",
    "layer_name",
    "layer_key",
    "shape",
    "dtype",
    "numel",
    "backend",
    "c_type",
    "c_codec",
    "cpu_payload_bytes",
    "cuda_v0_payload_bytes",
    "cuda_payload_bytes",
    "full_residual_bytes",
    "compact_intermediate_d2h_bytes",
    "compact_vs_full_residual_pct",
    "final_payload_bytes",
    "payload_growth_factor",
    "lost_compression_bytes",
    "ratio_delta",
    "decode_status",
    "codec_failure_reason",
    "state_committed",
    "cuda_kernel_ms",
    "cuda_payload_d2h_ms",
    "cpu_payload_assembly_ms",
    "cpu_reconstruct_input_ms",
    "cpu_encoder_ms",
    "cuda_state_commit_ms",
    "cpu_fallback_ms",
]


def classify_row(row: Mapping[str, object]) -> str:
    ratio_delta = abs(float(row["ratio_delta"]))
    model_total = float(row["model_total_ms"])
    speedup = float(row["speedup_vs_cpu_wrapper"])
    compact_ok = int(float(row["accepted_compact"])) == 1
    mode = str(row["codec_mode"])
    if mode == RESIDUAL_CODEC_FULL_PROBE:
        return "diagnostic_only"
    if ratio_delta > 0.01:
        return "stop_ratio_failed"
    if mode == RESIDUAL_CODEC_HYBRID and not compact_ok:
        return "research_only_compact_budget_failed"
    if model_total <= 500.0:
        return "strong_success"
    if speedup >= 1.2 and model_total <= 800.0:
        return "research_only"
    return "stop_speed_failed"


def run_cuda_mode(
    model_name: str,
    rounds: int,
    threshold: int,
    codec_mode: str,
    baseline: Mapping[int, Mapping[str, object]],
):
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaState(client_id=f"{model_name}_cuda_v1a_{codec_mode}_{threshold}", config=config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    model_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []
    for idx in range(rounds):
        _, _client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        try:
            compressed, stats = compress_batch_cuda(
                model,
                state_handle=state,
                options={
                    "client_id": state.client_id,
                    "cuda_min_numel": threshold,
                    "momentum_lr": 0.07,
                    "consistency_threshold": 0.5,
                    "residual_codec": codec_mode,
                },
            )
            payload = dumps_compressed_layers(compressed)
            codec_failure = str(stats.get("codec_failure_reason", ""))
        except Exception as exc:
            safe_sync()
            stats = {
                "cuda_layers": 0,
                "cpu_fallback_layers": 0,
                "full_residual_bytes": 0,
                "compact_intermediate_d2h_bytes": 0,
                "compact_vs_full_residual_pct": 0,
                "final_payload_bytes": 0,
                "compact_intermediate_d2h_mb": 0,
                "full_residual_d2h_avoided_mb": 0,
                "cpu_payload_assembly_ms": 0,
                "cpu_reconstruct_input_ms": 0,
                "cpu_encoder_ms": 0,
                "cuda_kernel_ms": 0,
                "cuda_state_commit_ms": 0,
                "state_committed": 0,
                "accepted_compact": 0,
                "layer_records": [],
            }
            payload = b""
            codec_failure = repr(exc)
        safe_sync()
        total_ms = (time.perf_counter() - t0) * 1000.0
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        base_ms = float(baseline[idx]["model_total_ms"])
        base_ratio = float(baseline[idx]["ratio"])
        raw_row = {
            "model": model_name,
            "threshold": threshold,
            "round_type": "round0" if idx == 0 else "hot",
            "round_index": idx,
            "codec_mode": codec_mode,
            "cuda_layers": int(stats["cuda_layers"]),
            "cpu_fallback_layers": int(stats["cpu_fallback_layers"]),
            "full_residual_bytes": int(stats["full_residual_bytes"]),
            "compact_intermediate_d2h_bytes": int(stats["compact_intermediate_d2h_bytes"]),
            "compact_vs_full_residual_pct": f"{float(stats['compact_vs_full_residual_pct']):.6f}",
            "final_payload_bytes": int(stats["final_payload_bytes"]),
            "compact_intermediate_d2h_mb": f"{float(stats['compact_intermediate_d2h_mb']):.6f}",
            "full_residual_d2h_avoided_mb": f"{float(stats['full_residual_d2h_avoided_mb']):.6f}",
            "cpu_payload_assembly_ms": f"{float(stats['cpu_payload_assembly_ms']):.6f}",
            "cpu_reconstruct_input_ms": f"{float(stats['cpu_reconstruct_input_ms']):.6f}",
            "cpu_encoder_ms": f"{float(stats['cpu_encoder_ms']):.6f}",
            "cuda_kernel_ms": f"{float(stats['cuda_kernel_ms']):.6f}",
            "cuda_state_commit_ms": f"{float(stats['cuda_state_commit_ms']):.6f}",
            "model_total_ms": f"{total_ms:.6f}",
            "cpu_baseline_model_total_ms": f"{base_ms:.6f}",
            "speedup_vs_cpu_wrapper": f"{(base_ms / total_ms) if total_ms else 0.0:.6f}",
            "speedup_vs_v0_raw": "1.000000",
            "compression_ratio": f"{ratio:.6f}",
            "ratio_delta": f"{((ratio - base_ratio) / base_ratio) if base_ratio else 0.0:.6f}",
            "codec_failure_reason": codec_failure,
            "state_committed": int(float(stats["state_committed"])),
            "accepted_compact": int(float(stats["accepted_compact"])),
        }
        raw_row["classification"] = classify_row(raw_row)
        model_rows.append(raw_row)

        cpu_sizes = baseline[idx]["payload_bytes"]
        for record in stats.get("layer_records", []):
            layer = dict(record)
            name = str(layer.get("layer_name", ""))
            cpu_payload = int(cpu_sizes.get(name, 0)) if isinstance(cpu_sizes, dict) else 0
            cuda_payload = int(layer.get("cuda_payload_bytes", 0))
            layer["model"] = model_name
            layer["threshold"] = threshold
            layer["round_index"] = idx
            layer["round_type"] = "round0" if idx == 0 else "hot"
            layer["codec_mode"] = codec_mode
            layer["cpu_payload_bytes"] = cpu_payload
            if codec_mode == RESIDUAL_CODEC_RAW:
                layer["cuda_v0_payload_bytes"] = cuda_payload
            layer["lost_compression_bytes"] = max(0, cuda_payload - cpu_payload)
            layer["payload_growth_factor"] = (float(cuda_payload) / float(cpu_payload)) if cpu_payload else 0.0
            if cpu_payload:
                layer["ratio_delta"] = (float(cpu_payload) - float(cuda_payload)) / float(cpu_payload)
            layer_rows.append({field: layer.get(field, "") for field in LAYER_FIELDS})
    return model_rows, layer_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def fill_speedup_vs_raw(rows: List[Dict[str, object]]) -> None:
    raw_ms: Dict[tuple, float] = {}
    for row in rows:
        if row["codec_mode"] == RESIDUAL_CODEC_RAW:
            raw_ms[(row["model"], row["threshold"], row["round_index"])] = float(row["model_total_ms"])
    for row in rows:
        key = (row["model"], row["threshold"], row["round_index"])
        base = raw_ms.get(key, 0.0)
        total = float(row["model_total_ms"])
        row["speedup_vs_v0_raw"] = f"{(base / total) if total else 0.0:.6f}"


def write_report(path: Path, rows: List[Dict[str, object]]) -> None:
    hot = [r for r in rows if r["round_type"] == "hot"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# CUDA v1a Hybrid Residual Codec Report\n\n")
        f.write("CUDA remains experimental behind `FALCOM_CUDA_EXPERIMENTAL=1`; the installed CPU `.so` and CPU wire format are unchanged.\n\n")
        f.write("## Acceptance Tiers\n\n")
        f.write("- Strong success: `abs(ratio_delta) <= 1%` and ResNet50 hot `<= 500 ms`.\n")
        f.write("- Research-only: ratio passes and speedup is >=20% vs CPU wrapper, but hot time is above 500 ms.\n")
        f.write("- Stop: ratio fails, decompressor compatibility fails, compact budget fails, or speedup is below 20%.\n\n")
        f.write("| model | threshold | codec | hot median ms | CPU hot median ms | speedup vs CPU | speedup vs raw | ratio delta median | compact % | classification |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        for model in sorted({str(r["model"]) for r in hot}):
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == model}):
                for codec in [RESIDUAL_CODEC_RAW, RESIDUAL_CODEC_FULL_PROBE, RESIDUAL_CODEC_HYBRID]:
                    subset = [r for r in hot if r["model"] == model and int(r["threshold"]) == threshold and r["codec_mode"] == codec]
                    if not subset:
                        continue
                    gpu_med = statistics.median(float(r["model_total_ms"]) for r in subset)
                    cpu_med = statistics.median(float(r["cpu_baseline_model_total_ms"]) for r in subset)
                    speed_cpu = cpu_med / gpu_med if gpu_med else 0.0
                    speed_raw = statistics.median(float(r["speedup_vs_v0_raw"]) for r in subset)
                    ratio_delta = statistics.median(float(r["ratio_delta"]) for r in subset)
                    compact_pct = statistics.median(float(r["compact_vs_full_residual_pct"]) for r in subset)
                    classes = sorted({str(r["classification"]) for r in subset})
                    f.write(
                        f"| {model} | {threshold} | {codec} | {gpu_med:.3f} | {cpu_med:.3f} | "
                        f"{speed_cpu:.3f} | {speed_raw:.3f} | {ratio_delta:.6f} | {compact_pct:.2f} | {','.join(classes)} |\n"
                    )
        f.write("\n## Decision Notes\n\n")
        f.write("- `cpu_codec_full_residual_probe` is diagnostic only because it transfers full residual bytes before CPU encoding.\n")
        f.write("- `hybrid_compact_v1a` is the only compact candidate; if its ratio fails, v1a should stop rather than invent a new GPU-only payload format.\n")
        f.write("- Benchmark timing is compressor-wrapper timing only, matching the project scope.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 524288, 1048576])
    parser.add_argument(
        "--codec-modes",
        nargs="+",
        default=[RESIDUAL_CODEC_RAW, RESIDUAL_CODEC_FULL_PROBE, RESIDUAL_CODEC_HYBRID],
    )
    args = parser.parse_args()

    all_model_rows: List[Dict[str, object]] = []
    all_layer_rows: List[Dict[str, object]] = []
    logs = PROJECT_ROOT / "logs" / "cuda_v1a"
    for model in args.models:
        baseline = cpu_baseline(model, args.rounds)
        model_rows: List[Dict[str, object]] = []
        layer_rows: List[Dict[str, object]] = []
        for threshold in args.thresholds:
            for codec_mode in args.codec_modes:
                rows, layers = run_cuda_mode(model, args.rounds, threshold, codec_mode, baseline)
                model_rows.extend(rows)
                layer_rows.extend(layers)
        fill_speedup_vs_raw(model_rows)
        all_model_rows.extend(model_rows)
        all_layer_rows.extend(layer_rows)
        write_csv(logs / f"{model}_perf.csv", model_rows, MODEL_FIELDS)
        write_csv(logs / f"{model}_layer_records.csv", layer_rows, LAYER_FIELDS)
    fill_speedup_vs_raw(all_model_rows)
    write_csv(logs / "cuda_v1a_perf_all.csv", all_model_rows, MODEL_FIELDS)
    write_csv(logs / "cuda_v1a_layer_records_all.csv", all_layer_rows, LAYER_FIELDS)
    write_report(logs / "final_cuda_v1a_report.md", all_model_rows)
    print(logs / "final_cuda_v1a_report.md")


if __name__ == "__main__":
    main()
