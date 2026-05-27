#!/usr/bin/env python3
"""Replay-based CUDA v4 training smoke proxy.

The transferred bundle contains real gradient/model rounds, not a full
train/eval dataloader. This smoke compares CPU safe grouped and CUDA v4 q8 on
the same replay rounds and records training-adjacent signals.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping

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
from falcom_cuda_v4_wrapper import (  # noqa: E402
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    decompress_batch_cuda_v4,
    dumps_cuda_v4_layers,
    loads_cuda_v4_layers,
)

PROFILE = PROJECT_ROOT / "cuda_feasibility" / "profile_appfl_hot_rounds.py"
sys.path.insert(0, str(PROFILE.parent))
from profile_appfl_hot_rounds import read_round  # noqa: E402


FIELDS = [
    "model",
    "round_index",
    "path",
    "threshold",
    "quant_mode",
    "round_time_ms",
    "compress_ms",
    "decompress_ms",
    "compression_ratio",
    "gradient_norm",
    "relative_l2_error",
    "finite_rate",
    "loss",
    "accuracy",
    "status",
]


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


def flatten_model(model: Mapping[str, torch.Tensor], keys: List[str] | None = None) -> torch.Tensor:
    pieces = []
    names = keys if keys is not None else list(model.keys())
    for name in names:
        tensor = model[name]
        t = tensor.detach()
        if t.is_cuda:
            t = t.cpu()
        pieces.append(t.reshape(-1).to(dtype=torch.float32))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def metrics(reference: Mapping[str, torch.Tensor], decoded: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    keys = list(reference.keys())
    if any(name not in decoded for name in keys):
        return {"gradient_norm": 0.0, "relative_l2_error": float("inf"), "finite_rate": 0.0}
    ref = flatten_model(reference, keys)
    out = flatten_model(decoded, keys)
    if ref.numel() != out.numel() or ref.numel() == 0:
        return {"gradient_norm": 0.0, "relative_l2_error": float("inf"), "finite_rate": 0.0}
    diff = ref - out
    denom = torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
    finite = torch.isfinite(out).to(torch.float32).mean().item()
    return {
        "gradient_norm": float(torch.linalg.vector_norm(ref).item()),
        "relative_l2_error": float((torch.linalg.vector_norm(diff) / denom).item()),
        "finite_rate": float(finite),
    }


def cpu_round(comp: FalComC, decomp: FalComC, model: OrderedDict, client_id: str, original_bytes: int) -> Dict[str, object]:
    safe_sync()
    t0 = time.perf_counter()
    payload = comp.compress_model(model, client_id=client_id)
    safe_sync()
    t1 = time.perf_counter()
    decoded = decomp.decompress_model(payload, client_id=client_id)
    safe_sync()
    t2 = time.perf_counter()
    m = metrics(model, decoded)
    return {
        "path": "cpu_safe_grouped",
        "compress_ms": (t1 - t0) * 1000.0,
        "decompress_ms": (t2 - t1) * 1000.0,
        "round_time_ms": (t2 - t0) * 1000.0,
        "compression_ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
        **m,
    }


def gpu_round(
    c_state: FalcomCudaV4State,
    d_state: FalcomCudaV4State,
    model: OrderedDict,
    original_bytes: int,
    threshold: int,
) -> Dict[str, object]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    safe_sync()
    t0 = time.perf_counter()
    layers, _stats = compress_batch_cuda_v4(
        model,
        state_handle=c_state,
        options={"cuda_min_numel": threshold, "quant_mode": "8"},
    )
    payload = dumps_cuda_v4_layers(layers)
    safe_sync()
    t1 = time.perf_counter()
    decoded, _dstats = decompress_batch_cuda_v4(loads_cuda_v4_layers(payload), state_handle=d_state)
    safe_sync()
    t2 = time.perf_counter()
    m = metrics(model, decoded)
    return {
        "path": "cuda_v4_q8",
        "compress_ms": (t1 - t0) * 1000.0,
        "decompress_ms": (t2 - t1) * 1000.0,
        "round_time_ms": (t2 - t0) * 1000.0,
        "compression_ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
        **m,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=262144)
    args = parser.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"

    rows: List[Dict[str, object]] = []
    for model_name in args.models:
        cpu_comp = FalComC(config())
        cpu_decomp = FalComC(config())
        gpu_c_state = FalcomCudaV4State(client_id=f"{model_name}_smoke_v4", config=config())
        gpu_d_state = FalcomCudaV4State(client_id=f"{model_name}_smoke_v4", config=config())
        data_dir = PROJECT_ROOT / "dataset" / model_name
        for idx in range(args.rounds):
            _round_idx, client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
            for result in (
                cpu_round(cpu_comp, cpu_decomp, model, client_id, original_bytes),
                gpu_round(gpu_c_state, gpu_d_state, model, original_bytes, args.threshold),
            ):
                status = "pass" if result["finite_rate"] == 1.0 and result["relative_l2_error"] <= 0.03 else "drift"
                rows.append(
                    {
                        "model": model_name,
                        "round_index": idx,
                        "threshold": args.threshold,
                        "quant_mode": "8",
                        "loss": "not_available",
                        "accuracy": "not_available",
                        "status": status,
                        **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in result.items()},
                    }
                )

    logs = PROJECT_ROOT / "logs" / "cuda_v4"
    write_csv(logs / "training_smoke.csv", rows)
    report = logs / "training_smoke.md"
    with report.open("w") as f:
        f.write("# CUDA v4 Training Smoke Proxy\n\n")
        f.write("This is a replay smoke, not a full train/eval run. Loss and accuracy are unavailable in the transferred dataset.\n\n")
        f.write("| model | round | path | round ms | ratio | grad norm | rel L2 | finite | status |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['round_index']} | {row['path']} | {row['round_time_ms']} | "
                f"{row['compression_ratio']} | {row['gradient_norm']} | {row['relative_l2_error']} | "
                f"{row['finite_rate']} | {row['status']} |\n"
            )
    print(report)


if __name__ == "__main__":
    main()
