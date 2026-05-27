#!/usr/bin/env python3
"""APPFL/FalComC hot-round CUDA tensor profiling.

This runner keeps the current CPU compressor path and measures the existing
CUDA tensor -> CPU numpy -> ctypes -> C compress flow over round0 and hot
rounds.  It does not import or use the CUDA v0 compressor.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPFL_SRC = PROJECT_ROOT / "EB-FaLCom" / "src"
if str(APPFL_SRC) not in sys.path:
    sys.path.insert(0, str(APPFL_SRC))

from appfl.compressor.FalComC import FalComC  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float64,
    2: torch.int32,
    3: torch.int64,
    4: torch.uint8,
}


def read_cstr(buf: bytes) -> str:
    return buf.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def read_round(path: Path, device: str = "cuda") -> Tuple[int, str, OrderedDict, int]:
    model = OrderedDict()
    original_bytes = 0
    with path.open("rb") as f:
        round_idx = struct.unpack("<I", f.read(4))[0]
        client_id = read_cstr(f.read(64))
        layer_count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(layer_count):
            name = read_cstr(f.read(256))
            shape8 = struct.unpack("<8Q", f.read(64))
            ndim = struct.unpack("<Q", f.read(8))[0]
            dtype_code = struct.unpack("<I", f.read(4))[0]
            data_size = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(data_size)
            shape = tuple(int(x) for x in shape8[:ndim])
            dtype = DTYPE_TO_TORCH[dtype_code]
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
            if device == "cuda":
                tensor = tensor.cuda(non_blocking=False).contiguous()
            else:
                tensor = tensor.contiguous()
            model[name] = tensor
            original_bytes += int(data_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return round_idx, client_id, model, original_bytes


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


def parse_profile_delta(path: Path, offset: int) -> Tuple[List[Dict[str, str]], int]:
    if not path.exists():
        return [], offset
    text = path.read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return rows[offset:], len(rows)


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_model(model_name: str, args: argparse.Namespace) -> List[Dict[str, object]]:
    logs = PROJECT_ROOT / "logs" / "cuda_v0"
    logs.mkdir(parents=True, exist_ok=True)
    layer_profile = logs / f"appfl_hot_round_layers_{model_name}.csv"
    if layer_profile.exists():
        layer_profile.unlink()

    os.environ["FALCOM_APPFL_PROFILE"] = "1"
    os.environ["FALCOM_APPFL_PROFILE_CSV"] = str(layer_profile)
    compressor = FalComC(config())

    rows: List[Dict[str, object]] = []
    profile_offset = 0
    data_dir = PROJECT_ROOT / "dataset" / model_name
    for idx in range(args.rounds):
        round_file = data_dir / f"round_{idx}_client_0.bin"
        _round_idx, client_id, model, original_bytes = read_round(round_file, "cuda")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        payload = compressor.compress_model(model, client_id=client_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_total_ms = (time.perf_counter() - t0) * 1000.0
        profile_rows, profile_offset = parse_profile_delta(layer_profile, profile_offset)
        gpu_to_cpu = sum(float(r["gpu_to_cpu_numpy_ms"]) for r in profile_rows)
        ctypes_ms = sum(float(r["ctypes_build_ms"]) for r in profile_rows)
        c_ms = sum(float(r["c_compress_ms"]) for r in profile_rows)
        payload_ms = sum(float(r["payload_copy_ms"]) for r in profile_rows)
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        rows.append(
            {
                "model": model_name,
                "round_index": idx,
                "round_type": "round0" if idx == 0 else "hot",
                "num_layers": len(model),
                "original_mb": f"{original_bytes / (1024.0 * 1024.0):.6f}",
                "model_total_ms": f"{model_total_ms:.6f}",
                "gpu_to_cpu_numpy_ms": f"{gpu_to_cpu:.6f}",
                "ctypes_ms": f"{ctypes_ms:.6f}",
                "c_compress_ms": f"{c_ms:.6f}",
                "payload_copy_ms": f"{payload_ms:.6f}",
                "compression_ratio": f"{ratio:.6f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []
    logs = PROJECT_ROOT / "logs" / "cuda_v0"
    for model_name in args.models:
        rows = profile_model(model_name, args)
        all_rows.extend(rows)
        write_rows(logs / f"appfl_hot_round_{model_name}.csv", rows)

    summary = logs / "appfl_hot_round_summary.md"
    with summary.open("w") as f:
        f.write("# APPFL Hot-Round CUDA Tensor Profiling\n\n")
        f.write("| model | round | type | total ms | gpu->cpu ms | ctypes ms | C compress ms | payload ms | ratio |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for r in all_rows:
            f.write(
                f"| {r['model']} | {r['round_index']} | {r['round_type']} | "
                f"{r['model_total_ms']} | {r['gpu_to_cpu_numpy_ms']} | {r['ctypes_ms']} | "
                f"{r['c_compress_ms']} | {r['payload_copy_ms']} | {r['compression_ratio']} |\n"
            )
    print(summary)


if __name__ == "__main__":
    main()
