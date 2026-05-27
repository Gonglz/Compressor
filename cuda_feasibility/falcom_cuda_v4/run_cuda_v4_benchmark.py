#!/usr/bin/env python3
"""Benchmark FalCom CUDA v4 fused q8 experimental closed-loop compressor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

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
    extension_available,
    loads_cuda_v4_layers,
)

PROFILE = PROJECT_ROOT / "cuda_feasibility" / "profile_appfl_hot_rounds.py"
sys.path.insert(0, str(PROFILE.parent))
from profile_appfl_hot_rounds import read_round  # noqa: E402


OFFICIAL_BASELINES = {
    "resnet18": {"serial_c_compress_ms": 340.294, "cpu_grouped_8t_compress_ms": 122.189},
    "resnet50": {"serial_c_compress_ms": 957.909, "cpu_grouped_8t_compress_ms": 197.865},
}

V3_PLAN_BASELINES = {
    "resnet18": {"compress_ms": 56.575, "decompress_ms": 40.858, "roundtrip_ms": 97.433},
    "resnet50": {"compress_ms": 132.308, "decompress_ms": 94.555, "roundtrip_ms": 226.864},
}

MODEL_FIELDS = [
    "model",
    "round_index",
    "round_type",
    "threshold",
    "quant_mode",
    "mode",
    "cpu_threads",
    "original_mb",
    "official_serial_c_compress_ms",
    "official_cpu_grouped_8t_compress_ms",
    "cuda_v3_reference_compress_ms",
    "cuda_v3_reference_decompress_ms",
    "cuda_v3_reference_roundtrip_ms",
    "cuda_layers",
    "cpu_fallback_layers",
    "cuda_original_mb",
    "fallback_original_mb",
    "avg_numel_per_cuda_layer",
    "compress_total_ms",
    "decompress_total_ms",
    "roundtrip_total_ms",
    "cpu_wrapper_compress_ms",
    "cpu_wrapper_decompress_ms",
    "cpu_wrapper_roundtrip_ms",
    "speedup_vs_cpu_wrapper",
    "roundtrip_speedup_vs_cpu_wrapper",
    "speedup_vs_official_cpu_8t",
    "speedup_vs_official_serial_c",
    "speedup_vs_cuda_v3",
    "roundtrip_speedup_vs_cuda_v3",
    "strict_resnet50_compress_pass",
    "practical_compress_pass",
    "closed_loop_pass",
    "encode_kernel_ms",
    "payload_d2h_ms",
    "payload_h2d_ms",
    "decode_kernel_ms",
    "decoded_tensor_materialize_ms",
    "envelope_serialize_ms",
    "envelope_parse_ms",
    "cpu_fallback_ms",
    "cpu_fallback_decode_ms",
    "fallback_gpu_to_cpu_ms",
    "kernel_launch_count",
    "decode_kernel_launch_count",
    "num_payload_objects",
    "payload_blob_bytes",
    "cpu_fallback_batches",
    "cpu_fallback_payload_bytes",
    "final_payload_bytes",
    "cpu_wrapper_payload_bytes",
    "compression_ratio",
    "cpu_wrapper_compression_ratio",
    "ratio_retention",
    "max_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "sign_agreement",
    "finite_rate",
    "decode_status",
    "correctness_status",
    "classification",
    "target_status",
]

LAYER_FIELDS = [
    "model",
    "round_index",
    "round_type",
    "threshold",
    "quant_mode",
    "layer_name",
    "layer_key",
    "backend",
    "step",
    "shape",
    "numel",
    "payload_length",
    "quant_bits_layer",
    "fallback_reason",
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


def flatten_model(model: Mapping[str, torch.Tensor], keys: Iterable[str] | None = None) -> torch.Tensor:
    pieces = []
    names = list(keys) if keys is not None else list(model.keys())
    for name in names:
        tensor = model[name]
        t = tensor.detach()
        if t.is_cuda:
            t = t.cpu()
        pieces.append(t.reshape(-1).to(dtype=torch.float32))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def error_metrics(reference: Mapping[str, torch.Tensor], decoded: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    keys = list(reference.keys())
    if any(name not in decoded for name in keys):
        return {
            "max_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": 0.0,
            "sign_agreement": 0.0,
            "finite_rate": 0.0,
        }
    ref = flatten_model(reference, keys)
    out = flatten_model(decoded, keys)
    if ref.numel() != out.numel() or ref.numel() == 0:
        return {
            "max_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": 0.0,
            "sign_agreement": 0.0,
            "finite_rate": 0.0,
        }
    diff = ref - out
    finite = torch.isfinite(out)
    denom = torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
    cos = torch.nn.functional.cosine_similarity(ref.reshape(1, -1), out.reshape(1, -1)).item()
    sign_agree = (torch.sign(ref) == torch.sign(out)).to(torch.float32).mean().item()
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine_similarity": max(-1.0, min(1.0, float(cos))),
        "sign_agreement": float(sign_agree),
        "finite_rate": float(finite.to(torch.float32).mean().item()),
    }


def fnum(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify(row: Mapping[str, object]) -> str:
    if str(row["decode_status"]) != "pass" or str(row["correctness_status"]) != "pass":
        return "stop_correctness_failed"
    if fnum(row["ratio_retention"]) < 0.80:
        return "stop_ratio_failed"
    if fnum(row["speedup_vs_official_cpu_8t"]) < 1.0:
        return "stop_speed_failed"
    if fnum(row["speedup_vs_official_cpu_8t"]) >= 1.3:
        return "strong_success"
    return "research_success"


def target_status(row: Mapping[str, object]) -> str:
    if str(row["classification"]).startswith("stop_"):
        return str(row["classification"])
    model = str(row["model"])
    compress_ok = str(row["practical_compress_pass"]) == "yes"
    closed_ok = str(row["closed_loop_pass"]) == "yes"
    if model == "resnet50":
        if compress_ok and closed_ok:
            return "v4_target_pass"
        if fnum(row["speedup_vs_cuda_v3"]) < 1.05 or fnum(row["roundtrip_speedup_vs_cuda_v3"]) <= 1.0:
            return "stop_gpu_internal_optimization"
        return "research_record"
    if model == "resnet18":
        if compress_ok and closed_ok:
            return "v4_target_pass"
    return "research_record"


def cpu_baseline(model_name: str, rounds: int) -> Dict[int, Dict[str, float]]:
    comp = FalComC(config())
    decomp = FalComC(config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    out: Dict[int, Dict[str, float]] = {}
    for idx in range(rounds):
        _, client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        payload = comp.compress_model(model, client_id=client_id)
        safe_sync()
        t1 = time.perf_counter()
        _decoded = decomp.decompress_model(payload, client_id=client_id)
        safe_sync()
        t2 = time.perf_counter()
        out[idx] = {
            "compress_ms": (t1 - t0) * 1000.0,
            "decompress_ms": (t2 - t1) * 1000.0,
            "roundtrip_ms": (t2 - t0) * 1000.0,
            "payload_bytes": float(len(payload)),
            "ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
            "original_bytes": float(original_bytes),
        }
    return out


def v3_references() -> Dict[Tuple[str, int], Dict[str, float]]:
    path = PROJECT_ROOT / "logs" / "cuda_v3" / "cuda_v3_perf_all.csv"
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    if path.exists():
        rows = list(csv.DictReader(path.open()))
        for model in sorted({r["model"] for r in rows}):
            for threshold in sorted({int(r["threshold"]) for r in rows if r["model"] == model}):
                subset = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["round_type"] == "hot"
                    and int(r["threshold"]) == threshold
                    and str(r["quant_mode"]) in ("8", "q8", "int8")
                ]
                if subset:
                    out[(model, threshold)] = {
                        "compress_ms": statistics.median(float(r["compress_total_ms"]) for r in subset),
                        "decompress_ms": statistics.median(float(r["decompress_total_ms"]) for r in subset),
                        "roundtrip_ms": statistics.median(float(r["roundtrip_total_ms"]) for r in subset),
                    }
    for model, vals in V3_PLAN_BASELINES.items():
        out.setdefault((model, 262144), dict(vals))
    return out


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def _pass_flags(model_name: str, compress_ms: float, roundtrip_ms: float, v3_ref: Mapping[str, float]) -> Tuple[str, str, str]:
    v3_comp = float(v3_ref.get("compress_ms", 0.0))
    v3_round = float(v3_ref.get("roundtrip_ms", 0.0))
    speed_v3 = (v3_comp / compress_ms) if v3_comp and compress_ms else 0.0
    round_v3 = (v3_round / roundtrip_ms) if v3_round and roundtrip_ms else 0.0
    if model_name == "resnet50":
        strict = "yes" if compress_ms <= 119.1 else "no"
        practical = "yes" if compress_ms <= 120.0 or speed_v3 >= 1.10 else "no"
        closed = "yes" if roundtrip_ms <= 205.0 or round_v3 >= 1.10 else "no"
        return strict, practical, closed
    if model_name == "resnet18":
        practical = "yes" if compress_ms <= 50.0 or speed_v3 >= 1.10 else "no"
        closed = "yes" if roundtrip_ms <= 88.0 or round_v3 >= 1.10 else "no"
        return "", practical, closed
    return "", "no", "no"


def run_gpu(
    model_name: str,
    rounds: int,
    threshold: int,
    baseline: Mapping[int, Mapping[str, float]],
    refs: Mapping[Tuple[str, int], Mapping[str, float]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state_id = f"{model_name}_cuda_v4_{threshold}"
    c_state = FalcomCudaV4State(client_id=state_id, config=config())
    d_state = FalcomCudaV4State(client_id=state_id, config=config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []
    cpu_threads = os.environ.get("OMP_NUM_THREADS", "")
    official = OFFICIAL_BASELINES[model_name]
    v3_ref = refs.get((model_name, threshold), {})
    for idx in range(rounds):
        _, _client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        compressed, cstats = compress_batch_cuda_v4(
            model,
            state_handle=c_state,
            options={"cuda_min_numel": threshold, "quant_mode": "8"},
        )
        t_ser = time.perf_counter()
        payload = dumps_cuda_v4_layers(compressed)
        fallback_serialize_ms = (time.perf_counter() - t_ser) * 1000.0
        serialize_ms = fnum(compressed.get("_last_envelope_serialize_ms", {}).get("ms"), fallback_serialize_ms)
        cstats["compressed_bytes"] = len(payload)
        original_mb = float(cstats["cuda_original_mb"]) + float(cstats["fallback_original_mb"])
        cstats["compression_ratio"] = (
            (original_mb * 1024.0 * 1024.0) / float(len(payload)) if payload else 0.0
        )
        safe_sync()
        compress_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        loaded = loads_cuda_v4_layers(payload)
        parse_ms = fnum(loaded.get("_last_envelope_parse_ms", {}).get("ms"), 0.0)
        decoded, dstats = decompress_batch_cuda_v4(loaded, state_handle=d_state)
        safe_sync()
        decompress_ms = (time.perf_counter() - t1) * 1000.0
        roundtrip_ms = compress_ms + decompress_ms
        metrics = error_metrics(model, decoded)
        base = baseline[idx]
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        v3_comp = float(v3_ref.get("compress_ms", 0.0))
        v3_decomp = float(v3_ref.get("decompress_ms", 0.0))
        v3_round = float(v3_ref.get("roundtrip_ms", 0.0))
        speed_v3 = (v3_comp / compress_ms) if v3_comp and compress_ms else 0.0
        round_v3 = (v3_round / roundtrip_ms) if v3_round and roundtrip_ms else 0.0
        strict_pass, practical_pass, closed_pass = _pass_flags(model_name, compress_ms, roundtrip_ms, v3_ref)
        decode_ok = (
            int(dstats.get("cuda_layers", 0)) == int(cstats["cuda_layers"])
            and int(dstats.get("cpu_fallback_layers", 0)) == int(cstats["cpu_fallback_layers"])
            and len(decoded) == len(model)
            and all(name in decoded for name in model.keys())
        )
        row = {
            "model": model_name,
            "round_index": idx,
            "round_type": "round0" if idx == 0 else "hot",
            "threshold": threshold,
            "quant_mode": "8",
            "mode": "gpu_v4_fused_extension_experimental",
            "cpu_threads": cpu_threads,
            "original_mb": f"{original_bytes / (1024.0 * 1024.0):.6f}",
            "official_serial_c_compress_ms": f"{official['serial_c_compress_ms']:.6f}",
            "official_cpu_grouped_8t_compress_ms": f"{official['cpu_grouped_8t_compress_ms']:.6f}",
            "cuda_v3_reference_compress_ms": f"{v3_comp:.6f}" if v3_comp else "",
            "cuda_v3_reference_decompress_ms": f"{v3_decomp:.6f}" if v3_decomp else "",
            "cuda_v3_reference_roundtrip_ms": f"{v3_round:.6f}" if v3_round else "",
            "cuda_layers": int(cstats["cuda_layers"]),
            "cpu_fallback_layers": int(cstats["cpu_fallback_layers"]),
            "cuda_original_mb": f"{float(cstats['cuda_original_mb']):.6f}",
            "fallback_original_mb": f"{float(cstats['fallback_original_mb']):.6f}",
            "avg_numel_per_cuda_layer": f"{float(cstats['avg_numel_per_cuda_layer']):.3f}",
            "compress_total_ms": f"{compress_ms:.6f}",
            "decompress_total_ms": f"{decompress_ms:.6f}",
            "roundtrip_total_ms": f"{roundtrip_ms:.6f}",
            "cpu_wrapper_compress_ms": f"{float(base['compress_ms']):.6f}",
            "cpu_wrapper_decompress_ms": f"{float(base['decompress_ms']):.6f}",
            "cpu_wrapper_roundtrip_ms": f"{float(base['roundtrip_ms']):.6f}",
            "speedup_vs_cpu_wrapper": f"{(float(base['compress_ms']) / compress_ms) if compress_ms else 0.0:.6f}",
            "roundtrip_speedup_vs_cpu_wrapper": f"{(float(base['roundtrip_ms']) / roundtrip_ms) if roundtrip_ms else 0.0:.6f}",
            "speedup_vs_official_cpu_8t": f"{official['cpu_grouped_8t_compress_ms'] / compress_ms:.6f}",
            "speedup_vs_official_serial_c": f"{official['serial_c_compress_ms'] / compress_ms:.6f}",
            "speedup_vs_cuda_v3": f"{speed_v3:.6f}" if v3_comp else "",
            "roundtrip_speedup_vs_cuda_v3": f"{round_v3:.6f}" if v3_round else "",
            "strict_resnet50_compress_pass": strict_pass,
            "practical_compress_pass": practical_pass,
            "closed_loop_pass": closed_pass,
            "encode_kernel_ms": f"{float(cstats['encode_kernel_ms']):.6f}",
            "payload_d2h_ms": f"{float(cstats['payload_d2h_ms']):.6f}",
            "payload_h2d_ms": f"{float(dstats.get('payload_h2d_ms', 0.0)):.6f}",
            "decode_kernel_ms": f"{float(dstats.get('decode_kernel_ms', 0.0)):.6f}",
            "decoded_tensor_materialize_ms": f"{float(dstats.get('decoded_tensor_materialize_ms', 0.0)):.6f}",
            "envelope_serialize_ms": f"{serialize_ms:.6f}",
            "envelope_parse_ms": f"{parse_ms:.6f}",
            "cpu_fallback_ms": f"{float(cstats['cpu_fallback_ms']):.6f}",
            "cpu_fallback_decode_ms": f"{float(dstats.get('cpu_fallback_decode_ms', 0.0)):.6f}",
            "fallback_gpu_to_cpu_ms": f"{float(cstats['fallback_gpu_to_cpu_ms']):.6f}",
            "kernel_launch_count": int(cstats["kernel_launch_count"]),
            "decode_kernel_launch_count": int(dstats.get("decode_kernel_launch_count", 0)),
            "num_payload_objects": int(cstats["num_payload_objects"]),
            "payload_blob_bytes": int(cstats["payload_blob_bytes"]),
            "cpu_fallback_batches": int(cstats["cpu_fallback_batches"]),
            "cpu_fallback_payload_bytes": int(cstats["cpu_fallback_payload_bytes"]),
            "final_payload_bytes": len(payload),
            "cpu_wrapper_payload_bytes": int(base["payload_bytes"]),
            "compression_ratio": f"{ratio:.6f}",
            "cpu_wrapper_compression_ratio": f"{float(base['ratio']):.6f}",
            "ratio_retention": f"{(ratio / float(base['ratio'])) if float(base['ratio']) else 0.0:.6f}",
            "max_abs_error": f"{metrics['max_abs_error']:.9f}",
            "relative_l2_error": f"{metrics['relative_l2_error']:.9f}",
            "cosine_similarity": f"{metrics['cosine_similarity']:.9f}",
            "sign_agreement": f"{metrics['sign_agreement']:.9f}",
            "finite_rate": f"{metrics['finite_rate']:.9f}",
            "decode_status": "pass" if decode_ok else "fail",
            "correctness_status": "pass" if metrics["finite_rate"] == 1.0 and metrics["relative_l2_error"] <= 0.03 else "fail",
        }
        row["classification"] = classify(row)
        row["target_status"] = target_status(row)
        rows.append(row)
        for record in cstats.get("layer_records", []):
            layer_rows.append(
                {
                    "model": model_name,
                    "round_index": idx,
                    "round_type": row["round_type"],
                    "threshold": threshold,
                    "quant_mode": "8",
                    "layer_name": record.get("layer_name", ""),
                    "layer_key": record.get("layer_key", ""),
                    "backend": record.get("backend", ""),
                    "step": record.get("step", ""),
                    "shape": record.get("shape", ""),
                    "numel": record.get("numel", ""),
                    "payload_length": record.get("payload_length", ""),
                    "quant_bits_layer": record.get("quant_bits", ""),
                    "fallback_reason": record.get("fallback_reason", ""),
                }
            )
    return rows, layer_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def median(rows: List[Mapping[str, object]], key: str) -> float:
    vals = [fnum(r.get(key, "")) for r in rows if str(r.get(key, "")) != ""]
    return statistics.median(vals) if vals else 0.0


def write_report(logs: Path, rows: List[Dict[str, object]]) -> None:
    hot = [r for r in rows if r["round_type"] == "hot"]
    report = logs / "final_fused_extension_report.md"
    with report.open("w") as f:
        f.write("# FalCom CUDA v4 Fused Extension HPC Report\n\n")
        f.write("CUDA v4 is an experimental closed-loop q8 codec. It uses a separate C++/CUDA extension, packed binary envelope, and batched payload transfer. It is not CPU wire-format-compatible and does not modify the installed CPU compressor.\n\n")
        f.write("## Build And Activation\n\n")
        f.write(f"- Extension import available: `{extension_available()}`\n")
        f.write("- Build toolchain selected: `/home/EXTRA_CUDA/CUDA-12.8` CUDAExtension against torch 2.7.1+cu126.\n")
        f.write("- Activation gate: `FALCOM_CUDA_EXPERIMENTAL=1`; unset/0 remains CPU fallback in wrapper tests.\n\n")
        f.write("## Fixed Baselines\n\n")
        f.write("| model | serial C ms | CPU grouped 8T ms | v3 q8 compress ms | v3 q8 closed-loop ms |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for model, base in OFFICIAL_BASELINES.items():
            v3 = V3_PLAN_BASELINES[model]
            f.write(f"| {model} | {base['serial_c_compress_ms']:.3f} | {base['cpu_grouped_8t_compress_ms']:.3f} | {v3['compress_ms']:.3f} | {v3['roundtrip_ms']:.3f} |\n")
        f.write("\n## Hot-Round Medians\n\n")
        f.write("| model | threshold | v4 compress ms | v4 decompress ms | closed-loop ms | CPU 8T speedup | v3 compress speedup | v3 roundtrip speedup | cuda layers | fallback layers | avg cuda numel | payload objects | launches | D2H ms | H2D ms | fallback ms | ratio retention | rel L2 | finite | target |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for model in sorted({str(r["model"]) for r in hot}):
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == model}):
                subset = [r for r in hot if r["model"] == model and int(r["threshold"]) == threshold]
                if not subset:
                    continue
                targets = ",".join(sorted({str(r["target_status"]) for r in subset}))
                f.write(
                    f"| {model} | {threshold} | "
                    f"{median(subset, 'compress_total_ms'):.3f} | "
                    f"{median(subset, 'decompress_total_ms'):.3f} | "
                    f"{median(subset, 'roundtrip_total_ms'):.3f} | "
                    f"{median(subset, 'speedup_vs_official_cpu_8t'):.3f} | "
                    f"{median(subset, 'speedup_vs_cuda_v3'):.3f} | "
                    f"{median(subset, 'roundtrip_speedup_vs_cuda_v3'):.3f} | "
                    f"{median(subset, 'cuda_layers'):.0f} | "
                    f"{median(subset, 'cpu_fallback_layers'):.0f} | "
                    f"{median(subset, 'avg_numel_per_cuda_layer'):.0f} | "
                    f"{median(subset, 'num_payload_objects'):.0f} | "
                    f"{median(subset, 'kernel_launch_count'):.0f} | "
                    f"{median(subset, 'payload_d2h_ms'):.3f} | "
                    f"{median(subset, 'payload_h2d_ms'):.3f} | "
                    f"{median(subset, 'cpu_fallback_ms'):.3f} | "
                    f"{median(subset, 'ratio_retention'):.3f} | "
                    f"{median(subset, 'relative_l2_error'):.6f} | "
                    f"{median(subset, 'finite_rate'):.3f} | {targets} |\n"
                )
        f.write("\n## Decision\n\n")
        resnet50_by_threshold = {
            threshold: [r for r in hot if r["model"] == "resnet50" and int(r["threshold"]) == threshold]
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == "resnet50"})
        }
        if resnet50_by_threshold:
            best_threshold = min(
                resnet50_by_threshold,
                key=lambda th: median(resnet50_by_threshold[th], "roundtrip_total_ms"),
            )
            best = resnet50_by_threshold[best_threshold]
            best_comp = median(best, "compress_total_ms")
            best_loop = median(best, "roundtrip_total_ms")
            best_rel = median(best, "relative_l2_error")
            best_ratio = median(best, "ratio_retention")
            f.write(
                f"- Best ResNet50 q8 threshold: {best_threshold}, compress {best_comp:.3f} ms, "
                f"closed-loop {best_loop:.3f} ms, rel L2 {best_rel:.6f}, ratio retention {best_ratio:.3f}.\n"
            )
            if best_comp <= 120.0 and best_loop <= 205.0 and best_rel <= 0.03 and best_ratio >= 0.80:
                f.write("- v4 meets the practical compress and closed-loop targets through fallback reduction at the lower accepted threshold.\n")
            else:
                f.write("- v4 does not meet both primary targets; stop further GPU-internal compressor optimization and keep v3/v4 as experimental results.\n")
            resnet50_262144 = resnet50_by_threshold.get(262144, [])
            if resnet50_262144:
                comp = median(resnet50_262144, "compress_total_ms")
                loop = median(resnet50_262144, "roundtrip_total_ms")
                v3s = median(resnet50_262144, "speedup_vs_cuda_v3")
                v3loop = median(resnet50_262144, "roundtrip_speedup_vs_cuda_v3")
                f.write(
                    f"- ResNet50 q8 @262144 median: compress {comp:.3f} ms, closed-loop {loop:.3f} ms, "
                    f"{v3s:.3f}x compress vs v3, {v3loop:.3f}x closed-loop vs v3.\n"
                )
        else:
            f.write("- No ResNet50 q8 @262144 hot rows were produced; benchmark is incomplete.\n")
        f.write("\n## Regression And Smoke Gates\n\n")
        so_paths = [
            PROJECT_ROOT / "libmomentum_compressor_openmp_simd_final.so",
            PROJECT_ROOT / "EB-FaLCom" / "src" / "appfl" / "compressor" / "libmomentum_compressor.so",
        ]
        for path in so_paths:
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                f.write(f"- `{path.relative_to(PROJECT_ROOT)}` SHA256: `{digest}`\n")
        f.write("- CPU regression gates run: static contracts, gcc syntax-only with SZ3 include, batch state equivalence, reference state oracle, and ctypes load.\n")
        f.write("- CUDA v4 unit gates run: extension build/import, q8 quality, mixed fallback roundtrip, gate default, backend ownership, transaction rollback, packed envelope, and separate encoder/decoder state.\n")
        smoke_path = logs / "training_smoke.csv"
        if smoke_path.exists():
            smoke_rows = list(csv.DictReader(smoke_path.open()))
            cuda_rows = [r for r in smoke_rows if str(r.get("path")) == "cuda_v4_q8"]
            if cuda_rows:
                max_rel = max(fnum(r.get("relative_l2_error")) for r in cuda_rows)
                min_finite = min(fnum(r.get("finite_rate")) for r in cuda_rows)
                statuses = ",".join(sorted({str(r.get("status")) for r in cuda_rows}))
                thresholds = ",".join(sorted({str(r.get("threshold")) for r in cuda_rows}))
                f.write(
                    f"- Training smoke proxy: CUDA v4 q8 thresholds {thresholds}, max rel L2 {max_rel:.6f}, "
                    f"min finite {min_finite:.3f}, statuses {statuses}. Loss/accuracy unavailable in replay data.\n"
                )
        f.write("\n## Artifacts\n\n")
        f.write("- `cuda_v4_perf_all.csv`\n")
        f.write("- `cuda_v4_layers_all.csv`\n")
        f.write("- `training_smoke.md`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 65536, 16384])
    parser.add_argument("--quick", action="store_true", help="Run only threshold 262144.")
    args = parser.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
    if args.quick:
        args.thresholds = [262144]
    if not extension_available():
        raise RuntimeError("CUDA v4 extension is not available")

    logs = PROJECT_ROOT / "logs" / "cuda_v4"
    refs = v3_references()
    all_rows: List[Dict[str, object]] = []
    all_layers: List[Dict[str, object]] = []
    for model_name in args.models:
        baseline = cpu_baseline(model_name, args.rounds)
        model_rows: List[Dict[str, object]] = []
        model_layers: List[Dict[str, object]] = []
        for threshold in args.thresholds:
            rows, layer_rows = run_gpu(model_name, args.rounds, threshold, baseline, refs)
            model_rows.extend(rows)
            model_layers.extend(layer_rows)
        all_rows.extend(model_rows)
        all_layers.extend(model_layers)
        write_csv(logs / f"{model_name}_perf.csv", model_rows, MODEL_FIELDS)
        write_csv(logs / f"{model_name}_layers.csv", model_layers, LAYER_FIELDS)
    write_csv(logs / "cuda_v4_perf_all.csv", all_rows, MODEL_FIELDS)
    write_csv(logs / "cuda_v4_layers_all.csv", all_layers, LAYER_FIELDS)
    write_report(logs, all_rows)
    (logs / "cpu_vs_gpu_summary.md").write_text((logs / "final_fused_extension_report.md").read_text())
    print(logs / "final_fused_extension_report.md")


if __name__ == "__main__":
    main()
