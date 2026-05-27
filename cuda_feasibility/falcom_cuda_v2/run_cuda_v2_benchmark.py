#!/usr/bin/env python3
"""Benchmark FalCom CUDA v2 experimental closed-loop compressor."""

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
from falcom_cuda_v2_wrapper import (  # noqa: E402
    FalcomCudaV2State,
    compress_batch_cuda_v2,
    decompress_batch_cuda_v2,
    dumps_cuda_v2_layers,
    loads_cuda_v2_layers,
)

PROFILE = PROJECT_ROOT / "cuda_feasibility" / "profile_appfl_hot_rounds.py"
sys.path.insert(0, str(PROFILE.parent))
from profile_appfl_hot_rounds import read_round  # noqa: E402


MODEL_FIELDS = [
    "model",
    "round_index",
    "round_type",
    "threshold",
    "quant_bits",
    "mode",
    "cpu_threads",
    "original_mb",
    "cuda_layers",
    "cpu_fallback_layers",
    "cuda_original_mb",
    "fallback_original_mb",
    "compress_total_ms",
    "decompress_total_ms",
    "roundtrip_total_ms",
    "cpu_safe_grouped_compress_ms",
    "cpu_safe_grouped_decompress_ms",
    "cpu_safe_grouped_roundtrip_ms",
    "speedup_vs_cpu_safe_grouped",
    "roundtrip_speedup_vs_cpu_safe_grouped",
    "kernel_ms",
    "payload_d2h_ms",
    "payload_serialize_ms",
    "payload_deserialize_ms",
    "cpu_fallback_ms",
    "fallback_gpu_to_cpu_ms",
    "experimental_payload_d2h_mb",
    "final_payload_bytes",
    "cpu_safe_grouped_payload_bytes",
    "compression_ratio",
    "cpu_safe_grouped_compression_ratio",
    "ratio_retention",
    "max_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "sign_agreement",
    "finite_rate",
    "decode_status",
    "correctness_status",
    "classification",
]

LAYER_FIELDS = [
    "model",
    "round_index",
    "round_type",
    "threshold",
    "quant_bits",
    "layer_name",
    "layer_key",
    "backend",
    "step",
    "shape",
    "numel",
    "payload_length",
    "quant_bits_layer",
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


def flatten_model(model: Mapping[str, torch.Tensor]) -> torch.Tensor:
    pieces = []
    for tensor in model.values():
        t = tensor.detach()
        if t.is_cuda:
            t = t.cpu()
        pieces.append(t.reshape(-1).to(dtype=torch.float32))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def error_metrics(reference: Mapping[str, torch.Tensor], decoded: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    ref = flatten_model(reference)
    out = flatten_model(decoded)
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
    cos = max(-1.0, min(1.0, float(cos)))
    sign_agree = (torch.sign(ref) == torch.sign(out)).to(torch.float32).mean().item()
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine_similarity": float(cos),
        "sign_agreement": float(sign_agree),
        "finite_rate": float(finite.to(torch.float32).mean().item()),
    }


def classify(row: Mapping[str, object]) -> str:
    if str(row["decode_status"]) != "pass" or str(row["correctness_status"]) != "pass":
        return "stop_correctness_failed"
    speedup = float(row["speedup_vs_cpu_safe_grouped"])
    roundtrip_speedup = float(row["roundtrip_speedup_vs_cpu_safe_grouped"])
    ratio_retention = float(row["ratio_retention"])
    rel_l2 = float(row["relative_l2_error"])
    finite_rate = float(row["finite_rate"])
    if speedup < 1.2:
        return "stop_speed_failed"
    if finite_rate < 1.0 or rel_l2 > 0.05:
        return "stop_error_failed"
    if speedup >= 2.0 and roundtrip_speedup >= 1.0 and ratio_retention >= 0.80:
        return "strong_success"
    return "research_success"


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
        compress_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        _decoded = decomp.decompress_model(payload, client_id=client_id)
        safe_sync()
        decompress_ms = (time.perf_counter() - t1) * 1000.0
        out[idx] = {
            "compress_ms": compress_ms,
            "decompress_ms": decompress_ms,
            "roundtrip_ms": compress_ms + decompress_ms,
            "payload_bytes": float(len(payload)),
            "ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
            "original_bytes": float(original_bytes),
        }
    return out


def run_gpu(
    model_name: str,
    rounds: int,
    threshold: int,
    quant_bits: int,
    baseline: Mapping[int, Mapping[str, float]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    c_state = FalcomCudaV2State(client_id=f"{model_name}_cuda_v2_{threshold}_{quant_bits}", config=config())
    d_state = FalcomCudaV2State(client_id=f"{model_name}_cuda_v2_{threshold}_{quant_bits}", config=config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []
    cpu_threads = os.environ.get("OMP_NUM_THREADS", "")
    for idx in range(rounds):
        _, _client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        compressed, cstats = compress_batch_cuda_v2(
            model,
            state_handle=c_state,
            options={"cuda_min_numel": threshold, "quant_bits": quant_bits},
        )
        payload = dumps_cuda_v2_layers(compressed)
        safe_sync()
        compress_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        t_load = time.perf_counter()
        loaded = loads_cuda_v2_layers(payload)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        decoded, dstats = decompress_batch_cuda_v2(loaded, state_handle=d_state)
        dstats["payload_deserialize_ms"] = float(dstats["payload_deserialize_ms"]) + load_ms
        safe_sync()
        decompress_ms = (time.perf_counter() - t1) * 1000.0
        roundtrip_ms = compress_ms + decompress_ms
        metrics = error_metrics(model, decoded)
        base = baseline[idx]
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        row = {
            "model": model_name,
            "round_index": idx,
            "round_type": "round0" if idx == 0 else "hot",
            "threshold": threshold,
            "quant_bits": quant_bits,
            "mode": "gpu_v2_experimental",
            "cpu_threads": cpu_threads,
            "original_mb": f"{original_bytes / (1024.0 * 1024.0):.6f}",
            "cuda_layers": int(cstats["cuda_layers"]),
            "cpu_fallback_layers": int(cstats["cpu_fallback_layers"]),
            "cuda_original_mb": f"{float(cstats['cuda_original_mb']):.6f}",
            "fallback_original_mb": f"{float(cstats['fallback_original_mb']):.6f}",
            "compress_total_ms": f"{compress_ms:.6f}",
            "decompress_total_ms": f"{decompress_ms:.6f}",
            "roundtrip_total_ms": f"{roundtrip_ms:.6f}",
            "cpu_safe_grouped_compress_ms": f"{float(base['compress_ms']):.6f}",
            "cpu_safe_grouped_decompress_ms": f"{float(base['decompress_ms']):.6f}",
            "cpu_safe_grouped_roundtrip_ms": f"{float(base['roundtrip_ms']):.6f}",
            "speedup_vs_cpu_safe_grouped": f"{(float(base['compress_ms']) / compress_ms) if compress_ms else 0.0:.6f}",
            "roundtrip_speedup_vs_cpu_safe_grouped": f"{(float(base['roundtrip_ms']) / roundtrip_ms) if roundtrip_ms else 0.0:.6f}",
            "kernel_ms": f"{float(cstats['kernel_ms']):.6f}",
            "payload_d2h_ms": f"{float(cstats['payload_d2h_ms']):.6f}",
            "payload_serialize_ms": f"{float(cstats['payload_serialize_ms']):.6f}",
            "payload_deserialize_ms": f"{float(dstats['payload_deserialize_ms']):.6f}",
            "cpu_fallback_ms": f"{float(cstats['cpu_fallback_ms']):.6f}",
            "fallback_gpu_to_cpu_ms": f"{float(cstats['fallback_gpu_to_cpu_ms']):.6f}",
            "experimental_payload_d2h_mb": f"{float(cstats['experimental_payload_d2h_mb']):.6f}",
            "final_payload_bytes": len(payload),
            "cpu_safe_grouped_payload_bytes": int(base["payload_bytes"]),
            "compression_ratio": f"{ratio:.6f}",
            "cpu_safe_grouped_compression_ratio": f"{float(base['ratio']):.6f}",
            "ratio_retention": f"{(ratio / float(base['ratio'])) if float(base['ratio']) else 0.0:.6f}",
            "max_abs_error": f"{metrics['max_abs_error']:.9f}",
            "relative_l2_error": f"{metrics['relative_l2_error']:.9f}",
            "cosine_similarity": f"{metrics['cosine_similarity']:.9f}",
            "sign_agreement": f"{metrics['sign_agreement']:.9f}",
            "finite_rate": f"{metrics['finite_rate']:.9f}",
            "decode_status": (
                "pass"
                if int(dstats["decoded_layers"]) == int(cstats["cuda_layers"])
                and int(dstats.get("cpu_fallback_decoded_layers", 0)) == int(cstats["cpu_fallback_layers"])
                and len(decoded) == len(model)
                else "fail"
            ),
            "correctness_status": "pass" if metrics["finite_rate"] == 1.0 and metrics["relative_l2_error"] <= 0.05 else "fail",
        }
        row["classification"] = classify(row)
        rows.append(row)
        for record in cstats.get("layer_records", []):
            rec = {
                "model": model_name,
                "round_index": idx,
                "round_type": row["round_type"],
                "threshold": threshold,
                "quant_bits": quant_bits,
                "layer_name": record.get("layer_name", ""),
                "layer_key": record.get("layer_key", ""),
                "backend": record.get("backend", ""),
                "step": record.get("step", ""),
                "shape": record.get("shape", ""),
                "numel": record.get("numel", ""),
                "payload_length": record.get("payload_length", ""),
                "quant_bits_layer": record.get("quant_bits", ""),
            }
            layer_rows.append(rec)
    return rows, layer_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_report(logs: Path, rows: List[Dict[str, object]]) -> None:
    hot = [r for r in rows if r["round_type"] == "hot"]
    report = logs / "final_gpu_acceleration_report.md"
    with report.open("w") as f:
        f.write("# FalCom CUDA v2 Experimental GPU Acceleration Report\n\n")
        f.write("CUDA v2 is an experimental closed-loop codec. It is not CPU-wire-compatible and does not replace the installed CPU compressor.\n\n")
        f.write("CPU baseline here is the FalComC wrapper using the installed safe grouped CPU `.so` on the same replay tensors, including wrapper and CUDA-tensor-to-CPU fallback costs where applicable. It is not the raw C-only microbenchmark number.\n\n")
        f.write("| model | threshold | qbits | hot compress ms | CPU compress ms | speedup | roundtrip speedup | ratio retention | rel L2 | finite | classification |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for model in sorted({str(r["model"]) for r in hot}):
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == model}):
                for qbits in sorted({int(r["quant_bits"]) for r in hot if r["model"] == model and int(r["threshold"]) == threshold}):
                    subset = [r for r in hot if r["model"] == model and int(r["threshold"]) == threshold and int(r["quant_bits"]) == qbits]
                    if not subset:
                        continue
                    med_gpu = statistics.median(float(r["compress_total_ms"]) for r in subset)
                    med_cpu = statistics.median(float(r["cpu_safe_grouped_compress_ms"]) for r in subset)
                    med_speed = statistics.median(float(r["speedup_vs_cpu_safe_grouped"]) for r in subset)
                    med_roundtrip = statistics.median(float(r["roundtrip_speedup_vs_cpu_safe_grouped"]) for r in subset)
                    med_ratio = statistics.median(float(r["ratio_retention"]) for r in subset)
                    med_l2 = statistics.median(float(r["relative_l2_error"]) for r in subset)
                    med_finite = statistics.median(float(r["finite_rate"]) for r in subset)
                    classes = ",".join(sorted({str(r["classification"]) for r in subset}))
                    f.write(
                        f"| {model} | {threshold} | {qbits} | {med_gpu:.3f} | {med_cpu:.3f} | "
                        f"{med_speed:.3f} | {med_roundtrip:.3f} | {med_ratio:.3f} | {med_l2:.6f} | {med_finite:.3f} | {classes} |\n"
                    )
        f.write("\n## Decision\n\n")
        f.write("- Strong success requires >=2x compress speed, ratio retention >=0.80, acceptable decode error, and CPU default unchanged.\n")
        f.write("- Research success requires >=20% compress speedup with passing state/decode tests and explicit research-only marking.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 524288, 1048576])
    parser.add_argument("--quant-bits", nargs="+", type=int, default=[8, 4])
    args = parser.parse_args()

    logs = PROJECT_ROOT / "logs" / "cuda_v2"
    all_rows: List[Dict[str, object]] = []
    all_layers: List[Dict[str, object]] = []
    for model_name in args.models:
        baseline = cpu_baseline(model_name, args.rounds)
        model_rows: List[Dict[str, object]] = []
        model_layers: List[Dict[str, object]] = []
        for threshold in args.thresholds:
            for qbits in args.quant_bits:
                rows, layer_rows = run_gpu(model_name, args.rounds, threshold, qbits, baseline)
                model_rows.extend(rows)
                model_layers.extend(layer_rows)
        all_rows.extend(model_rows)
        all_layers.extend(model_layers)
        write_csv(logs / f"{model_name}_perf.csv", model_rows, MODEL_FIELDS)
        write_csv(logs / f"{model_name}_layers.csv", model_layers, LAYER_FIELDS)
    write_csv(logs / "cuda_v2_perf_all.csv", all_rows, MODEL_FIELDS)
    write_csv(logs / "cuda_v2_layers_all.csv", all_layers, LAYER_FIELDS)
    write_report(logs, all_rows)
    (logs / "cpu_vs_gpu_summary.md").write_text((logs / "final_gpu_acceleration_report.md").read_text())
    print(logs / "final_gpu_acceleration_report.md")


if __name__ == "__main__":
    main()
