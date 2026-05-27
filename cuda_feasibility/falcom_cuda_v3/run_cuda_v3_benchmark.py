#!/usr/bin/env python3
"""Benchmark FalCom CUDA v3 experimental closed-loop compressor."""

from __future__ import annotations

import argparse
import csv
import os
import re
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
from falcom_cuda_v3_wrapper import (  # noqa: E402
    FalcomCudaV3State,
    compress_batch_cuda_v3,
    decompress_batch_cuda_v3,
    dumps_cuda_v3_layers,
    loads_cuda_v3_layers,
)

PROFILE = PROJECT_ROOT / "cuda_feasibility" / "profile_appfl_hot_rounds.py"
sys.path.insert(0, str(PROFILE.parent))
from profile_appfl_hot_rounds import read_round  # noqa: E402


OFFICIAL_BASELINES = {
    "resnet18": {"serial_c_compress_ms": 340.294, "cpu_grouped_8t_compress_ms": 122.189},
    "resnet50": {"serial_c_compress_ms": 957.909, "cpu_grouped_8t_compress_ms": 197.865},
}

MODEL_FIELDS = [
    "model",
    "round_index",
    "round_type",
    "threshold",
    "quant_mode",
    "quant_bits",
    "mode",
    "cpu_threads",
    "original_mb",
    "official_serial_c_compress_ms",
    "official_cpu_grouped_8t_compress_ms",
    "cuda_v2_reference_compress_ms",
    "cuda_layers",
    "cpu_fallback_layers",
    "cuda_original_mb",
    "fallback_original_mb",
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
    "speedup_vs_cuda_v2",
    "kernel_ms",
    "kernel_launch_count",
    "decompress_kernel_launch_count",
    "payload_d2h_ms",
    "payload_serialize_ms",
    "payload_deserialize_ms",
    "python_wrapper_ms",
    "state_commit_ms",
    "cpu_fallback_ms",
    "cpu_fallback_decompress_ms",
    "fallback_gpu_to_cpu_ms",
    "cpu_fallback_batches",
    "cpu_fallback_payload_bytes",
    "experimental_payload_d2h_mb",
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
    "quant_mode_layer",
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
    if str(row["model"]) == "resnet50" and str(row["quant_mode"]) in ("8", "q8", "int8"):
        compress_ms = fnum(row["compress_total_ms"])
        speedup_vs_v2 = fnum(row["speedup_vs_cuda_v2"])
        speedup_vs_cpu = fnum(row["speedup_vs_official_cpu_8t"])
        if compress_ms <= 150.0 or speedup_vs_v2 >= 1.10:
            return "v3_target_pass"
        if speedup_vs_cpu < 1.3:
            return "stop_gpu_internal_optimization"
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


def v2_references() -> Dict[Tuple[str, int, str], float]:
    path = PROJECT_ROOT / "logs" / "cuda_v2" / "cuda_v2_perf_all.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    out: Dict[Tuple[str, int, str], float] = {}
    for model in sorted({r["model"] for r in rows}):
        for threshold in sorted({int(r["threshold"]) for r in rows if r["model"] == model}):
            for qbits in sorted({str(r["quant_bits"]) for r in rows if r["model"] == model and int(r["threshold"]) == threshold}):
                vals = [
                    float(r["compress_total_ms"])
                    for r in rows
                    if r["model"] == model
                    and r["round_type"] == "hot"
                    and int(r["threshold"]) == threshold
                    and str(r["quant_bits"]) == qbits
                ]
                if vals:
                    out[(model, threshold, qbits)] = statistics.median(vals)
    return out


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def run_gpu(
    model_name: str,
    rounds: int,
    threshold: int,
    quant_mode: str,
    baseline: Mapping[int, Mapping[str, float]],
    refs: Mapping[Tuple[str, int, str], float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state_id = f"{model_name}_cuda_v3_{threshold}_{clean_id(quant_mode)}"
    c_state = FalcomCudaV3State(client_id=state_id, config=config())
    d_state = FalcomCudaV3State(client_id=state_id, config=config())
    data_dir = PROJECT_ROOT / "dataset" / model_name
    rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []
    cpu_threads = os.environ.get("OMP_NUM_THREADS", "")
    official = OFFICIAL_BASELINES[model_name]
    v2_ref = refs.get((model_name, threshold, str(quant_mode)), 0.0)
    for idx in range(rounds):
        _, _client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        safe_sync()
        t0 = time.perf_counter()
        compressed, cstats = compress_batch_cuda_v3(
            model,
            state_handle=c_state,
            options={"cuda_min_numel": threshold, "quant_mode": quant_mode},
        )
        t_ser = time.perf_counter()
        payload = dumps_cuda_v3_layers(compressed)
        serialize_ms = (time.perf_counter() - t_ser) * 1000.0
        cstats["payload_serialize_ms"] = serialize_ms
        cstats["compressed_bytes"] = len(payload)
        original_mb = float(cstats["cuda_original_mb"]) + float(cstats["fallback_original_mb"])
        cstats["compression_ratio"] = (
            (original_mb * 1024.0 * 1024.0) / float(len(payload)) if payload else 0.0
        )
        safe_sync()
        compress_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        t_load = time.perf_counter()
        loaded = loads_cuda_v3_layers(payload)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        decoded, dstats = decompress_batch_cuda_v3(loaded, state_handle=d_state)
        dstats["payload_deserialize_ms"] = float(dstats["payload_deserialize_ms"]) + load_ms
        safe_sync()
        decompress_ms = (time.perf_counter() - t1) * 1000.0
        roundtrip_ms = compress_ms + decompress_ms
        metrics = error_metrics(model, decoded)
        base = baseline[idx]
        ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
        speedup_v2 = (v2_ref / compress_ms) if v2_ref else 0.0
        quant_bits_used = ",".join(sorted({str(r.get("quant_bits", "")) for r in cstats.get("layer_records", []) if r.get("backend") == "cuda"}))
        row = {
            "model": model_name,
            "round_index": idx,
            "round_type": "round0" if idx == 0 else "hot",
            "threshold": threshold,
            "quant_mode": quant_mode,
            "quant_bits": quant_bits_used,
            "mode": "gpu_v3_experimental",
            "cpu_threads": cpu_threads,
            "original_mb": f"{original_bytes / (1024.0 * 1024.0):.6f}",
            "official_serial_c_compress_ms": f"{official['serial_c_compress_ms']:.6f}",
            "official_cpu_grouped_8t_compress_ms": f"{official['cpu_grouped_8t_compress_ms']:.6f}",
            "cuda_v2_reference_compress_ms": f"{v2_ref:.6f}" if v2_ref else "",
            "cuda_layers": int(cstats["cuda_layers"]),
            "cpu_fallback_layers": int(cstats["cpu_fallback_layers"]),
            "cuda_original_mb": f"{float(cstats['cuda_original_mb']):.6f}",
            "fallback_original_mb": f"{float(cstats['fallback_original_mb']):.6f}",
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
            "speedup_vs_cuda_v2": f"{speedup_v2:.6f}" if v2_ref else "",
            "kernel_ms": f"{float(cstats['kernel_ms']):.6f}",
            "kernel_launch_count": int(cstats["kernel_launch_count"]),
            "decompress_kernel_launch_count": int(dstats.get("kernel_launch_count", 0)),
            "payload_d2h_ms": f"{float(cstats['payload_d2h_ms']):.6f}",
            "payload_serialize_ms": f"{float(cstats['payload_serialize_ms']):.6f}",
            "payload_deserialize_ms": f"{float(dstats['payload_deserialize_ms']):.6f}",
            "python_wrapper_ms": f"{float(cstats['python_wrapper_ms']):.6f}",
            "state_commit_ms": f"{float(cstats['state_commit_ms']):.6f}",
            "cpu_fallback_ms": f"{float(cstats['cpu_fallback_ms']):.6f}",
            "cpu_fallback_decompress_ms": f"{float(dstats.get('cpu_fallback_decompress_ms', 0.0)):.6f}",
            "fallback_gpu_to_cpu_ms": f"{float(cstats['fallback_gpu_to_cpu_ms']):.6f}",
            "cpu_fallback_batches": int(cstats["cpu_fallback_batches"]),
            "cpu_fallback_payload_bytes": int(cstats["cpu_fallback_payload_bytes"]),
            "experimental_payload_d2h_mb": f"{float(cstats['experimental_payload_d2h_mb']):.6f}",
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
            "decode_status": (
                "pass"
                if int(dstats["decoded_layers"]) == int(cstats["cuda_layers"])
                and int(dstats.get("cpu_fallback_decoded_layers", 0)) == int(cstats["cpu_fallback_layers"])
                and len(decoded) == len(model)
                else "fail"
            ),
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
                    "quant_mode": quant_mode,
                    "layer_name": record.get("layer_name", ""),
                    "layer_key": record.get("layer_key", ""),
                    "backend": record.get("backend", ""),
                    "step": record.get("step", ""),
                    "shape": record.get("shape", ""),
                    "numel": record.get("numel", ""),
                    "payload_length": record.get("payload_length", ""),
                    "quant_bits_layer": record.get("quant_bits", ""),
                    "quant_mode_layer": record.get("quant_mode", ""),
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
    report = logs / "final_hpc_optimization_report.md"
    with report.open("w") as f:
        f.write("# FalCom CUDA v3 Closed-Loop HPC Optimization Report\n\n")
        f.write("CUDA v3 is experimental, closed-loop, and not CPU wire-format-compatible. The installed CPU compressor and CPU ABI remain unchanged.\n\n")
        f.write("## Fixed Production Baselines\n\n")
        f.write("| model | serial C ms | CPU grouped 8T ms |\n")
        f.write("|---|---:|---:|\n")
        for model, base in OFFICIAL_BASELINES.items():
            f.write(f"| {model} | {base['serial_c_compress_ms']:.3f} | {base['cpu_grouped_8t_compress_ms']:.3f} |\n")
        f.write("\n## Hot-Round Medians\n\n")
        f.write("| model | threshold | quant mode | bits | v3 compress ms | v3 decompress ms | closed-loop ms | CPU 8T speedup | v2 speedup | ratio retention | rel L2 | fallback layers | fallback ms | launches | target |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for model in sorted({str(r["model"]) for r in hot}):
            for threshold in sorted({int(r["threshold"]) for r in hot if r["model"] == model}):
                for quant_mode in sorted({str(r["quant_mode"]) for r in hot if r["model"] == model and int(r["threshold"]) == threshold}):
                    subset = [r for r in hot if r["model"] == model and int(r["threshold"]) == threshold and str(r["quant_mode"]) == quant_mode]
                    if not subset:
                        continue
                    bits = ",".join(sorted({str(r["quant_bits"]) for r in subset if str(r["quant_bits"])}))
                    targets = ",".join(sorted({str(r["target_status"]) for r in subset}))
                    f.write(
                        f"| {model} | {threshold} | {quant_mode} | {bits} | "
                        f"{median(subset, 'compress_total_ms'):.3f} | "
                        f"{median(subset, 'decompress_total_ms'):.3f} | "
                        f"{median(subset, 'roundtrip_total_ms'):.3f} | "
                        f"{median(subset, 'speedup_vs_official_cpu_8t'):.3f} | "
                        f"{median(subset, 'speedup_vs_cuda_v2'):.3f} | "
                        f"{median(subset, 'ratio_retention'):.3f} | "
                        f"{median(subset, 'relative_l2_error'):.6f} | "
                        f"{median(subset, 'cpu_fallback_layers'):.0f} | "
                        f"{median(subset, 'cpu_fallback_ms'):.3f} | "
                        f"{median(subset, 'kernel_launch_count'):.0f} | {targets} |\n"
                    )
        resnet50_q8 = [
            r
            for r in hot
            if r["model"] == "resnet50" and int(r["threshold"]) == 262144 and str(r["quant_mode"]) in ("8", "q8", "int8")
        ]
        f.write("\n## Decision\n\n")
        if resnet50_q8:
            speed_v2 = median(resnet50_q8, "speedup_vs_cuda_v2")
            speed_cpu = median(resnet50_q8, "speedup_vs_official_cpu_8t")
            comp = median(resnet50_q8, "compress_total_ms")
            if comp <= 150.0 or speed_v2 >= 1.10:
                f.write("- Continue GPU v3 evaluation: ResNet50 q8 met the <=150 ms or >=10% vs v2 target.\n")
            elif speed_cpu < 1.3:
                f.write("- Stop GPU-internal optimization after this round: ResNet50 q8 did not reach >=10% vs v2 or >=1.3x vs CPU grouped 8T.\n")
            else:
                f.write("- Keep GPU v3 as a research result; further work should be justified by training smoke results.\n")
            f.write(f"- ResNet50 q8 median: {comp:.3f} ms, {speed_cpu:.3f}x vs CPU 8T, {speed_v2:.3f}x vs CUDA v2.\n")
        else:
            f.write("- No ResNet50 q8 hot rows were produced; benchmark is incomplete.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 524288, 1048576])
    parser.add_argument(
        "--quant-modes",
        nargs="+",
        default=["8", "7", "6", "4", "adaptive_q6_q8", "outlier_q6_q8"],
    )
    parser.add_argument("--quick", action="store_true", help="Run only q8 at threshold 262144.")
    args = parser.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
    if args.quick:
        args.thresholds = [262144]
        args.quant_modes = ["8"]

    logs = PROJECT_ROOT / "logs" / "cuda_v3"
    refs = v2_references()
    all_rows: List[Dict[str, object]] = []
    all_layers: List[Dict[str, object]] = []
    for model_name in args.models:
        baseline = cpu_baseline(model_name, args.rounds)
        model_rows: List[Dict[str, object]] = []
        model_layers: List[Dict[str, object]] = []
        for threshold in args.thresholds:
            for quant_mode in args.quant_modes:
                rows, layer_rows = run_gpu(model_name, args.rounds, threshold, str(quant_mode), baseline, refs)
                model_rows.extend(rows)
                model_layers.extend(layer_rows)
        all_rows.extend(model_rows)
        all_layers.extend(model_layers)
        write_csv(logs / f"{model_name}_perf.csv", model_rows, MODEL_FIELDS)
        write_csv(logs / f"{model_name}_layers.csv", model_layers, LAYER_FIELDS)
    write_csv(logs / "cuda_v3_perf_all.csv", all_rows, MODEL_FIELDS)
    write_csv(logs / "cuda_v3_layers_all.csv", all_layers, LAYER_FIELDS)
    write_report(logs, all_rows)
    (logs / "cpu_vs_gpu_summary.md").write_text((logs / "final_hpc_optimization_report.md").read_text())
    print(logs / "final_hpc_optimization_report.md")


if __name__ == "__main__":
    main()
