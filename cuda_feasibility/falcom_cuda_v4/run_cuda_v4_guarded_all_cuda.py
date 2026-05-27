#!/usr/bin/env python3
"""Guard-safe all-valid-CUDA official path probe for FalCom CUDA v4.

This is a narrow validation runner, not v5. It uses the existing v4 q8
wrapper/extension, writes under logs/cuda_v4_guarded_all_cuda, and keeps the
CPU compressor path untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
APPFL_SRC = PROJECT_ROOT / "EB-FaLCom" / "src"
PROFILE_DIR = PROJECT_ROOT / "cuda_feasibility"
for path in (THIS_DIR, APPFL_SRC, PROFILE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from appfl.compressor.FalComC import FalComC  # noqa: E402
from falcom_cuda_v4_wrapper import (  # noqa: E402
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    cuda_guard_status,
    decompress_batch_cuda_v4,
    default_config,
    dumps_cuda_v4_layers,
    extension_available,
    loads_cuda_v4_layers,
)
from profile_appfl_hot_rounds import read_round  # noqa: E402


RESNET50_TARGET_MS = 55.5
QUALITY_REL_L2_MAX = 0.03
QUALITY_RATIO_MIN = 0.8

OFFICIAL_FIELDS = [
    "model",
    "threshold",
    "guarded_all_cuda",
    "sample_index",
    "timing_kind",
    "warmup_round",
    "measured_round",
    "warmup_excluded",
    "fresh_state",
    "compress_ms",
    "decompress_ms",
    "closed_loop_ms",
    "original_mb",
    "cuda_layers",
    "fallback_layers",
    "cuda_mb",
    "fallback_mb",
    "avg_numel_per_cuda_layer",
    "kernel_launch_count",
    "decode_kernel_launch_count",
    "num_payload_objects",
    "payload_blob_bytes",
    "final_payload_bytes",
    "compression_ratio",
    "cpu_wrapper_compression_ratio",
    "ratio_retention",
    "max_abs_error",
    "relative_l2_error",
    "finite_rate",
    "decode_status",
    "correctness_status",
    "payload_object_status",
]

QUALITY_FIELDS = [
    "model",
    "threshold",
    "guarded_all_cuda",
    "samples",
    "closed_loop_median_ms",
    "closed_loop_iqr_ms",
    "closed_loop_min_ms",
    "closed_loop_max_ms",
    "compress_median_ms",
    "decompress_median_ms",
    "cuda_layers_median",
    "fallback_layers_median",
    "num_payload_objects_median",
    "payload_blob_bytes_median",
    "rel_l2_median",
    "finite_rate_min",
    "ratio_retention_median",
    "quality_pass",
    "acceptance_status",
]

LAYER_DIFF_FIELDS = [
    "model",
    "layer_name",
    "layer_key",
    "shape",
    "numel",
    "official_threshold_2048_backend",
    "valid_full_cuda_backend",
    "fallback_reason_official",
    "guard_status",
    "dtype_ok",
    "contiguous_ok",
    "shape_supported",
    "state_owner_stable",
    "q8_supported",
    "rel_l2_contribution",
    "payload_bytes",
    "estimated_time_ms",
]


def config() -> Any:
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


def fnum(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    q1, _q2, q3 = statistics.quantiles(list(values), n=4, method="inclusive")
    return float(q3 - q1)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def flatten_model(model: Mapping[str, torch.Tensor], keys: Iterable[str] | None = None) -> torch.Tensor:
    pieces = []
    for name in list(keys) if keys is not None else list(model.keys()):
        tensor = model[name].detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        pieces.append(tensor.reshape(-1).to(dtype=torch.float32))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def error_metrics(reference: Mapping[str, torch.Tensor], decoded: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    keys = list(reference.keys())
    if any(name not in decoded for name in keys):
        return {"max_abs_error": float("inf"), "relative_l2_error": float("inf"), "finite_rate": 0.0}
    ref = flatten_model(reference, keys)
    out = flatten_model(decoded, keys)
    if ref.numel() != out.numel() or ref.numel() == 0:
        return {"max_abs_error": float("inf"), "relative_l2_error": float("inf"), "finite_rate": 0.0}
    diff = ref - out
    denom = torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(diff) / denom).item()),
        "finite_rate": float(torch.isfinite(out).to(torch.float32).mean().item()),
    }


def available_rounds(model_name: str) -> List[int]:
    data_dir = PROJECT_ROOT / "dataset" / model_name
    rounds = []
    for path in data_dir.glob("round_*_client_0.bin"):
        try:
            rounds.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(rounds)


def load_models(model_name: str, rounds: Sequence[int]) -> Dict[int, Tuple[str, OrderedDict, int]]:
    data_dir = PROJECT_ROOT / "dataset" / model_name
    out = {}
    for idx in rounds:
        _round_idx, client_id, model, original_bytes = read_round(data_dir / f"round_{idx}_client_0.bin", "cuda")
        out[int(idx)] = (client_id, model, int(original_bytes))
    return out


def cpu_ratio_baselines(models: Mapping[int, Tuple[str, OrderedDict, int]]) -> Dict[int, Dict[str, float]]:
    comp = FalComC(config())
    out: Dict[int, Dict[str, float]] = {}
    for idx, (client_id, model, original_bytes) in models.items():
        safe_sync()
        payload = comp.compress_model(model, client_id=client_id)
        safe_sync()
        out[idx] = {
            "payload_bytes": float(len(payload)),
            "compression_ratio": float(original_bytes) / float(len(payload)) if payload else 0.0,
        }
    return out


def threshold_label(threshold: int, guarded: bool) -> str:
    return "guarded_all_cuda" if guarded else str(int(threshold))


def run_one_sample(
    model_name: str,
    threshold: int,
    guarded: bool,
    sample_index: int,
    models: Mapping[int, Tuple[str, OrderedDict, int]],
    cpu_ratios: Mapping[int, Mapping[str, float]],
    warmup_round: int,
    measured_round: int,
    return_details: bool = False,
) -> Tuple[Dict[str, object], Dict[str, Any]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    if guarded:
        os.environ["FALCOM_CUDA_V4_GUARDED_ALL_CUDA"] = "1"
    else:
        os.environ.pop("FALCOM_CUDA_V4_GUARDED_ALL_CUDA", None)

    state_id = f"v4_guarded_{model_name}_{threshold_label(threshold, guarded)}_{sample_index}"
    c_state = FalcomCudaV4State(client_id=state_id, config=config())
    d_state = FalcomCudaV4State(client_id=state_id, config=config())
    options = {"cuda_min_numel": threshold, "quant_mode": "8", "guarded_all_cuda": guarded}

    warm_client, warm_model, _warm_original = models[warmup_round]
    warm_layers, _warm_cstats = compress_batch_cuda_v4(
        warm_model,
        state_handle=c_state,
        options={**options, "client_id": warm_client},
    )
    warm_payload = dumps_cuda_v4_layers(warm_layers)
    _warm_decoded, _warm_dstats = decompress_batch_cuda_v4(loads_cuda_v4_layers(warm_payload), state_handle=d_state)
    safe_sync()

    client_id, model, original_bytes = models[measured_round]
    safe_sync()
    t0 = time.perf_counter()
    layers, cstats = compress_batch_cuda_v4(
        model,
        state_handle=c_state,
        options={**options, "client_id": client_id},
    )
    payload = dumps_cuda_v4_layers(layers)
    safe_sync()
    t1 = time.perf_counter()
    loaded = loads_cuda_v4_layers(payload)
    decoded, dstats = decompress_batch_cuda_v4(loaded, state_handle=d_state)
    safe_sync()
    t2 = time.perf_counter()

    compress_ms = (t1 - t0) * 1000.0
    decompress_ms = (t2 - t1) * 1000.0
    closed_loop_ms = compress_ms + decompress_ms
    metrics = error_metrics(model, decoded)
    ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
    cpu_ratio = float(cpu_ratios[measured_round]["compression_ratio"])
    decode_ok = (
        int(dstats.get("cuda_layers", 0)) == int(cstats.get("cuda_layers", 0))
        and int(dstats.get("cpu_fallback_layers", 0)) == int(cstats.get("cpu_fallback_layers", 0))
        and len(decoded) == len(model)
        and all(name in decoded for name in model.keys())
    )
    num_payload_objects = int(cstats.get("num_payload_objects", 0))
    row = {
        "model": model_name,
        "threshold": threshold_label(threshold, guarded),
        "guarded_all_cuda": "yes" if guarded else "no",
        "sample_index": sample_index,
        "timing_kind": "official_minimal",
        "warmup_round": warmup_round,
        "measured_round": measured_round,
        "warmup_excluded": "yes",
        "fresh_state": "yes",
        "compress_ms": f"{compress_ms:.6f}",
        "decompress_ms": f"{decompress_ms:.6f}",
        "closed_loop_ms": f"{closed_loop_ms:.6f}",
        "original_mb": f"{original_bytes / (1024.0 * 1024.0):.6f}",
        "cuda_layers": int(cstats.get("cuda_layers", 0)),
        "fallback_layers": int(cstats.get("cpu_fallback_layers", 0)),
        "cuda_mb": f"{float(cstats.get('cuda_original_mb', 0.0)):.6f}",
        "fallback_mb": f"{float(cstats.get('fallback_original_mb', 0.0)):.6f}",
        "avg_numel_per_cuda_layer": f"{float(cstats.get('avg_numel_per_cuda_layer', 0.0)):.3f}",
        "kernel_launch_count": int(cstats.get("kernel_launch_count", 0)),
        "decode_kernel_launch_count": int(dstats.get("decode_kernel_launch_count", 0)),
        "num_payload_objects": num_payload_objects,
        "payload_blob_bytes": int(cstats.get("payload_blob_bytes", 0)),
        "final_payload_bytes": int(len(payload)),
        "compression_ratio": f"{ratio:.6f}",
        "cpu_wrapper_compression_ratio": f"{cpu_ratio:.6f}",
        "ratio_retention": f"{(ratio / cpu_ratio) if cpu_ratio else 0.0:.6f}",
        "max_abs_error": f"{metrics['max_abs_error']:.9f}",
        "relative_l2_error": f"{metrics['relative_l2_error']:.9f}",
        "finite_rate": f"{metrics['finite_rate']:.9f}",
        "decode_status": "pass" if decode_ok else "fail",
        "correctness_status": "pass"
        if metrics["finite_rate"] == 1.0 and metrics["relative_l2_error"] <= QUALITY_REL_L2_MAX
        else "fail",
        "payload_object_status": "constant_small" if num_payload_objects <= 2 else "layer_count_sized",
    }
    details = {
        "model": model,
        "decoded": decoded,
        "cstats": cstats,
        "dstats": dstats,
        "payload": payload,
        "row": row,
    }
    return row, details if return_details else {}


def summarize_quality(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    keys = sorted({(str(r["model"]), str(r["threshold"]), str(r["guarded_all_cuda"])) for r in rows})
    for model, threshold, guarded in keys:
        subset = [r for r in rows if r["model"] == model and r["threshold"] == threshold and r["guarded_all_cuda"] == guarded]
        loops = [fnum(r["closed_loop_ms"]) for r in subset]
        rels = [fnum(r["relative_l2_error"]) for r in subset]
        finite = [fnum(r["finite_rate"]) for r in subset]
        ratios = [fnum(r["ratio_retention"]) for r in subset]
        quality_pass = bool(max(rels) <= QUALITY_REL_L2_MAX and min(finite) == 1.0 and min(ratios) >= QUALITY_RATIO_MIN)
        payload_ok = all(str(r["payload_object_status"]) == "constant_small" for r in subset)
        accepted = model == "resnet50" and threshold == "guarded_all_cuda" and median(loops) <= RESNET50_TARGET_MS and quality_pass and payload_ok
        if accepted:
            acceptance_status = "accepted_v4_final_selected_path"
        elif guarded == "yes" and quality_pass and payload_ok:
            acceptance_status = "supporting_pass" if model != "resnet50" else "quality_pass_target_miss"
        elif quality_pass:
            acceptance_status = "comparison_quality_pass"
        else:
            acceptance_status = "not_accepted"
        out.append(
            {
                "model": model,
                "threshold": threshold,
                "guarded_all_cuda": guarded,
                "samples": len(subset),
                "closed_loop_median_ms": f"{median(loops):.6f}",
                "closed_loop_iqr_ms": f"{iqr(loops):.6f}",
                "closed_loop_min_ms": f"{min(loops):.6f}",
                "closed_loop_max_ms": f"{max(loops):.6f}",
                "compress_median_ms": f"{median([fnum(r['compress_ms']) for r in subset]):.6f}",
                "decompress_median_ms": f"{median([fnum(r['decompress_ms']) for r in subset]):.6f}",
                "cuda_layers_median": f"{median([fnum(r['cuda_layers']) for r in subset]):.0f}",
                "fallback_layers_median": f"{median([fnum(r['fallback_layers']) for r in subset]):.0f}",
                "num_payload_objects_median": f"{median([fnum(r['num_payload_objects']) for r in subset]):.0f}",
                "payload_blob_bytes_median": f"{median([fnum(r['payload_blob_bytes']) for r in subset]):.0f}",
                "rel_l2_median": f"{median(rels):.9f}",
                "finite_rate_min": f"{min(finite):.9f}",
                "ratio_retention_median": f"{median(ratios):.6f}",
                "quality_pass": "yes" if quality_pass else "no",
                "acceptance_status": acceptance_status,
            }
        )
    return out


def layer_record_map(cstats: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("layer_name")): dict(r) for r in cstats.get("layer_records", [])}


def estimate_layer_time_ms(layer_name: str, cstats: Mapping[str, Any], dstats: Mapping[str, Any]) -> float:
    records = layer_record_map(cstats)
    record = records.get(layer_name, {})
    if str(record.get("backend")) == "cuda":
        payload = fnum(record.get("payload_length"))
        total_payload = sum(fnum(r.get("payload_length")) for r in records.values() if str(r.get("backend")) == "cuda")
        total_ms = (
            fnum(cstats.get("encode_kernel_ms"))
            + fnum(cstats.get("payload_d2h_ms"))
            + fnum(dstats.get("payload_h2d_ms"))
            + fnum(dstats.get("decode_kernel_ms"))
            + fnum(dstats.get("decoded_tensor_materialize_ms"))
        )
        return total_ms * payload / total_payload if total_payload else 0.0
    fallback = {str(r.get("layer_name")): dict(r) for r in cstats.get("fallback_records", [])}
    fb = fallback.get(layer_name, {})
    nbytes = fnum(fb.get("nbytes"))
    total_bytes = sum(fnum(r.get("nbytes")) for r in fallback.values())
    total_ms = fnum(cstats.get("fallback_gpu_to_cpu_ms")) + fnum(cstats.get("cpu_fallback_ms")) + fnum(dstats.get("cpu_fallback_decode_ms"))
    return total_ms * nbytes / total_bytes if total_bytes else 0.0


def layer_rel_l2_contrib(
    layer_name: str,
    model: Mapping[str, torch.Tensor],
    decoded: Mapping[str, torch.Tensor],
    global_norm: float,
) -> float:
    if layer_name not in model or layer_name not in decoded or global_norm <= 0.0:
        return float("inf")
    ref = model[layer_name].detach()
    out = decoded[layer_name].detach()
    if ref.is_cuda:
        ref = ref.cpu()
    if out.is_cuda:
        out = out.cpu()
    return float(torch.linalg.vector_norm((ref.reshape(-1) - out.reshape(-1)).to(torch.float32)).item() / global_norm)


def build_layer_diff(
    model_name: str,
    models: Mapping[int, Tuple[str, OrderedDict, int]],
    cpu_ratios: Mapping[int, Mapping[str, float]],
) -> List[Dict[str, object]]:
    rounds = sorted(models.keys())
    warmup_round = rounds[0]
    measured_round = (rounds[1:] or rounds)[0]
    _official_row, official = run_one_sample(
        model_name,
        threshold=2048,
        guarded=False,
        sample_index=0,
        models=models,
        cpu_ratios=cpu_ratios,
        warmup_round=warmup_round,
        measured_round=measured_round,
        return_details=True,
    )
    _guarded_row, guarded = run_one_sample(
        model_name,
        threshold=2048,
        guarded=True,
        sample_index=0,
        models=models,
        cpu_ratios=cpu_ratios,
        warmup_round=warmup_round,
        measured_round=measured_round,
        return_details=True,
    )
    model = official["model"]
    official_records = layer_record_map(official["cstats"])
    guarded_records = layer_record_map(guarded["cstats"])
    global_norm = float(torch.linalg.vector_norm(flatten_model(model)).item())
    rows = []
    for layer_name, tensor in model.items():
        off = official_records.get(layer_name, {})
        val = guarded_records.get(layer_name, {})
        guard = cuda_guard_status(tensor, min_numel=2048, guarded_all_cuda=True)
        rows.append(
            {
                "model": model_name,
                "layer_name": layer_name,
                "layer_key": off.get("layer_key") or val.get("layer_key") or "",
                "shape": off.get("shape") or val.get("shape") or "x".join(str(x) for x in tensor.shape),
                "numel": int(tensor.numel()),
                "official_threshold_2048_backend": off.get("backend", ""),
                "valid_full_cuda_backend": val.get("backend", ""),
                "fallback_reason_official": off.get("fallback_reason", ""),
                "guard_status": guard.get("guard_status", ""),
                "dtype_ok": guard.get("dtype_ok", False),
                "contiguous_ok": guard.get("contiguous_ok", False),
                "shape_supported": guard.get("shape_supported", False),
                "state_owner_stable": guard.get("state_owner_stable", False),
                "q8_supported": guard.get("q8_supported", False),
                "rel_l2_contribution": f"{layer_rel_l2_contrib(layer_name, model, guarded['decoded'], global_norm):.9f}",
                "payload_bytes": int(fnum(val.get("payload_length"), fnum(off.get("payload_length")))),
                "estimated_time_ms": f"{estimate_layer_time_ms(layer_name, official['cstats'], official['dstats']):.6f}",
            }
        )
    return rows


def so_hashes() -> List[Tuple[str, str]]:
    out = []
    for rel in (
        "libmomentum_compressor_openmp_simd_final.so",
        "EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so",
    ):
        path = PROJECT_ROOT / rel
        if path.exists():
            out.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def write_report(logs: Path, quality_rows: List[Dict[str, object]], layer_diff_rows: List[Dict[str, object]]) -> None:
    report = logs / "final_guarded_all_cuda_report.md"
    resnet50 = [r for r in quality_rows if r["model"] == "resnet50"]
    accepted = [r for r in resnet50 if r["acceptance_status"] == "accepted_v4_final_selected_path"]
    quality_pass = [r for r in resnet50 if r["quality_pass"] == "yes"]
    best = min(quality_pass or resnet50, key=lambda r: fnum(r["closed_loop_median_ms"])) if resnet50 else None
    official_2048 = next((r for r in resnet50 if r["threshold"] == "2048"), None)
    guarded = next((r for r in resnet50 if r["threshold"] == "guarded_all_cuda"), None)
    converted = [
        r
        for r in layer_diff_rows
        if r["model"] == "resnet50"
        and str(r["official_threshold_2048_backend"]).startswith("cpu")
        and str(r["valid_full_cuda_backend"]) == "cuda"
    ]
    blocked = [
        r
        for r in layer_diff_rows
        if r["model"] == "resnet50"
        and str(r["official_threshold_2048_backend"]).startswith("cpu")
        and str(r["valid_full_cuda_backend"]) != "cuda"
    ]
    with report.open("w") as f:
        f.write("# CUDA v4 Guarded All-Valid-CUDA Report\n\n")
        f.write("This is the final narrow guard-validation patch for CUDA v4 q8. It does not create v5 and does not modify the CPU compressor ABI, wire format, installed `.so`, v2, or v3.\n\n")
        f.write("## Official Minimal Timing\n\n")
        f.write("| model | threshold | guarded | closed-loop median ms | IQR | compress ms | decompress ms | CUDA layers | fallback layers | payload objects | rel L2 | finite | ratio retention | quality | acceptance |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in quality_rows:
            f.write(
                f"| {row['model']} | {row['threshold']} | {row['guarded_all_cuda']} | "
                f"{fnum(row['closed_loop_median_ms']):.3f} | {fnum(row['closed_loop_iqr_ms']):.3f} | "
                f"{fnum(row['compress_median_ms']):.3f} | {fnum(row['decompress_median_ms']):.3f} | "
                f"{row['cuda_layers_median']} | {row['fallback_layers_median']} | {row['num_payload_objects_median']} | "
                f"{fnum(row['rel_l2_median']):.6f} | {fnum(row['finite_rate_min']):.3f} | "
                f"{fnum(row['ratio_retention_median']):.3f} | {row['quality_pass']} | {row['acceptance_status']} |\n"
            )
        if accepted:
            row = accepted[0]
            f.write("\n## Selected GPU Experimental Config\n\n")
            f.write("- codec: `cuda_v4_q8`\n")
            f.write("- mode: `guarded_all_cuda`\n")
            f.write("- threshold: `N/A / all valid CUDA`\n")
            f.write(f"- ResNet50 closed-loop median: `{fnum(row['closed_loop_median_ms']):.3f} ms`\n")
            f.write(f"- ResNet50 compress/decompress median: `{fnum(row['compress_median_ms']):.3f} / {fnum(row['decompress_median_ms']):.3f} ms`\n")
            f.write(f"- CUDA/fallback layers: `{row['cuda_layers_median']} / {row['fallback_layers_median']}`\n")
            f.write(f"- payload objects: `{row['num_payload_objects_median']}`\n")
            f.write("- The numeric thresholds are comparison controls only, not the selected final path.\n")
        f.write("\n## Layer Backend Diff\n\n")
        f.write(f"- ResNet50 official threshold 2048 fallback layers converted to CUDA by guarded mode: {len(converted)}.\n")
        f.write(f"- ResNet50 official threshold 2048 fallback layers still blocked in guarded mode: {len(blocked)}.\n")
        if converted:
            reasons = {}
            for row in converted:
                reason = str(row["fallback_reason_official"] or "none")
                reasons[reason] = reasons.get(reason, 0) + 1
            f.write("- Converted fallback reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) + ".\n")
        f.write("\n## Decision\n\n")
        if accepted:
            row = accepted[0]
            f.write(
                f"- Decision: guarded all-valid-CUDA is accepted as the v4 final selected experimental path. "
                f"ResNet50 closed-loop median is {fnum(row['closed_loop_median_ms']):.3f} ms, meeting <= {RESNET50_TARGET_MS:.1f} ms with quality passing.\n"
            )
            f.write("- No v5, new codec, kernel-math optimization, or envelope writer/parser patch is recommended for compressor-internal work.\n")
            f.write("- Revisit envelope work only if future end-to-end APPFL/pipeline profiling proves it is the system bottleneck.\n")
        elif best:
            best_ms = fnum(best["closed_loop_median_ms"])
            f.write(
                f"- Decision: stop GPU compressor-internal optimization. Best quality-passing ResNet50 official median is "
                f"{best_ms:.3f} ms at threshold `{best['threshold']}`.\n"
            )
            if guarded:
                f.write(
                    f"- Guarded all-valid-CUDA median is {fnum(guarded['closed_loop_median_ms']):.3f} ms; "
                    "it did not satisfy the <=55.5 ms acceptance gate.\n"
                )
            if official_2048:
                f.write(f"- Official 2048 reference in this run is {fnum(official_2048['closed_loop_median_ms']):.3f} ms.\n")
        else:
            f.write("- Decision: no ResNet50 quality-passing result; stop and investigate correctness before optimization.\n")
        f.write("\n## Regression Notes\n\n")
        for rel, digest in so_hashes():
            f.write(f"- `{rel}` SHA256: `{digest}`\n")
        f.write("- CPU and CUDA gate command results are reported by the execution transcript/final answer after this runner completes.\n")
        f.write("\n## Artifacts\n\n")
        f.write("- `official_timing.csv`\n")
        f.write("- `quality.csv`\n")
        f.write("- `layer_backend_diff.csv`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--thresholds", nargs="+", type=int, default=[2048, 1024, 512])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.models = ["resnet50"]
        args.thresholds = [2048]
        args.samples = 2
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
    if not extension_available():
        raise RuntimeError("CUDA v4 extension is not available")

    logs = PROJECT_ROOT / "logs" / "cuda_v4_guarded_all_cuda"
    logs.mkdir(parents=True, exist_ok=True)
    before_hashes = so_hashes()

    models_by_name: Dict[str, Dict[int, Tuple[str, OrderedDict, int]]] = {}
    cpu_ratios_by_name: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model_name in args.models:
        rounds = available_rounds(model_name)
        if not rounds:
            raise RuntimeError(f"No replay rounds found for {model_name}")
        models = load_models(model_name, rounds)
        models_by_name[model_name] = models
        cpu_ratios_by_name[model_name] = cpu_ratio_baselines(models)

    rows: List[Dict[str, object]] = []
    for model_name in args.models:
        rounds = sorted(models_by_name[model_name])
        warmup_round = rounds[0]
        measured_candidates = rounds[1:] or rounds
        configs: List[Tuple[int, bool]] = [(threshold, False) for threshold in args.thresholds]
        configs.append((2048, True))
        for threshold, guarded in configs:
            for sample in range(args.samples):
                measured_round = measured_candidates[sample % len(measured_candidates)]
                row, _details = run_one_sample(
                    model_name,
                    threshold,
                    guarded,
                    sample,
                    models_by_name[model_name],
                    cpu_ratios_by_name[model_name],
                    warmup_round,
                    measured_round,
                )
                rows.append(row)

    quality_rows = summarize_quality(rows)
    layer_diff_rows: List[Dict[str, object]] = []
    for model_name in args.models:
        layer_diff_rows.extend(build_layer_diff(model_name, models_by_name[model_name], cpu_ratios_by_name[model_name]))

    after_hashes = so_hashes()
    write_csv(logs / "official_timing.csv", rows, OFFICIAL_FIELDS)
    write_csv(logs / "quality.csv", quality_rows, QUALITY_FIELDS)
    write_csv(logs / "layer_backend_diff.csv", layer_diff_rows, LAYER_DIFF_FIELDS)
    write_csv(
        logs / "cpu_so_hashes.csv",
        [
            {"phase": "before", "path": rel, "sha256": digest}
            for rel, digest in before_hashes
        ]
        + [
            {"phase": "after", "path": rel, "sha256": digest}
            for rel, digest in after_hashes
        ],
        ["phase", "path", "sha256"],
    )
    write_report(logs, quality_rows, layer_diff_rows)
    print(logs / "final_guarded_all_cuda_report.md")


if __name__ == "__main__":
    main()
