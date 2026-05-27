#!/usr/bin/env python3
"""CUDA v4 HPC probe with separated official and diagnostic timing.

This runner is intentionally a probe, not a v5 implementation. It only uses the
existing CUDA v4 wrapper/extension and writes new artifacts under
logs/cuda_v4_hpc_probe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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

CURRENT_V4_BEST_CLOSED_LOOP_MS = {"resnet50": 61.664, "resnet18": 29.228}
RESNET50_CONTINUE_TARGET_MS = 55.5

OFFICIAL_FIELDS = [
    "model",
    "threshold",
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
]

DIAGNOSTIC_FIELDS = OFFICIAL_FIELDS + [
    "encode_kernel_ms",
    "decode_kernel_ms",
    "payload_d2h_ms",
    "payload_h2d_ms",
    "envelope_serialize_ms",
    "envelope_parse_ms",
    "cpu_fallback_ms",
    "cpu_fallback_decode_ms",
    "fallback_gpu_to_cpu_ms",
    "decoded_tensor_alloc_ms",
    "clone_count",
    "clone_bytes",
    "clone_ms",
    "cudaMalloc_count",
    "cudaFree_count",
    "state_pointer_swap_possible",
    "unaccounted_wrapper_ms",
]

SUMMARY_FIELDS = [
    "model",
    "threshold",
    "timing_kind",
    "samples",
    "closed_loop_median_ms",
    "closed_loop_iqr_ms",
    "closed_loop_min_ms",
    "closed_loop_max_ms",
    "compress_median_ms",
    "decompress_median_ms",
    "cuda_layers_median",
    "fallback_layers_median",
    "cuda_mb_median",
    "fallback_mb_median",
    "rel_l2_median",
    "finite_rate_min",
    "ratio_retention_median",
    "quality_pass",
]

MARGINAL_FIELDS = [
    "model",
    "from_threshold",
    "to_threshold",
    "added_cuda_layers",
    "added_cuda_mb",
    "closed_loop_delta_ms",
    "rel_l2_delta",
    "marginal_closed_loop_gain_pct",
    "saturation_flag",
]

ENVELOPE_FIELDS = [
    "model",
    "threshold",
    "repeat_index",
    "payload_bytes",
    "python_wrapper_dumps_ms",
    "cpp_envelope_pack_ms",
    "python_wrapper_loads_ms",
    "cpp_envelope_parse_ms",
    "object_rebuild_ms",
    "tensor_materialization_ms",
    "notes",
]

TRANSFER_FIELDS = [
    "payload_mb",
    "direction",
    "host_buffer_kind",
    "allocation_policy",
    "copy_api",
    "sync_location",
    "repeat_index",
    "wall_ms",
    "cuda_event_ms",
    "gb_per_s_wall",
    "gb_per_s_event",
    "status",
]

KERNEL_FIELDS = [
    "model",
    "threshold",
    "encode_kernel_ms",
    "decode_kernel_ms",
    "kernel_launch_count",
    "decode_kernel_launch_count",
    "encode_ms_per_launch",
    "decode_ms_per_launch",
    "generic_launch_event_ms",
    "event_method",
]

ALLOCATION_FIELDS = [
    "model",
    "threshold",
    "round_index",
    "clone_count",
    "clone_bytes",
    "clone_ms",
    "decoded_tensor_alloc_count",
    "decoded_tensor_alloc_bytes",
    "decoded_tensor_alloc_ms",
    "cudaMalloc_count",
    "cudaFree_count",
    "state_pointer_swap_possible",
]

FALLBACK_FIELDS = [
    "model",
    "threshold",
    "sample_index",
    "layer_name",
    "layer_key",
    "shape",
    "numel",
    "nbytes",
    "fallback_reason",
    "estimated_encode_ms",
    "estimated_decode_ms",
]

STRESS_FIELDS = OFFICIAL_FIELDS + ["stress_label", "stress_candidate"]


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
        return {"max_abs_error": float("inf"), "relative_l2_error": float("inf"), "finite_rate": 0.0}
    ref = flatten_model(reference, keys)
    out = flatten_model(decoded, keys)
    if ref.numel() != out.numel() or ref.numel() == 0:
        return {"max_abs_error": float("inf"), "relative_l2_error": float("inf"), "finite_rate": 0.0}
    diff = ref - out
    denom = torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
    finite = torch.isfinite(out).to(torch.float32).mean().item()
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(diff) / denom).item()),
        "finite_rate": float(finite),
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


def cpu_ratio_baselines(model_name: str, models: Mapping[int, Tuple[str, OrderedDict, int]]) -> Dict[int, Dict[str, float]]:
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


def cuda_eligible_items(model: Mapping[str, torch.Tensor], threshold: int) -> List[Tuple[str, torch.Tensor]]:
    out = []
    for name, tensor in model.items():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
            and int(tensor.numel()) >= int(threshold)
        ):
            out.append((name, tensor))
    return out


def measure_clone_and_alloc(model: Mapping[str, torch.Tensor], threshold: int) -> Dict[str, object]:
    items = cuda_eligible_items(model, threshold)
    clone_bytes = sum(int(t.numel() * t.element_size()) for _name, t in items)
    safe_sync()
    t0 = time.perf_counter()
    clones = [t.detach().clone() for _name, t in items]
    safe_sync()
    clone_ms = (time.perf_counter() - t0) * 1000.0
    del clones
    safe_sync()
    t1 = time.perf_counter()
    outs = [torch.empty_like(t) for _name, t in items]
    safe_sync()
    alloc_ms = (time.perf_counter() - t1) * 1000.0
    del outs
    return {
        "clone_count": len(items),
        "clone_bytes": clone_bytes,
        "clone_ms": clone_ms,
        "decoded_tensor_alloc_count": len(items),
        "decoded_tensor_alloc_bytes": clone_bytes,
        "decoded_tensor_alloc_ms": alloc_ms,
        "cudaMalloc_count": "not_available_from_torch_allocator",
        "cudaFree_count": "not_available_from_torch_allocator",
        "state_pointer_swap_possible": "yes_with_double_buffered_transaction_commit",
    }


def run_one_measured_sample(
    model_name: str,
    threshold: int,
    sample_index: int,
    timing_kind: str,
    models: Mapping[int, Tuple[str, OrderedDict, int]],
    cpu_ratios: Mapping[int, Mapping[str, float]],
    warmup_round: int,
    measured_round: int,
    include_diagnostics: bool,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state_id = f"hpc_probe_{model_name}_{threshold}_{timing_kind}_{sample_index}"
    c_state = FalcomCudaV4State(client_id=state_id, config=config())
    d_state = FalcomCudaV4State(client_id=state_id, config=config())

    warm_client, warm_model, _warm_bytes = models[warmup_round]
    warm_layers, _warm_cstats = compress_batch_cuda_v4(
        warm_model,
        state_handle=c_state,
        options={"cuda_min_numel": threshold, "quant_mode": "8", "client_id": warm_client},
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
        options={"cuda_min_numel": threshold, "quant_mode": "8", "client_id": client_id},
    )
    t_ser = time.perf_counter()
    payload = dumps_cuda_v4_layers(layers)
    serialize_wall_ms = (time.perf_counter() - t_ser) * 1000.0
    serialize_ms = fnum(layers.get("_last_envelope_serialize_ms", {}).get("ms"), serialize_wall_ms)
    safe_sync()
    t1 = time.perf_counter()
    loaded = loads_cuda_v4_layers(payload)
    parse_ms = fnum(loaded.get("_last_envelope_parse_ms", {}).get("ms"), 0.0)
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
    base = {
        "model": model_name,
        "threshold": threshold,
        "sample_index": sample_index,
        "timing_kind": timing_kind,
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
        "payload_blob_bytes": int(cstats.get("payload_blob_bytes", 0)),
        "final_payload_bytes": int(len(payload)),
        "compression_ratio": f"{ratio:.6f}",
        "cpu_wrapper_compression_ratio": f"{cpu_ratio:.6f}",
        "ratio_retention": f"{(ratio / cpu_ratio) if cpu_ratio else 0.0:.6f}",
        "max_abs_error": f"{metrics['max_abs_error']:.9f}",
        "relative_l2_error": f"{metrics['relative_l2_error']:.9f}",
        "finite_rate": f"{metrics['finite_rate']:.9f}",
        "decode_status": "pass" if decode_ok else "fail",
        "correctness_status": "pass" if metrics["finite_rate"] == 1.0 and metrics["relative_l2_error"] <= 0.03 else "fail",
    }
    diagnostic = dict(base)
    if include_diagnostics:
        alloc = measure_clone_and_alloc(model, threshold)
        known_ms = (
            fnum(cstats.get("encode_kernel_ms"))
            + fnum(dstats.get("decode_kernel_ms"))
            + fnum(cstats.get("payload_d2h_ms"))
            + fnum(dstats.get("payload_h2d_ms"))
            + serialize_ms
            + parse_ms
            + fnum(cstats.get("cpu_fallback_ms"))
            + fnum(dstats.get("cpu_fallback_decode_ms"))
            + fnum(cstats.get("fallback_gpu_to_cpu_ms"))
            + fnum(dstats.get("decoded_tensor_materialize_ms"))
            + fnum(alloc["clone_ms"])
        )
        diagnostic.update(
            {
                "encode_kernel_ms": f"{fnum(cstats.get('encode_kernel_ms')):.6f}",
                "decode_kernel_ms": f"{fnum(dstats.get('decode_kernel_ms')):.6f}",
                "payload_d2h_ms": f"{fnum(cstats.get('payload_d2h_ms')):.6f}",
                "payload_h2d_ms": f"{fnum(dstats.get('payload_h2d_ms')):.6f}",
                "envelope_serialize_ms": f"{serialize_ms:.6f}",
                "envelope_parse_ms": f"{parse_ms:.6f}",
                "cpu_fallback_ms": f"{fnum(cstats.get('cpu_fallback_ms')):.6f}",
                "cpu_fallback_decode_ms": f"{fnum(dstats.get('cpu_fallback_decode_ms')):.6f}",
                "fallback_gpu_to_cpu_ms": f"{fnum(cstats.get('fallback_gpu_to_cpu_ms')):.6f}",
                "decoded_tensor_alloc_ms": f"{fnum(dstats.get('decoded_tensor_materialize_ms')):.6f}",
                "clone_count": alloc["clone_count"],
                "clone_bytes": alloc["clone_bytes"],
                "clone_ms": f"{fnum(alloc['clone_ms']):.6f}",
                "cudaMalloc_count": alloc["cudaMalloc_count"],
                "cudaFree_count": alloc["cudaFree_count"],
                "state_pointer_swap_possible": alloc["state_pointer_swap_possible"],
                "unaccounted_wrapper_ms": f"{max(0.0, closed_loop_ms - known_ms):.6f}",
            }
        )

    fallback_rows = []
    total_fallback_bytes = sum(int(r.get("nbytes", 0)) for r in cstats.get("fallback_records", []))
    for record in cstats.get("fallback_records", []):
        nbytes = int(record.get("nbytes", 0))
        share = (float(nbytes) / float(total_fallback_bytes)) if total_fallback_bytes else 0.0
        fallback_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "sample_index": sample_index,
                "layer_name": record.get("layer_name", ""),
                "layer_key": record.get("layer_key", ""),
                "shape": "x".join(str(x) for x in record.get("shape", ())),
                "numel": int(record.get("numel", 0)),
                "nbytes": nbytes,
                "fallback_reason": record.get("reason", ""),
                "estimated_encode_ms": f"{share * fnum(cstats.get('cpu_fallback_ms')):.6f}",
                "estimated_decode_ms": f"{share * fnum(dstats.get('cpu_fallback_decode_ms')):.6f}",
            }
        )
    return base, diagnostic, fallback_rows


def summarize(rows: List[Dict[str, object]], timing_kind: str) -> List[Dict[str, object]]:
    out = []
    for model in sorted({str(r["model"]) for r in rows}):
        for threshold in sorted({int(r["threshold"]) for r in rows if r["model"] == model}):
            subset = [r for r in rows if r["model"] == model and int(r["threshold"]) == threshold]
            if not subset:
                continue
            vals = [fnum(r["closed_loop_ms"]) for r in subset]
            rels = [fnum(r["relative_l2_error"]) for r in subset]
            finite = [fnum(r["finite_rate"]) for r in subset]
            ratios = [fnum(r["ratio_retention"]) for r in subset]
            out.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "timing_kind": timing_kind,
                    "samples": len(subset),
                    "closed_loop_median_ms": f"{median(vals):.6f}",
                    "closed_loop_iqr_ms": f"{iqr(vals):.6f}",
                    "closed_loop_min_ms": f"{min(vals):.6f}",
                    "closed_loop_max_ms": f"{max(vals):.6f}",
                    "compress_median_ms": f"{median([fnum(r['compress_ms']) for r in subset]):.6f}",
                    "decompress_median_ms": f"{median([fnum(r['decompress_ms']) for r in subset]):.6f}",
                    "cuda_layers_median": f"{median([fnum(r['cuda_layers']) for r in subset]):.0f}",
                    "fallback_layers_median": f"{median([fnum(r['fallback_layers']) for r in subset]):.0f}",
                    "cuda_mb_median": f"{median([fnum(r['cuda_mb']) for r in subset]):.6f}",
                    "fallback_mb_median": f"{median([fnum(r['fallback_mb']) for r in subset]):.6f}",
                    "rel_l2_median": f"{median(rels):.9f}",
                    "finite_rate_min": f"{min(finite):.9f}",
                    "ratio_retention_median": f"{median(ratios):.6f}",
                    "quality_pass": "yes" if max(rels) <= 0.03 and min(finite) == 1.0 and min(ratios) >= 0.8 else "no",
                }
            )
    return out


def marginal_rows(summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    by_model = defaultdict(list)
    for row in summary_rows:
        by_model[str(row["model"])].append(row)
    for model, rows in by_model.items():
        ordered = sorted(rows, key=lambda r: int(r["threshold"]), reverse=True)
        for prev, cur in zip(ordered, ordered[1:]):
            prev_loop = fnum(prev["closed_loop_median_ms"])
            cur_loop = fnum(cur["closed_loop_median_ms"])
            delta = prev_loop - cur_loop
            gain_pct = (delta / prev_loop * 100.0) if prev_loop else 0.0
            added_layers = fnum(cur["cuda_layers_median"]) - fnum(prev["cuda_layers_median"])
            added_mb = fnum(cur["cuda_mb_median"]) - fnum(prev["cuda_mb_median"])
            rel_delta = fnum(cur["rel_l2_median"]) - fnum(prev["rel_l2_median"])
            saturated = "yes" if added_layers > 0 and gain_pct < 2.0 else "no"
            out.append(
                {
                    "model": model,
                    "from_threshold": prev["threshold"],
                    "to_threshold": cur["threshold"],
                    "added_cuda_layers": f"{added_layers:.0f}",
                    "added_cuda_mb": f"{added_mb:.6f}",
                    "closed_loop_delta_ms": f"{delta:.6f}",
                    "rel_l2_delta": f"{rel_delta:.9f}",
                    "marginal_closed_loop_gain_pct": f"{gain_pct:.6f}",
                    "saturation_flag": saturated,
                }
            )
    return out


def make_payload_for_isolation(
    model_name: str,
    threshold: int,
    models: Mapping[int, Tuple[str, OrderedDict, int]],
    warmup_round: int,
    measured_round: int,
) -> Tuple[OrderedDict, bytes]:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state_id = f"hpc_probe_isolation_{model_name}_{threshold}"
    c_state = FalcomCudaV4State(client_id=state_id, config=config())
    d_state = FalcomCudaV4State(client_id=state_id, config=config())
    warm_client, warm_model, _warm_bytes = models[warmup_round]
    warm_layers, _stats = compress_batch_cuda_v4(
        warm_model,
        state_handle=c_state,
        options={"cuda_min_numel": threshold, "quant_mode": "8", "client_id": warm_client},
    )
    _out, _dstats = decompress_batch_cuda_v4(loads_cuda_v4_layers(dumps_cuda_v4_layers(warm_layers)), state_handle=d_state)
    client_id, model, _original_bytes = models[measured_round]
    layers, _stats = compress_batch_cuda_v4(
        model,
        state_handle=c_state,
        options={"cuda_min_numel": threshold, "quant_mode": "8", "client_id": client_id},
    )
    payload = dumps_cuda_v4_layers(layers)
    return layers, payload


def run_envelope_isolation(
    models_by_name: Mapping[str, Mapping[int, Tuple[str, OrderedDict, int]]],
    thresholds: Sequence[int],
    repeats: int,
) -> List[Dict[str, object]]:
    rows = []
    for model_name, models in models_by_name.items():
        round_ids = sorted(models.keys())
        warmup_round = round_ids[0]
        measured_round = round_ids[1] if len(round_ids) > 1 else round_ids[0]
        for threshold in thresholds:
            layers, payload = make_payload_for_isolation(model_name, threshold, models, warmup_round, measured_round)
            for idx in range(repeats):
                t0 = time.perf_counter()
                dumped = dumps_cuda_v4_layers(layers)
                dumps_ms = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                loaded = loads_cuda_v4_layers(payload)
                loads_ms = (time.perf_counter() - t1) * 1000.0
                parse_ms = fnum(loaded.get("_last_envelope_parse_ms", {}).get("ms"), loads_ms)
                rows.append(
                    {
                        "model": model_name,
                        "threshold": threshold,
                        "repeat_index": idx,
                        "payload_bytes": len(dumped),
                        "python_wrapper_dumps_ms": f"{dumps_ms:.6f}",
                        "cpp_envelope_pack_ms": "0.000000",
                        "python_wrapper_loads_ms": f"{loads_ms:.6f}",
                        "cpp_envelope_parse_ms": "0.000000",
                        "object_rebuild_ms": f"{parse_ms:.6f}",
                        "tensor_materialization_ms": f"{parse_ms:.6f}",
                        "notes": "current envelope pack/parse is Python wrapper code; no C++ envelope API exists",
                    }
                )
    return rows


def run_transfer_isolation(payload_mbs: Sequence[int], repeats: int) -> List[Dict[str, object]]:
    rows = []
    if not torch.cuda.is_available():
        return rows
    for payload_mb in payload_mbs:
        n = int(payload_mb * 1024 * 1024)
        device_src = torch.empty((n,), device="cuda", dtype=torch.uint8)
        device_dst = torch.empty((n,), device="cuda", dtype=torch.uint8)
        buffer_specs = [
            ("pageable", "new_each_time", False),
            ("pinned", "new_each_time", True),
            ("pinned", "reused", True),
        ]
        reused = None
        for host_kind, allocation_policy, pinned in buffer_specs:
            status = "ok"
            if allocation_policy == "reused":
                try:
                    reused = torch.empty((n,), device="cpu", dtype=torch.uint8, pin_memory=pinned)
                except RuntimeError as exc:
                    status = f"pin_alloc_failed:{exc.__class__.__name__}"
                    reused = None
            for direction in ("d2h", "h2d"):
                for idx in range(repeats):
                    try:
                        if allocation_policy == "new_each_time":
                            host = torch.empty((n,), device="cpu", dtype=torch.uint8, pin_memory=pinned)
                        else:
                            if reused is None:
                                raise RuntimeError(status)
                            host = reused
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        safe_sync()
                        wall0 = time.perf_counter()
                        start.record()
                        if direction == "d2h":
                            host.copy_(device_src, non_blocking=pinned)
                        else:
                            device_dst.copy_(host, non_blocking=pinned)
                        end.record()
                        end.synchronize()
                        wall_ms = (time.perf_counter() - wall0) * 1000.0
                        event_ms = float(start.elapsed_time(end))
                        gb = float(n) / (1024.0**3)
                        rows.append(
                            {
                                "payload_mb": payload_mb,
                                "direction": direction,
                                "host_buffer_kind": host_kind,
                                "allocation_policy": allocation_policy,
                                "copy_api": "torch.Tensor.copy_",
                                "sync_location": "cuda_event_end_synchronize",
                                "repeat_index": idx,
                                "wall_ms": f"{wall_ms:.6f}",
                                "cuda_event_ms": f"{event_ms:.6f}",
                                "gb_per_s_wall": f"{(gb / (wall_ms / 1000.0)) if wall_ms else 0.0:.6f}",
                                "gb_per_s_event": f"{(gb / (event_ms / 1000.0)) if event_ms else 0.0:.6f}",
                                "status": "ok",
                            }
                        )
                    except RuntimeError as exc:
                        rows.append(
                            {
                                "payload_mb": payload_mb,
                                "direction": direction,
                                "host_buffer_kind": host_kind,
                                "allocation_policy": allocation_policy,
                                "copy_api": "torch.Tensor.copy_",
                                "sync_location": "cuda_event_end_synchronize",
                                "repeat_index": idx,
                                "wall_ms": "",
                                "cuda_event_ms": "",
                                "gb_per_s_wall": "",
                                "gb_per_s_event": "",
                                "status": f"failed:{exc.__class__.__name__}",
                            }
                        )
    return rows


def generic_launch_event_ms(repeats: int = 500) -> float:
    if not torch.cuda.is_available():
        return 0.0
    x = torch.zeros((1,), device="cuda")
    safe_sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        x.add_(1.0)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(repeats)


def kernel_rows_from_diagnostic(diagnostic_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    launch_ms = generic_launch_event_ms()
    summary = summarize(diagnostic_rows, "diagnostic")
    by_key = {(r["model"], int(r["threshold"])): r for r in summary}
    for model, threshold in sorted(by_key):
        subset = [r for r in diagnostic_rows if r["model"] == model and int(r["threshold"]) == threshold]
        enc = median([fnum(r.get("encode_kernel_ms")) for r in subset])
        dec = median([fnum(r.get("decode_kernel_ms")) for r in subset])
        launches = median([fnum(r.get("kernel_launch_count")) for r in subset])
        decode_launches = median([fnum(r.get("decode_kernel_launch_count")) for r in subset])
        out.append(
            {
                "model": model,
                "threshold": threshold,
                "encode_kernel_ms": f"{enc:.6f}",
                "decode_kernel_ms": f"{dec:.6f}",
                "kernel_launch_count": f"{launches:.0f}",
                "decode_kernel_launch_count": f"{decode_launches:.0f}",
                "encode_ms_per_launch": f"{(enc / launches) if launches else 0.0:.6f}",
                "decode_ms_per_launch": f"{(dec / decode_launches) if decode_launches else 0.0:.6f}",
                "generic_launch_event_ms": f"{launch_ms:.9f}",
                "event_method": "generic_torch_event; extension does not expose per-kernel CUDA events without code changes",
            }
        )
    return out


def allocation_rows_from_diagnostic(diagnostic_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in diagnostic_rows:
        rows.append(
            {
                "model": row["model"],
                "threshold": row["threshold"],
                "round_index": row["measured_round"],
                "clone_count": row.get("clone_count", ""),
                "clone_bytes": row.get("clone_bytes", ""),
                "clone_ms": row.get("clone_ms", ""),
                "decoded_tensor_alloc_count": row.get("cuda_layers", ""),
                "decoded_tensor_alloc_bytes": row.get("payload_blob_bytes", ""),
                "decoded_tensor_alloc_ms": row.get("decoded_tensor_alloc_ms", ""),
                "cudaMalloc_count": row.get("cudaMalloc_count", ""),
                "cudaFree_count": row.get("cudaFree_count", ""),
                "state_pointer_swap_possible": row.get("state_pointer_swap_possible", ""),
            }
        )
    return rows


def run_fallback_stress(
    models_by_name: Mapping[str, Mapping[int, Tuple[str, OrderedDict, int]]],
    cpu_ratios_by_name: Mapping[str, Mapping[int, Mapping[str, float]]],
    samples: int,
) -> List[Dict[str, object]]:
    rows = []
    for model_name, models in models_by_name.items():
        round_ids = sorted(models.keys())
        warmup_round = round_ids[0]
        measured_candidates = round_ids[1:] or round_ids
        for sample in range(samples):
            measured_round = measured_candidates[sample % len(measured_candidates)]
            base, _diag, _fallback = run_one_measured_sample(
                model_name,
                threshold=1,
                sample_index=sample,
                timing_kind="full_cuda_stress",
                models=models,
                cpu_ratios=cpu_ratios_by_name[model_name],
                warmup_round=warmup_round,
                measured_round=measured_round,
                include_diagnostics=False,
            )
            label = "valid_full_cuda" if int(base["fallback_layers"]) == 0 else "unsafe_full_cuda_stress"
            base.update({"stress_label": label, "stress_candidate": "no" if label.startswith("unsafe") else "yes"})
            rows.append(base)
    return rows


def write_report(
    logs: Path,
    official_summary: List[Dict[str, object]],
    diagnostic_summary: List[Dict[str, object]],
    marginal: List[Dict[str, object]],
    diagnostic_rows: List[Dict[str, object]],
) -> None:
    report = logs / "final_hpc_probe_report.md"
    best_resnet50 = None
    resnet50 = [r for r in official_summary if r["model"] == "resnet50" and r["quality_pass"] == "yes"]
    if resnet50:
        best_resnet50 = min(resnet50, key=lambda r: fnum(r["closed_loop_median_ms"]))
    with report.open("w") as f:
        f.write("# CUDA v4 HPC Probe Report\n\n")
        f.write("This probe separates official minimal timing from diagnostic timing. It uses existing CUDA v4 q8 code only and writes under `logs/cuda_v4_hpc_probe/`.\n\n")
        f.write("## Official Timing Medians\n\n")
        f.write("| model | threshold | closed-loop median ms | IQR | compress ms | decompress ms | CUDA layers | fallback layers | rel L2 | ratio retention | quality |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in official_summary:
            f.write(
                f"| {row['model']} | {row['threshold']} | {fnum(row['closed_loop_median_ms']):.3f} | "
                f"{fnum(row['closed_loop_iqr_ms']):.3f} | {fnum(row['compress_median_ms']):.3f} | "
                f"{fnum(row['decompress_median_ms']):.3f} | {row['cuda_layers_median']} | "
                f"{row['fallback_layers_median']} | {fnum(row['rel_l2_median']):.6f} | "
                f"{fnum(row['ratio_retention_median']):.3f} | {row['quality_pass']} |\n"
            )
        f.write("\n## Diagnostic Overhead Check\n\n")
        by_diag = {(r["model"], int(r["threshold"])): r for r in diagnostic_summary}
        f.write("| model | threshold | official closed-loop | diagnostic closed-loop | overhead % | diagnostic usable as official |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for off in official_summary:
            key = (off["model"], int(off["threshold"]))
            diag = by_diag.get(key)
            if not diag:
                continue
            off_ms = fnum(off["closed_loop_median_ms"])
            diag_ms = fnum(diag["closed_loop_median_ms"])
            overhead = ((diag_ms - off_ms) / off_ms * 100.0) if off_ms else 0.0
            f.write(
                f"| {off['model']} | {off['threshold']} | {off_ms:.3f} | {diag_ms:.3f} | "
                f"{overhead:.3f} | {'yes' if abs(overhead) < 2.0 else 'no'} |\n"
            )
        f.write("\n## Marginal Threshold Analysis\n\n")
        f.write("| model | from | to | added CUDA layers | closed-loop delta ms | gain % | saturation |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for row in marginal:
            f.write(
                f"| {row['model']} | {row['from_threshold']} | {row['to_threshold']} | "
                f"{row['added_cuda_layers']} | {fnum(row['closed_loop_delta_ms']):.3f} | "
                f"{fnum(row['marginal_closed_loop_gain_pct']):.3f} | {row['saturation_flag']} |\n"
            )
        f.write("\n## Bottleneck Signals\n\n")
        if diagnostic_rows:
            best_diag_subset = diagnostic_rows
            if best_resnet50:
                best_diag_subset = [
                    r
                    for r in diagnostic_rows
                    if r["model"] == "resnet50" and int(r["threshold"]) == int(best_resnet50["threshold"])
                ]
            if best_diag_subset:
                med_loop = median([fnum(r["closed_loop_ms"]) for r in best_diag_subset])
                items = [
                    ("envelope", median([fnum(r["envelope_serialize_ms"]) + fnum(r["envelope_parse_ms"]) for r in best_diag_subset])),
                    ("fallback", median([fnum(r["cpu_fallback_ms"]) + fnum(r["cpu_fallback_decode_ms"]) for r in best_diag_subset])),
                    ("transfer", median([fnum(r["payload_d2h_ms"]) + fnum(r["payload_h2d_ms"]) for r in best_diag_subset])),
                    ("kernel", median([fnum(r["encode_kernel_ms"]) + fnum(r["decode_kernel_ms"]) for r in best_diag_subset])),
                    ("clone", median([fnum(r["clone_ms"]) for r in best_diag_subset])),
                ]
                for name, ms in items:
                    pct = (ms / med_loop * 100.0) if med_loop else 0.0
                    f.write(f"- {name}: {ms:.3f} ms, {pct:.1f}% of diagnostic closed-loop at selected point.\n")
        stress_path = logs / "fallback_stress.csv"
        valid_stress = []
        if stress_path.exists():
            stress_rows = list(csv.DictReader(stress_path.open()))
            valid_stress = [
                r
                for r in stress_rows
                if r.get("model") == "resnet50"
                and r.get("stress_label") == "valid_full_cuda"
                and r.get("correctness_status") == "pass"
                and fnum(r.get("ratio_retention")) >= 0.8
                and fnum(r.get("finite_rate")) == 1.0
                and fnum(r.get("relative_l2_error")) <= 0.03
            ]
        if valid_stress:
            stress_loop = median([fnum(r["closed_loop_ms"]) for r in valid_stress])
            stress_rel = median([fnum(r["relative_l2_error"]) for r in valid_stress])
            stress_ratio = median([fnum(r["ratio_retention"]) for r in valid_stress])
            f.write("\n## Full-CUDA Stress\n\n")
            f.write(
                f"- Valid full-CUDA stress median for ResNet50: {stress_loop:.3f} ms closed-loop, "
                f"rel L2 {stress_rel:.6f}, ratio retention {stress_ratio:.3f}.\n"
            )
            f.write("- This is not a production default by itself, but it is safe evidence that lower-threshold/all-CUDA coverage has headroom.\n")
        f.write("\n## Decision\n\n")
        if best_resnet50:
            best_ms = fnum(best_resnet50["closed_loop_median_ms"])
            best_threshold = int(best_resnet50["threshold"])
            f.write(f"- Best official ResNet50 threshold: {best_threshold}, closed-loop {best_ms:.3f} ms.\n")
            if best_ms <= RESNET50_CONTINUE_TARGET_MS:
                f.write("- Decision: continue with a very narrow patch only if it selects this threshold and preserves quality gates.\n")
            elif valid_stress and median([fnum(r["closed_loop_ms"]) for r in valid_stress]) <= RESNET50_CONTINUE_TARGET_MS:
                f.write(
                    "- Decision: continue with one very narrow low-threshold/all-CUDA guard validation patch. "
                    "Do not open a new codec; prove threshold <2048 in official timing, preserve transaction semantics, and keep CPU default unchanged.\n"
                )
            else:
                over_15 = False
                if diagnostic_rows:
                    subset = [r for r in diagnostic_rows if r["model"] == "resnet50" and int(r["threshold"]) == best_threshold]
                    if subset:
                        med_loop = median([fnum(r["closed_loop_ms"]) for r in subset])
                        for key in ("cpu_fallback_ms", "cpu_fallback_decode_ms", "envelope_serialize_ms", "envelope_parse_ms", "clone_ms"):
                            if med_loop and median([fnum(r.get(key)) for r in subset]) / med_loop > 0.15:
                                over_15 = True
                if over_15:
                    f.write("- Decision: plan at most one targeted patch for the >15% overhead component; do not open a new codec.\n")
                else:
                    f.write("- Decision: stop GPU compressor-internal optimization; v4 is the final experimental compressor result.\n")
        else:
            f.write("- Decision: no quality-passing ResNet50 official row; stop and investigate correctness before optimization.\n")
        f.write("\n## Regression Notes\n\n")
        for rel in (
            "libmomentum_compressor_openmp_simd_final.so",
            "EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so",
        ):
            path = PROJECT_ROOT / rel
            if path.exists():
                f.write(f"- `{rel}` SHA256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`\n")
        f.write("- See CSV artifacts for envelope, transfer, allocation, fallback, and kernel-launch isolation details.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--thresholds", nargs="+", type=int, default=[262144, 65536, 16384, 8192, 4096, 2048])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--envelope-repeats", type=int, default=100)
    parser.add_argument("--transfer-repeats", type=int, default=20)
    parser.add_argument("--transfer-mb", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--stress-samples", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.thresholds = [16384, 8192]
        args.samples = 2
        args.envelope_repeats = 5
        args.transfer_repeats = 3
        args.transfer_mb = [8, 24]
        args.stress_samples = 1
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
    if not extension_available():
        raise RuntimeError("CUDA v4 extension is not available")

    logs = PROJECT_ROOT / "logs" / "cuda_v4_hpc_probe"
    logs.mkdir(parents=True, exist_ok=True)

    models_by_name: Dict[str, Dict[int, Tuple[str, OrderedDict, int]]] = {}
    cpu_ratios_by_name: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model_name in args.models:
        rounds = available_rounds(model_name)
        if not rounds:
            raise RuntimeError(f"No replay rounds found for {model_name}")
        models = load_models(model_name, rounds)
        models_by_name[model_name] = models
        cpu_ratios_by_name[model_name] = cpu_ratio_baselines(model_name, models)

    official_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    fallback_rows: List[Dict[str, object]] = []
    for timing_kind, include_diagnostics in (("official", False), ("diagnostic", True)):
        for model_name in args.models:
            rounds = sorted(models_by_name[model_name].keys())
            warmup_round = rounds[0]
            measured_candidates = rounds[1:] or rounds
            for threshold in args.thresholds:
                for sample in range(args.samples):
                    measured_round = measured_candidates[sample % len(measured_candidates)]
                    official, diagnostic, fallback = run_one_measured_sample(
                        model_name,
                        threshold,
                        sample,
                        timing_kind,
                        models_by_name[model_name],
                        cpu_ratios_by_name[model_name],
                        warmup_round,
                        measured_round,
                        include_diagnostics,
                    )
                    if include_diagnostics:
                        diagnostic_rows.append(diagnostic)
                        fallback_rows.extend(fallback)
                    else:
                        official_rows.append(official)

    official_summary = summarize(official_rows, "official")
    diagnostic_summary = summarize(diagnostic_rows, "diagnostic")
    marginal = marginal_rows(official_summary)
    envelope_rows = run_envelope_isolation(models_by_name, args.thresholds, args.envelope_repeats)
    transfer_rows = run_transfer_isolation(args.transfer_mb, args.transfer_repeats)
    kernel_rows = kernel_rows_from_diagnostic(diagnostic_rows)
    allocation_rows = allocation_rows_from_diagnostic(diagnostic_rows)
    stress_rows = run_fallback_stress(models_by_name, cpu_ratios_by_name, args.stress_samples)

    write_csv(logs / "official_timing.csv", official_rows, OFFICIAL_FIELDS)
    write_csv(logs / "official_summary.csv", official_summary, SUMMARY_FIELDS)
    write_csv(logs / "diagnostic_breakdown.csv", diagnostic_rows, DIAGNOSTIC_FIELDS)
    write_csv(logs / "diagnostic_summary.csv", diagnostic_summary, SUMMARY_FIELDS)
    write_csv(logs / "threshold_marginal.csv", marginal, MARGINAL_FIELDS)
    write_csv(logs / "envelope_isolation.csv", envelope_rows, ENVELOPE_FIELDS)
    write_csv(logs / "transfer_isolation.csv", transfer_rows, TRANSFER_FIELDS)
    write_csv(logs / "kernel_launch_isolation.csv", kernel_rows, KERNEL_FIELDS)
    write_csv(logs / "allocation_state_copy.csv", allocation_rows, ALLOCATION_FIELDS)
    write_csv(logs / "fallback_ranking.csv", fallback_rows, FALLBACK_FIELDS)
    write_csv(logs / "fallback_stress.csv", stress_rows, STRESS_FIELDS)
    write_report(logs, official_summary, diagnostic_summary, marginal, diagnostic_rows)
    print(logs / "final_hpc_probe_report.md")


if __name__ == "__main__":
    main()
