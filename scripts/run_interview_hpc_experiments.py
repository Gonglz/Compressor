#!/usr/bin/env python3
"""Build the lightweight HPC interview evidence package.

The public GitHub snapshot intentionally omits replay `.bin` files and local
build products. This runner writes the committed CSV evidence under
`logs/interview_hpc/` while using the full local experiment tree for optional
CUDA guarded diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = Path("/scratch2/lgong1/Compressor/compressor_gpu_experiment")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "interview_hpc"
DEFAULT_SELECTED_QUALITY = REPO_ROOT / "logs" / "cuda_v4_guarded_all_cuda" / "quality.csv"
DEFAULT_SELECTED_TIMING = REPO_ROOT / "logs" / "cuda_v4_guarded_all_cuda" / "official_timing.csv"

QUALITY_REL_L2_MAX = 0.03
QUALITY_RATIO_MIN = 0.8


CPU_AUDIT_ROWS = {
    "resnet50": {
        1: (936.56, 1098.56, 907.36, 0.85, 62.44),
        2: (913.18, 548.04, 472.43, 1.67, 62.44),
        4: (803.11, 304.25, 262.88, 2.64, 62.44),
        8: (797.62, 211.36, 178.27, 3.77, 62.44),
        16: (901.40, 265.33, 238.53, 3.40, 62.44),
    },
    "resnet18": {
        1: (323.64, 395.77, 299.90, 0.82, 220.71),
        2: (324.39, 221.24, 181.71, 1.47, 220.71),
        4: (311.53, 132.54, 110.62, 2.35, 220.71),
        8: (337.84, 128.66, 108.06, 2.63, 220.71),
        16: (355.88, 133.58, 108.89, 2.66, 220.71),
    },
}

CPU_AUDIT_BEST_DECOMPRESS_MS = {
    "resnet50": {8: 273.43},
    "resnet18": {8: 118.81},
}

CPU_RELEASE_GATE_DECOMPRESS_MS = {
    "resnet50": {1: 248.897, 2: 261.913, 4: 267.179, 8: 273.707},
    "resnet18": {1: 109.961, 2: 116.568, 4: 120.434, 8: 125.354},
}


def fnum(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{fnum(value):.{digits}f}"


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def materialize_cpu_thread_scaling(output_dir: Path) -> None:
    fields = [
        "model",
        "threads",
        "pinning",
        "omp_num_threads",
        "omp_proc_bind",
        "omp_places",
        "warmup_round",
        "hot_rounds",
        "measurement_status",
        "compress_source",
        "decompress_source",
        "safe_serial_compress_median_ms",
        "safe_grouped_compress_median_ms",
        "safe_grouped_decompress_median_ms",
        "safe_grouped_closed_loop_median_ms",
        "unsafe_v22_compress_median_ms",
        "compression_ratio",
        "speedup_vs_safe_serial_compress",
        "notes",
    ]
    rows: List[Dict[str, object]] = []
    for model, by_thread in CPU_AUDIT_ROWS.items():
        for threads, values in by_thread.items():
            serial_ms, grouped_ms, unsafe_ms, speedup, ratio = values
            audit_decompress = CPU_AUDIT_BEST_DECOMPRESS_MS.get(model, {}).get(threads)
            release_decompress = CPU_RELEASE_GATE_DECOMPRESS_MS.get(model, {}).get(threads)
            decomp_ms = audit_decompress if audit_decompress is not None else release_decompress
            if audit_decompress is not None:
                decomp_source = "reports/final_audit_report.md best 8T hot-round median"
            elif release_decompress is not None:
                decomp_source = "reports/final_cpu_logging_gate_gpu_feasibility_report.md installed benchmark"
            else:
                decomp_source = "not measured in committed audit for this thread count"
            rows.append(
                {
                    "model": model,
                    "threads": threads,
                    "pinning": "pinned",
                    "omp_num_threads": threads,
                    "omp_proc_bind": "close",
                    "omp_places": "cores",
                    "warmup_round": "round_0",
                    "hot_rounds": "round_1,round_2",
                    "measurement_status": "audited",
                    "compress_source": "reports/final_audit_report.md thread scan",
                    "decompress_source": decomp_source,
                    "safe_serial_compress_median_ms": fmt(serial_ms, 3),
                    "safe_grouped_compress_median_ms": fmt(grouped_ms, 3),
                    "safe_grouped_decompress_median_ms": fmt(decomp_ms, 3) if decomp_ms is not None else "",
                    "safe_grouped_closed_loop_median_ms": fmt(grouped_ms + decomp_ms, 3) if decomp_ms is not None else "",
                    "unsafe_v22_compress_median_ms": fmt(unsafe_ms, 3),
                    "compression_ratio": fmt(ratio, 3),
                    "speedup_vs_safe_serial_compress": fmt(speedup, 3),
                    "notes": "Use 8T rows for the headline direct audit closed-loop comparison."
                    if threads == 8
                    else "Compress thread scan is direct audit evidence; non-8T decompress comes from the release logging gate when available.",
                }
            )
    write_csv(output_dir / "cpu_thread_scaling.csv", rows, fields)

    unpinned_rows: List[Dict[str, object]] = []
    for model in CPU_AUDIT_ROWS:
        for threads in (1, 2, 4, 8, 16):
            unpinned_rows.append(
                {
                    "model": model,
                    "threads": threads,
                    "pinning": "unpinned",
                    "omp_num_threads": threads,
                    "omp_proc_bind": "",
                    "omp_places": "",
                    "warmup_round": "round_0",
                    "hot_rounds": "round_1,round_2",
                    "measurement_status": "not_available_in_committed_audit",
                    "compress_source": "",
                    "decompress_source": "",
                    "safe_serial_compress_median_ms": "",
                    "safe_grouped_compress_median_ms": "",
                    "safe_grouped_decompress_median_ms": "",
                    "safe_grouped_closed_loop_median_ms": "",
                    "unsafe_v22_compress_median_ms": "",
                    "compression_ratio": "",
                    "speedup_vs_safe_serial_compress": "",
                    "notes": "Committed headline uses pinned OpenMP results. Re-run the full CPU benchmark with OMP_PROC_BIND/OMP_PLACES unset to fill this backup table.",
                }
            )
    write_csv(output_dir / "cpu_thread_scaling_unpinned.csv", unpinned_rows, fields)


def selected_quality_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    selected = {}
    for row in rows:
        if row.get("threshold") == "guarded_all_cuda" and row.get("guarded_all_cuda") == "yes":
            selected[row["model"]] = row
    missing = {"resnet50", "resnet18"} - set(selected)
    if missing:
        raise RuntimeError(f"Missing guarded_all_cuda quality rows for: {sorted(missing)}")
    return selected


def selected_timing_rows(path: Path) -> Dict[str, List[Dict[str, str]]]:
    rows = read_csv(path)
    selected: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("threshold") == "guarded_all_cuda" and row.get("guarded_all_cuda") == "yes":
            selected.setdefault(row["model"], []).append(row)
    return selected


def materialize_gpu_speedup(
    output_dir: Path,
    quality_by_model: Mapping[str, Mapping[str, str]],
    timing_by_model: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    speed_fields = [
        "model",
        "cpu_source",
        "cpu_grouped_threads",
        "cpu_pinning",
        "cpu_grouped_compress_ms",
        "cpu_grouped_decompress_ms",
        "cpu_grouped_closed_loop_ms",
        "gpu_source",
        "gpu_codec",
        "gpu_mode",
        "gpu_compress_median_ms",
        "gpu_decompress_median_ms",
        "gpu_closed_loop_median_ms",
        "compress_only_speedup_vs_cpu_grouped",
        "closed_loop_speedup_vs_cpu_grouped",
        "speedup_label",
        "timing_scope",
    ]
    rows: List[Dict[str, object]] = []
    for model in ("resnet50", "resnet18"):
        q = quality_by_model[model]
        cpu_compress = CPU_AUDIT_ROWS[model][8][1]
        cpu_decompress = CPU_AUDIT_BEST_DECOMPRESS_MS[model][8]
        cpu_closed = cpu_compress + cpu_decompress
        gpu_compress = fnum(q["compress_median_ms"])
        gpu_decompress = fnum(q["decompress_median_ms"])
        gpu_closed = fnum(q["closed_loop_median_ms"])
        rows.append(
            {
                "model": model,
                "cpu_source": "reports/final_audit_report.md safe grouped OpenMP/SIMD 8T",
                "cpu_grouped_threads": 8,
                "cpu_pinning": "OMP_PROC_BIND=close; OMP_PLACES=cores",
                "cpu_grouped_compress_ms": fmt(cpu_compress, 3),
                "cpu_grouped_decompress_ms": fmt(cpu_decompress, 3),
                "cpu_grouped_closed_loop_ms": fmt(cpu_closed, 3),
                "gpu_source": "logs/cuda_v4_guarded_all_cuda/quality.csv",
                "gpu_codec": "cuda_v4_q8",
                "gpu_mode": "guarded_all_cuda",
                "gpu_compress_median_ms": fmt(gpu_compress, 6),
                "gpu_decompress_median_ms": fmt(gpu_decompress, 6),
                "gpu_closed_loop_median_ms": fmt(gpu_closed, 6),
                "compress_only_speedup_vs_cpu_grouped": fmt(cpu_compress / gpu_compress, 3),
                "closed_loop_speedup_vs_cpu_grouped": fmt(cpu_closed / gpu_closed, 3),
                "speedup_label": "closed-loop headline" if model == "resnet50" else "supporting closed-loop",
                "timing_scope": "Replay file load excluded; timed CUDA closed-loop includes codec kernel, internal payload D2H/H2D, and envelope serialize/parse.",
            }
        )
    write_csv(output_dir / "gpu_speedup_breakdown.csv", rows, speed_fields)

    quality_fields = [
        "model",
        "cpu_production_wire_format",
        "cuda_v4_wire_format",
        "cuda_experimental_gate",
        "relative_l2_median",
        "finite_rate_min",
        "ratio_retention_median",
        "decode_status",
        "correctness_status",
        "quality_pass",
        "cuda_layers_median",
        "fallback_layers_median",
        "payload_objects_median",
        "quality_constraints",
        "source",
    ]
    quality_rows: List[Dict[str, object]] = []
    for model in ("resnet50", "resnet18"):
        q = quality_by_model[model]
        timing_rows = list(timing_by_model.get(model, []))
        decode_status = "pass" if timing_rows and all(r.get("decode_status") == "pass" for r in timing_rows) else "fail"
        correctness_status = "pass" if timing_rows and all(r.get("correctness_status") == "pass" for r in timing_rows) else "fail"
        quality_rows.append(
            {
                "model": model,
                "cpu_production_wire_format": "preserved CPU ABI and CPU wire format",
                "cuda_v4_wire_format": "experimental packed payload envelope; not CPU-wire-compatible",
                "cuda_experimental_gate": "FALCOM_CUDA_EXPERIMENTAL=1",
                "relative_l2_median": fmt(q["rel_l2_median"], 9),
                "finite_rate_min": fmt(q["finite_rate_min"], 9),
                "ratio_retention_median": fmt(q["ratio_retention_median"], 6),
                "decode_status": decode_status,
                "correctness_status": correctness_status,
                "quality_pass": q["quality_pass"],
                "cuda_layers_median": q["cuda_layers_median"],
                "fallback_layers_median": q["fallback_layers_median"],
                "payload_objects_median": q["num_payload_objects_median"],
                "quality_constraints": f"relative_l2_error<={QUALITY_REL_L2_MAX}; finite_rate==1.0; ratio_retention>={QUALITY_RATIO_MIN}",
                "source": "logs/cuda_v4_guarded_all_cuda/quality.csv and official_timing.csv",
            }
        )
    write_csv(output_dir / "wire_format_quality.csv", quality_rows, quality_fields)


def import_guarded_runner(experiment_root: Path) -> Any:
    runner_path = experiment_root / "cuda_feasibility" / "falcom_cuda_v4" / "run_cuda_v4_guarded_all_cuda.py"
    if not runner_path.exists():
        raise FileNotFoundError(runner_path)
    spec = importlib.util.spec_from_file_location("interview_guarded_cuda_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_guarded_diagnostic_sample(
    guarded: Any,
    model_name: str,
    sample_index: int,
    loaded_models: Mapping[int, Any],
    cpu_ratios: Mapping[int, Mapping[str, float]],
    warmup_round: int,
    measured_round: int,
) -> Dict[str, Any]:
    state_id = f"interview_hpc_{model_name}_{sample_index}"
    c_state = guarded.FalcomCudaV4State(client_id=state_id, config=guarded.config())
    d_state = guarded.FalcomCudaV4State(client_id=state_id, config=guarded.config())
    options = {
        "cuda_min_numel": 2048,
        "quant_mode": "8",
        "guarded_all_cuda": True,
    }

    warm_client, warm_model, _warm_original = loaded_models[warmup_round]
    warm_layers, _warm_cstats = guarded.compress_batch_cuda_v4(
        warm_model,
        state_handle=c_state,
        options={**options, "client_id": warm_client},
    )
    warm_payload = guarded.dumps_cuda_v4_layers(warm_layers)
    warm_loaded = guarded.loads_cuda_v4_layers(warm_payload)
    _warm_decoded, _warm_dstats = guarded.decompress_batch_cuda_v4(warm_loaded, state_handle=d_state)
    guarded.safe_sync()

    client_id, model, original_bytes = loaded_models[measured_round]
    guarded.safe_sync()
    t0 = time.perf_counter()
    layers, cstats = guarded.compress_batch_cuda_v4(
        model,
        state_handle=c_state,
        options={**options, "client_id": client_id},
    )
    payload = guarded.dumps_cuda_v4_layers(layers)
    envelope_serialize_ms = fnum(layers.get("_last_envelope_serialize_ms", {}).get("ms"))
    guarded.safe_sync()
    t1 = time.perf_counter()
    loaded = guarded.loads_cuda_v4_layers(payload)
    envelope_parse_ms = fnum(loaded.get("_last_envelope_parse_ms", {}).get("ms"))
    decoded, dstats = guarded.decompress_batch_cuda_v4(loaded, state_handle=d_state)
    guarded.safe_sync()
    t2 = time.perf_counter()

    metrics = guarded.error_metrics(model, decoded)
    ratio = float(original_bytes) / float(len(payload)) if payload else 0.0
    cpu_ratio = float(cpu_ratios[measured_round]["compression_ratio"])
    decode_ok = (
        int(dstats.get("cuda_layers", 0)) == int(cstats.get("cuda_layers", 0))
        and int(dstats.get("cpu_fallback_layers", 0)) == int(cstats.get("cpu_fallback_layers", 0))
        and len(decoded) == len(model)
        and all(name in decoded for name in model.keys())
    )
    compress_ms = (t1 - t0) * 1000.0
    decompress_ms = (t2 - t1) * 1000.0
    return {
        "model": model_name,
        "sample_index": sample_index,
        "warmup_round": warmup_round,
        "measured_round": measured_round,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "closed_loop_ms": compress_ms + decompress_ms,
        "encode_kernel_ms": fnum(cstats.get("encode_kernel_ms")),
        "decode_kernel_ms": fnum(dstats.get("decode_kernel_ms")),
        "payload_d2h_ms": fnum(cstats.get("payload_d2h_ms")),
        "payload_h2d_ms": fnum(dstats.get("payload_h2d_ms")),
        "envelope_serialize_ms": envelope_serialize_ms,
        "envelope_parse_ms": envelope_parse_ms,
        "decoded_tensor_materialize_ms": fnum(dstats.get("decoded_tensor_materialize_ms")),
        "cpu_fallback_ms": fnum(cstats.get("cpu_fallback_ms")),
        "cpu_fallback_decode_ms": fnum(dstats.get("cpu_fallback_decode_ms")),
        "fallback_gpu_to_cpu_ms": fnum(cstats.get("fallback_gpu_to_cpu_ms")),
        "kernel_launch_count": fnum(cstats.get("kernel_launch_count")),
        "decode_kernel_launch_count": fnum(dstats.get("decode_kernel_launch_count")),
        "cuda_layers": fnum(cstats.get("cuda_layers")),
        "fallback_layers": fnum(cstats.get("cpu_fallback_layers")),
        "payload_blob_bytes": fnum(cstats.get("payload_blob_bytes")),
        "final_payload_bytes": len(payload),
        "compression_ratio": ratio,
        "cpu_wrapper_compression_ratio": cpu_ratio,
        "ratio_retention": ratio / cpu_ratio if cpu_ratio else 0.0,
        "relative_l2_error": fnum(metrics["relative_l2_error"]),
        "finite_rate": fnum(metrics["finite_rate"]),
        "decode_status": "pass" if decode_ok else "fail",
        "correctness_status": "pass"
        if metrics["finite_rate"] == 1.0 and metrics["relative_l2_error"] <= QUALITY_REL_L2_MAX
        else "fail",
    }


def run_guarded_transfer_diagnostics(
    output_dir: Path,
    experiment_root: Path,
    models: Sequence[str],
    samples: int,
    gpu: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    os.environ["FALCOM_CUDA_V4_GUARDED_ALL_CUDA"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    guarded = import_guarded_runner(experiment_root)
    if not guarded.extension_available():
        raise RuntimeError("CUDA v4 extension is not available")

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in models:
        rounds = guarded.available_rounds(model_name)
        if not rounds:
            raise RuntimeError(f"No replay rounds found for {model_name} under {experiment_root}")
        loaded_models = guarded.load_models(model_name, rounds)
        cpu_ratios = guarded.cpu_ratio_baselines(loaded_models)
        warmup_round = rounds[0]
        measured_candidates = rounds[1:] or rounds
        for sample in range(samples):
            measured_round = measured_candidates[sample % len(measured_candidates)]
            row = run_guarded_diagnostic_sample(
                guarded,
                model_name,
                sample,
                loaded_models,
                cpu_ratios,
                warmup_round,
                measured_round,
            )
            rows_by_model.setdefault(model_name, []).append(row)

    fields = [
        "model",
        "samples",
        "diagnostic_source",
        "cuda_visible_devices",
        "timing_scope",
        "compress_median_ms",
        "decompress_median_ms",
        "closed_loop_median_ms",
        "encode_kernel_ms",
        "decode_kernel_ms",
        "payload_d2h_ms",
        "payload_h2d_ms",
        "envelope_serialize_ms",
        "envelope_parse_ms",
        "decoded_tensor_materialize_ms",
        "cpu_fallback_ms",
        "cpu_fallback_decode_ms",
        "fallback_gpu_to_cpu_ms",
        "kernel_launch_count",
        "decode_kernel_launch_count",
        "cuda_layers",
        "fallback_layers",
        "payload_blob_bytes",
        "final_payload_bytes",
        "decode_status",
        "correctness_status",
    ]
    summary_rows: List[Dict[str, object]] = []
    for model_name, rows in rows_by_model.items():
        def med(key: str) -> float:
            return median([fnum(r[key]) for r in rows])

        summary_rows.append(
            {
                "model": model_name,
                "samples": len(rows),
                "diagnostic_source": str(experiment_root),
                "cuda_visible_devices": gpu,
                "timing_scope": "Replay loading excluded; model tensors already on CUDA; codec internal payload D2H/H2D, kernels, and envelope overhead included.",
                "compress_median_ms": fmt(med("compress_ms"), 6),
                "decompress_median_ms": fmt(med("decompress_ms"), 6),
                "closed_loop_median_ms": fmt(med("closed_loop_ms"), 6),
                "encode_kernel_ms": fmt(med("encode_kernel_ms"), 6),
                "decode_kernel_ms": fmt(med("decode_kernel_ms"), 6),
                "payload_d2h_ms": fmt(med("payload_d2h_ms"), 6),
                "payload_h2d_ms": fmt(med("payload_h2d_ms"), 6),
                "envelope_serialize_ms": fmt(med("envelope_serialize_ms"), 6),
                "envelope_parse_ms": fmt(med("envelope_parse_ms"), 6),
                "decoded_tensor_materialize_ms": fmt(med("decoded_tensor_materialize_ms"), 6),
                "cpu_fallback_ms": fmt(med("cpu_fallback_ms"), 6),
                "cpu_fallback_decode_ms": fmt(med("cpu_fallback_decode_ms"), 6),
                "fallback_gpu_to_cpu_ms": fmt(med("fallback_gpu_to_cpu_ms"), 6),
                "kernel_launch_count": fmt(med("kernel_launch_count"), 0),
                "decode_kernel_launch_count": fmt(med("decode_kernel_launch_count"), 0),
                "cuda_layers": fmt(med("cuda_layers"), 0),
                "fallback_layers": fmt(med("fallback_layers"), 0),
                "payload_blob_bytes": fmt(med("payload_blob_bytes"), 0),
                "final_payload_bytes": fmt(med("final_payload_bytes"), 0),
                "decode_status": "pass" if all(r["decode_status"] == "pass" for r in rows) else "fail",
                "correctness_status": "pass" if all(r["correctness_status"] == "pass" for r in rows) else "fail",
            }
        )
    write_csv(output_dir / "gpu_transfer_breakdown.csv", summary_rows, fields)


def write_failed_transfer_breakdown(output_dir: Path, message: str) -> None:
    fields = [
        "model",
        "samples",
        "diagnostic_source",
        "cuda_visible_devices",
        "timing_scope",
        "compress_median_ms",
        "decompress_median_ms",
        "closed_loop_median_ms",
        "encode_kernel_ms",
        "decode_kernel_ms",
        "payload_d2h_ms",
        "payload_h2d_ms",
        "envelope_serialize_ms",
        "envelope_parse_ms",
        "decoded_tensor_materialize_ms",
        "cpu_fallback_ms",
        "cpu_fallback_decode_ms",
        "fallback_gpu_to_cpu_ms",
        "kernel_launch_count",
        "decode_kernel_launch_count",
        "cuda_layers",
        "fallback_layers",
        "payload_blob_bytes",
        "final_payload_bytes",
        "decode_status",
        "correctness_status",
    ]
    rows = [
        {
            "model": model,
            "samples": 0,
            "diagnostic_source": "diagnostic_failed",
            "cuda_visible_devices": "",
            "timing_scope": message,
            "compress_median_ms": "",
            "decompress_median_ms": "",
            "closed_loop_median_ms": "",
            "encode_kernel_ms": "",
            "decode_kernel_ms": "",
            "payload_d2h_ms": "",
            "payload_h2d_ms": "",
            "envelope_serialize_ms": "",
            "envelope_parse_ms": "",
            "decoded_tensor_materialize_ms": "",
            "cpu_fallback_ms": "",
            "cpu_fallback_decode_ms": "",
            "fallback_gpu_to_cpu_ms": "",
            "kernel_launch_count": "",
            "decode_kernel_launch_count": "",
            "cuda_layers": "",
            "fallback_layers": "",
            "payload_blob_bytes": "",
            "final_payload_bytes": "",
            "decode_status": "not_run",
            "correctness_status": "not_run",
        }
        for model in ("resnet50", "resnet18")
    ]
    write_csv(output_dir / "gpu_transfer_breakdown.csv", rows, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-quality", type=Path, default=DEFAULT_SELECTED_QUALITY)
    parser.add_argument("--selected-timing", type=Path, default=DEFAULT_SELECTED_TIMING)
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet18"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--skip-gpu-diagnostic", action="store_true")
    parser.add_argument("--allow-gpu-diagnostic-failure", action="store_true")
    args = parser.parse_args()

    materialize_cpu_thread_scaling(args.output_dir)
    quality_by_model = selected_quality_rows(args.selected_quality)
    timing_by_model = selected_timing_rows(args.selected_timing)
    materialize_gpu_speedup(args.output_dir, quality_by_model, timing_by_model)

    if args.skip_gpu_diagnostic:
        write_failed_transfer_breakdown(args.output_dir, "skipped by --skip-gpu-diagnostic")
    else:
        try:
            run_guarded_transfer_diagnostics(args.output_dir, args.experiment_root, args.models, args.samples, args.gpu)
        except Exception as exc:
            if not args.allow_gpu_diagnostic_failure:
                raise
            write_failed_transfer_breakdown(args.output_dir, str(exc))

    print(args.output_dir)


if __name__ == "__main__":
    main()
