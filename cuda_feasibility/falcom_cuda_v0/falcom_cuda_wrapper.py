#!/usr/bin/env python3
"""Experimental device-resident FalCom CUDA v0 wrapper.

This module is deliberately outside the default CPU compressor path.  It only
activates when callers use it directly and set FALCOM_CUDA_EXPERIMENTAL=1.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import pickle
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
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

try:
    import _falcom_cuda_v0  # type: ignore  # noqa: E402
except Exception as exc:  # pragma: no cover - exercised by build sanity
    _falcom_cuda_v0 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MAX_NDIM = 8
RESIDUAL_CODEC_RAW = "raw_v0"
RESIDUAL_CODEC_FULL_PROBE = "cpu_codec_full_residual_probe"
RESIDUAL_CODEC_HYBRID = "hybrid_compact_v1a"
RESIDUAL_CODECS = {RESIDUAL_CODEC_RAW, RESIDUAL_CODEC_FULL_PROBE, RESIDUAL_CODEC_HYBRID}


def cuda_enabled() -> bool:
    value = os.environ.get("FALCOM_CUDA_EXPERIMENTAL", "")
    return value not in ("", "0", "false", "False", "FALSE", "off", "OFF")


def default_config() -> Any:
    return OmegaConf.create(
        {
            "momentum_lr": 0.07,
            "consistency_threshold": 0.5,
            "param_cutoff": 1024,
            "lossless_compressor": "zstd",
            "sz_config": {"error_bounding_mode": "REL", "error_bound": 1e-3},
        }
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _shape_tuple(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(x) for x in t.shape)


def layer_key(layer_name: str, tensor: torch.Tensor) -> str:
    shape = "x".join(str(x) for x in _shape_tuple(tensor))
    return f"{layer_name}|{str(tensor.dtype)}|{shape}"


def _dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float32:
        return "float32"
    if tensor.dtype == torch.float64:
        return "float64"
    if tensor.dtype == torch.int32:
        return "int32"
    if tensor.dtype == torch.int64:
        return "int64"
    if tensor.dtype == torch.uint8:
        return "uint8"
    return str(tensor.dtype)


def _pack_bitmap(flags: np.ndarray) -> bytes:
    bits = np.asarray(flags != 0, dtype=np.uint8)
    return np.packbits(bits, bitorder="little").tobytes()


def _pack_dense_signs(flags: np.ndarray, signs: np.ndarray) -> bytes:
    dense = np.asarray(signs[flags != 0] > 0, dtype=np.uint8)
    if dense.size == 0:
        return b""
    return np.packbits(dense, bitorder="little").tobytes()


class _Zstd:
    def __init__(self) -> None:
        lib_name = ctypes.util.find_library("zstd") or "libzstd.so.1"
        self.lib = ctypes.CDLL(lib_name)
        self.lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self.lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.ZSTD_compress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint
        self.lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p

    def compress(self, data: bytes, level: int = 10) -> bytes:
        src = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        bound = self.lib.ZSTD_compressBound(len(data))
        dst = (ctypes.c_uint8 * bound)()
        size = self.lib.ZSTD_compress(dst, bound, src, len(data), level)
        if self.lib.ZSTD_isError(size):
            msg = self.lib.ZSTD_getErrorName(size).decode("utf-8", errors="replace")
            raise RuntimeError(f"ZSTD_compress failed: {msg}")
        return bytes(dst[:size])


_ZSTD: Optional[_Zstd] = None


def zstd_compress(data: bytes) -> bytes:
    global _ZSTD
    if _ZSTD is None:
        _ZSTD = _Zstd()
    return _ZSTD.compress(data)


def _payload_bytes(layer: Mapping[str, Any]) -> int:
    return len(layer.get("data", b"")) + len(layer.get("bitmap", b"")) + len(layer.get("dominant_signs", b""))


def _codec_mode(options: Mapping[str, Any]) -> str:
    mode = str(options.get("residual_codec", RESIDUAL_CODEC_RAW))
    if mode not in RESIDUAL_CODECS:
        raise ValueError(f"Unsupported residual_codec={mode!r}; expected one of {sorted(RESIDUAL_CODECS)}")
    return mode


def _fault(options: Mapping[str, Any], stage: str, layer_name: str) -> None:
    requested = str(options.get("fault_inject_stage", ""))
    requested_layer = options.get("fault_inject_layer")
    if requested == stage and (requested_layer is None or str(requested_layer) == layer_name):
        raise RuntimeError(f"Injected CUDA v1a transaction failure at {stage} for {layer_name}")


def _record_layer(
    stats: MutableMapping[str, Any],
    *,
    layer_name: str,
    key: str,
    tensor: torch.Tensor,
    backend: str,
    codec_mode: str,
    c_type: str,
    c_codec: str,
    payload_bytes: int,
    full_residual_bytes: int = 0,
    compact_intermediate_d2h_bytes: int = 0,
    cpu_payload_bytes: int = 0,
    codec_failure_reason: str = "",
    state_committed: bool = True,
    cuda_kernel_ms: float = 0.0,
    cuda_payload_d2h_ms: float = 0.0,
    cpu_payload_assembly_ms: float = 0.0,
    cpu_reconstruct_input_ms: float = 0.0,
    cpu_encoder_ms: float = 0.0,
    cuda_state_commit_ms: float = 0.0,
    cpu_fallback_ms: float = 0.0,
) -> None:
    records = stats.setdefault("layer_records", [])
    compact_pct = (
        100.0 * float(compact_intermediate_d2h_bytes) / float(full_residual_bytes)
        if full_residual_bytes
        else 0.0
    )
    records.append(
        {
            "layer_name": layer_name,
            "layer_key": key,
            "shape": "x".join(str(x) for x in _shape_tuple(tensor)),
            "dtype": _dtype_name(tensor),
            "numel": int(tensor.numel()),
            "backend": backend,
            "codec_mode": codec_mode,
            "c_type": c_type,
            "c_codec": c_codec,
            "cpu_payload_bytes": int(cpu_payload_bytes),
            "cuda_v0_payload_bytes": int(payload_bytes) if codec_mode == RESIDUAL_CODEC_RAW else 0,
            "cuda_payload_bytes": int(payload_bytes),
            "full_residual_bytes": int(full_residual_bytes),
            "compact_intermediate_d2h_bytes": int(compact_intermediate_d2h_bytes),
            "compact_vs_full_residual_pct": compact_pct,
            "final_payload_bytes": int(payload_bytes),
            "payload_growth_factor": (
                float(payload_bytes) / float(cpu_payload_bytes) if cpu_payload_bytes else 0.0
            ),
            "ratio_delta": 0.0,
            "decode_status": "not_checked",
            "codec_failure_reason": codec_failure_reason,
            "state_committed": int(bool(state_committed)),
            "cuda_kernel_ms": float(cuda_kernel_ms),
            "cuda_payload_d2h_ms": float(cuda_payload_d2h_ms),
            "cpu_payload_assembly_ms": float(cpu_payload_assembly_ms),
            "cpu_reconstruct_input_ms": float(cpu_reconstruct_input_ms),
            "cpu_encoder_ms": float(cpu_encoder_ms),
            "cuda_state_commit_ms": float(cuda_state_commit_ms),
            "cpu_fallback_ms": float(cpu_fallback_ms),
        }
    )


@dataclass
class FalcomCudaState:
    client_id: str = "Client1"
    cpu_compressor: Optional[FalComC] = None
    config: Any = None
    backend_by_key: Dict[str, str] = field(default_factory=dict)
    prev_grad: Dict[str, torch.Tensor] = field(default_factory=dict)
    prediction_memory: Dict[str, torch.Tensor] = field(default_factory=dict)
    steps: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = default_config()
        if self.cpu_compressor is None:
            self.cpu_compressor = FalComC(self.config)
        self.cpu_compressor.set_client_id(self.client_id)


def _normalize_inputs(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]],
) -> Tuple[List[str], List[torch.Tensor]]:
    if isinstance(torch_tensors, Mapping):
        names = [str(k) for k in torch_tensors.keys()]
        tensors = list(torch_tensors.values())
        return names, tensors
    if layer_names is None:
        raise ValueError("layer_names is required when torch_tensors is not a mapping")
    if len(layer_names) != len(torch_tensors):
        raise ValueError("layer_names length does not match tensor count")
    return [str(x) for x in layer_names], list(torch_tensors)


def _is_cuda_candidate(tensor: torch.Tensor, min_numel: int) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        and tensor.dim() == 4
        and int(tensor.numel()) >= int(min_numel)
    )


def _cpu_fallback_layer(
    state: FalcomCudaState,
    layer_name: str,
    tensor: torch.Tensor,
    stats: MutableMapping[str, float],
) -> Dict[str, Any]:
    assert state.cpu_compressor is not None
    t0 = time.perf_counter()
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            _sync()
        arr = tensor.detach().cpu().numpy()
        if tensor.is_cuda:
            _sync()
    else:
        arr = np.asarray(tensor)
    payload = state.cpu_compressor.compress_model(
        OrderedDict([(layer_name, arr)]),
        client_id=state.client_id,
    )
    layer = pickle.loads(payload)[layer_name]
    fallback_ms = (time.perf_counter() - t0) * 1000.0
    stats["cpu_fallback_ms"] += fallback_ms
    stats["cpu_fallback_layers"] += 1
    stats["cpu_original_mb"] += float(arr.nbytes) / (1024.0 * 1024.0)
    _record_layer(
        stats,
        layer_name=layer_name,
        key=layer_key(layer_name, tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(arr)),
        tensor=tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(arr),
        backend="cpu",
        codec_mode=str(stats.get("codec_mode", RESIDUAL_CODEC_RAW)),
        c_type=str(layer.get("c_type", "")),
        c_codec=str(layer.get("c_codec", "")),
        payload_bytes=_payload_bytes(layer),
        cpu_payload_bytes=_payload_bytes(layer),
        cpu_fallback_ms=fallback_ms,
    )
    return layer


def _encode_cuda_float_payload(
    tensor: torch.Tensor,
    codec_mode: str,
    options: Mapping[str, Any],
    stats: MutableMapping[str, Any],
    layer_name: str,
) -> Tuple[bytes, str, Dict[str, float]]:
    full_residual_bytes = int(tensor.numel() * tensor.element_size())
    timings = {
        "payload_d2h_ms": 0.0,
        "cpu_reconstruct_input_ms": 0.0,
        "cpu_encoder_ms": 0.0,
        "cpu_payload_assembly_ms": 0.0,
        "compact_intermediate_d2h_bytes": 0.0,
        "full_residual_bytes": float(full_residual_bytes),
    }

    if codec_mode == RESIDUAL_CODEC_RAW:
        _sync()
        t_d2h = time.perf_counter()
        raw = tensor.detach().cpu().numpy().astype(np.float32, copy=False).tobytes()
        _sync()
        timings["payload_d2h_ms"] = (time.perf_counter() - t_d2h) * 1000.0
        timings["compact_intermediate_d2h_bytes"] = float(full_residual_bytes)
        return raw, "sz3_memcpy", timings

    if codec_mode == RESIDUAL_CODEC_FULL_PROBE:
        _sync()
        t_d2h = time.perf_counter()
        raw = tensor.detach().cpu().numpy().astype(np.float32, copy=False).tobytes()
        _sync()
        timings["payload_d2h_ms"] = (time.perf_counter() - t_d2h) * 1000.0
        timings["compact_intermediate_d2h_bytes"] = float(full_residual_bytes)
        _fault(options, "after_compact_d2h", layer_name)
        t_enc = time.perf_counter()
        _fault(options, "payload_assembly", layer_name)
        encoded = zstd_compress(raw)
        timings["cpu_encoder_ms"] = (time.perf_counter() - t_enc) * 1000.0
        timings["cpu_payload_assembly_ms"] = timings["cpu_encoder_ms"]
        _fault(options, "decode_check", layer_name)
        return encoded, "zstd", timings

    if codec_mode != RESIDUAL_CODEC_HYBRID:
        raise ValueError(f"Unsupported residual codec {codec_mode}")

    _sync()
    t_d2h = time.perf_counter()
    max_abs = float(tensor.detach().abs().max().item())
    if max_abs == 0.0:
        scale = 1.0
        quantized = torch.zeros_like(tensor, dtype=torch.int8)
    else:
        scale = max_abs / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    compact = quantized.detach().cpu().numpy().astype(np.int8, copy=False)
    _sync()
    timings["payload_d2h_ms"] = (time.perf_counter() - t_d2h) * 1000.0
    timings["compact_intermediate_d2h_bytes"] = float(compact.nbytes)
    _fault(options, "after_compact_d2h", layer_name)

    t_reconstruct = time.perf_counter()
    reconstructed = (compact.astype(np.float32) * np.float32(scale)).astype(np.float32, copy=False)
    residual_bytes = reconstructed.tobytes()
    timings["cpu_reconstruct_input_ms"] = (time.perf_counter() - t_reconstruct) * 1000.0

    t_enc = time.perf_counter()
    _fault(options, "payload_assembly", layer_name)
    encoded = zstd_compress(residual_bytes)
    timings["cpu_encoder_ms"] = (time.perf_counter() - t_enc) * 1000.0
    timings["cpu_payload_assembly_ms"] = timings["cpu_reconstruct_input_ms"] + timings["cpu_encoder_ms"]
    _fault(options, "decode_check", layer_name)
    return encoded, "zstd", timings


def _direct_cuda_layer(
    state: FalcomCudaState,
    key: str,
    layer_name: str,
    tensor: torch.Tensor,
    step: int,
    options: Mapping[str, Any],
    stats: MutableMapping[str, float],
) -> Dict[str, Any]:
    shape = _shape_tuple(tensor)
    codec_mode = _codec_mode(options)
    t_payload0 = time.perf_counter()
    data, c_codec, payload_stats = _encode_cuda_float_payload(tensor, codec_mode, options, stats, layer_name)
    payload_assembly_ms = (time.perf_counter() - t_payload0) * 1000.0
    t_commit = time.perf_counter()
    state.prev_grad[key] = tensor.detach().clone()
    state.prediction_memory[key] = torch.zeros_like(tensor)
    _sync()
    state.steps[key] = step
    commit_ms = (time.perf_counter() - t_commit) * 1000.0
    stats["state_commit_ms"] += commit_ms
    stats["payload_d2h_ms"] += payload_stats["payload_d2h_ms"]
    stats["payload_assembly_ms"] += payload_assembly_ms
    stats["cpu_payload_assembly_ms"] += payload_stats["cpu_payload_assembly_ms"]
    stats["cpu_reconstruct_input_ms"] += payload_stats["cpu_reconstruct_input_ms"]
    stats["cpu_encoder_ms"] += payload_stats["cpu_encoder_ms"]
    stats["full_residual_bytes"] += payload_stats["full_residual_bytes"]
    stats["compact_intermediate_d2h_bytes"] += payload_stats["compact_intermediate_d2h_bytes"]
    stats["final_payload_bytes"] += len(data)
    stats["D2H_payload_mb"] += len(data) / (1024.0 * 1024.0)
    stats["full_gradient_D2H_avoided_mb"] += max(
        0.0,
        (payload_stats["full_residual_bytes"] - payload_stats["compact_intermediate_d2h_bytes"])
        / (1024.0 * 1024.0),
    )
    layer = {
        "codec": "c_struct",
        "c_type": "direct",
        "c_codec": c_codec,
        "data": data,
        "bitmap": b"",
        "dominant_signs": b"",
        "shape": shape,
        "ndim": len(shape),
        "original_dtype": "float32",
        "stored_dtype": "float32",
        "step": step,
        "num_predicted_kernels": 0,
        "prediction_ratio": 0.0,
        "sign_mismatch_ratio": 0.0,
        "current_mean": 0.0,
        "current_std": 0.0,
        "prev_mean": 0.0,
        "prev_std": 0.0,
        "global_min": 0.0,
        "global_max": 0.0,
        "breakdown_stats_time": 0.0,
        "breakdown_normalize_time": 0.0,
        "breakdown_consistency_time": 0.0,
        "breakdown_prediction_time": 0.0,
        "breakdown_residual_compress_time": payload_stats["payload_d2h_ms"] / 1000.0,
        "breakdown_bitmap_compress_time": 0.0,
        "breakdown_metadata_time": 0.0,
        "breakdown_total_time": payload_assembly_ms / 1000.0,
    }
    _record_layer(
        stats,
        layer_name=layer_name,
        key=key,
        tensor=tensor,
        backend="cuda",
        codec_mode=codec_mode,
        c_type="direct",
        c_codec=c_codec,
        payload_bytes=_payload_bytes(layer),
        full_residual_bytes=int(payload_stats["full_residual_bytes"]),
        compact_intermediate_d2h_bytes=int(payload_stats["compact_intermediate_d2h_bytes"]),
        state_committed=True,
        cuda_payload_d2h_ms=payload_stats["payload_d2h_ms"],
        cpu_payload_assembly_ms=payload_stats["cpu_payload_assembly_ms"],
        cpu_reconstruct_input_ms=payload_stats["cpu_reconstruct_input_ms"],
        cpu_encoder_ms=payload_stats["cpu_encoder_ms"],
        cuda_state_commit_ms=commit_ms,
    )
    return layer


def _cuda_momentum_layer(
    state: FalcomCudaState,
    key: str,
    layer_name: str,
    tensor: torch.Tensor,
    step: int,
    options: Mapping[str, Any],
    stats: MutableMapping[str, float],
) -> Dict[str, Any]:
    if _falcom_cuda_v0 is None:
        raise RuntimeError(f"_falcom_cuda_v0 extension is not built: {_IMPORT_ERROR}")
    prev = state.prev_grad[key]
    predmem = state.prediction_memory[key]
    if tuple(prev.shape) != tuple(tensor.shape) or tuple(predmem.shape) != tuple(tensor.shape):
        raise RuntimeError(f"layout mismatch for CUDA key {key}")

    t_stats0 = time.perf_counter()
    abs_cur = tensor.abs()
    abs_prev = prev.abs()
    current_mean = float(abs_cur.mean().item())
    current_std = float(abs_cur.std(unbiased=False).item())
    prev_mean = float(abs_prev.mean().item())
    prev_std = float(abs_prev.std(unbiased=False).item())
    global_min = float(tensor.min().item())
    global_max = float(tensor.max().item())
    _sync()
    stats["state_lookup_ms"] += (time.perf_counter() - t_stats0) * 1000.0

    momentum_lr = float(options.get("momentum_lr", 0.07))
    consistency_threshold = float(options.get("consistency_threshold", 0.5))
    codec_mode = _codec_mode(options)

    _sync()
    t_kernel0 = time.perf_counter()
    residual, next_predmem, flags, signs, counters, kernel_ms_tensor = _falcom_cuda_v0.momentum_pack_cuda(
        tensor,
        prev,
        predmem,
        current_mean,
        current_std,
        prev_mean,
        prev_std,
        global_min,
        global_max,
        momentum_lr,
        consistency_threshold,
        int(step),
    )
    _sync()
    stats["cuda_wall_ms"] += (time.perf_counter() - t_kernel0) * 1000.0
    kernel_ms = float(kernel_ms_tensor.item())
    stats["cuda_kernel_ms"] += kernel_ms
    stats["kernel_launch_count"] += 1

    try:
        t_payload0 = time.perf_counter()
        residual_bytes, c_codec, payload_stats = _encode_cuda_float_payload(
            residual,
            codec_mode,
            options,
            stats,
            layer_name,
        )
        _sync()
        t_d2h0 = time.perf_counter()
        flags_np = flags.detach().cpu().numpy()
        signs_np = signs.detach().cpu().numpy()
        counters_np = counters.detach().cpu().numpy()
        _sync()
        metadata_d2h_ms = (time.perf_counter() - t_d2h0) * 1000.0

        bitmap_raw = _pack_bitmap(flags_np)
        signs_raw = _pack_dense_signs(flags_np, signs_np)
        bitmap_zstd = zstd_compress(bitmap_raw)
        signs_zstd = zstd_compress(signs_raw) if signs_raw else b""
        payload_assembly_ms = (time.perf_counter() - t_payload0) * 1000.0

        predicted_count = int(counters_np[0])
        sign_mismatch_count = int(counters_np[1])
        predicted_elements = int(counters_np[2])
        total_kernels = int(tensor.shape[0] * tensor.shape[1])
        prediction_ratio = float(predicted_count) / float(total_kernels) if total_kernels else 0.0
        sign_mismatch_ratio = (
            float(sign_mismatch_count) / float(predicted_elements) if predicted_elements else 0.0
        )

        layer = {
            "codec": "c_struct",
            "c_type": "momentum_predicted",
            "c_codec": c_codec,
            "data": residual_bytes,
            "bitmap": bitmap_zstd,
            "dominant_signs": signs_zstd,
            "shape": _shape_tuple(tensor),
            "ndim": int(tensor.dim()),
            "original_dtype": "float32",
            "stored_dtype": "float32",
            "step": int(step),
            "num_predicted_kernels": predicted_count,
            "prediction_ratio": prediction_ratio,
            "sign_mismatch_ratio": sign_mismatch_ratio,
            "current_mean": current_mean,
            "current_std": current_std,
            "prev_mean": prev_mean,
            "prev_std": prev_std,
            "global_min": global_min,
            "global_max": global_max,
            "breakdown_stats_time": 0.0,
            "breakdown_normalize_time": 0.0,
            "breakdown_consistency_time": kernel_ms / 1000.0,
            "breakdown_prediction_time": kernel_ms / 1000.0,
            "breakdown_residual_compress_time": payload_stats["payload_d2h_ms"] / 1000.0,
            "breakdown_bitmap_compress_time": 0.0,
            "breakdown_metadata_time": 0.0,
            "breakdown_total_time": (kernel_ms + payload_assembly_ms) / 1000.0,
        }
    except Exception:
        raise

    t_commit = time.perf_counter()
    state.prev_grad[key] = tensor.detach().clone()
    state.prediction_memory[key] = next_predmem.detach()
    _sync()
    state.steps[key] = step
    commit_ms = (time.perf_counter() - t_commit) * 1000.0
    stats["state_commit_ms"] += commit_ms
    stats["payload_d2h_ms"] += payload_stats["payload_d2h_ms"] + metadata_d2h_ms
    stats["payload_assembly_ms"] += payload_assembly_ms
    stats["cpu_payload_assembly_ms"] += payload_stats["cpu_payload_assembly_ms"]
    stats["cpu_reconstruct_input_ms"] += payload_stats["cpu_reconstruct_input_ms"]
    stats["cpu_encoder_ms"] += payload_stats["cpu_encoder_ms"]
    stats["full_residual_bytes"] += payload_stats["full_residual_bytes"]
    stats["compact_intermediate_d2h_bytes"] += payload_stats["compact_intermediate_d2h_bytes"]
    stats["final_payload_bytes"] += _payload_bytes(layer)
    stats["D2H_payload_mb"] += _payload_bytes(layer) / (1024.0 * 1024.0)
    stats["full_gradient_D2H_avoided_mb"] += max(
        0.0,
        (payload_stats["full_residual_bytes"] - payload_stats["compact_intermediate_d2h_bytes"])
        / (1024.0 * 1024.0),
    )
    _record_layer(
        stats,
        layer_name=layer_name,
        key=key,
        tensor=tensor,
        backend="cuda",
        codec_mode=codec_mode,
        c_type="momentum_predicted",
        c_codec=c_codec,
        payload_bytes=_payload_bytes(layer),
        full_residual_bytes=int(payload_stats["full_residual_bytes"]),
        compact_intermediate_d2h_bytes=int(payload_stats["compact_intermediate_d2h_bytes"]),
        state_committed=True,
        cuda_kernel_ms=kernel_ms,
        cuda_payload_d2h_ms=payload_stats["payload_d2h_ms"] + metadata_d2h_ms,
        cpu_payload_assembly_ms=payload_stats["cpu_payload_assembly_ms"],
        cpu_reconstruct_input_ms=payload_stats["cpu_reconstruct_input_ms"],
        cpu_encoder_ms=payload_stats["cpu_encoder_ms"],
        cuda_state_commit_ms=commit_ms,
    )
    return layer


def compress_batch_cuda(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    state_handle: Optional[FalcomCudaState] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, float]]:
    options = dict(options or {})
    state = state_handle or FalcomCudaState(client_id=str(options.get("client_id", "Client1")))
    names, tensors = _normalize_inputs(torch_tensors, layer_names)
    min_numel = int(options.get("cuda_min_numel", 524288))
    codec_mode = _codec_mode(options)

    stats: Dict[str, Any] = {
        "cuda_layers": 0.0,
        "cpu_fallback_layers": 0.0,
        "cuda_original_mb": 0.0,
        "cpu_original_mb": 0.0,
        "D2H_payload_mb": 0.0,
        "full_gradient_D2H_avoided_mb": 0.0,
        "kernel_launch_count": 0.0,
        "cuda_kernel_ms": 0.0,
        "cuda_wall_ms": 0.0,
        "state_lookup_ms": 0.0,
        "payload_d2h_ms": 0.0,
        "payload_assembly_ms": 0.0,
        "cpu_payload_assembly_ms": 0.0,
        "cpu_reconstruct_input_ms": 0.0,
        "cpu_encoder_ms": 0.0,
        "state_commit_ms": 0.0,
        "cuda_state_commit_ms": 0.0,
        "cpu_fallback_ms": 0.0,
        "full_residual_bytes": 0.0,
        "compact_intermediate_d2h_bytes": 0.0,
        "compact_vs_full_residual_pct": 0.0,
        "final_payload_bytes": 0.0,
        "compact_intermediate_d2h_mb": 0.0,
        "full_residual_d2h_avoided_mb": 0.0,
        "codec_mode": codec_mode,
        "codec_failure_reason": "",
        "state_committed": 0.0,
        "accepted_compact": 0.0,
        "layer_records": [],
    }

    out = OrderedDict()
    use_cuda = cuda_enabled()

    for layer_name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        key = layer_key(layer_name, tensor)
        candidate = use_cuda and _is_cuda_candidate(tensor, min_numel)
        assigned = state.backend_by_key.get(key)
        if assigned is None:
            assigned = "cuda" if candidate else "cpu"
            state.backend_by_key[key] = assigned
        elif assigned == "cuda" and not candidate:
            raise RuntimeError(f"CUDA key {key} became ineligible; v0 does not migrate state")

        if assigned == "cpu":
            out[layer_name] = _cpu_fallback_layer(state, layer_name, tensor, stats)
            continue

        stats["cuda_layers"] += 1
        stats["cuda_original_mb"] += float(tensor.numel() * tensor.element_size()) / (1024.0 * 1024.0)
        step = int(state.steps.get(key, 0)) + 1
        if step == 1:
            out[layer_name] = _direct_cuda_layer(state, key, layer_name, tensor, step, options, stats)
        else:
            out[layer_name] = _cuda_momentum_layer(state, key, layer_name, tensor, step, options, stats)

    total_original_mb = stats["cuda_original_mb"] + stats["cpu_original_mb"]
    stats["original_mb"] = total_original_mb
    if stats["full_residual_bytes"] > 0:
        stats["compact_vs_full_residual_pct"] = (
            100.0 * float(stats["compact_intermediate_d2h_bytes"]) / float(stats["full_residual_bytes"])
        )
    stats["compact_intermediate_d2h_mb"] = float(stats["compact_intermediate_d2h_bytes"]) / (1024.0 * 1024.0)
    stats["full_residual_d2h_avoided_mb"] = max(
        0.0,
        (float(stats["full_residual_bytes"]) - float(stats["compact_intermediate_d2h_bytes"]))
        / (1024.0 * 1024.0),
    )
    stats["cuda_state_commit_ms"] = stats["state_commit_ms"]
    stats["state_committed"] = 1.0 if stats["cuda_layers"] > 0 and not stats["codec_failure_reason"] else 0.0
    stats["accepted_compact"] = (
        1.0
        if codec_mode == RESIDUAL_CODEC_HYBRID
        and stats["full_residual_bytes"] > 0
        and stats["compact_vs_full_residual_pct"] <= 35.0
        else 0.0
    )
    if options.get("measure_serialized_size", False):
        stats["compressed_bytes"] = float(len(pickle.dumps(out)))
        stats["compression_ratio"] = (
            (total_original_mb * 1024.0 * 1024.0) / stats["compressed_bytes"]
            if stats["compressed_bytes"] > 0
            else 0.0
        )
    else:
        stats["compressed_bytes"] = 0.0
        stats["compression_ratio"] = 0.0
    return out, stats


def dumps_compressed_layers(layers: Mapping[str, Dict[str, Any]]) -> bytes:
    return pickle.dumps(OrderedDict(layers))
