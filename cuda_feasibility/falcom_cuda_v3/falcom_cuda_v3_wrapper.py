#!/usr/bin/env python3
"""FalCom CUDA v3 experimental closed-loop compressor.

This module is intentionally separate from the installed CPU compressor.  It
only activates when callers import it directly and set FALCOM_CUDA_EXPERIMENTAL.
The payload format is experimental and decoded only by this module.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
APPFL_SRC = PROJECT_ROOT / "EB-FaLCom" / "src"
if str(APPFL_SRC) not in sys.path:
    sys.path.insert(0, str(APPFL_SRC))

from appfl.compressor.FalComC import FalComC  # noqa: E402


CODEC_NAME = "cuda_v3_experimental"
CODEC_VERSION = 1
ENVELOPE_CODEC = "cuda_v3_batch"
ENVELOPE_VERSION = 1
CPU_PAYLOAD_WRAPPER = "cpu_fallback_pickle"
CPU_BATCH_CODEC = "cpu_fallback_batch_pickle"
CPU_BATCH_KEY = "__cpu_fallback_batch__"
SUPPORTED_BITS = (4, 6, 7, 8)


def cuda_v3_enabled() -> bool:
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


def _shape_tuple(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _shape_string(tensor: torch.Tensor) -> str:
    return "x".join(str(x) for x in _shape_tuple(tensor))


def layer_key(layer_name: str, tensor: torch.Tensor) -> str:
    return f"{layer_name}|{str(tensor.dtype)}|{_shape_string(tensor)}"


def _dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float32:
        return "float32"
    return str(tensor.dtype)


def _normalize_inputs(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]],
) -> Tuple[list[str], list[torch.Tensor]]:
    if isinstance(torch_tensors, Mapping):
        return [str(k) for k in torch_tensors.keys()], list(torch_tensors.values())
    if layer_names is None:
        raise ValueError("layer_names is required when torch_tensors is not a mapping")
    if len(layer_names) != len(torch_tensors):
        raise ValueError("layer_names length does not match tensor count")
    return [str(x) for x in layer_names], list(torch_tensors)


def _is_cuda_candidate(tensor: torch.Tensor, min_numel: int) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and cuda_v3_enabled()
        and tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        and int(tensor.numel()) >= int(min_numel)
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _quant_mode(options: Mapping[str, Any]) -> str:
    return str(options.get("quant_mode", options.get("quant_bits", 8))).lower()


def _resolve_bits(mode: str, max_abs: float, step: int) -> Tuple[int, str]:
    if mode in ("4", "int4", "q4"):
        return 4, "symmetric_int4"
    if mode in ("6", "int6", "q6"):
        return 6, "symmetric_int6"
    if mode in ("7", "int7", "q7"):
        return 7, "symmetric_int7"
    if mode in ("8", "int8", "q8"):
        return 8, "symmetric_int8"
    if mode == "adaptive_q6_q8":
        # Preserve first-round payload quality, then test cheaper residual precision on hot rounds.
        return (6 if step > 1 and max_abs < 0.25 else 8), mode
    if mode == "outlier_q6_q8":
        # Outlier-heavy layers stay q8; compact low-range residuals are allowed to try q6.
        return (6 if step > 1 and max_abs < 0.10 else 8), mode
    raise ValueError(f"Unsupported cuda_v3 quant_mode={mode}")


def _quantize_tensor(tensor: torch.Tensor, mode: str, step: int, stats: MutableMapping[str, Any]) -> Tuple[torch.Tensor, float, int, str, float]:
    t0 = time.perf_counter()
    max_abs = float(tensor.detach().abs().max().item())
    bits, quantization_mode = _resolve_bits(mode, max_abs, step)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"cuda_v3 quant_bits must be one of {SUPPORTED_BITS}")
    max_q = float((1 << (bits - 1)) - 1)
    if max_abs == 0.0:
        scale = 1.0
        q = torch.zeros_like(tensor, dtype=torch.int8)
    else:
        scale = max_abs / max_q
        q = torch.clamp(torch.round(tensor / scale), -max_q, max_q).to(torch.int8)
    _sync()
    stats["kernel_launch_count"] += 5
    return q.contiguous(), float(scale), int(bits), quantization_mode, (time.perf_counter() - t0) * 1000.0


def _fault(options: Mapping[str, Any], stage: str, layer_name: str) -> None:
    requested = str(options.get("fault_inject_stage", ""))
    requested_layer = options.get("fault_inject_layer")
    if requested == stage and (requested_layer is None or str(requested_layer) == layer_name):
        raise RuntimeError(f"Injected CUDA v3 transaction failure at {stage} for {layer_name}")


def _pack_int4(q: torch.Tensor) -> bytes:
    # Values are expected in [-7, 7]. Store sign/magnitude biased into 0..15.
    arr = q.detach().cpu().numpy().astype(np.int8, copy=False).reshape(-1)
    biased = np.clip(arr + 8, 0, 15).astype(np.uint8, copy=False)
    if biased.size % 2:
        biased = np.concatenate([biased, np.zeros(1, dtype=np.uint8)])
    packed = (biased[0::2] | (biased[1::2] << 4)).astype(np.uint8, copy=False)
    return packed.tobytes()


def _unpack_int4(payload: bytes, numel: int, device: torch.device) -> torch.Tensor:
    packed = np.frombuffer(payload, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    biased = np.empty(int(packed.size) * 2, dtype=np.uint8)
    biased[0::2] = low
    biased[1::2] = high
    signed = biased[:numel].astype(np.int16) - 8
    return torch.from_numpy(signed.astype(np.int8, copy=False)).to(device=device)


def _pack_int_bits(q: torch.Tensor, bits: int) -> bytes:
    if bits not in (6, 7):
        raise ValueError(f"Unsupported packed bit width: {bits}")
    arr = q.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
    biased = np.clip(arr + (1 << (bits - 1)), 0, (1 << bits) - 1).astype(np.uint8, copy=False)
    bit_planes = ((biased[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8, copy=False)
    return np.packbits(bit_planes.reshape(-1), bitorder="little").tobytes()


def _unpack_int_bits(payload: bytes, numel: int, bits: int, device: torch.device) -> torch.Tensor:
    if bits not in (6, 7):
        raise ValueError(f"Unsupported packed bit width: {bits}")
    packed = np.frombuffer(payload, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")[: int(numel) * bits].reshape(-1, bits)
    weights = (1 << np.arange(bits, dtype=np.uint16)).reshape(1, bits)
    biased = (unpacked.astype(np.uint16, copy=False) * weights).sum(axis=1).astype(np.int16, copy=False)
    signed = biased - (1 << (bits - 1))
    return torch.from_numpy(signed.astype(np.int8, copy=False)).to(device=device)


def _int8_payload_bytes(q: torch.Tensor, state: "FalcomCudaV3State", use_pinned: bool) -> bytes:
    flat = q.detach().reshape(-1)
    if use_pinned and flat.is_cuda and torch.cuda.is_available():
        numel = int(flat.numel())
        host = state.pinned_int8_buffers.get(numel)
        if host is None:
            try:
                host = torch.empty(numel, dtype=torch.int8, device="cpu", pin_memory=True)
                state.pinned_int8_buffers[numel] = host
            except RuntimeError:
                host = None
        if host is not None:
            host.copy_(flat, non_blocking=True)
            _sync()
            return host.numpy().tobytes()
    return flat.cpu().numpy().astype(np.int8, copy=False).tobytes()


@dataclass
class FalcomCudaV3State:
    client_id: str = "Client1"
    cpu_compressor: Optional[FalComC] = None
    config: Any = None
    backend_by_key: Dict[str, str] = field(default_factory=dict)
    prev_grad: Dict[str, torch.Tensor] = field(default_factory=dict)
    prediction_memory: Dict[str, torch.Tensor] = field(default_factory=dict)
    steps: Dict[str, int] = field(default_factory=dict)
    pinned_int8_buffers: Dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = default_config()
        if self.cpu_compressor is None:
            self.cpu_compressor = FalComC(self.config)
        self.cpu_compressor.set_client_id(self.client_id)


def _empty_stats() -> Dict[str, Any]:
    return {
        "cuda_layers": 0,
        "cpu_fallback_layers": 0,
        "cuda_original_mb": 0.0,
        "fallback_original_mb": 0.0,
        "experimental_payload_d2h_mb": 0.0,
        "fallback_gpu_to_cpu_ms": 0.0,
        "cpu_fallback_ms": 0.0,
        "cpu_fallback_payload_bytes": 0,
        "cpu_fallback_batches": 0,
        "payload_d2h_ms": 0.0,
        "payload_serialize_ms": 0.0,
        "payload_deserialize_ms": 0.0,
        "kernel_ms": 0.0,
        "kernel_launch_count": 0,
        "python_wrapper_ms": 0.0,
        "state_commit_ms": 0.0,
        "compressed_bytes": 0,
        "compression_ratio": 0.0,
        "state_committed": 0,
        "layer_records": [],
        "fallback_records": [],
    }


def _cpu_fallback_batch(
    state: FalcomCudaV3State,
    items: Sequence[Tuple[str, torch.Tensor, str]],
    stats: MutableMapping[str, Any],
) -> Dict[str, Any]:
    assert state.cpu_compressor is not None
    arrays = OrderedDict()
    layer_meta = []
    any_cuda = any(isinstance(t, torch.Tensor) and t.is_cuda for _name, t, _key in items)
    if any_cuda:
        _sync()
    t_copy = time.perf_counter()
    for layer_name, tensor, key in items:
        if tensor.is_cuda:
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        arrays[layer_name] = arr
        stats["fallback_original_mb"] += float(arr.nbytes) / (1024.0 * 1024.0)
        layer_meta.append(
            {
                "layer_name": layer_name,
                "layer_key": key,
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "numel": int(arr.size),
                "nbytes": int(arr.nbytes),
                "reason": "below_cuda_min_numel_or_gate",
            }
        )
    if any_cuda:
        _sync()
    copy_ms = (time.perf_counter() - t_copy) * 1000.0
    t_cpu = time.perf_counter()
    payload = state.cpu_compressor.compress_model(arrays, client_id=state.client_id)
    cpu_ms = (time.perf_counter() - t_cpu) * 1000.0
    stats["cpu_fallback_layers"] += len(items)
    stats["cpu_fallback_batches"] += 1
    stats["fallback_gpu_to_cpu_ms"] += copy_ms
    stats["cpu_fallback_ms"] += cpu_ms
    stats["cpu_fallback_payload_bytes"] += len(payload)
    stats["fallback_records"].extend(layer_meta)
    return {
        "codec": CPU_BATCH_CODEC,
        "codec_version": CODEC_VERSION,
        "layers": layer_meta,
        "payload": payload,
        "payload_length": len(payload),
    }


def _make_payload(
    *,
    state: FalcomCudaV3State,
    layer_name: str,
    key: str,
    tensor: torch.Tensor,
    step: int,
    residual: torch.Tensor,
    base_tensor: Optional[torch.Tensor],
    base_kind: str,
    quant_mode: str,
    state_candidate: torch.Tensor,
    pred_candidate: torch.Tensor,
    stats: MutableMapping[str, Any],
    options: Mapping[str, Any],
) -> Dict[str, Any]:
    q, scale, bits, quantization_mode, kernel_ms = _quantize_tensor(residual, quant_mode, step, stats)
    _fault(options, "after_encode", layer_name)
    stats["kernel_ms"] += kernel_ms
    _sync()
    t_d2h = time.perf_counter()
    if bits == 4:
        payload_bytes = _pack_int4(q)
    elif bits in (6, 7):
        payload_bytes = _pack_int_bits(q, bits)
    else:
        payload_bytes = _int8_payload_bytes(q, state, bool(options.get("use_pinned_d2h", True)))
    _sync()
    d2h_ms = (time.perf_counter() - t_d2h) * 1000.0
    _fault(options, "after_payload_d2h", layer_name)
    stats["payload_d2h_ms"] += d2h_ms
    stats["experimental_payload_d2h_mb"] += len(payload_bytes) / (1024.0 * 1024.0)

    payload = {
        "codec": CODEC_NAME,
        "codec_version": CODEC_VERSION,
        "layer_name": layer_name,
        "layer_key": key,
        "shape": _shape_tuple(tensor),
        "dtype": _dtype_name(tensor),
        "step": int(step),
        "numel": int(tensor.numel()),
        "payload_format": f"int{bits}_symmetric_residual",
        "quantization_mode": quantization_mode,
        "requested_quant_mode": quant_mode,
        "quant_bits": int(bits),
        "scale": float(scale),
        "base_kind": base_kind,
        "momentum_lr": float(options.get("momentum_lr", 0.07)),
        "payload": payload_bytes,
        "payload_length": len(payload_bytes),
    }

    # Optional decode check before committing state. Benchmarks do the full
    # closed-loop decode once per model, so the hot path avoids a per-layer
    # decode replay unless explicitly requested.
    _fault(options, "decode_check", layer_name)
    if bool(options.get("decode_check", False)):
        decoded = _decode_cuda_payload(payload, state_candidate.device, base_tensor)
        if tuple(decoded.shape) != tuple(tensor.shape) or not torch.isfinite(decoded).all().item():
            raise RuntimeError(f"CUDA v3 decode check failed for {layer_name}")
    return payload


def _decode_cuda_payload(
    payload: Mapping[str, Any],
    device: torch.device | str,
    base_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    if payload.get("codec") != CODEC_NAME:
        raise ValueError(f"Unsupported CUDA v3 payload codec: {payload.get('codec')}")
    if int(payload.get("codec_version", -1)) != CODEC_VERSION:
        raise ValueError(f"Unsupported CUDA v3 payload version: {payload.get('codec_version')}")
    shape = tuple(int(x) for x in payload["shape"])
    numel = int(payload["numel"])
    bits = int(payload.get("quant_bits", 8))
    raw = payload["payload"]
    if bits == 8:
        q = torch.from_numpy(np.frombuffer(raw, dtype=np.int8).copy()).to(device=device)
    elif bits in (6, 7):
        q = _unpack_int_bits(raw, numel, bits, torch.device(device))
    elif bits == 4:
        q = _unpack_int4(raw, numel, torch.device(device))
    else:
        raise ValueError(f"Unsupported CUDA v3 quant_bits={bits}")
    q = q.reshape(shape).to(dtype=torch.float32)
    residual = q * float(payload["scale"])
    base_kind = str(payload.get("base_kind", "zero"))
    if base_kind == "zero":
        return residual.contiguous()
    if base_kind == "prediction":
        if base_tensor is None:
            raise ValueError("CUDA v3 prediction payload requires decoder state")
        return (base_tensor + residual).contiguous()
    raise ValueError(f"Unsupported CUDA v3 base_kind={base_kind}")


def compress_batch_cuda_v3(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    state_handle: Optional[FalcomCudaV3State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    wrapper_start = time.perf_counter()
    options = dict(options or {})
    state = state_handle or FalcomCudaV3State(client_id=str(options.get("client_id", "Client1")))
    names, tensors = _normalize_inputs(torch_tensors, layer_names)
    min_numel = int(options.get("cuda_min_numel", 524288))
    quant_mode = _quant_mode(options)
    stats = _empty_stats()
    out = OrderedDict()
    pending = []
    cpu_items: List[Tuple[str, torch.Tensor, str]] = []
    staged_backends = dict(state.backend_by_key)
    staged_prev = {k: v for k, v in state.prev_grad.items()}
    staged_pred = {k: v for k, v in state.prediction_memory.items()}
    staged_steps = dict(state.steps)

    for layer_name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        key = layer_key(layer_name, tensor)
        candidate = _is_cuda_candidate(tensor, min_numel)
        assigned = staged_backends.get(key)
        if assigned is None:
            assigned = "cuda" if candidate else "cpu"
            staged_backends[key] = assigned
        elif assigned == "cuda" and not candidate:
            raise RuntimeError(f"CUDA v3 key {key} became ineligible; state migration is not implemented")
        if assigned == "cpu":
            pending.append(("cpu", layer_name, tensor, key, None, None, None))
            continue

        stats["cuda_layers"] += 1
        stats["cuda_original_mb"] += _tensor_bytes(tensor) / (1024.0 * 1024.0)
        step = int(staged_steps.get(key, 0)) + 1
        if step == 1:
            base = None
            residual = tensor
            base_kind = "zero"
        else:
            prev = staged_prev[key]
            pred = staged_pred[key]
            if tuple(prev.shape) != tuple(tensor.shape) or tuple(pred.shape) != tuple(tensor.shape):
                raise RuntimeError(f"CUDA v3 layout mismatch for {key}")
            base = pred
            residual = tensor - pred
            base_kind = "prediction"
        next_pred = tensor.detach().clone()
        payload = _make_payload(
            state=state,
            layer_name=layer_name,
            key=key,
            tensor=tensor,
            step=step,
            residual=residual,
            base_tensor=base,
            base_kind=base_kind,
            quant_mode=quant_mode,
            state_candidate=tensor,
            pred_candidate=next_pred,
            stats=stats,
            options=options,
        )
        staged_prev[key] = next_pred
        staged_pred[key] = next_pred
        staged_steps[key] = step
        pending.append(("cuda", layer_name, tensor, key, payload, step, int(payload["quant_bits"])))

    for backend, layer_name, tensor, key, payload, step, qbits in pending:
        if backend == "cpu":
            cpu_items.append((layer_name, tensor, key))
            continue
        assert payload is not None
        out[layer_name] = payload
        stats["layer_records"].append(
            {
                "layer_name": layer_name,
                "layer_key": key,
                "backend": "cuda",
                "step": step,
                "shape": _shape_string(tensor),
                "numel": int(tensor.numel()),
                "payload_length": int(payload["payload_length"]),
                "quant_bits": qbits,
                "quant_mode": payload.get("quantization_mode", ""),
                "fallback_reason": "",
            }
        )
    if cpu_items:
        out[CPU_BATCH_KEY] = _cpu_fallback_batch(state, cpu_items, stats)
        for record in stats["fallback_records"]:
            stats["layer_records"].append(
                {
                    "layer_name": record["layer_name"],
                    "layer_key": record["layer_key"],
                    "backend": "cpu_batch",
                    "step": "",
                    "shape": "x".join(str(x) for x in record["shape"]),
                    "numel": int(record["numel"]),
                    "payload_length": 0,
                    "quant_bits": "",
                    "quant_mode": "",
                    "fallback_reason": record["reason"],
                }
            )

    t_commit = time.perf_counter()
    state.backend_by_key = staged_backends
    state.prev_grad = staged_prev
    state.prediction_memory = staged_pred
    state.steps = staged_steps
    stats["state_commit_ms"] += (time.perf_counter() - t_commit) * 1000.0
    stats["state_committed"] = stats["cuda_layers"]
    total_ms = (time.perf_counter() - wrapper_start) * 1000.0
    accounted_ms = (
        float(stats["kernel_ms"])
        + float(stats["payload_d2h_ms"])
        + float(stats["fallback_gpu_to_cpu_ms"])
        + float(stats["cpu_fallback_ms"])
        + float(stats["state_commit_ms"])
    )
    stats["python_wrapper_ms"] = max(0.0, total_ms - accounted_ms)
    return out, stats


def decompress_batch_cuda_v3(
    compressed_layers: Mapping[str, Mapping[str, Any]],
    state_handle: Optional[FalcomCudaV3State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    options = dict(options or {})
    state = state_handle or FalcomCudaV3State(client_id=str(options.get("client_id", "Client1")))
    out = OrderedDict()
    stats: Dict[str, Any] = {
        "decompress_total_ms": 0.0,
        "payload_deserialize_ms": 0.0,
        "cpu_fallback_decompress_ms": 0.0,
        "kernel_launch_count": 0,
        "decoded_layers": 0,
        "cpu_fallback_decoded_layers": 0,
    }
    t_total = time.perf_counter()
    staged_prev = {k: v for k, v in state.prev_grad.items()}
    staged_pred = {k: v for k, v in state.prediction_memory.items()}
    staged_steps = dict(state.steps)
    pending = []
    for layer_name, payload in compressed_layers.items():
        if layer_name == CPU_BATCH_KEY and payload.get("codec") == CPU_BATCH_CODEC:
            if int(payload.get("codec_version", -1)) != CODEC_VERSION:
                raise ValueError(f"Unsupported CPU fallback batch payload version: {payload.get('codec_version')}")
            pending.append(("cpu_batch", layer_name, payload, None, None))
            stats["cpu_fallback_decoded_layers"] += len(payload.get("layers", ()))
            continue
        if payload.get("codec") == CPU_PAYLOAD_WRAPPER:
            if int(payload.get("codec_version", -1)) != CODEC_VERSION:
                raise ValueError(f"Unsupported CPU fallback payload version: {payload.get('codec_version')}")
            pending.append(("cpu_payload", layer_name, payload, None, None))
            stats["cpu_fallback_decoded_layers"] += 1
            continue
        key = str(payload["layer_key"])
        step = int(payload["step"])
        if step == 1:
            base = None
        else:
            base = staged_pred.get(key)
            if base is None:
                raise RuntimeError(f"CUDA v3 decoder missing state for {key}")
        t0 = time.perf_counter()
        decoded = _decode_cuda_payload(payload, "cuda" if torch.cuda.is_available() else "cpu", base)
        _sync()
        stats["payload_deserialize_ms"] += (time.perf_counter() - t0) * 1000.0
        stats["kernel_launch_count"] += 2
        staged_prev[key] = decoded.detach().clone()
        staged_pred[key] = decoded.detach().clone()
        staged_steps[key] = step
        pending.append(("cuda", layer_name, decoded, key, step))
        stats["decoded_layers"] += 1

    decoded_pending = []
    for backend, layer_name, decoded, _key, _step in pending:
        if backend == "cpu_batch":
            assert state.cpu_compressor is not None
            t0 = time.perf_counter()
            decomp = state.cpu_compressor.decompress_model(decoded["payload"], client_id=state.client_id)
            for fallback_name, fallback_tensor in decomp.items():
                out_tensor = fallback_tensor.cuda() if torch.cuda.is_available() else fallback_tensor
                decoded_pending.append((fallback_name, out_tensor))
            stats["cpu_fallback_decompress_ms"] += (time.perf_counter() - t0) * 1000.0
            continue
        if backend == "cpu_payload":
            assert state.cpu_compressor is not None
            t0 = time.perf_counter()
            decomp = state.cpu_compressor.decompress_model(decoded["payload"], client_id=state.client_id)
            decoded = decomp[layer_name].cuda() if torch.cuda.is_available() else decomp[layer_name]
            stats["cpu_fallback_decompress_ms"] += (time.perf_counter() - t0) * 1000.0
        decoded_pending.append((layer_name, decoded))
    for layer_name, decoded in decoded_pending:
        out[layer_name] = decoded
    state.prev_grad = staged_prev
    state.prediction_memory = staged_pred
    state.steps = staged_steps
    stats["decompress_total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return out, stats


def dumps_cuda_v3_layers(layers: Mapping[str, Mapping[str, Any]]) -> bytes:
    return pickle.dumps(
        {
            "codec": ENVELOPE_CODEC,
            "codec_version": ENVELOPE_VERSION,
            "layers": OrderedDict(layers),
        }
    )


def loads_cuda_v3_layers(payload: bytes) -> OrderedDict:
    obj = pickle.loads(payload)
    if isinstance(obj, OrderedDict):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("CUDA v3 payload envelope is not a mapping")
    if obj.get("codec") != ENVELOPE_CODEC:
        raise ValueError(f"Unsupported CUDA v3 envelope codec: {obj.get('codec')}")
    if int(obj.get("codec_version", -1)) != ENVELOPE_VERSION:
        raise ValueError(f"Unsupported CUDA v3 envelope version: {obj.get('codec_version')}")
    layers = obj.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("CUDA v3 envelope missing layers mapping")
    return OrderedDict(layers)
