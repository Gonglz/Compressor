#!/usr/bin/env python3
"""FalCom CUDA v2 experimental closed-loop compressor.

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
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
APPFL_SRC = PROJECT_ROOT / "EB-FaLCom" / "src"
if str(APPFL_SRC) not in sys.path:
    sys.path.insert(0, str(APPFL_SRC))

from appfl.compressor.FalComC import FalComC  # noqa: E402


CODEC_NAME = "cuda_v2_experimental"
CODEC_VERSION = 1
ENVELOPE_CODEC = "cuda_v2_batch"
ENVELOPE_VERSION = 1
CPU_PAYLOAD_WRAPPER = "cpu_fallback_pickle"


def cuda_v2_enabled() -> bool:
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
        and cuda_v2_enabled()
        and tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        and int(tensor.numel()) >= int(min_numel)
        and torch.isfinite(tensor).all().item()
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _quantize_tensor(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float, float]:
    if bits not in (4, 8):
        raise ValueError("cuda_v2 quant_bits must be 4 or 8")
    max_q = float((1 << (bits - 1)) - 1)
    _sync()
    t0 = time.perf_counter()
    max_abs = float(tensor.detach().abs().max().item())
    if max_abs == 0.0:
        scale = 1.0
        q = torch.zeros_like(tensor, dtype=torch.int8)
    else:
        scale = max_abs / max_q
        q = torch.clamp(torch.round(tensor / scale), -max_q, max_q).to(torch.int8)
    _sync()
    return q.contiguous(), float(scale), (time.perf_counter() - t0) * 1000.0


def _fault(options: Mapping[str, Any], stage: str, layer_name: str) -> None:
    requested = str(options.get("fault_inject_stage", ""))
    requested_layer = options.get("fault_inject_layer")
    if requested == stage and (requested_layer is None or str(requested_layer) == layer_name):
        raise RuntimeError(f"Injected CUDA v2 transaction failure at {stage} for {layer_name}")


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


@dataclass
class FalcomCudaV2State:
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


def _empty_stats() -> Dict[str, Any]:
    return {
        "cuda_layers": 0,
        "cpu_fallback_layers": 0,
        "cuda_original_mb": 0.0,
        "fallback_original_mb": 0.0,
        "experimental_payload_d2h_mb": 0.0,
        "fallback_gpu_to_cpu_ms": 0.0,
        "cpu_fallback_ms": 0.0,
        "payload_d2h_ms": 0.0,
        "payload_serialize_ms": 0.0,
        "payload_deserialize_ms": 0.0,
        "kernel_ms": 0.0,
        "state_commit_ms": 0.0,
        "compressed_bytes": 0,
        "compression_ratio": 0.0,
        "state_committed": 0,
        "layer_records": [],
    }


def _cpu_fallback_layer(
    state: FalcomCudaV2State,
    layer_name: str,
    tensor: torch.Tensor,
    stats: MutableMapping[str, Any],
) -> Dict[str, Any]:
    assert state.cpu_compressor is not None
    copy_ms = 0.0
    if tensor.is_cuda:
        _sync()
        t_copy = time.perf_counter()
        arr = tensor.detach().cpu().numpy()
        _sync()
        copy_ms = (time.perf_counter() - t_copy) * 1000.0
    else:
        arr = tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    t_cpu = time.perf_counter()
    payload = state.cpu_compressor.compress_model(OrderedDict([(layer_name, arr)]), client_id=state.client_id)
    cpu_ms = (time.perf_counter() - t_cpu) * 1000.0
    stats["cpu_fallback_layers"] += 1
    stats["fallback_gpu_to_cpu_ms"] += copy_ms
    stats["cpu_fallback_ms"] += cpu_ms
    stats["fallback_original_mb"] += float(arr.nbytes) / (1024.0 * 1024.0)
    return {
        "codec": CPU_PAYLOAD_WRAPPER,
        "codec_version": CODEC_VERSION,
        "layer_key": layer_key(layer_name, tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(arr)),
        "payload": payload,
        "payload_length": len(payload),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "step": None,
    }


def _make_payload(
    *,
    layer_name: str,
    key: str,
    tensor: torch.Tensor,
    step: int,
    residual: torch.Tensor,
    base_tensor: Optional[torch.Tensor],
    base_kind: str,
    bits: int,
    state_candidate: torch.Tensor,
    pred_candidate: torch.Tensor,
    stats: MutableMapping[str, Any],
    options: Mapping[str, Any],
) -> Dict[str, Any]:
    q, scale, kernel_ms = _quantize_tensor(residual, bits)
    _fault(options, "after_encode", layer_name)
    stats["kernel_ms"] += kernel_ms
    _sync()
    t_d2h = time.perf_counter()
    if bits == 8:
        payload_bytes = q.detach().cpu().numpy().astype(np.int8, copy=False).tobytes()
    else:
        payload_bytes = _pack_int4(q)
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
        "quantization_mode": f"symmetric_int{bits}",
        "quant_bits": int(bits),
        "scale": float(scale),
        "base_kind": base_kind,
        "momentum_lr": float(options.get("momentum_lr", 0.07)),
        "payload": payload_bytes,
        "payload_length": len(payload_bytes),
    }

    # Decode check before committing state.
    _fault(options, "decode_check", layer_name)
    decoded = _decode_cuda_payload(payload, state_candidate.device, base_tensor)
    if tuple(decoded.shape) != tuple(tensor.shape) or not torch.isfinite(decoded).all().item():
        raise RuntimeError(f"CUDA v2 decode check failed for {layer_name}")
    return payload


def _decode_cuda_payload(
    payload: Mapping[str, Any],
    device: torch.device | str,
    base_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    if payload.get("codec") != CODEC_NAME:
        raise ValueError(f"Unsupported CUDA v2 payload codec: {payload.get('codec')}")
    if int(payload.get("codec_version", -1)) != CODEC_VERSION:
        raise ValueError(f"Unsupported CUDA v2 payload version: {payload.get('codec_version')}")
    shape = tuple(int(x) for x in payload["shape"])
    numel = int(payload["numel"])
    bits = int(payload.get("quant_bits", 8))
    raw = payload["payload"]
    if bits == 8:
        q = torch.from_numpy(np.frombuffer(raw, dtype=np.int8).copy()).to(device=device)
    elif bits == 4:
        q = _unpack_int4(raw, numel, torch.device(device))
    else:
        raise ValueError(f"Unsupported CUDA v2 quant_bits={bits}")
    q = q.reshape(shape).to(dtype=torch.float32)
    residual = q * float(payload["scale"])
    base_kind = str(payload.get("base_kind", "zero"))
    if base_kind == "zero":
        return residual.contiguous()
    if base_kind == "prediction":
        if base_tensor is None:
            raise ValueError("CUDA v2 prediction payload requires decoder state")
        return (base_tensor + residual).contiguous()
    raise ValueError(f"Unsupported CUDA v2 base_kind={base_kind}")


def compress_batch_cuda_v2(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    state_handle: Optional[FalcomCudaV2State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    options = dict(options or {})
    state = state_handle or FalcomCudaV2State(client_id=str(options.get("client_id", "Client1")))
    names, tensors = _normalize_inputs(torch_tensors, layer_names)
    min_numel = int(options.get("cuda_min_numel", 524288))
    quant_bits = int(options.get("quant_bits", 8))
    stats = _empty_stats()
    out = OrderedDict()
    pending = []
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
            raise RuntimeError(f"CUDA v2 key {key} became ineligible; state migration is not implemented")
        if assigned == "cpu":
            pending.append(("cpu", layer_name, tensor, key, None, None, None))
            continue

        stats["cuda_layers"] += 1
        stats["cuda_original_mb"] += _tensor_bytes(tensor) / (1024.0 * 1024.0)
        step = int(staged_steps.get(key, 0)) + 1
        if step == 1:
            base = torch.zeros_like(tensor)
            residual = tensor
            base_kind = "zero"
        else:
            prev = staged_prev[key]
            pred = staged_pred[key]
            if tuple(prev.shape) != tuple(tensor.shape) or tuple(pred.shape) != tuple(tensor.shape):
                raise RuntimeError(f"CUDA v2 layout mismatch for {key}")
            base = pred
            residual = tensor - pred
            base_kind = "prediction"
        next_pred = tensor.detach().clone()
        payload = _make_payload(
            layer_name=layer_name,
            key=key,
            tensor=tensor,
            step=step,
            residual=residual,
            base_tensor=base,
            base_kind=base_kind,
            bits=quant_bits,
            state_candidate=tensor,
            pred_candidate=next_pred,
            stats=stats,
            options=options,
        )
        staged_prev[key] = tensor.detach().clone()
        staged_pred[key] = next_pred
        staged_steps[key] = step
        pending.append(("cuda", layer_name, tensor, key, payload, step, quant_bits))

    for backend, layer_name, tensor, key, payload, step, qbits in pending:
        if backend == "cpu":
            out[layer_name] = _cpu_fallback_layer(state, layer_name, tensor, stats)
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
            }
        )

    t_commit = time.perf_counter()
    state.backend_by_key = staged_backends
    state.prev_grad = staged_prev
    state.prediction_memory = staged_pred
    _sync()
    state.steps = staged_steps
    stats["state_commit_ms"] += (time.perf_counter() - t_commit) * 1000.0
    stats["state_committed"] = stats["cuda_layers"]

    t_ser = time.perf_counter()
    serialized = dumps_cuda_v2_layers(out)
    stats["payload_serialize_ms"] += (time.perf_counter() - t_ser) * 1000.0
    stats["compressed_bytes"] = len(serialized)
    original_mb = stats["cuda_original_mb"] + stats["fallback_original_mb"]
    stats["compression_ratio"] = (
        (original_mb * 1024.0 * 1024.0) / float(len(serialized)) if serialized else 0.0
    )
    return out, stats


def decompress_batch_cuda_v2(
    compressed_layers: Mapping[str, Mapping[str, Any]],
    state_handle: Optional[FalcomCudaV2State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    options = dict(options or {})
    state = state_handle or FalcomCudaV2State(client_id=str(options.get("client_id", "Client1")))
    out = OrderedDict()
    stats: Dict[str, Any] = {
        "decompress_total_ms": 0.0,
        "payload_deserialize_ms": 0.0,
        "cpu_fallback_decompress_ms": 0.0,
        "decoded_layers": 0,
        "cpu_fallback_decoded_layers": 0,
    }
    t_total = time.perf_counter()
    staged_prev = {k: v for k, v in state.prev_grad.items()}
    staged_pred = {k: v for k, v in state.prediction_memory.items()}
    staged_steps = dict(state.steps)
    pending = []
    for layer_name, payload in compressed_layers.items():
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
                raise RuntimeError(f"CUDA v2 decoder missing state for {key}")
        t0 = time.perf_counter()
        decoded = _decode_cuda_payload(payload, "cuda" if torch.cuda.is_available() else "cpu", base)
        _sync()
        stats["payload_deserialize_ms"] += (time.perf_counter() - t0) * 1000.0
        staged_prev[key] = decoded.detach().clone()
        staged_pred[key] = decoded.detach().clone()
        staged_steps[key] = step
        pending.append(("cuda", layer_name, decoded, key, step))
        stats["decoded_layers"] += 1

    decoded_pending = []
    for backend, layer_name, decoded, _key, _step in pending:
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


def dumps_cuda_v2_layers(layers: Mapping[str, Mapping[str, Any]]) -> bytes:
    return pickle.dumps(
        {
            "codec": ENVELOPE_CODEC,
            "codec_version": ENVELOPE_VERSION,
            "layers": OrderedDict(layers),
        }
    )


def loads_cuda_v2_layers(payload: bytes) -> OrderedDict:
    obj = pickle.loads(payload)
    if isinstance(obj, OrderedDict):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("CUDA v2 payload envelope is not a mapping")
    if obj.get("codec") != ENVELOPE_CODEC:
        raise ValueError(f"Unsupported CUDA v2 envelope codec: {obj.get('codec')}")
    if int(obj.get("codec_version", -1)) != ENVELOPE_VERSION:
        raise ValueError(f"Unsupported CUDA v2 envelope version: {obj.get('codec_version')}")
    layers = obj.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("CUDA v2 envelope missing layers mapping")
    return OrderedDict(layers)
