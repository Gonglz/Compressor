#!/usr/bin/env python3
"""FalCom CUDA v4 experimental fused q8 closed-loop compressor."""

from __future__ import annotations

import importlib
import os
import pickle
import struct
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
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from appfl.compressor.FalComC import FalComC  # noqa: E402


CODEC_NAME = "cuda_v4_experimental"
CODEC_VERSION = 1
ENVELOPE_MAGIC = b"FCV4B001"
ENVELOPE_VERSION = 1
CUDA_BATCH_KEY = "__cuda_q8_blob__"
CPU_BATCH_KEY = "__cpu_fallback_batch__"
CPU_BATCH_CODEC = "cpu_fallback_batch_pickle"

try:
    _EXT = importlib.import_module("_falcom_cuda_v4")
except Exception:
    _EXT = None


def cuda_v4_enabled() -> bool:
    value = os.environ.get("FALCOM_CUDA_EXPERIMENTAL", "")
    return value not in ("", "0", "false", "False", "FALSE", "off", "OFF")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) not in ("", "0", "false", "False", "FALSE", "off", "OFF", "none", "None")


def guarded_all_cuda_enabled(options: Optional[Mapping[str, Any]] = None) -> bool:
    if options is not None and "guarded_all_cuda" in options:
        return _truthy(options.get("guarded_all_cuda"))
    return _truthy(os.environ.get("FALCOM_CUDA_V4_GUARDED_ALL_CUDA", ""))


def extension_available() -> bool:
    return _EXT is not None and bool(_EXT.is_cuda_build())


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


def _shape_string(shape: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def layer_key(layer_name: str, tensor: torch.Tensor) -> str:
    return f"{layer_name}|{str(tensor.dtype)}|{_shape_string(_shape_tuple(tensor))}"


def _normalize_inputs(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]],
) -> Tuple[List[str], List[torch.Tensor]]:
    if isinstance(torch_tensors, Mapping):
        return [str(k) for k in torch_tensors.keys()], list(torch_tensors.values())
    if layer_names is None:
        raise ValueError("layer_names is required when torch_tensors is not a mapping")
    if len(layer_names) != len(torch_tensors):
        raise ValueError("layer_names length does not match tensor count")
    return [str(x) for x in layer_names], list(torch_tensors)


def cuda_guard_status(
    tensor: Any,
    min_numel: int,
    guarded_all_cuda: bool = False,
    assigned_backend: Optional[str] = None,
) -> Dict[str, Any]:
    is_tensor = isinstance(tensor, torch.Tensor)
    numel = int(tensor.numel()) if is_tensor else 0
    dtype_ok = bool(is_tensor and tensor.dtype == torch.float32)
    cuda_tensor_ok = bool(is_tensor and tensor.is_cuda)
    contiguous_ok = bool(is_tensor and tensor.is_contiguous())
    shape_supported = bool(is_tensor and numel > 0)
    q8_supported = bool(extension_available() and cuda_v4_enabled() and dtype_ok and shape_supported)
    threshold_ok = bool(numel >= int(min_numel))
    state_owner_stable = assigned_backend != "cpu"
    mandatory_ok = bool(
        extension_available()
        and cuda_v4_enabled()
        and is_tensor
        and cuda_tensor_ok
        and dtype_ok
        and contiguous_ok
        and shape_supported
        and q8_supported
    )
    candidate = bool(mandatory_ok and state_owner_stable and (threshold_ok or guarded_all_cuda))
    if candidate:
        reason = ""
    elif not extension_available():
        reason = "unsupported_codec_path"
    elif not cuda_v4_enabled():
        reason = "cpu_fallback_policy_only"
    elif not is_tensor:
        reason = "unsupported_codec_path"
    elif not cuda_tensor_ok or not dtype_ok:
        reason = "dtype/layout_not_supported"
    elif not contiguous_ok:
        reason = "non_contiguous"
    elif not shape_supported:
        reason = "shape_guard_failed"
    elif not q8_supported:
        reason = "unsupported_codec_path"
    elif not threshold_ok and not guarded_all_cuda:
        reason = "numel_below_threshold"
    elif not state_owner_stable:
        reason = "backend_stability_guard"
    else:
        reason = "cpu_fallback_policy_only"
    return {
        "candidate": candidate,
        "guard_status": "pass" if candidate else "fallback",
        "fallback_reason": reason,
        "dtype_ok": dtype_ok,
        "cuda_tensor_ok": cuda_tensor_ok,
        "contiguous_ok": contiguous_ok,
        "shape_supported": shape_supported,
        "state_owner_stable": state_owner_stable,
        "q8_supported": q8_supported,
        "threshold_ok": threshold_ok,
        "numel": numel,
    }


def _is_cuda_candidate(tensor: torch.Tensor, min_numel: int, guarded_all_cuda: bool = False) -> bool:
    return bool(cuda_guard_status(tensor, min_numel, guarded_all_cuda=guarded_all_cuda)["candidate"])


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _empty_stats() -> Dict[str, Any]:
    return {
        "cuda_layers": 0,
        "cpu_fallback_layers": 0,
        "cuda_original_mb": 0.0,
        "fallback_original_mb": 0.0,
        "avg_numel_per_cuda_layer": 0.0,
        "encode_kernel_ms": 0.0,
        "payload_d2h_ms": 0.0,
        "payload_blob_bytes": 0,
        "num_payload_objects": 0,
        "kernel_launch_count": 0,
        "extension_encode_total_ms": 0.0,
        "envelope_serialize_ms": 0.0,
        "envelope_parse_ms": 0.0,
        "payload_h2d_ms": 0.0,
        "decode_kernel_ms": 0.0,
        "decoded_tensor_materialize_ms": 0.0,
        "cpu_fallback_ms": 0.0,
        "cpu_fallback_decode_ms": 0.0,
        "fallback_gpu_to_cpu_ms": 0.0,
        "cpu_fallback_payload_bytes": 0,
        "cpu_fallback_batches": 0,
        "state_commit_ms": 0.0,
        "compressed_bytes": 0,
        "compression_ratio": 0.0,
        "state_committed": 0,
        "layer_records": [],
        "fallback_records": [],
    }


@dataclass
class FalcomCudaV4State:
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


def _fault(options: Mapping[str, Any], stage: str, layer_name: str = "") -> None:
    requested = str(options.get("fault_inject_stage", ""))
    requested_layer = options.get("fault_inject_layer")
    if requested == stage and (requested_layer is None or str(requested_layer) == layer_name):
        where = f" for {layer_name}" if layer_name else ""
        raise RuntimeError(f"Injected CUDA v4 transaction failure at {stage}{where}")


def _cpu_fallback_batch(
    state: FalcomCudaV4State,
    items: Sequence[Tuple[str, torch.Tensor, str, str, Dict[str, Any]]],
    stats: MutableMapping[str, Any],
) -> Dict[str, Any]:
    assert state.cpu_compressor is not None
    arrays = OrderedDict()
    layer_meta = []
    any_cuda = any(isinstance(t, torch.Tensor) and t.is_cuda for _name, t, _key, _reason, _guard in items)
    if any_cuda:
        _sync()
    t_copy = time.perf_counter()
    for layer_name, tensor, key, reason, guard in items:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        arrays[layer_name] = arr
        stats["fallback_original_mb"] += float(arr.nbytes) / (1024.0 * 1024.0)
        layer_meta.append(
            {
                "layer_name": layer_name,
                "layer_key": key,
                "shape": tuple(int(x) for x in arr.shape),
                "dtype": str(arr.dtype),
                "numel": int(arr.size),
                "nbytes": int(arr.nbytes),
                "reason": reason or "cpu_fallback_policy_only",
                "guard_status": guard.get("guard_status", "fallback"),
                "dtype_ok": bool(guard.get("dtype_ok", False)),
                "contiguous_ok": bool(guard.get("contiguous_ok", False)),
                "shape_supported": bool(guard.get("shape_supported", False)),
                "state_owner_stable": bool(guard.get("state_owner_stable", True)),
                "q8_supported": bool(guard.get("q8_supported", False)),
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


def _ext_stats_encode(stats_tensor: torch.Tensor) -> Dict[str, float]:
    vals = [float(x) for x in stats_tensor.cpu().tolist()]
    return {
        "encode_kernel_ms": vals[0],
        "payload_d2h_ms": vals[1],
        "num_payload_objects": vals[2],
        "payload_blob_bytes": vals[3],
        "kernel_launch_count": vals[4],
        "extension_encode_total_ms": vals[5],
    }


def _ext_stats_decode(stats_tensor: torch.Tensor) -> Dict[str, float]:
    vals = [float(x) for x in stats_tensor.cpu().tolist()]
    return {
        "payload_h2d_ms": vals[0],
        "decode_kernel_ms": vals[1],
        "decoded_tensor_materialize_ms": vals[2],
        "decode_kernel_launch_count": vals[3],
        "payload_blob_bytes": vals[4],
    }


def _cuda_payload_from_ext(
    records: List[Dict[str, Any]],
    payload_blob_cpu: torch.Tensor,
    offsets_cpu: torch.Tensor,
    lengths_cpu: torch.Tensor,
    scales_cpu: torch.Tensor,
) -> Dict[str, Any]:
    return {
        "codec": CODEC_NAME,
        "codec_version": CODEC_VERSION,
        "records": records,
        "payload_blob_cpu": payload_blob_cpu.contiguous(),
        "offsets_cpu": offsets_cpu.contiguous(),
        "lengths_cpu": lengths_cpu.contiguous(),
        "scales_cpu": scales_cpu.contiguous(),
    }


def compress_batch_cuda_v4(
    torch_tensors: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    state_handle: Optional[FalcomCudaV4State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    options = dict(options or {})
    if str(options.get("quant_mode", "8")).lower() not in ("8", "q8", "int8"):
        raise ValueError("CUDA v4 implements q8 only")
    state = state_handle or FalcomCudaV4State(client_id=str(options.get("client_id", "Client1")))
    names, tensors = _normalize_inputs(torch_tensors, layer_names)
    min_numel = int(options.get("cuda_min_numel", 262144))
    guarded_all_cuda = guarded_all_cuda_enabled(options)
    stats = _empty_stats()
    stats["guarded_all_cuda"] = int(guarded_all_cuda)
    out = OrderedDict()
    pending = []
    cpu_items: List[Tuple[str, torch.Tensor, str, str, Dict[str, Any]]] = []
    cuda_names: List[str] = []
    cuda_keys: List[str] = []
    currents: List[torch.Tensor] = []
    bases: List[torch.Tensor] = []
    flags: List[int] = []
    steps: List[int] = []
    base_kinds: List[str] = []
    staged_backends = dict(state.backend_by_key)
    staged_prev = {k: v for k, v in state.prev_grad.items()}
    staged_pred = {k: v for k, v in state.prediction_memory.items()}
    staged_steps = dict(state.steps)

    for layer_name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        key = layer_key(layer_name, tensor)
        unowned_guard = cuda_guard_status(tensor, min_numel, guarded_all_cuda=guarded_all_cuda)
        assigned = staged_backends.get(key)
        assigned_before = assigned
        if assigned is None:
            assigned = "cuda" if unowned_guard["candidate"] else "cpu"
            staged_backends[key] = assigned
        owned_guard = cuda_guard_status(
            tensor,
            min_numel,
            guarded_all_cuda=guarded_all_cuda,
            assigned_backend=assigned,
        )
        if assigned == "cuda" and not owned_guard["candidate"]:
            raise RuntimeError(f"CUDA v4 key {key} became ineligible; state migration is not implemented")
        if assigned == "cpu":
            reason = str(unowned_guard.get("fallback_reason", ""))
            if assigned_before is not None and bool(unowned_guard.get("candidate", False)):
                reason = "backend_stability_guard"
            guard = dict(owned_guard)
            guard["fallback_reason"] = reason
            pending.append(("cpu", layer_name, tensor, key, reason, guard))
            continue

        step = int(staged_steps.get(key, 0)) + 1
        if step == 1:
            base = tensor
            has_base = 0
            base_kind = "zero"
        else:
            prev = staged_prev[key]
            pred = staged_pred[key]
            if tuple(prev.shape) != tuple(tensor.shape) or tuple(pred.shape) != tuple(tensor.shape):
                raise RuntimeError(f"CUDA v4 layout mismatch for {key}")
            base = pred
            has_base = 1
            base_kind = "prediction"
        stats["cuda_layers"] += 1
        stats["cuda_original_mb"] += _tensor_bytes(tensor) / (1024.0 * 1024.0)
        currents.append(tensor)
        bases.append(base)
        flags.append(has_base)
        steps.append(step)
        base_kinds.append(base_kind)
        cuda_names.append(layer_name)
        cuda_keys.append(key)

    for backend, layer_name, tensor, key, reason, guard in pending:
        if backend == "cpu":
            cpu_items.append((layer_name, tensor, key, reason, guard))

    candidates: List[torch.Tensor] = []
    if currents:
        if _EXT is None:
            raise RuntimeError("CUDA v4 extension is not available")
        flag_tensor = torch.tensor(flags, dtype=torch.uint8, device="cpu")
        (
            payload_blob_cpu,
            offsets_cpu,
            lengths_cpu,
            scales_cpu,
            candidates,
            ext_stats_tensor,
        ) = _EXT.encode_q8_batch(currents, bases, flag_tensor)
        ext_stats = _ext_stats_encode(ext_stats_tensor)
        for key, value in ext_stats.items():
            stats[key] = value
        _fault(options, "after_encode")
        records = []
        offsets = offsets_cpu.cpu().tolist()
        lengths = lengths_cpu.cpu().tolist()
        scales = scales_cpu.cpu().tolist()
        for idx, (layer_name, key, tensor, step, base_kind) in enumerate(
            zip(cuda_names, cuda_keys, currents, steps, base_kinds)
        ):
            record = {
                "layer_name": layer_name,
                "layer_key": key,
                "shape": _shape_tuple(tensor),
                "dtype": "float32",
                "step": int(step),
                "numel": int(tensor.numel()),
                "offset": int(offsets[idx]),
                "length": int(lengths[idx]),
                "scale": float(scales[idx]),
                "base_kind": base_kind,
                "quant_bits": 8,
            }
            records.append(record)
            stats["layer_records"].append(
                {
                    "layer_name": layer_name,
                    "layer_key": key,
                    "backend": "cuda",
                    "step": step,
                    "shape": _shape_string(record["shape"]),
                    "numel": int(tensor.numel()),
                    "payload_length": int(record["length"]),
                    "quant_bits": 8,
                    "fallback_reason": "",
                    "guard_status": "pass",
                    "dtype_ok": True,
                    "contiguous_ok": True,
                    "shape_supported": True,
                    "state_owner_stable": True,
                    "q8_supported": True,
                }
            )
        out[CUDA_BATCH_KEY] = _cuda_payload_from_ext(records, payload_blob_cpu, offsets_cpu, lengths_cpu, scales_cpu)
        stats["avg_numel_per_cuda_layer"] = (
            float(sum(int(t.numel()) for t in currents)) / float(len(currents)) if currents else 0.0
        )

    if cpu_items:
        out[CPU_BATCH_KEY] = _cpu_fallback_batch(state, cpu_items, stats)
        stats["num_payload_objects"] += 1
        for record in stats["fallback_records"]:
            stats["layer_records"].append(
                {
                    "layer_name": record["layer_name"],
                    "layer_key": record["layer_key"],
                    "backend": "cpu_batch",
                    "step": "",
                    "shape": _shape_string(record["shape"]),
                    "numel": int(record["numel"]),
                    "payload_length": 0,
                    "quant_bits": "",
                    "fallback_reason": record["reason"],
                    "guard_status": record.get("guard_status", "fallback"),
                    "dtype_ok": record.get("dtype_ok", False),
                    "contiguous_ok": record.get("contiguous_ok", False),
                    "shape_supported": record.get("shape_supported", False),
                    "state_owner_stable": record.get("state_owner_stable", True),
                    "q8_supported": record.get("q8_supported", False),
                }
            )

    _fault(options, "decode_check")
    t_commit = time.perf_counter()
    for key, candidate, step in zip(cuda_keys, candidates, steps):
        staged_prev[key] = candidate
        staged_pred[key] = candidate
        staged_steps[key] = step
    state.backend_by_key = staged_backends
    state.prev_grad = staged_prev
    state.prediction_memory = staged_pred
    state.steps = staged_steps
    stats["state_commit_ms"] += (time.perf_counter() - t_commit) * 1000.0
    stats["state_committed"] = stats["cuda_layers"]
    return out, stats


def decompress_batch_cuda_v4(
    compressed_layers: Mapping[str, Mapping[str, Any]],
    state_handle: Optional[FalcomCudaV4State] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    options = dict(options or {})
    state = state_handle or FalcomCudaV4State(client_id=str(options.get("client_id", "Client1")))
    out = OrderedDict()
    stats = _empty_stats()
    staged_prev = {k: v for k, v in state.prev_grad.items()}
    staged_pred = {k: v for k, v in state.prediction_memory.items()}
    staged_steps = dict(state.steps)

    cuda_payload = compressed_layers.get(CUDA_BATCH_KEY)
    if cuda_payload is not None:
        records = list(cuda_payload["records"])
        bases = []
        flags = []
        shapes = []
        for record in records:
            key = str(record["layer_key"])
            step = int(record["step"])
            shapes.append([int(x) for x in record["shape"]])
            if step == 1:
                bases.append(torch.empty((1,), device="cuda", dtype=torch.float32))
                flags.append(0)
            else:
                base = staged_pred.get(key)
                if base is None:
                    raise RuntimeError(f"CUDA v4 decoder missing state for {key}")
                bases.append(base)
                flags.append(1)
        flag_tensor = torch.tensor(flags, dtype=torch.uint8, device="cpu")
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        decoded, ext_stats_tensor = _EXT.decode_q8_batch(
            cuda_payload["payload_blob_cpu"],
            cuda_payload["offsets_cpu"],
            cuda_payload["lengths_cpu"],
            cuda_payload["scales_cpu"],
            shapes,
            bases,
            flag_tensor,
            int(device_index),
        )
        ext_stats = _ext_stats_decode(ext_stats_tensor)
        stats.update(ext_stats)
        for record, tensor in zip(records, decoded):
            layer_name = str(record["layer_name"])
            key = str(record["layer_key"])
            step = int(record["step"])
            out[layer_name] = tensor
            staged_prev[key] = tensor.detach().clone()
            staged_pred[key] = tensor.detach().clone()
            staged_steps[key] = step
            stats["cuda_layers"] += 1

    cpu_payload = compressed_layers.get(CPU_BATCH_KEY)
    if cpu_payload is not None:
        if int(cpu_payload.get("codec_version", -1)) != CODEC_VERSION:
            raise ValueError(f"Unsupported CPU fallback batch payload version: {cpu_payload.get('codec_version')}")
        assert state.cpu_compressor is not None
        t0 = time.perf_counter()
        decomp = state.cpu_compressor.decompress_model(cpu_payload["payload"], client_id=state.client_id)
        for layer_name, tensor in decomp.items():
            out[layer_name] = tensor.cuda() if torch.cuda.is_available() else tensor
        stats["cpu_fallback_decode_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["cpu_fallback_layers"] = len(cpu_payload.get("layers", ()))

    state.prev_grad = staged_prev
    state.prediction_memory = staged_pred
    state.steps = staged_steps
    return out, stats


def _pack_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _read_str(buf: memoryview, pos: int) -> Tuple[str, int]:
    (n,) = struct.unpack_from("<H", buf, pos)
    pos += 2
    raw = bytes(buf[pos : pos + n])
    pos += n
    return raw.decode("utf-8"), pos


def dumps_cuda_v4_layers(layers: Mapping[str, Mapping[str, Any]]) -> bytes:
    t0 = time.perf_counter()
    cuda_payload = layers.get(CUDA_BATCH_KEY)
    cpu_payload = layers.get(CPU_BATCH_KEY)
    parts = [ENVELOPE_MAGIC, struct.pack("<II", ENVELOPE_VERSION, 1 if cuda_payload else 0)]
    if cuda_payload:
        records = cuda_payload["records"]
        blob = cuda_payload["payload_blob_cpu"].contiguous().numpy().tobytes()
        parts.append(struct.pack("<Q", len(blob)))
        parts.append(blob)
        parts.append(struct.pack("<I", len(records)))
        for record in records:
            parts.append(_pack_str(str(record["layer_name"])))
            parts.append(_pack_str(str(record["layer_key"])))
            shape = tuple(int(x) for x in record["shape"])
            parts.append(struct.pack("<B", len(shape)))
            parts.append(struct.pack("<" + "Q" * len(shape), *shape))
            parts.append(
                struct.pack(
                    "<qQQfB",
                    int(record["step"]),
                    int(record["offset"]),
                    int(record["length"]),
                    float(record["scale"]),
                    1 if str(record["base_kind"]) == "prediction" else 0,
                )
            )
    fallback_blob = cpu_payload["payload"] if cpu_payload else b""
    fallback_layers = int(len(cpu_payload.get("layers", ()))) if cpu_payload else 0
    parts.append(struct.pack("<IQ", fallback_layers, len(fallback_blob)))
    parts.append(fallback_blob)
    result = b"".join(parts)
    # Attach timing to the mutable source object for benchmark accounting.
    if hasattr(layers, "__setitem__"):
        layers["_last_envelope_serialize_ms"] = {"ms": (time.perf_counter() - t0) * 1000.0}
    return result


def loads_cuda_v4_layers(payload: bytes) -> OrderedDict:
    t0 = time.perf_counter()
    buf = memoryview(payload)
    pos = 0
    if bytes(buf[: len(ENVELOPE_MAGIC)]) != ENVELOPE_MAGIC:
        raise ValueError("Unsupported CUDA v4 envelope magic")
    pos += len(ENVELOPE_MAGIC)
    version, has_cuda = struct.unpack_from("<II", buf, pos)
    pos += 8
    if int(version) != ENVELOPE_VERSION:
        raise ValueError(f"Unsupported CUDA v4 envelope version: {version}")
    out = OrderedDict()
    if has_cuda:
        (blob_len,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        blob_bytes = bytes(buf[pos : pos + blob_len])
        pos += blob_len
        payload_blob_cpu = torch.frombuffer(bytearray(blob_bytes), dtype=torch.uint8).contiguous()
        (count,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        records = []
        offsets = []
        lengths = []
        scales = []
        for _ in range(count):
            layer_name, pos = _read_str(buf, pos)
            key, pos = _read_str(buf, pos)
            (ndim,) = struct.unpack_from("<B", buf, pos)
            pos += 1
            shape = struct.unpack_from("<" + "Q" * ndim, buf, pos)
            pos += 8 * ndim
            step, offset, length, scale, base_flag = struct.unpack_from("<qQQfB", buf, pos)
            pos += struct.calcsize("<qQQfB")
            records.append(
                {
                    "layer_name": layer_name,
                    "layer_key": key,
                    "shape": tuple(int(x) for x in shape),
                    "dtype": "float32",
                    "step": int(step),
                    "numel": int(length),
                    "offset": int(offset),
                    "length": int(length),
                    "scale": float(scale),
                    "base_kind": "prediction" if int(base_flag) else "zero",
                    "quant_bits": 8,
                }
            )
            offsets.append(int(offset))
            lengths.append(int(length))
            scales.append(float(scale))
        out[CUDA_BATCH_KEY] = {
            "codec": CODEC_NAME,
            "codec_version": CODEC_VERSION,
            "records": records,
            "payload_blob_cpu": payload_blob_cpu,
            "offsets_cpu": torch.tensor(offsets, dtype=torch.int64),
            "lengths_cpu": torch.tensor(lengths, dtype=torch.int64),
            "scales_cpu": torch.tensor(scales, dtype=torch.float32),
        }
    fallback_count, fallback_len = struct.unpack_from("<IQ", buf, pos)
    pos += struct.calcsize("<IQ")
    fallback_payload = bytes(buf[pos : pos + fallback_len])
    if fallback_len:
        out[CPU_BATCH_KEY] = {
            "codec": CPU_BATCH_CODEC,
            "codec_version": CODEC_VERSION,
            "layers": [None] * int(fallback_count),
            "payload": fallback_payload,
            "payload_length": int(fallback_len),
        }
    out["_last_envelope_parse_ms"] = {"ms": (time.perf_counter() - t0) * 1000.0}
    return out
