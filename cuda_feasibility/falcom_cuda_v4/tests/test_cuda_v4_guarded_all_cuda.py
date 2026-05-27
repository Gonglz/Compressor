#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "cuda_feasibility" / "falcom_cuda_v4"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V4))
sys.path.insert(0, str(APPFL))

from falcom_cuda_v4_wrapper import FalcomCudaV4State, compress_batch_cuda_v4, default_config  # noqa: E402


def main() -> None:
    small = torch.ones((16,), device="cuda", dtype=torch.float32).contiguous()

    os.environ.pop("FALCOM_CUDA_EXPERIMENTAL", None)
    os.environ["FALCOM_CUDA_V4_GUARDED_ALL_CUDA"] = "1"
    off_state = FalcomCudaV4State(client_id="v4_guarded_gate_off", config=default_config())
    _layers, stats = compress_batch_cuda_v4(
        OrderedDict([("small.bias", small)]),
        state_handle=off_state,
        options={"cuda_min_numel": 2048, "guarded_all_cuda": True},
    )
    assert int(stats["cuda_layers"]) == 0
    assert int(stats["cpu_fallback_layers"]) == 1

    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    on_state = FalcomCudaV4State(client_id="v4_guarded_gate_on", config=default_config())
    _layers, stats = compress_batch_cuda_v4(
        OrderedDict([("small.bias", small)]),
        state_handle=on_state,
        options={"cuda_min_numel": 2048, "guarded_all_cuda": True},
    )
    assert int(stats["cuda_layers"]) == 1
    assert int(stats["cpu_fallback_layers"]) == 0
    assert int(stats["num_payload_objects"]) <= 1

    cpu_owned = FalcomCudaV4State(client_id="v4_guarded_owner_stable", config=default_config())
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    os.environ.pop("FALCOM_CUDA_V4_GUARDED_ALL_CUDA", None)
    compress_batch_cuda_v4(
        OrderedDict([("small.bias", small)]),
        state_handle=cpu_owned,
        options={"cuda_min_numel": 2048},
    )
    _layers, stats = compress_batch_cuda_v4(
        OrderedDict([("small.bias", small)]),
        state_handle=cpu_owned,
        options={"cuda_min_numel": 2048, "guarded_all_cuda": True},
    )
    assert int(stats["cuda_layers"]) == 0
    assert int(stats["cpu_fallback_layers"]) == 1
    assert stats["fallback_records"][0]["reason"] == "backend_stability_guard"
    print("ok - cuda v4 guarded all-valid-CUDA gate and backend ownership")


if __name__ == "__main__":
    main()
