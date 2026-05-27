#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
V0 = ROOT / "cuda_feasibility" / "falcom_cuda_v0"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V0))
sys.path.insert(0, str(APPFL))

from falcom_cuda_wrapper import FalcomCudaState, compress_batch_cuda, default_config, layer_key  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaState(client_id="shape_dtype", config=default_config())
    f32 = torch.ones((64, 64, 3, 3), device="cuda", dtype=torch.float32)
    f64 = torch.ones((64, 64, 3, 3), device="cuda", dtype=torch.float64)
    changed = torch.ones((32, 64, 3, 3), device="cuda", dtype=torch.float32)
    compress_batch_cuda(OrderedDict([("layer.weight", f32)]), state_handle=state, options={"cuda_min_numel": 1})
    compress_batch_cuda(OrderedDict([("layer.weight", f64)]), state_handle=state, options={"cuda_min_numel": 1})
    compress_batch_cuda(OrderedDict([("layer.weight", changed)]), state_handle=state, options={"cuda_min_numel": 1})
    assert state.backend_by_key[layer_key("layer.weight", f32)] == "cuda"
    assert state.backend_by_key[layer_key("layer.weight", f64)] == "cpu"
    assert state.backend_by_key[layer_key("layer.weight", changed)] == "cuda"
    assert layer_key("layer.weight", f32) != layer_key("layer.weight", changed)
    print("ok - shape/dtype guard")


if __name__ == "__main__":
    main()
