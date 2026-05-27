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
    state = FalcomCudaState(client_id="rounds", config=default_config())
    x0 = torch.linspace(-1.0, 1.0, 64 * 64 * 3 * 3, device="cuda", dtype=torch.float32).reshape(64, 64, 3, 3)
    x1 = (x0 * 1.02).contiguous()
    c0, _ = compress_batch_cuda(OrderedDict([("conv.weight", x0.contiguous())]), state_handle=state, options={"cuda_min_numel": 1})
    c1, s1 = compress_batch_cuda(OrderedDict([("conv.weight", x1)]), state_handle=state, options={"cuda_min_numel": 1})
    assert c0["conv.weight"]["c_type"] == "direct"
    assert c0["conv.weight"]["step"] == 1
    assert c1["conv.weight"]["c_type"] == "momentum_predicted"
    assert c1["conv.weight"]["step"] == 2
    assert s1["kernel_launch_count"] == 1
    assert state.steps[layer_key("conv.weight", x0)] == 2
    print("ok - round0 direct and hot momentum CUDA path")


if __name__ == "__main__":
    main()
