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
    state = FalcomCudaState(client_id="same_shape", config=default_config())
    a0 = torch.ones((64, 64, 3, 3), device="cuda", dtype=torch.float32)
    b0 = torch.full((64, 64, 3, 3), 2.0, device="cuda", dtype=torch.float32)
    a1 = a0 * 1.1
    b1 = b0 * 0.9
    compress_batch_cuda(OrderedDict([("convA.weight", a0), ("convB.weight", b0)]), state_handle=state, options={"cuda_min_numel": 1})
    compress_batch_cuda(OrderedDict([("convA.weight", a1), ("convB.weight", b1)]), state_handle=state, options={"cuda_min_numel": 1})
    key_a = layer_key("convA.weight", a0)
    key_b = layer_key("convB.weight", b0)
    assert key_a != key_b
    assert state.backend_by_key[key_a] == "cuda"
    assert state.backend_by_key[key_b] == "cuda"
    assert state.steps[key_a] == 2
    assert state.steps[key_b] == 2
    assert not torch.equal(state.prev_grad[key_a], state.prev_grad[key_b])
    print("ok - same-shape different-layer CUDA state isolation")


if __name__ == "__main__":
    main()
