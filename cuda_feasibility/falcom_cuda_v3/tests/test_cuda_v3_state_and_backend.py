#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "cuda_feasibility" / "falcom_cuda_v3"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(APPFL))

from falcom_cuda_v3_wrapper import FalcomCudaV3State, compress_batch_cuda_v3, default_config, layer_key  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaV3State(client_id="state_backend", config=default_config())
    a0 = torch.ones((64, 64, 4, 4), device="cuda", dtype=torch.float32)
    b0 = torch.full_like(a0, 2.0)
    a1 = (a0 * 1.01).contiguous()
    b1 = (b0 * 0.99).contiguous()
    compress_batch_cuda_v3(OrderedDict([("convA.weight", a0), ("convB.weight", b0)]), state_handle=state, options={"cuda_min_numel": 1})
    compress_batch_cuda_v3(OrderedDict([("convA.weight", a1), ("convB.weight", b1)]), state_handle=state, options={"cuda_min_numel": 1})
    key_a = layer_key("convA.weight", a0)
    key_b = layer_key("convB.weight", b0)
    assert key_a != key_b
    assert state.backend_by_key[key_a] == "cuda"
    assert state.backend_by_key[key_b] == "cuda"
    assert state.steps[key_a] == 2
    assert state.steps[key_b] == 2
    assert not torch.equal(state.prev_grad[key_a], state.prev_grad[key_b])

    cpu_owned = FalcomCudaV3State(client_id="owner_stable", config=default_config())
    small = torch.ones((8,), device="cuda", dtype=torch.float32)
    compress_batch_cuda_v3(OrderedDict([("small.bias", small)]), state_handle=cpu_owned, options={"cuda_min_numel": 1024})
    key_small = layer_key("small.bias", small)
    assert cpu_owned.backend_by_key[key_small] == "cpu"
    compress_batch_cuda_v3(OrderedDict([("small.bias", small)]), state_handle=cpu_owned, options={"cuda_min_numel": 1})
    assert cpu_owned.backend_by_key[key_small] == "cpu"
    print("ok - cuda v3 state and backend ownership")


if __name__ == "__main__":
    main()
