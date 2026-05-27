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
    state = FalcomCudaV3State(client_id="txn", config=default_config())
    x0 = torch.ones((64, 64, 4, 4), device="cuda", dtype=torch.float32)
    x1 = (x0 + 0.01).contiguous()
    compress_batch_cuda_v3(OrderedDict([("conv.weight", x0)]), state_handle=state, options={"cuda_min_numel": 1})
    key = layer_key("conv.weight", x0)
    step = state.steps[key]
    prev = state.prev_grad[key].clone()
    pred = state.prediction_memory[key].clone()
    try:
        compress_batch_cuda_v3(
            OrderedDict([("conv.weight", x1)]),
            state_handle=state,
            options={"cuda_min_numel": 1, "fault_inject_stage": "decode_check"},
        )
    except RuntimeError as exc:
        assert "decode_check" in str(exc)
    else:
        raise AssertionError("expected decode_check failure")
    assert state.steps[key] == step
    assert torch.equal(state.prev_grad[key], prev)
    assert torch.equal(state.prediction_memory[key], pred)
    print("ok - cuda v3 transaction failure leaves state unchanged")


if __name__ == "__main__":
    main()
