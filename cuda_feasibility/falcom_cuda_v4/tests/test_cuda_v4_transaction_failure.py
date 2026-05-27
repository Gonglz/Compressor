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

from falcom_cuda_v4_wrapper import FalcomCudaV4State, compress_batch_cuda_v4, default_config, layer_key  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaV4State(client_id="v4_txn", config=default_config())
    x0 = torch.randn((64, 64, 4, 4), device="cuda", dtype=torch.float32).contiguous()
    x1 = (x0 * 1.01).contiguous()
    compress_batch_cuda_v4(OrderedDict([("conv.weight", x0)]), state_handle=state, options={"cuda_min_numel": 1})
    key = layer_key("conv.weight", x0)
    before_step = state.steps[key]
    before_prev = state.prev_grad[key].detach().clone()
    try:
        compress_batch_cuda_v4(
            OrderedDict([("conv.weight", x1)]),
            state_handle=state,
            options={"cuda_min_numel": 1, "fault_inject_stage": "decode_check"},
        )
        raise AssertionError("expected transaction failure")
    except RuntimeError as exc:
        assert "transaction failure" in str(exc)
    assert state.steps[key] == before_step
    assert torch.equal(state.prev_grad[key], before_prev)
    print("ok - cuda v4 transaction failure leaves state unchanged")


if __name__ == "__main__":
    main()
