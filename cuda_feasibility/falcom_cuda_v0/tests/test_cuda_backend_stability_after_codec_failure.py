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
    state = FalcomCudaState(client_id="backend_stability", config=default_config())
    x0 = torch.ones((64, 64, 8, 8), device="cuda", dtype=torch.float32)
    x1 = (x0 + 0.01).contiguous()
    compress_batch_cuda(
        OrderedDict([("conv.weight", x0)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
    )
    key = layer_key("conv.weight", x0)
    assert state.backend_by_key[key] == "cuda"
    try:
        compress_batch_cuda(
            OrderedDict([("conv.weight", x1)]),
            state_handle=state,
            options={
                "cuda_min_numel": 1,
                "residual_codec": "hybrid_compact_v1a",
                "fault_inject_stage": "payload_assembly",
            },
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected injected codec failure")
    assert state.backend_by_key[key] == "cuda"
    assert key in state.prev_grad
    assert key in state.prediction_memory
    print("ok - codec failure does not switch backend or drop CUDA state")


if __name__ == "__main__":
    main()
