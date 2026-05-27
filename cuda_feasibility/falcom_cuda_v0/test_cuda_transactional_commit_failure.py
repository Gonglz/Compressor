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


def _snapshot(state: FalcomCudaState, key: str):
    return (
        int(state.steps[key]),
        state.prev_grad[key].detach().clone(),
        state.prediction_memory[key].detach().clone(),
        dict(state.backend_by_key),
    )


def _assert_snapshot_unchanged(state: FalcomCudaState, key: str, snap) -> None:
    step, prev, pred, backends = snap
    assert state.steps[key] == step
    assert torch.equal(state.prev_grad[key], prev)
    assert torch.equal(state.prediction_memory[key], pred)
    assert state.backend_by_key == backends


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    stages = ["after_compact_d2h", "payload_assembly", "decode_check"]
    for stage in stages:
        state = FalcomCudaState(client_id=f"txn_{stage}", config=default_config())
        x0 = torch.ones((64, 64, 8, 8), device="cuda", dtype=torch.float32)
        x1 = (x0 * 1.01 + 0.004).contiguous()
        compress_batch_cuda(
            OrderedDict([("conv.weight", x0)]),
            state_handle=state,
            options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
        )
        key = layer_key("conv.weight", x0)
        snap = _snapshot(state, key)
        try:
            compress_batch_cuda(
                OrderedDict([("conv.weight", x1)]),
                state_handle=state,
                options={
                    "cuda_min_numel": 1,
                    "residual_codec": "hybrid_compact_v1a",
                    "fault_inject_stage": stage,
                },
            )
        except RuntimeError as exc:
            assert stage in str(exc)
        else:
            raise AssertionError(f"fault stage {stage} did not fail")
        _assert_snapshot_unchanged(state, key, snap)
        compress_batch_cuda(
            OrderedDict([("conv.weight", x1)]),
            state_handle=state,
            options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
        )
        assert state.steps[key] == snap[0] + 1
    print("ok - transaction failures leave CUDA state unchanged")


if __name__ == "__main__":
    main()
