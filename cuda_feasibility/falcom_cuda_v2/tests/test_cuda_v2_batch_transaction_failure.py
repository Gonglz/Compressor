#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "cuda_feasibility" / "falcom_cuda_v2"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(APPFL))

from falcom_cuda_v2_wrapper import FalcomCudaV2State, compress_batch_cuda_v2, default_config, layer_key  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaV2State(client_id="batch_txn", config=default_config())
    a = torch.ones((64, 64, 4, 4), device="cuda", dtype=torch.float32)
    b = torch.full_like(a, 2.0)
    key_a = layer_key("a.weight", a)
    key_b = layer_key("b.weight", b)
    try:
        compress_batch_cuda_v2(
            OrderedDict([("a.weight", a), ("b.weight", b)]),
            state_handle=state,
            options={
                "cuda_min_numel": 1,
                "fault_inject_stage": "decode_check",
                "fault_inject_layer": "b.weight",
            },
        )
    except RuntimeError as exc:
        assert "decode_check" in str(exc)
    else:
        raise AssertionError("expected batch failure")
    assert key_a not in state.steps
    assert key_b not in state.steps
    assert key_a not in state.prev_grad
    assert key_b not in state.prev_grad
    assert key_a not in state.prediction_memory
    assert key_b not in state.prediction_memory
    assert key_a not in state.backend_by_key
    assert key_b not in state.backend_by_key
    print("ok - cuda v2 batch failure does not partially commit CUDA state")


if __name__ == "__main__":
    main()
