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

from falcom_cuda_wrapper import FalcomCudaState, compress_batch_cuda, default_config  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaState(client_id="mixed", config=default_config())
    big = torch.ones((64, 64, 3, 3), device="cuda", dtype=torch.float32)
    small = torch.ones((8,), device="cuda", dtype=torch.float32)
    layers = OrderedDict([("big.weight", big), ("small.bias", small)])
    compressed, stats = compress_batch_cuda(layers, state_handle=state, options={"cuda_min_numel": 1024})
    assert compressed["big.weight"]["codec"] == "c_struct"
    assert compressed["small.bias"]["codec"] == "c_struct"
    assert int(stats["cuda_layers"]) == 1
    assert int(stats["cpu_fallback_layers"]) == 1
    assert "cpu" in set(state.backend_by_key.values())
    assert "cuda" in set(state.backend_by_key.values())
    print("ok - mixed CUDA/CPU fallback")


if __name__ == "__main__":
    main()
