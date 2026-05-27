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

from falcom_cuda_v2_wrapper import FalcomCudaV2State, compress_batch_cuda_v2, default_config  # noqa: E402


def main() -> None:
    os.environ.pop("FALCOM_CUDA_EXPERIMENTAL", None)
    state = FalcomCudaV2State(client_id="gate", config=default_config())
    x = torch.ones((64, 64, 4, 4), device="cuda", dtype=torch.float32)
    _layers, stats = compress_batch_cuda_v2(OrderedDict([("conv.weight", x)]), state_handle=state, options={"cuda_min_numel": 1})
    assert int(stats["cuda_layers"]) == 0
    assert int(stats["cpu_fallback_layers"]) == 1
    print("ok - cuda v2 experimental gate defaults to CPU fallback")


if __name__ == "__main__":
    main()
