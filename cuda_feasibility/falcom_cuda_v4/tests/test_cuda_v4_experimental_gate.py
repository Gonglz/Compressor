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

from falcom_cuda_v4_wrapper import FalcomCudaV4State, compress_batch_cuda_v4, default_config  # noqa: E402


def _run_with_gate(value: str | None) -> None:
    if value is None:
        os.environ.pop("FALCOM_CUDA_EXPERIMENTAL", None)
    else:
        os.environ["FALCOM_CUDA_EXPERIMENTAL"] = value
    state = FalcomCudaV4State(client_id=f"v4_gate_{value}", config=default_config())
    x = torch.ones((64, 64, 4, 4), device="cuda", dtype=torch.float32)
    _layers, stats = compress_batch_cuda_v4(
        OrderedDict([("conv.weight", x.contiguous())]),
        state_handle=state,
        options={"cuda_min_numel": 1},
    )
    assert int(stats["cuda_layers"]) == 0
    assert int(stats["cpu_fallback_layers"]) == 1


def main() -> None:
    _run_with_gate(None)
    _run_with_gate("0")
    print("ok - cuda v4 experimental gate defaults to CPU fallback")


if __name__ == "__main__":
    main()
