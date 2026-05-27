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
    state = FalcomCudaState(client_id="payload_budget", config=default_config())
    x0 = torch.linspace(-1.0, 1.0, 64 * 64 * 8 * 8, device="cuda", dtype=torch.float32).reshape(64, 64, 8, 8)
    x1 = (x0 * 1.01 + 0.001).contiguous()
    compress_batch_cuda(
        OrderedDict([("conv.weight", x0.contiguous())]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
    )
    _layers, stats = compress_batch_cuda(
        OrderedDict([("conv.weight", x1)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
    )
    assert stats["codec_mode"] == "hybrid_compact_v1a"
    assert stats["state_committed"] == 1
    assert stats["full_residual_bytes"] > 0
    assert stats["compact_intermediate_d2h_bytes"] > 0
    assert stats["compact_vs_full_residual_pct"] <= 35.0
    assert stats["accepted_compact"] == 1
    print("ok - hybrid compact residual respects <=35% D2H budget")


if __name__ == "__main__":
    main()
