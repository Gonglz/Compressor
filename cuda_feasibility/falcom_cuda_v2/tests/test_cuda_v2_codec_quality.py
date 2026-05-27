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

from falcom_cuda_v2_wrapper import (  # noqa: E402
    CODEC_NAME,
    FalcomCudaV2State,
    compress_batch_cuda_v2,
    decompress_batch_cuda_v2,
    default_config,
)


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(a), min=1e-12))


def _case(name: str, tensor: torch.Tensor) -> None:
    c_state = FalcomCudaV2State(client_id=f"codec_{name}", config=default_config())
    d_state = FalcomCudaV2State(client_id=f"codec_{name}", config=default_config())
    layers, stats = compress_batch_cuda_v2(
        OrderedDict([(f"{name}.weight", tensor.contiguous())]),
        state_handle=c_state,
        options={"cuda_min_numel": 1, "quant_bits": 8},
    )
    assert layers[f"{name}.weight"]["codec"] == CODEC_NAME
    out, dstats = decompress_batch_cuda_v2(layers, state_handle=d_state)
    decoded = out[f"{name}.weight"]
    assert tuple(decoded.shape) == tuple(tensor.shape)
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    assert _relative_l2(tensor, decoded) <= 0.02
    assert int(stats["cuda_layers"]) == 1
    assert int(dstats["decoded_layers"]) == 1


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    shape = (64, 64, 4, 4)
    _case("zeros", torch.zeros(shape, device="cuda", dtype=torch.float32))
    torch.manual_seed(7)
    _case("normal", torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1)
    sparse = torch.zeros(shape, device="cuda", dtype=torch.float32)
    sparse.reshape(-1)[::17] = 0.25
    _case("sparse", sparse)
    signs = torch.sign(torch.randn(shape, device="cuda", dtype=torch.float32)) * 0.05
    _case("signs", signs)
    print("ok - cuda v2 codec quality")


if __name__ == "__main__":
    main()
