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

from falcom_cuda_v4_wrapper import (  # noqa: E402
    CUDA_BATCH_KEY,
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    decompress_batch_cuda_v4,
    default_config,
)


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(a), min=1e-12))


def _case(name: str, tensor: torch.Tensor) -> None:
    c_state = FalcomCudaV4State(client_id=f"v4_quality_{name}", config=default_config())
    d_state = FalcomCudaV4State(client_id=f"v4_quality_{name}", config=default_config())
    layers, stats = compress_batch_cuda_v4(
        OrderedDict([(f"{name}.weight", tensor.contiguous())]),
        state_handle=c_state,
        options={"cuda_min_numel": 1},
    )
    assert list(layers.keys()) == [CUDA_BATCH_KEY]
    assert int(stats["cuda_layers"]) == 1
    assert int(stats["num_payload_objects"]) == 1
    assert int(stats["payload_blob_bytes"]) == tensor.numel()
    out, dstats = decompress_batch_cuda_v4(layers, state_handle=d_state)
    decoded = out[f"{name}.weight"]
    assert int(dstats["cuda_layers"]) == 1
    assert tuple(decoded.shape) == tuple(tensor.shape)
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    assert _relative_l2(tensor, decoded) <= 0.02


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    shape = (64, 64, 4, 4)
    _case("zeros", torch.zeros(shape, device="cuda", dtype=torch.float32))
    torch.manual_seed(17)
    _case("normal", torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1)
    sparse = torch.zeros(shape, device="cuda", dtype=torch.float32)
    sparse.reshape(-1)[::17] = 0.25
    _case("sparse", sparse)
    signs = torch.sign(torch.randn(shape, device="cuda", dtype=torch.float32)) * 0.05
    _case("signs", signs)
    print("ok - cuda v4 q8 codec quality")


if __name__ == "__main__":
    main()
