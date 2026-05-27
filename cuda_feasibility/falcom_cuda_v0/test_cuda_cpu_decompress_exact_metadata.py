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

from appfl.compressor.FalComC import FalComC  # noqa: E402
from falcom_cuda_wrapper import FalcomCudaState, compress_batch_cuda, default_config, dumps_compressed_layers  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    state = FalcomCudaState(client_id="exact_metadata", config=default_config())
    decomp = FalComC(default_config())
    x0 = torch.ones((64, 64, 8, 8), device="cuda", dtype=torch.float32)
    x1 = (x0 * 0.99 - 0.003).contiguous()
    compress_batch_cuda(
        OrderedDict([("conv.weight", x0)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
    )
    layers, _stats = compress_batch_cuda(
        OrderedDict([("conv.weight", x1)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": "hybrid_compact_v1a"},
    )
    layer = layers["conv.weight"]
    assert layer["codec"] == "c_struct"
    assert layer["c_type"] == "momentum_predicted"
    assert layer["c_codec"] == "zstd"
    assert layer["shape"] == (64, 64, 8, 8)
    assert layer["ndim"] == 4
    assert layer["stored_dtype"] == "float32"
    assert layer["step"] == 2
    out = decomp.decompress_model(dumps_compressed_layers(layers), client_id="exact_metadata")
    assert tuple(out["conv.weight"].shape) == layer["shape"]
    assert str(out["conv.weight"].dtype).endswith("float32")
    assert torch.isfinite(out["conv.weight"]).all()
    print("ok - hybrid compact payload keeps CPU decompressor metadata contract")


if __name__ == "__main__":
    main()
