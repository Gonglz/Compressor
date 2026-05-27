#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "cuda_feasibility" / "falcom_cuda_v3"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(APPFL))

from falcom_cuda_v3_wrapper import CPU_BATCH_KEY, FalcomCudaV3State, compress_batch_cuda_v3, decompress_batch_cuda_v3, default_config  # noqa: E402


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    c_state = FalcomCudaV3State(client_id="mixed_roundtrip", config=default_config())
    d_state = FalcomCudaV3State(client_id="mixed_roundtrip", config=default_config())
    big0 = torch.linspace(-0.5, 0.5, 64 * 64 * 4 * 4, device="cuda", dtype=torch.float32).reshape(64, 64, 4, 4)
    small0 = torch.ones((8,), device="cuda", dtype=torch.float32)
    layers, stats = compress_batch_cuda_v3(
        OrderedDict([("big.weight", big0.contiguous()), ("small.bias", small0)]),
        state_handle=c_state,
        options={"cuda_min_numel": 1024, "quant_bits": 8},
    )
    out, dstats = decompress_batch_cuda_v3(layers, state_handle=d_state)
    assert int(stats["cuda_layers"]) == 1
    assert int(stats["cpu_fallback_layers"]) == 1
    assert CPU_BATCH_KEY in layers
    assert layers[CPU_BATCH_KEY]["codec"] == "cpu_fallback_batch_pickle"
    assert int(dstats["decoded_layers"]) == 1
    assert tuple(out["big.weight"].shape) == tuple(big0.shape)
    assert torch.isfinite(out["big.weight"]).all()
    assert tuple(out["small.bias"].shape) == tuple(small0.shape)
    print("ok - cuda v3 mixed fallback roundtrip")


if __name__ == "__main__":
    main()
