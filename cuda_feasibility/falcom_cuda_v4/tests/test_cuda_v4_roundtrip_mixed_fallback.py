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
    CPU_BATCH_KEY,
    CUDA_BATCH_KEY,
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    decompress_batch_cuda_v4,
    default_config,
    dumps_cuda_v4_layers,
    loads_cuda_v4_layers,
)


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    torch.manual_seed(23)
    big = (torch.randn((64, 64, 4, 4), device="cuda", dtype=torch.float32) * 0.01).contiguous()
    small = torch.randn((16,), device="cuda", dtype=torch.float32).contiguous()
    c_state = FalcomCudaV4State(client_id="v4_mixed", config=default_config())
    d_state = FalcomCudaV4State(client_id="v4_mixed", config=default_config())
    layers, stats = compress_batch_cuda_v4(
        OrderedDict([("big.weight", big), ("small.bias", small)]),
        state_handle=c_state,
        options={"cuda_min_numel": 1024},
    )
    assert CUDA_BATCH_KEY in layers
    assert CPU_BATCH_KEY in layers
    assert int(stats["cuda_layers"]) == 1
    assert int(stats["cpu_fallback_layers"]) == 1
    blob = dumps_cuda_v4_layers(layers)
    loaded = loads_cuda_v4_layers(blob)
    out, dstats = decompress_batch_cuda_v4(loaded, state_handle=d_state)
    assert set(out.keys()) == {"big.weight", "small.bias"}
    assert int(dstats["cuda_layers"]) == 1
    assert int(dstats["cpu_fallback_layers"]) == 1
    assert torch.isfinite(out["big.weight"]).all()
    assert torch.isfinite(out["small.bias"]).all()
    print("ok - cuda v4 mixed CUDA and CPU fallback roundtrip")


if __name__ == "__main__":
    main()
