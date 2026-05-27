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
    ENVELOPE_CODEC,
    ENVELOPE_VERSION,
    FalcomCudaV2State,
    compress_batch_cuda_v2,
    decompress_batch_cuda_v2,
    default_config,
    dumps_cuda_v2_layers,
    loads_cuda_v2_layers,
)


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    c_state = FalcomCudaV2State(client_id="serialized", config=default_config())
    d_state = FalcomCudaV2State(client_id="serialized", config=default_config())
    x = torch.randn((64, 64, 4, 4), device="cuda", dtype=torch.float32) * 0.01
    layers, _stats = compress_batch_cuda_v2(
        OrderedDict([("conv.weight", x.contiguous())]),
        state_handle=c_state,
        options={"cuda_min_numel": 1},
    )
    blob = dumps_cuda_v2_layers(layers)
    loaded = loads_cuda_v2_layers(blob)
    assert isinstance(loaded, OrderedDict)
    out, dstats = decompress_batch_cuda_v2(loaded, state_handle=d_state)
    assert int(dstats["decoded_layers"]) == 1
    assert tuple(out["conv.weight"].shape) == tuple(x.shape)
    assert torch.isfinite(out["conv.weight"]).all()
    assert loaded["conv.weight"]["codec_version"] == 1
    print(f"ok - {ENVELOPE_CODEC} v{ENVELOPE_VERSION} serialized roundtrip")


if __name__ == "__main__":
    main()
