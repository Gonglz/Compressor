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
    ENVELOPE_MAGIC,
    ENVELOPE_VERSION,
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    decompress_batch_cuda_v4,
    default_config,
    dumps_cuda_v4_layers,
    loads_cuda_v4_layers,
)


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    c_state = FalcomCudaV4State(client_id="v4_serialized", config=default_config())
    d_state = FalcomCudaV4State(client_id="v4_serialized", config=default_config())
    x = (torch.randn((64, 64, 4, 4), device="cuda", dtype=torch.float32) * 0.01).contiguous()
    layers, _stats = compress_batch_cuda_v4(
        OrderedDict([("conv.weight", x)]),
        state_handle=c_state,
        options={"cuda_min_numel": 1},
    )
    blob = dumps_cuda_v4_layers(layers)
    assert blob.startswith(ENVELOPE_MAGIC)
    loaded = loads_cuda_v4_layers(blob)
    assert isinstance(loaded, OrderedDict)
    assert loaded[CUDA_BATCH_KEY]["codec_version"] == ENVELOPE_VERSION
    assert "_last_envelope_parse_ms" in loaded
    out, dstats = decompress_batch_cuda_v4(loaded, state_handle=d_state)
    assert int(dstats["cuda_layers"]) == 1
    assert tuple(out["conv.weight"].shape) == tuple(x.shape)
    assert torch.isfinite(out["conv.weight"]).all()
    print("ok - cuda v4 packed envelope serialized roundtrip")


if __name__ == "__main__":
    main()
