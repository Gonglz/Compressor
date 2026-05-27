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
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    decompress_batch_cuda_v4,
    default_config,
    dumps_cuda_v4_layers,
    loads_cuda_v4_layers,
    layer_key,
)


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(a), min=1e-12))


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    torch.manual_seed(41)
    encoder_state = FalcomCudaV4State(client_id="v4_separate", config=default_config())
    decoder_state = FalcomCudaV4State(client_id="v4_separate", config=default_config())
    base = (torch.randn((64, 64, 4, 4), device="cuda", dtype=torch.float32) * 0.01).contiguous()
    updates = [base, (base * 1.01 + 0.0001).contiguous()]
    decoded_last = None
    for idx, tensor in enumerate(updates):
        layers, _stats = compress_batch_cuda_v4(
            OrderedDict([("conv.weight", tensor)]),
            state_handle=encoder_state,
            options={"cuda_min_numel": 1},
        )
        loaded = loads_cuda_v4_layers(dumps_cuda_v4_layers(layers))
        out, _dstats = decompress_batch_cuda_v4(loaded, state_handle=decoder_state)
        decoded_last = out["conv.weight"]
        assert _relative_l2(tensor, decoded_last) <= 0.03
        key = layer_key("conv.weight", tensor)
        assert encoder_state.steps[key] == idx + 1
        assert decoder_state.steps[key] == idx + 1
        assert encoder_state.prev_grad[key].data_ptr() != decoder_state.prev_grad[key].data_ptr()
    assert decoded_last is not None
    print("ok - cuda v4 separate encoder decoder state")


if __name__ == "__main__":
    main()
