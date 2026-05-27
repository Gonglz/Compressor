#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import pickle
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
    state = FalcomCudaState(client_id="cuda_state_equiv", config=default_config())
    decomp = FalComC(default_config())
    decomp.set_client_id("cuda_state_equiv")

    base = torch.linspace(-1.0, 1.0, 64 * 64 * 3 * 3, device="cuda", dtype=torch.float32).reshape(64, 64, 3, 3)
    round0 = OrderedDict([("convA.weight", base.contiguous())])
    round1 = OrderedDict([("convA.weight", (base * 1.01 + 0.001).contiguous())])

    c0, s0 = compress_batch_cuda(round0, state_handle=state, options={"cuda_min_numel": 1})
    c1, s1 = compress_batch_cuda(round1, state_handle=state, options={"cuda_min_numel": 1})
    assert c0["convA.weight"]["c_type"] == "direct"
    assert c1["convA.weight"]["c_type"] == "momentum_predicted"
    assert c1["convA.weight"]["step"] == 2
    assert c1["convA.weight"]["num_predicted_kernels"] > 0

    d0 = decomp.decompress_model(dumps_compressed_layers(c0), client_id="cuda_state_equiv")
    d1 = decomp.decompress_model(dumps_compressed_layers(c1), client_id="cuda_state_equiv")
    assert tuple(d0["convA.weight"].shape) == tuple(round0["convA.weight"].shape)
    assert tuple(d1["convA.weight"].shape) == tuple(round1["convA.weight"].shape)
    assert str(d0["convA.weight"].dtype).endswith("float32")
    assert str(d1["convA.weight"].dtype).endswith("float32")
    assert math.isfinite(float(d1["convA.weight"].mean()))
    assert s1["cuda_layers"] == 1
    print("ok - cuda state equivalence/decompress compatibility")


if __name__ == "__main__":
    main()
