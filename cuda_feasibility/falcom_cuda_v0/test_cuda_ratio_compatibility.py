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


def _two_hot_rounds(codec_mode: str):
    state = FalcomCudaState(client_id=f"ratio_{codec_mode}", config=default_config())
    x0 = torch.ones((64, 64, 8, 8), device="cuda", dtype=torch.float32)
    x1 = (x0 * 1.015 + 0.002).contiguous()
    compress_batch_cuda(
        OrderedDict([("conv.weight", x0)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": codec_mode},
    )
    return compress_batch_cuda(
        OrderedDict([("conv.weight", x1)]),
        state_handle=state,
        options={"cuda_min_numel": 1, "residual_codec": codec_mode},
    )


def main() -> None:
    os.environ["FALCOM_CUDA_EXPERIMENTAL"] = "1"
    raw_layers, raw_stats = _two_hot_rounds("raw_v0")
    probe_layers, probe_stats = _two_hot_rounds("cpu_codec_full_residual_probe")

    raw = raw_layers["conv.weight"]
    probe = probe_layers["conv.weight"]
    assert raw["c_codec"] == "sz3_memcpy"
    assert probe["c_codec"] == "zstd"
    assert probe_stats["codec_mode"] == "cpu_codec_full_residual_probe"
    assert probe_stats["accepted_compact"] == 0
    assert probe_stats["compact_vs_full_residual_pct"] == 100.0
    assert len(probe["data"]) < len(raw["data"])

    decomp = FalComC(default_config())
    out = decomp.decompress_model(
        dumps_compressed_layers(probe_layers),
        client_id="ratio_cpu_decompress",
    )
    assert tuple(out["conv.weight"].shape) == tuple(x for x in (64, 64, 8, 8))
    assert str(out["conv.weight"].dtype).endswith("float32")
    assert torch.isfinite(out["conv.weight"]).all()
    print("ok - full residual probe restores CPU-decompressor-compatible compressed payload")


if __name__ == "__main__":
    main()
