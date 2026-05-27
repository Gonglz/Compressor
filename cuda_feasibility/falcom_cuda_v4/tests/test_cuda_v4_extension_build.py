#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "cuda_feasibility" / "falcom_cuda_v4"
APPFL = ROOT / "EB-FaLCom" / "src"
sys.path.insert(0, str(V4))
sys.path.insert(0, str(APPFL))

import _falcom_cuda_v4  # noqa: E402
from falcom_cuda_v4_wrapper import extension_available  # noqa: E402


def main() -> None:
    assert _falcom_cuda_v4.is_cuda_build() is True
    assert extension_available() is True
    print("ok - cuda v4 extension build")


if __name__ == "__main__":
    main()
