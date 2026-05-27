#!/usr/bin/env python3
import argparse
import csv
import os
import struct
import subprocess
from pathlib import Path


DTYPE_SIZE = {0: 4, 1: 8, 2: 4, 3: 8, 4: 1}


def read_cstr(buf: bytes) -> str:
    return buf.split(b"\0", 1)[0].decode("utf-8")


def layer_records(path: Path):
    with path.open("rb") as f:
        round_num = struct.unpack("<I", f.read(4))[0]
        client_id = read_cstr(f.read(64))
        layer_count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(layer_count):
            name = read_cstr(f.read(256))
            shape8 = struct.unpack("<8Q", f.read(64))
            ndim = struct.unpack("<Q", f.read(8))[0]
            dtype = struct.unpack("<I", f.read(4))[0]
            data_size = struct.unpack("<Q", f.read(8))[0]
            f.seek(data_size, os.SEEK_CUR)
            shape = tuple(int(x) for x in shape8[:ndim])
            numel = 1
            for dim in shape:
                numel *= dim
            yield {
                "round": round_num,
                "client_id": client_id,
                "layer_name": name,
                "shape": "x".join(str(x) for x in shape),
                "ndim": ndim,
                "dtype": dtype,
                "numel": numel,
                "data_size": data_size,
            }


def choose_layers(records):
    float_records = [r for r in records if r["dtype"] in (0, 1) and r["numel"] > 0]
    ordered = sorted(float_records, key=lambda r: r["numel"])
    if not ordered:
        return []
    idxs = {
        0,
        len(ordered) // 4,
        len(ordered) // 2,
        (3 * len(ordered)) // 4,
        max(0, len(ordered) - 1),
        max(0, len(ordered) - 2),
        max(0, len(ordered) - 3),
    }
    chosen = []
    seen = set()
    for idx in sorted(idxs):
        r = ordered[idx]
        if r["numel"] not in seen:
            chosen.append(r)
            seen.add(r["numel"])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    exe = Path(args.exe).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ("resnet18", "resnet50"):
        data_file = dataset_root / model / "round_0_client_0.bin"
        selected = choose_layers(list(layer_records(data_file)))
        for layer in selected:
            proc = subprocess.run(
                [str(exe), "--numel", str(layer["numel"]), "--iters", str(args.iters)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            reader = csv.DictReader(proc.stdout.splitlines())
            for bench in reader:
                rows.append({
                    "model": model,
                    "layer_name": layer["layer_name"],
                    "shape": layer["shape"],
                    "numel": layer["numel"],
                    **bench,
                })

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
