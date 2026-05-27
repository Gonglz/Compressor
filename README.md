# FalCom Compressor: Python/C/CPU/GPU Acceleration Report Bundle

CPU/GPU acceleration experiments for a federated learning compressor. This repository publishes the source code, tests, environment manifests, key CSV outputs, and report material from the FalCom compressor optimization work. Raw replay binaries, compiled artifacts, virtual environments, local caches, and transfer archives are intentionally excluded.

## Summary

The project has four implementation lines:

| Area | Goal | Key files | Result |
|---|---|---|---|
| Python integration | Connect the C compressor to the APPFL/Python side and measure wrapper overhead. | `EB-FaLCom/src/appfl/compressor/FalComC.py`, `appfl_data_residency*.csv` | Python and `ctypes` overhead is small; the main cost is C compression and data movement. |
| C baseline | Keep an auditable fallback implementation and wire-format reference. | `momentum_compressor_final.c`, `test_runtime_contracts.c` | Functional baseline and ABI/wire-format reference, not the fastest path. |
| CPU acceleration | Use OpenMP/SIMD grouped batch compression as the production/default acceleration path. | `momentum_compressor_openmp_simd_final.c`, `reports/final_audit_report.md` | ResNet50 8-thread safe path reaches `3.77x` over safe serial; ResNet18 reaches `2.63x`. |
| GPU feasibility | Explore a CUDA v4 q8 closed-loop codec as an experimental path. | `cuda_feasibility/falcom_cuda_v4/`, `reports/final_cuda_v4_selected_path.md` | ResNet50 guarded all-CUDA closed-loop median is `44.865 ms`, with quality checks passing. |

The CPU grouped OpenMP/SIMD implementation is the default path. CUDA v4 is an experimental closed-loop codec and does not replace the CPU shared library, CPU ABI, or CPU wire format.

## Report Entry Points

- [Deliverables index](DELIVERABLES.md)
- [Python-to-C integration](docs/py-to-c-integration.md)
- [C baseline](docs/c-baseline.md)
- [CPU OpenMP/SIMD acceleration](docs/cpu-acceleration.md)
- [GPU CUDA v4 acceleration](docs/gpu-acceleration.md)
- [Performance comparison](docs/performance-comparison.md)
- [HPC interview evidence](docs/interview-hpc-evidence.md)

## Performance Overview

### Python-to-C profiling

These measurements profile the APPFL/Python wrapper path calling the C compressor through `FalComC.py`. They isolate Python wrapping, `ctypes` marshalling, GPU tensor to CPU NumPy transfer, C compression, and payload-copy overhead.

| Model/input | Layers | Raw size | Model total | GPU->CPU NumPy | `ctypes` | C compress | Payload copy | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet18 NumPy CPU | 101 | 42.66 MB | 323.868 ms | 0.000 ms | 1.195 ms | 310.631 ms | 3.761 ms | `ctypes` is not the bottleneck. |
| ResNet18 Torch CPU | 101 | 42.66 MB | 303.459 ms | 1.140 ms | 1.493 ms | 290.474 ms | 3.031 ms | Python wrapping overhead is small. |
| ResNet18 Torch CUDA | 101 | 42.66 MB | 323.091 ms | 15.083 ms | 1.836 ms | 294.274 ms | 3.335 ms | CUDA tensors add GPU-to-CPU copy cost. |
| ResNet50 Torch CUDA | 266 | 89.93 MB | 739.525 ms | 38.502 ms | 4.619 ms | 663.696 ms | 6.732 ms | The main bottleneck is still C compression. |

Sources: `MIGRATION_EXPERIMENT_SUMMARY.md`, `appfl_data_residency.csv`, and `appfl_data_residency_resnet50_cuda.csv`.

### C baseline and CPU grouped acceleration

These results compare the safe serial C baseline and safe grouped OpenMP/SIMD implementation from the same audit scan. The 8-thread numbers are the recommended reporting point because they were the stable best path in this experiment.

| Model | C safe serial compress | CPU grouped compress | CPU grouped decompress | Compression ratio | Grouped vs serial |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 797.62 ms | 211.36 ms | 273.43 ms | 62.44:1 | 3.77x |
| ResNet18 | 337.84 ms | 128.66 ms | 118.81 ms | 220.71:1 | 2.63x |

Source: `reports/final_audit_report.md`.

The installed shared-library logging gate also reported:

| Model | Grouped 8-thread hot compress | Release baseline | Improvement | Gate |
|---|---:|---:|---:|---|
| ResNet50 | 197.865 ms | 208.325 ms | 5.02% | PASS |
| ResNet18 | 122.189 ms | 125.641 ms | 2.75% | MISS strict threshold, no regression |

Source: `reports/final_cpu_logging_gate_gpu_feasibility_report.md`.

### GPU CUDA v4 final path

The final CUDA v4 selected path is `cuda_v4_q8 + guarded_all_cuda`. It is an experimental closed-loop codec, so the timing is GPU feasibility evidence rather than a drop-in replacement claim for the production CPU wire format.

| Model | Closed-loop median | Compress median | Decompress median | Relative L2 | Finite | Ratio retention | CUDA/fallback layers | Payload objects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 44.865 ms | 22.550 ms | 22.635 ms | 0.019808 | 1.000 | 1.063 | 266 / 0 | 1 |
| ResNet18 | 20.808 ms | 10.237 ms | 10.455 ms | 0.020187 | 1.000 | 1.047 | 101 / 0 | 1 |

Sources: `logs/cuda_v4_guarded_all_cuda/final_guarded_all_cuda_report.md` and `logs/cuda_v4_guarded_all_cuda/quality.csv`.

### CUDA microbenchmarks

On a representative large layer with `numel=2359296`, GPU-only kernels are much faster, while transfer-inclusive timing shows that packing and sign packing are the clear GPU-friendly operations.

| Kernel | CPU ms | GPU-only ms | H2D+GPU+D2H ms | GPU-only speedup | Transfer-inclusive speedup |
|---|---:|---:|---:|---:|---:|
| Momentum update | 1.6150 | 0.0320 | 6.4951 | 50.5x | 0.25x |
| Residual/prediction-memory update | 2.6756 | 0.0421 | 6.4983 | 63.6x | 0.41x |
| Threshold scan | 2.0723 | 0.0193 | 2.6728 | 107.3x | 0.78x |
| Bitmap pack | 6.5086 | 0.0053 | 0.6446 | 1218.8x | 10.10x |
| Dominant-sign pack | 6.5496 | 0.0209 | 2.1081 | 313.2x | 3.11x |
| Decompress reconstruct | 1.5613 | 0.0317 | 6.4878 | 49.2x | 0.24x |

Sources: `MIGRATION_EXPERIMENT_SUMMARY.md` and `cuda_microbench_results.csv`.

## Repository Structure

| Path | Contents |
|---|---|
| `EB-FaLCom/src/appfl/compressor/` | APPFL/FalCom Python wrappers, including `FalComC.py`. |
| `momentum_compressor_final.c` | C baseline. |
| `momentum_compressor_openmp_simd_final.c` | Safe grouped OpenMP/SIMD CPU production path. |
| `cuda_feasibility/falcom_cuda_v0/` to `falcom_cuda_v4/` | CUDA experiment evolution, wrappers, benchmarks, and tests. |
| `logs/cuda_v4_guarded_all_cuda/` | CUDA v4 final timing, quality, and layer-backend evidence. |
| `reports/` | CPU audit, logging gate, GPU feasibility, and selected CUDA path reports. |
| `dataset/*.md` | Replay dataset notes without raw `.bin` data. |
| `env/` | Conda, pip, CUDA, and host environment manifests. |
| `docs/` | Report sections used for writeups. |

## Reproduction Notes

CPU default path:

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash build_openmp_simd_final.sh
```

CUDA v4 experimental path:

```bash
export FALCOM_CUDA_EXPERIMENTAL=1
export FALCOM_CUDA_V4_GUARDED_ALL_CUDA=1
```

Without `FALCOM_CUDA_EXPERIMENTAL`, the system should remain on the CPU fallback path.

## Excluded Data

- Raw replay data under `dataset/**/*.bin`.
- Compiled CPU/CUDA shared libraries, binaries, and object files.
- Virtual environments.
- Transfer archives, backup archives, local caches, and temporary logs.

The repository is intended to keep source code, experiment flow, evidence files, and report material readable on GitHub.
