# Microway Compressor GPU Experiment Summary

## Location

- Workdir: `/scratch2/lgong1/Compressor`
- Project: `/scratch2/lgong1/Compressor/compressor_gpu_experiment`
- Env: `/scratch2/lgong1/Compressor/envs/falcom_gpu`
- GPU used: `CUDA_VISIBLE_DEVICES=2` -> one `Tesla V100S-PCIE-32GB`

## Setup Status

- Transfer archive sha256 verified: `c8c6b3b13702053c6e9db72734f72c2215bd0de0bbff57aeb294d22616ef517c`
- PyTorch CUDA verified: `torch 2.7.1+cu126`, `cuda_available=True`, `device_count=1`
- APPFL deps available: `appfl`, `omegaconf`, `blosc`
- SZ3 rebuilt locally for Ubuntu 20.04/glibc 2.31: `/scratch2/lgong1/Compressor/deps/SZ3_local`
- Final CPU library rebuilt locally:
  - `libmomentum_compressor_openmp_simd_final.so`
  - `EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so`
  - sha256 both: `fba66a70ea2f4fb0f6bbfdd4c9eedd3fa94ebd0d2704e1087fe4fb97b4df4b23`

## Correctness Gates

Passed:

- `python3 test_static_contracts.py`
- `gcc -fsyntax-only momentum_compressor_openmp_simd_final.c`
- `bin/test_batch_state_equivalence_local`
- `bin/test_reference_state_oracle_local`
- `ctypes.CDLL` local and APPFL installed `.so`

Log: `/scratch2/lgong1/Compressor/logs/correctness_gates_after_local_rebuild.log`

## APPFL Data Residency Profiling

ResNet18 round0, 101 layers, 42.66 MB original:

| Input mode | model_total_ms | GPU->CPU numpy ms | ctypes ms | C compress ms | payload copy ms |
|---|---:|---:|---:|---:|---:|
| numpy CPU | 323.868 | 0.000 | 1.195 | 310.631 | 3.761 |
| torch CPU | 303.459 | 1.140 | 1.493 | 290.474 | 3.031 |
| torch CUDA | 323.091 | 15.083 | 1.836 | 294.274 | 3.335 |

ResNet50 round0 CUDA tensor, 266 layers, 89.93 MB original:

| Input mode | model_total_ms | GPU->CPU numpy ms | ctypes ms | C compress ms | payload copy ms |
|---|---:|---:|---:|---:|---:|
| torch CUDA | 739.525 | 38.502 | 4.619 | 663.696 | 6.732 |

CSV outputs:

- `/scratch2/lgong1/Compressor/appfl_data_residency.csv`
- `/scratch2/lgong1/Compressor/appfl_data_residency_resnet50_cuda.csv`

## CUDA Microbench

Standalone kernels compiled with CUDA 11.6 / `sm_70`:

- `momentum_update`
- `residual_predmem_update`
- `threshold_scan`
- `bitmap_pack`
- `dominant_sign_pack`
- `decompress_reconstruct`

Largest representative layer, `numel=2359296`:

| Kernel | CPU ms | GPU-only ms | H2D+GPU+D2H ms | GPU-only speedup | Transfer-inclusive speedup |
|---|---:|---:|---:|---:|---:|
| momentum_update | 1.6150 | 0.0320 | 6.4951 | 50.5x | 0.25x |
| residual_predmem_update | 2.6756 | 0.0421 | 6.4983 | 63.6x | 0.41x |
| threshold_scan | 2.0723 | 0.0193 | 2.6728 | 107.3x | 0.78x |
| bitmap_pack | 6.5086 | 0.0053 | 0.6446 | 1218.8x | 10.10x |
| dominant_sign_pack | 6.5496 | 0.0209 | 2.1081 | 313.2x | 3.11x |
| decompress_reconstruct | 1.5613 | 0.0317 | 6.4878 | 49.2x | 0.24x |

Full CSV: `/scratch2/lgong1/Compressor/cuda_microbench_results.csv`
Summary log: `/scratch2/lgong1/Compressor/logs/cuda_microbench_summary.log`

## Initial Conclusion

- GPU-only kernels are much faster for large tensors, especially packing/sign packing.
- If data must move CPU<->GPU per layer, momentum/update/reconstruct kernels lose to CPU because transfer dominates.
- Bitmap/sign packing remains promising even with transfer, but integrating only packing would require careful wire-format/state-compatible design.
- APPFL CUDA tensor profiling shows GPU->CPU copy is measurable but not the dominant cost in the current Python wrapper replay: ResNet50 CUDA copy is ~38.5 ms vs C compression ~663.7 ms.
- The next worthwhile experiment is a device-resident fused CUDA path that avoids per-layer H2D/D2H, not a host-pointer CUDA wrapper.

## Activation Commands

```bash
ssh lgong1@155.246.227.35
source /scratch2/lgong1/Compressor/envs/falcom_gpu/bin/activate
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/scratch2/lgong1/Compressor/compressor_gpu_experiment/EB-FaLCom/src:$PYTHONPATH
export LD_LIBRARY_PATH=/scratch2/lgong1/Compressor/deps/SZ3_local/lib:$LD_LIBRARY_PATH
cd /scratch2/lgong1/Compressor/compressor_gpu_experiment
```
