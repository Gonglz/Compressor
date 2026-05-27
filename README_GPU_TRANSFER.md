# Compressor GPU Experiment Transfer Bundle

This bundle contains the audited CPU safe grouped compressor, APPFL integration code, real ResNet18/ResNet50 gradient datasets, correctness tests, CPU baseline binaries, and GPU feasibility reports.

## What to Copy to the New GPU Server

Copy the archive to the GPU server and unpack it:

```bash
tar --zstd -xf compressor_gpu_experiment_*.tar.zst
cd compressor_gpu_experiment
```

If `tar --zstd` is unavailable:

```bash
zstd -dc compressor_gpu_experiment_*.tar.zst | tar -xf -
```

## Recommended Environment

Create or reuse a conda env compatible with `env/conda_falcom_environment.yml`. The original working env was named `falcom` and used PyTorch `2.9.1+cu128`.

```bash
conda env create -n falcom_gpu -f env/conda_falcom_environment.yml
conda activate falcom_gpu
export PYTHONPATH=$PWD/EB-FaLCom/src:$PYTHONPATH
```

If the exact env solve is difficult, install the essentials manually: Python 3.10, numpy, torch with CUDA, zstd development/runtime libraries, gcc, OpenMP, and appfl dependencies from `EB-FaLCom`.

## Sanity Checks

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

CPU correctness gates:

```bash
python3 test_static_contracts.py
gcc -std=c99 -O3 -march=native -ffast-math -fopenmp -Wall -Wextra -I. -fsyntax-only momentum_compressor_openmp_simd_final.c
./bin/test_batch_state_equivalence
./bin/test_reference_state_oracle
```

Build the CPU compressor on the new host if needed:

```bash
bash build_openmp_simd_final.sh
cp libmomentum_compressor_openmp_simd_final.so EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so
```

## APPFL Data Residency Profiling

The transferred `FalComC.py` supports env-gated profiling:

```bash
export FALCOM_APPFL_PROFILE=1
export FALCOM_APPFL_PROFILE_CSV=$PWD/appfl_data_residency.csv
```

It records tensor type/device/dtype/shape/numel, `.detach().cpu().numpy()` time, ctypes build time, C call time, payload-copy time, and per-layer/model totals. CUDA tensors are synchronized around CPU-copy timing.

## GPU Experiment Scope

Do not replace the CPU default path initially. First run feasibility:

1. Confirm whether APPFL gradients are CUDA tensors or already CPU/numpy.
2. Measure GPU-to-CPU copy and ctypes marshalling cost.
3. Implement standalone CUDA microbenchmarks for hot kernels only.
4. Compare GPU-only and H2D+GPU+D2H against the CPU safe grouped baseline.

GPU path should only become a serious candidate if ResNet50 beats the CPU logging-off safe path by at least 20%, keeps compression ratio delta <= 1%, preserves state equivalence, and improves APPFL end-to-end round time.

## Important Reports

See `reports/` for the CPU logging gate result, current GPU blocked diagnosis, and audit report.
