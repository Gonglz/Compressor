# Deliverables Map

This repository contains the requested delivery tracks: PY integration, C baseline, CPU acceleration, and GPU acceleration. Large local-only artifacts such as replay `.bin` files, compiled binaries, virtual environments, and transfer archives are intentionally excluded.

## 1. PY To C Integration

- `EB-FaLCom/src/appfl/compressor/FalCom.py`: original Python-facing compressor path.
- `EB-FaLCom/src/appfl/compressor/FalComC.py`: ctypes bridge into the C compressor, including profiling hooks.
- `EB-FaLCom/C_INTEGRATION.md`: integration notes for the C-backed APPFL compressor.
- `docs/py-to-c-integration.md`: report-ready explanation of the Python wrapper, ctypes bridge, profiling hooks, and observed overhead.

## 2. C Baseline

- `momentum_compressor_final.c`: baseline C compressor implementation.
- `test_static_contracts.py`, `test_runtime_contracts.c`, `test_reference_state_oracle.c`: contract and correctness tests.
- `docs/c-baseline.md`: C baseline semantics, ABI boundary, state ownership, and report wording.

## 3. CPU Acceleration

- `momentum_compressor_openmp_simd_final.c`: production/default OpenMP/SIMD grouped CPU compressor.
- `simd_helpers_v23.h`, `momentum_compressor.h`, `ompv22sz3.c`: CPU implementation support files.
- `build_openmp_simd_final.sh`: CPU library build script.
- `benchmark_threads.sh`, `run_benchmark_threads.sh`, `benchmark_real.c`, `benchmark_decomp.py`: CPU benchmark and replay tooling.
- `reports/final_cpu_logging_gate_gpu_feasibility_report.md`: CPU baseline and logging-gate evidence.
- `docs/cpu-acceleration.md`: detailed CPU grouped OpenMP/SIMD design, correctness gates, and performance tables.

## 4. GPU Acceleration

- `cuda_feasibility/falcom_cuda_microbench.cu`: standalone CUDA hot-kernel microbenchmark.
- `cuda_feasibility/falcom_cuda_v0/` through `cuda_feasibility/falcom_cuda_v4/`: CUDA experiment generations, wrappers, setup files, benchmarks, and tests.
- `cuda_feasibility/falcom_cuda_v4/run_cuda_v4_guarded_all_cuda.py`: final guarded all-CUDA experiment driver.
- `reports/final_cuda_v4_selected_path.md`: selected CUDA v4 path and final conclusion.
- `logs/cuda_v4_guarded_all_cuda/`: final CUDA quality/timing/backend evidence.
- `cuda_microbench_results.csv`: GPU kernel microbenchmark result table.
- `docs/gpu-acceleration.md`: detailed CUDA microbench, v4 final path, quality gates, and boundary notes.

## Performance Comparison

- `README.md`: concise root-level PY/C/CPU/GPU summary and performance tables.
- `docs/performance-comparison.md`: report-ready comparison tables and wording guidance.

## Supporting Evidence

- `MIGRATION_EXPERIMENT_SUMMARY.md`: server setup, correctness gates, CPU/GPU profiling summary, and activation commands.
- `README_GPU_TRANSFER.md`: transfer-bundle usage notes and sanity checks.
- `dataset/*.md`: dataset documentation only; large replay `.bin` files are not uploaded.
- `appfl_data_residency.csv` and `appfl_data_residency_resnet50_cuda.csv`: APPFL tensor residency profiling.
- `env/`: conda, pip, CUDA, and host environment manifests.
