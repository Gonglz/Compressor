# CPU Logging Gate + GPU Feasibility Report

## Patch Summary

Implemented the release logging gate and APPFL profiling instrumentation.

- legacy `[BREAKDOWN]` default: off
- grouped `[BATCH_BREAKDOWN]` default: off
- default compressor log level: `WARNING`
- `FALCOM_LEGACY_BREAKDOWN=1`: legacy breakdown only
- `FALCOM_BATCH_BREAKDOWN=1`: grouped batch breakdown only
- `FALCOM_LOG_LEVEL=INFO|DEBUG`: batch INFO/DEBUG logging only
- `momentum_compressor_enable_breakdown(1)`: legacy API behavior preserved
- APPFL profiling added behind `FALCOM_APPFL_PROFILE=1`

The second timing-gate fix also disables default `clock_gettime()` collection for legacy/batch timing fields. Timers are re-enabled only when legacy or batch breakdown is explicitly active.

## Hashes

Old installed hash before this task: `949d2c7499d9a4de8202f9ab9d51532f547962585c577b8d7fb4d0a1dcf96fb2`

Current hashes:

```text
97818e72bcbcd3e64b4b8081ad3c2649c2f56fe4deef345c7792c2c7f7696f23  momentum_compressor_openmp_simd_final.c
2c313e6830725f97fc7e6386aea60fdf3df0252c80d77a9ea5836b5c161467f2  libmomentum_compressor_openmp_simd_final.so
2c313e6830725f97fc7e6386aea60fdf3df0252c80d77a9ea5836b5c161467f2  EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so
a891bb7d4a033434a2077dbe4963e71307fa5a153831791b91f48cd8a2e61ae1  test_static_contracts.py
d8b40104475aa24032e234a3bee2da213bc8a10ce241f4230ff838a81db3e820  EB-FaLCom/src/appfl/compressor/FalComC.py
```

Local and installed `.so` hashes match: `2c313e6830725f97fc7e6386aea60fdf3df0252c80d77a9ea5836b5c161467f2`.

## Correctness Gates

- Static contracts: PASS (`21 static contract checks passed`)
- `gcc -fsyntax-only`: PASS
- Runtime contracts: PASS
- Batch state equivalence: PASS
- Reference state oracle: PASS
- ASAN/UBSAN runtime contract: PASS
- ctypes load local/installed `.so`: PASS
- Signature equivalence default vs disabled API vs legacy env vs batch env: PASS

Logs:

- `/home/exouser/compressor/final/logs/cpu_logging_gate_20260525_040946/gates/correctness_after_timing_gate.log`
- `/home/exouser/compressor/final/logs/cpu_logging_gate_20260525_040946/sanitizer/runtime_asan_after_timing_gate.log`
- `/home/exouser/compressor/final/logs/cpu_logging_gate_20260525_040946/signature/equivalence.txt`

## Behavior Gates

- Default run prints no legacy `[BREAKDOWN]`, no `[BATCH_BREAKDOWN]`, no batch INFO lines.
- `FALCOM_LEGACY_BREAKDOWN=1` prints legacy `[BREAKDOWN]` only.
- `FALCOM_BATCH_BREAKDOWN=1` prints `[BATCH_BREAKDOWN]` only.
- `FALCOM_LOG_LEVEL=INFO` prints batch INFO without forcing breakdown output.
- `momentum_compressor_enable_breakdown(1)` still prints legacy `[BREAKDOWN]`.

Behavior logs are under `/home/exouser/compressor/final/logs/cpu_logging_gate_20260525_040946/behavior`.

## Official Installed Benchmark

# CPU Logging Gate Benchmark Summary

Official release median source: installed .

| model | mode | threads | hot compress median ms | hot compress runs ms | hot decompress median ms | full ratio median | vs baseline |
|---|---:|---:|---:|---|---:|---:|---:|
| resnet18 | grouped | 1 | 432.276 | 466.799, 428.730, 435.575, 430.301, 432.276 | 109.961 | 220.714:1 |  |
| resnet18 | grouped | 2 | 247.588 | 229.219, 240.431, 247.588, 250.541, 248.925 | 116.568 | 220.714:1 |  |
| resnet18 | grouped | 4 | 146.246 | 147.838, 141.965, 139.148, 146.246, 148.041 | 120.434 | 220.714:1 |  |
| resnet18 | grouped | 8 | 122.189 | 120.602, 123.547, 118.087, 122.189, 122.815 | 125.354 | 220.714:1 | 2.75% |
| resnet18 | safe_serial | 1 | 340.294 | 344.265, 319.989, 336.691, 340.294, 340.728 | 114.928 | 220.714:1 |  |
| resnet18 | safe_serial | 2 | 324.127 | 324.127, 321.144, 333.807, 329.779, 318.621 | 117.512 | 220.714:1 |  |
| resnet18 | safe_serial | 4 | 323.657 | 323.657, 305.096, 324.465, 319.150, 325.158 | 118.685 | 220.714:1 |  |
| resnet18 | safe_serial | 8 | 346.072 | 333.835, 345.963, 346.072, 348.117, 347.216 | 128.909 | 220.714:1 |  |
| resnet50 | grouped | 1 | 1065.170 | 1155.193, 1077.119, 869.245, 1065.170, 889.340 | 248.897 | 62.436:1 |  |
| resnet50 | grouped | 2 | 531.458 | 565.470, 556.905, 506.069, 531.458, 466.585 | 261.913 | 62.436:1 |  |
| resnet50 | grouped | 4 | 313.849 | 346.460, 313.849, 293.850, 320.964, 283.778 | 267.179 | 62.436:1 |  |
| resnet50 | grouped | 8 | 197.865 | 223.670, 218.363, 185.620, 197.865, 165.846 | 273.707 | 62.436:1 | 5.02% |
| resnet50 | safe_serial | 1 | 957.909 | 973.417, 957.909, 880.767, 965.447, 862.175 | 266.412 | 62.436:1 |  |
| resnet50 | safe_serial | 2 | 945.765 | 953.909, 955.800, 851.949, 945.765, 852.264 | 268.233 | 62.436:1 |  |
| resnet50 | safe_serial | 4 | 842.451 | 854.172, 842.451, 762.940, 864.267, 758.160 | 269.550 | 62.436:1 |  |
| resnet50 | safe_serial | 8 | 774.179 | 791.044, 894.356, 704.683, 774.179, 687.858 | 296.311 | 62.436:1 |  |

- resnet50 grouped 8T: 197.865 ms vs baseline 208.325 ms, improvement 5.02%, acceptance=PASS.
- resnet18 grouped 8T: 122.189 ms vs baseline 125.641 ms, improvement 2.75%, acceptance=MISS.


Acceptance notes:

- ResNet50 grouped 8T passed the strict CPU logging gate target: `197.865 ms`, `5.02%` faster than the `208.325 ms` baseline.
- ResNet18 grouped 8T improved to `122.189 ms`, but did not meet the strict `<=118 ms` or `>=4%` threshold. This patch remains acceptable as release cleanup because diagnostic logging/timing must not be default-on, correctness passed, ratio stayed unchanged, and ResNet18 did not regress.
- Full compression ratios remained unchanged within the required `<=1%` relative delta: ResNet50 `62.436:1`, ResNet18 `220.714:1`.

## GPU Feasibility

CUDA microbench was not run on this host.

- `nvidia-smi`: present but cannot communicate with a running NVIDIA driver.
- `nvcc`: unavailable.
- audit Python lacks `torch`, so PyTorch CUDA probing is blocked.

GPU files:

- `/home/exouser/compressor/final/logs/gpu_feasibility_20260525_042142/cuda_probe.txt`
- `/home/exouser/compressor/final/logs/gpu_feasibility_20260525_042142/cuda_hardware_blocked.md`
- `/home/exouser/compressor/final/logs/gpu_feasibility_20260525_042142/appfl_data_residency.csv`
- `/home/exouser/compressor/final/logs/gpu_feasibility_20260525_042142/appfl_data_residency.md`

APPFL source audit confirms that torch-like tensors go through `array.detach().cpu().numpy()` before ctypes conversion, so CUDA tensors would incur GPU-to-CPU copy in the current host-pointer path. The new profiling instrumentation will measure this with CUDA synchronization when run in a real APPFL environment with torch/CUDA available.

## Freeze Recommendation

Freeze the CPU safe grouped default path after this patch, except for bug fixes. The only recommended next step is to run `FALCOM_APPFL_PROFILE=1` inside a real APPFL training/replay environment with the actual Python dependencies and, if available, CUDA tensors. Do not start CUDA compressor implementation until that confirms GPU-to-CPU copy or Python/ctypes marshalling is a meaningful end-to-end bottleneck.
