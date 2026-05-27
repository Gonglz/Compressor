# FalCom Compressor: PY / C / CPU / GPU Acceleration Report Bundle

这个仓库是 `/scratch2/lgong1/Compressor/compressor_gpu_experiment` 的公开发布快照，面向后续论文/课程报告整理。仓库保留源码、测试、环境清单、关键 CSV 和结论报告；原始 replay `.bin`、编译产物、虚拟环境、本地缓存和压缩包没有上传。

## 一句话结论

本项目完成了四条交付线：

| 方向 | 目标 | 关键文件 | 结论 |
|---|---|---|---|
| PY | 在 APPFL/Python 侧接入 C compressor，并测量 Python 到 C 的桥接开销 | `EB-FaLCom/src/appfl/compressor/FalComC.py`, `appfl_data_residency*.csv` | Python/ctypes 桥接开销很小；当前主要耗时在 C compression 和数据搬运 |
| C | 保留可审计、可回退的 C baseline | `momentum_compressor_final.c`, `test_runtime_contracts.c` | 作为功能 baseline 和 ABI/wire-format 参考，不作为最快路径 |
| CPU | 用 OpenMP/SIMD grouped batch 做 production/default 加速 | `momentum_compressor_openmp_simd_final.c`, `reports/final_audit_report.md` | ResNet50 8T 安全路径 `3.77x` vs safe serial；ResNet18 8T `2.63x` |
| GPU | 用 CUDA v4 q8 closed-loop codec 做 experimental 加速 | `cuda_feasibility/falcom_cuda_v4/`, `reports/final_cuda_v4_selected_path.md` | ResNet50 guarded all-CUDA closed-loop median `44.865 ms`，质量通过 |

CPU safe grouped OpenMP/SIMD 是 production/default 路径。CUDA v4 是独立 experimental closed-loop codec，不替代 CPU `.so`、CPU ABI 或 CPU wire format。

## 报告入口

- [交付物索引](DELIVERABLES.md)
- [PY 到 C 集成详解](docs/py-to-c-integration.md)
- [C baseline 详解](docs/c-baseline.md)
- [CPU OpenMP/SIMD 加速详解](docs/cpu-acceleration.md)
- [GPU CUDA v4 加速详解](docs/gpu-acceleration.md)
- [性能对比与报告引用口径](docs/performance-comparison.md)
- [HPC 面试证据口径](docs/interview-hpc-evidence.md)

## 性能总览

### 1. PY 到 C profiling

这组数据不是“纯 Python compressor vs C compressor”的端到端对比，而是 `FalComC.py` 在 APPFL/Python 侧调用 C compressor 时的分解 profiling。它回答的问题是：Python 包装、ctypes marshalling、GPU tensor 转 CPU numpy 各占多少。

| 模型/输入 | 层数 | 原始大小 | model total | GPU->CPU numpy | ctypes | C compress | payload copy | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet18 numpy CPU | 101 | 42.66 MB | 323.868 ms | 0.000 ms | 1.195 ms | 310.631 ms | 3.761 ms | ctypes 不是瓶颈 |
| ResNet18 torch CPU | 101 | 42.66 MB | 303.459 ms | 1.140 ms | 1.493 ms | 290.474 ms | 3.031 ms | Python 包装开销小 |
| ResNet18 torch CUDA | 101 | 42.66 MB | 323.091 ms | 15.083 ms | 1.836 ms | 294.274 ms | 3.335 ms | CUDA tensor 会产生 GPU->CPU copy |
| ResNet50 torch CUDA | 266 | 89.93 MB | 739.525 ms | 38.502 ms | 4.619 ms | 663.696 ms | 6.732 ms | 主要瓶颈仍在 C compression |

来源：`MIGRATION_EXPERIMENT_SUMMARY.md`, `appfl_data_residency.csv`, `appfl_data_residency_resnet50_cuda.csv`。

### 2. C baseline 与 CPU grouped 加速

这组来自同一 audit thread scan，可直接比较 safe serial C baseline 和 safe grouped OpenMP/SIMD。推荐报告里使用 8 threads，因为它是本实验的稳定最佳点。

| 模型 | C safe serial compress | CPU grouped compress | CPU grouped decompress | 压缩比 | grouped vs serial |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 797.62 ms | 211.36 ms | 273.43 ms | 62.44:1 | 3.77x |
| ResNet18 | 337.84 ms | 128.66 ms | 118.81 ms | 220.71:1 | 2.63x |

来源：`reports/final_audit_report.md`。

另外，release logging gate 的 installed `.so` 基准显示：

| 模型 | grouped 8T hot compress | release baseline | improvement | gate |
|---|---:|---:|---:|---|
| ResNet50 | 197.865 ms | 208.325 ms | 5.02% | PASS |
| ResNet18 | 122.189 ms | 125.641 ms | 2.75% | MISS strict threshold, no regression |

来源：`reports/final_cpu_logging_gate_gpu_feasibility_report.md`。

### 3. GPU CUDA v4 final path

CUDA v4 final selected path 是 `cuda_v4_q8 + guarded_all_cuda`。它是 experimental closed-loop codec，所以性能可以作为 GPU 可行性与上限证据，但不能直接说它替换了 production CPU wire format。

| 模型 | closed-loop median | compress median | decompress median | rel L2 | finite | ratio retention | CUDA/fallback 层数 | payload objects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 44.865 ms | 22.550 ms | 22.635 ms | 0.019808 | 1.000 | 1.063 | 266 / 0 | 1 |
| ResNet18 | 20.808 ms | 10.237 ms | 10.455 ms | 0.020187 | 1.000 | 1.047 | 101 / 0 | 1 |

来源：`logs/cuda_v4_guarded_all_cuda/final_guarded_all_cuda_report.md`, `logs/cuda_v4_guarded_all_cuda/quality.csv`。

### 4. CUDA microbench

最大代表层 `numel=2359296` 上，GPU-only kernel 明显快，但含 H2D/D2H 后只有 packing/sign packing 仍然明显占优。

| Kernel | CPU ms | GPU-only ms | H2D+GPU+D2H ms | GPU-only speedup | Transfer-inclusive speedup |
|---|---:|---:|---:|---:|---:|
| momentum_update | 1.6150 | 0.0320 | 6.4951 | 50.5x | 0.25x |
| residual_predmem_update | 2.6756 | 0.0421 | 6.4983 | 63.6x | 0.41x |
| threshold_scan | 2.0723 | 0.0193 | 2.6728 | 107.3x | 0.78x |
| bitmap_pack | 6.5086 | 0.0053 | 0.6446 | 1218.8x | 10.10x |
| dominant_sign_pack | 6.5496 | 0.0209 | 2.1081 | 313.2x | 3.11x |
| decompress_reconstruct | 1.5613 | 0.0317 | 6.4878 | 49.2x | 0.24x |

来源：`MIGRATION_EXPERIMENT_SUMMARY.md`, `cuda_microbench_results.csv`。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `EB-FaLCom/src/appfl/compressor/` | APPFL/FalCom Python wrapper、`FalComC.py`、compressor 集成代码 |
| `momentum_compressor_final.c` | C baseline |
| `momentum_compressor_openmp_simd_final.c` | safe grouped OpenMP/SIMD CPU production path |
| `cuda_feasibility/falcom_cuda_v0/` 到 `falcom_cuda_v4/` | CUDA 实验演进、wrapper、benchmark、tests |
| `logs/cuda_v4_guarded_all_cuda/` | CUDA v4 final timing、quality、layer backend diff |
| `reports/` | CPU audit、logging gate、GPU feasibility、CUDA selected path |
| `dataset/*.md` | replay dataset 说明文档，不含原始 `.bin` 数据 |
| `env/` | conda/pip/CUDA/host 环境清单 |
| `docs/` | 报告写作用的分章节说明 |

## 复现提示

CPU default path：

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash build_openmp_simd_final.sh
```

CUDA v4 experimental path：

```bash
export FALCOM_CUDA_EXPERIMENTAL=1
export FALCOM_CUDA_V4_GUARDED_ALL_CUDA=1
```

未设置 `FALCOM_CUDA_EXPERIMENTAL` 时，默认应保持 CPU fallback。

## 未上传内容

- `dataset/**/*.bin` 原始 replay 数据。
- CPU/CUDA 编译后的 `.so`、二进制、object 文件。
- `envs/` 虚拟环境。
- transfer archive、backup archive、本地缓存和临时日志。

这样做是为了让 GitHub 仓库聚焦于源码、实验流程、结果证据和报告材料。
