# HPC 面试证据口径

这份文档专门回答面试里最容易被追问的实验边界：`3.77x` 相对谁、`10.8x` 是什么 timing、GPU 是否包含 H2D/D2H、CPU 是几线程、CUDA path 是否保持同一 wire format。

## 一句话版本

FalCom/APPFL 梯度压缩器项目完成了从 Python 调用、C baseline、多核 CPU 到 CUDA experimental codec 的完整优化链路。CPU 侧使用 safe grouped OpenMP/SIMD batch compressor，在 ResNet50 上以 8 threads、`OMP_PROC_BIND=close`、`OMP_PLACES=cores` 达到 `211.36 ms` compress，相比 safe serial C baseline `797.62 ms` 是 `3.77x` compress-only 加速。GPU 侧使用 CUDA v4 q8 `guarded_all_cuda` experimental closed-loop codec，在 ResNet50 上 `22.550 ms` compress、`22.635 ms` decompress、`44.865 ms` closed-loop；相比 CPU grouped 8T closed-loop `211.36 + 273.43 = 484.79 ms` 是 `10.8x` closed-loop 加速，compress-only 则是 `211.36 / 22.550 = 9.37x`。

## 面试官追问

### 1. `3.77x` 是相对谁？

`3.77x` 是 ResNet50 safe grouped OpenMP/SIMD 8T compress 相对 safe serial C 8T compress：

```text
safe serial C baseline compress = 797.62 ms
safe grouped OpenMP/SIMD 8T compress = 211.36 ms
speedup = 797.62 / 211.36 = 3.77x
```

这是 compress-only 数字，不是 compress+decompress。线程扫描使用 `round_0` warmup，`round_1/round_2` hot-round median。主结果使用 pinning：

```bash
OMP_NUM_THREADS=8
OMP_PROC_BIND=close
OMP_PLACES=cores
```

对应证据：`reports/final_audit_report.md` 和 `logs/interview_hpc/cpu_thread_scaling.csv`。

### 2. `10.8x` 是 compress-only 还是 compress+decompress？

`10.8x` 只用于 GPU closed-loop，也就是 compress+decompress：

```text
CPU grouped 8T closed-loop = 211.36 + 273.43 = 484.79 ms
GPU CUDA v4 guarded closed-loop = 44.865 ms
closed-loop speedup = 484.79 / 44.865 = 10.8x
```

如果只讲 compress-only，ResNet50 是：

```text
CPU grouped 8T compress = 211.36 ms
GPU CUDA v4 guarded compress = 22.550 ms
compress-only speedup = 211.36 / 22.550 = 9.37x
```

所以简历和面试里建议分开写：`3.77x CPU compress-only`、`9.37x GPU compress-only`、`10.8x GPU closed-loop`。

对应证据：`logs/interview_hpc/gpu_speedup_breakdown.csv`。

### 3. GPU timing 是否包含 H2D/D2H？

主口径不是 replay 文件加载，也不是把模型从磁盘读进 GPU 的时间。timed region 里模型 tensor 已经在 CUDA 上，`round_0` 是 warmup，hot rounds 才统计。

timed CUDA closed-loop 包含 CUDA codec 内部开销：

- encode/decode kernel time
- payload D2H
- payload H2D
- envelope serialize/parse
- decoded tensor materialize

也就是说：不包含 replay file loading；包含 codec 内部 payload transfer 和 envelope 开销。这个边界是为了回答 HPC 优化本体，而不是把数据集 IO 混进 codec timing。

对应证据：`logs/interview_hpc/gpu_transfer_breakdown.csv`。

### 4. CPU 是几线程？有没有 pinning？

主结果是 8 threads，并且有 pinning：

```bash
OMP_NUM_THREADS=8
OMP_PROC_BIND=close
OMP_PLACES=cores
```

完整 pinned thread scan 覆盖 `1,2,4,8,16` threads。ResNet50 在 8T 达到最稳定的主结果，16T 反而因为 overhead 和 imbalance 回落，所以面试里把 8T 作为默认 benchmark point。

未 pinning 的表 `logs/interview_hpc/cpu_thread_scaling_unpinned.csv` 作为备用占位；当前 committed audit headline 不使用 unpinned 数字。换句话说，面试里只主动讲 pinned 结果，除非对方追问实验控制变量，再说明本轮公开证据包没有把 unpinned 作为结论。

### 5. CUDA path 是否保持同一 wire format？

不把两个路径混在一起讲：

- CPU production/default path：保持 CPU ABI 和 CPU wire format。
- CUDA v4 path：experimental codec，使用独立 packed payload envelope，不保持 CPU wire format。
- CUDA path 必须显式启用：`FALCOM_CUDA_EXPERIMENTAL=1`，final guarded path 还使用 `FALCOM_CUDA_V4_GUARDED_ALL_CUDA=1`。

CUDA v4 的结论应表述为 experimental closed-loop codec 的 GPU 可行性和加速证据，不能说它直接替换 production CPU wire format。

质量约束：

```text
relative_l2_error <= 0.03
finite_rate == 1.0
ratio_retention >= 0.8
decode_status == pass
correctness_status == pass
fallback_layers == 0 for guarded_all_cuda
```

对应证据：`logs/interview_hpc/wire_format_quality.csv`。

## 推荐简历三点

1. 面向联邦学习 ResNet18/ResNet50 梯度压缩场景，完成从 Python/APPFL 调用、C 原生 compressor、多核 CPU 优化到 CUDA experimental codec 的完整性能优化链路，并建立可复现实验与正确性证据。
2. 设计 safe grouped OpenMP/SIMD batch compressor，将 layer 压缩任务分组并行化，在保持状态语义和 CPU wire format 正确的前提下实现多核 CPU 加速；ResNet50 8T compress 从 safe serial C `797.62 ms` 降到 `211.36 ms`，达到 `3.77x` compress-only 加速。
3. 设计 CUDA v4 q8 guarded all-CUDA experimental codec，将压缩/解压核心算子迁移到 GPU，并显式记录 kernel、payload transfer、envelope 开销；ResNet50 GPU compress-only 达到 `9.37x`，closed-loop compress+decompress 相比 CPU grouped 8T 达到约 `10.8x`，同时满足 rel L2、finite rate、ratio retention 和 decode/correctness gates。

## 证据文件

| 文件 | 用途 |
|---|---|
| `logs/interview_hpc/cpu_thread_scaling.csv` | CPU pinned thread scan；8T 是直接 audit headline，非 8T 主要用于 compress scaling 辅助说明 |
| `logs/interview_hpc/cpu_thread_scaling_unpinned.csv` | unpinned 备用说明表，当前不作为 headline |
| `logs/interview_hpc/gpu_speedup_breakdown.csv` | GPU compress-only 与 closed-loop speedup 口径 |
| `logs/interview_hpc/gpu_transfer_breakdown.csv` | CUDA codec 内部 kernel、payload D2H/H2D、envelope 开销 |
| `logs/interview_hpc/wire_format_quality.csv` | wire format 状态与质量 gate |
| `scripts/run_interview_hpc_experiments.py` | 生成上述轻量 CSV 的 runner |

默认 runner 使用 `CUDA_VISIBLE_DEVICES=2`；如果该卡被其他进程占用，可以传 `--gpu <id>` 重新生成 transfer diagnostic。本次 committed transfer breakdown 使用 GPU 3，因为 GPU 2 当时显存不足；headline speedup 仍来自已选定的 `logs/cuda_v4_guarded_all_cuda/quality.csv`。
