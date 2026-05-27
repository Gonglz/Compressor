# 性能对比与报告引用口径

## 先说明可比性

本项目里有几类性能数字，报告中最好分清：

1. `PY profiling`：衡量 Python wrapper、ctypes、GPU->CPU numpy、C compression 各自占比。
2. `C safe serial vs CPU grouped`：同一 audit benchmark 内的直接可比 CPU 加速。
3. `release logging gate`：installed `.so` 在 logging/timing gate 修复后的 release benchmark。
4. `CUDA v4 closed-loop`：experimental GPU codec 的端到端 encode+decode 闭环，不替代 CPU wire format。
5. `CUDA microbench`：kernel-level 性能，用来解释为什么需要避免 H2D/D2H。

最稳的写法是分别列这些数字，不把不同 benchmark 的结果硬合成一个绝对 speedup。

面试追问版证据见：[HPC 面试证据口径](interview-hpc-evidence.md)。这份补充文档把 `3.77x`、`9.37x`、`10.8x` 的对照 baseline、timing scope、OpenMP pinning、CUDA wire-format 边界和质量 gate 单独列出。

## 推荐主表

| 路径 | ResNet50 | ResNet18 | 性质 | 建议表述 |
|---|---:|---:|---|---|
| PY->C wrapper profiling | 739.525 ms model total, CUDA tensor | 323.091 ms model total, CUDA tensor | Python/ctypes/C 分解 | ctypes 不是瓶颈，C compression 和 GPU->CPU copy 更重要 |
| C safe serial | 797.62 ms compress | 337.84 ms compress | C baseline | CPU grouped 的对照 baseline |
| CPU grouped OpenMP/SIMD | 211.36 ms compress | 128.66 ms compress | production/default CPU path | ResNet50 `3.77x`, ResNet18 `2.63x` vs safe serial |
| CPU grouped release gate | 197.865 ms hot compress | 122.189 ms hot compress | installed `.so` release gate | ResNet50 `5.02%` faster than release baseline，ResNet18 no regression |
| GPU CUDA v4 guarded | 44.865 ms closed-loop | 20.808 ms closed-loop | experimental closed-loop codec | 质量通过，但不替代 CPU wire format |

## PY 到 C profiling 表

| 模型/输入 | model total | GPU->CPU numpy | ctypes | C compress | payload copy |
|---|---:|---:|---:|---:|---:|
| ResNet18 numpy CPU | 323.868 ms | 0.000 ms | 1.195 ms | 310.631 ms | 3.761 ms |
| ResNet18 torch CPU | 303.459 ms | 1.140 ms | 1.493 ms | 290.474 ms | 3.031 ms |
| ResNet18 torch CUDA | 323.091 ms | 15.083 ms | 1.836 ms | 294.274 ms | 3.335 ms |
| ResNet50 torch CUDA | 739.525 ms | 38.502 ms | 4.619 ms | 663.696 ms | 6.732 ms |

报告解释：

- ResNet50 CUDA tensor 的 `ctypes=4.619 ms`，远小于 `C compress=663.696 ms`。
- GPU->CPU copy 是可见开销，但仍不是当前总耗时最大项。
- 这支持“GPU path 要 device-resident，不能只做每层 H2D/D2H wrapper”的结论。

## CPU 直接可比表

| 模型 | safe serial compress | safe grouped compress | safe grouped decompress | ratio | speedup vs safe serial |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 797.62 ms | 211.36 ms | 273.43 ms | 62.44:1 | 3.77x |
| ResNet18 | 337.84 ms | 128.66 ms | 118.81 ms | 220.71:1 | 2.63x |

报告解释：

- 这是同一 `reports/final_audit_report.md` thread scan 内的直接对比。
- 8 threads 是推荐默认 benchmark point。
- ratio 没有恶化，正确性 gates 通过。

## CPU release gate 表

| 模型 | grouped 8T hot compress | release baseline | improvement | acceptance |
|---|---:|---:|---:|---|
| ResNet50 | 197.865 ms | 208.325 ms | 5.02% | PASS |
| ResNet18 | 122.189 ms | 125.641 ms | 2.75% | strict threshold MISS, no regression |

报告解释：

- 这组数据更适合写在“release cleanup / logging gate”段落。
- 它证明默认关闭 diagnostic logging/timing 后，installed `.so` 不被 instrumentation 拖慢。
- ResNet18 虽然未达到 strict threshold，但没有 regression，ratio 不变。

## GPU CUDA v4 表

| 模型 | closed-loop median | compress | decompress | rel L2 | finite | ratio retention | CUDA/fallback | payload objects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 44.865 ms | 22.550 ms | 22.635 ms | 0.019808 | 1.000 | 1.063 | 266 / 0 | 1 |
| ResNet18 | 20.808 ms | 10.237 ms | 10.455 ms | 0.020187 | 1.000 | 1.047 | 101 / 0 | 1 |

报告解释：

- `closed-loop` 包含 encode/decode 闭环。
- quality pass 的核心是 finite rate、relative L2、ratio retention。
- `guarded_all_cuda` 把 fallback layers 降为 0。
- 它是 experimental codec，不是 production CPU `.so` 的直接替换。

## GPU microbench 表

| Kernel | CPU ms | GPU-only ms | H2D+GPU+D2H ms | GPU-only speedup | Transfer-inclusive speedup |
|---|---:|---:|---:|---:|---:|
| momentum_update | 1.6150 | 0.0320 | 6.4951 | 50.5x | 0.25x |
| residual_predmem_update | 2.6756 | 0.0421 | 6.4983 | 63.6x | 0.41x |
| threshold_scan | 2.0723 | 0.0193 | 2.6728 | 107.3x | 0.78x |
| bitmap_pack | 6.5086 | 0.0053 | 0.6446 | 1218.8x | 10.10x |
| dominant_sign_pack | 6.5496 | 0.0209 | 2.1081 | 313.2x | 3.11x |
| decompress_reconstruct | 1.5613 | 0.0317 | 6.4878 | 49.2x | 0.24x |

报告解释：

- GPU-only 并不等于端到端收益。
- transfer-inclusive 结果显示 H2D/D2H 是主要约束。
- packing/sign packing 是最有潜力的 kernel 类别。

## 推荐报告段落

可以直接改写下面这段：

> The project follows a staged acceleration path. First, the Python APPFL wrapper was connected to a native C compressor through ctypes, and profiling showed that the Python-to-C bridge itself was not the bottleneck. Second, the C baseline established stable ABI, wire-format, and state semantics. Third, the production CPU path used a safe grouped OpenMP/SIMD implementation, reaching 3.77x compression speedup on ResNet50 and 2.63x on ResNet18 against the safe serial C baseline while preserving compression ratio and correctness. Finally, an experimental CUDA v4 q8 closed-loop codec achieved 44.865 ms median encode/decode time on ResNet50 and 20.808 ms on ResNet18 with all layers on CUDA and quality checks passing. The CUDA path is kept experimental because it uses a separate payload envelope and does not replace the production CPU wire format.

## 引用来源

- `MIGRATION_EXPERIMENT_SUMMARY.md`
- `reports/final_audit_report.md`
- `reports/final_cpu_logging_gate_gpu_feasibility_report.md`
- `reports/final_cuda_v4_selected_path.md`
- `logs/cuda_v4_guarded_all_cuda/final_guarded_all_cuda_report.md`
- `logs/cuda_v4_guarded_all_cuda/quality.csv`
- `cuda_microbench_results.csv`
