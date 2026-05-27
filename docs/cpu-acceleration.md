# CPU OpenMP/SIMD 加速详解

## 目标

CPU 部分的目标是在不破坏 C ABI、CPU wire format 和状态语义的前提下，把 compressor 的 production/default path 做快。最终采用的是 safe grouped OpenMP/SIMD path。

报告里可以把它写成：

> CPU 加速通过 batch grouping、OpenMP parallel compute、SIMD-friendly hot loops 和 serial commit 组合实现。它保留 CPU wire format 和 fallback 能力，是当前 production/default 推荐路径。

## 关键文件

| 文件 | 作用 |
|---|---|
| `momentum_compressor_openmp_simd_final.c` | final safe grouped OpenMP/SIMD compressor |
| `simd_helpers_v23.h` | SIMD helper |
| `ompv22sz3.c` | unsafe v22 performance ceiling reference |
| `build_openmp_simd_final.sh` | final CPU `.so` build script |
| `benchmark_threads.sh`, `run_benchmark_threads.sh` | thread scan / benchmark scripts |
| `benchmark_real.c`, `benchmark_decomp.py` | replay and decompression benchmark tooling |
| `reports/final_audit_report.md` | audit, correctness, thread scan |
| `reports/final_cpu_logging_gate_gpu_feasibility_report.md` | installed release gate |

## 方法

CPU grouped path 把 batch 内的 layer 按状态 key 分组：

1. prepass 为每个 item 生成 history/prediction-memory key。
2. group-local compute 在 OpenMP threads 中并行执行。
3. 每个 group 保持自己的 local state chain。
4. commit 阶段按原 batch 顺序串行提交到全局 state。
5. 如果中间失败，临时 payload 清理，public output 不半提交。

这个设计的重点是“并行计算”和“串行状态提交”分开。它比直接在 public state 上并行写更稳。

## Correctness gates

最终路径通过：

- static contracts。
- `gcc -fsyntax-only`。
- runtime contracts。
- batch state equivalence。
- reference state oracle。
- local/installed `.so` ctypes load。
- ASAN/UBSAN runtime contracts。
- signature equivalence under default/disabled API/env modes。

同时确认：

- default run 不打印 legacy `[BREAKDOWN]`。
- default run 不打印 grouped `[BATCH_BREAKDOWN]`。
- diagnostic logging 必须显式开启。
- compression ratio delta 保持在要求范围内。

## Thread scan 性能

来自 `reports/final_audit_report.md`，每个配置 5 次 run，hot-round median 使用 rounds 1 和 2。

### ResNet50

| threads | safe serial ms | safe grouped ms | unsafe v22 ms | speedup vs serial | grouped/v22 gap | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 936.56 | 1098.56 | 907.36 | 0.85x | 1.21x | 62.44 |
| 2 | 913.18 | 548.04 | 472.43 | 1.67x | 1.16x | 62.44 |
| 4 | 803.11 | 304.25 | 262.88 | 2.64x | 1.16x | 62.44 |
| 8 | 797.62 | 211.36 | 178.27 | 3.77x | 1.19x | 62.44 |
| 16 | 901.40 | 265.33 | 238.53 | 3.40x | 1.11x | 62.44 |

Best safe grouped setting：

```text
threads=8
compress median = 211.36 ms
decompress median = 273.43 ms
compression ratio = 62.44:1
speedup vs safe serial = 3.77x
gap vs unsafe v22 = 1.19x slower
```

### ResNet18

| threads | safe serial ms | safe grouped ms | unsafe v22 ms | speedup vs serial | grouped/v22 gap | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 323.64 | 395.77 | 299.90 | 0.82x | 1.32x | 220.71 |
| 2 | 324.39 | 221.24 | 181.71 | 1.47x | 1.22x | 220.71 |
| 4 | 311.53 | 132.54 | 110.62 | 2.35x | 1.20x | 220.71 |
| 8 | 337.84 | 128.66 | 108.06 | 2.63x | 1.19x | 220.71 |
| 16 | 355.88 | 133.58 | 108.89 | 2.66x | 1.23x | 220.71 |

Best safe grouped setting：

```text
threads=8
compress median = 128.66 ms
decompress median = 118.81 ms
compression ratio = 220.71:1
speedup vs safe serial = 2.63x
gap vs unsafe v22 = 1.19x slower
```

## Release logging gate

installed `.so` benchmark 里，logging/timing gate 关闭默认诊断开销后：

| 模型 | grouped 8T hot compress | release baseline | improvement | acceptance |
|---|---:|---:|---:|---|
| ResNet50 | 197.865 ms | 208.325 ms | 5.02% | PASS |
| ResNet18 | 122.189 ms | 125.641 ms | 2.75% | strict gate MISS, no regression |

这里的 ResNet18 没过 strict `<=118 ms` 或 `>=4%` threshold，但 correctness 通过、ratio 不变、没有 regression，因此作为 release cleanup 是可接受的。

## 为什么没有采用 unsafe v22

`ompv22sz3.c` 只保留为性能上限参考，不作为默认路径。原因：

- prediction memory key 曾经只有 shape，可能同 shape 不同 layer 冲突。
- pure compute 会在 OpenMP region 内修改 global step。
- batch compute 直接并行调用 pure function，状态语义不安全。
- commit 对 prediction-memory persistence 有未完成的实现备注。

因此 final path 选择 safe grouped，牺牲一部分极限速度，换取可审计 correctness。

## 推荐复现环境

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash build_openmp_simd_final.sh
```

## 报告写法建议

可以这样写：

1. CPU production path 选择 safe grouped OpenMP/SIMD，而不是 unsafe v22。
2. ResNet50 在 8 threads 下相对 safe serial 达到 `3.77x` compress speedup。
3. ResNet18 在 8 threads 下达到 `2.63x` compress speedup。
4. 压缩比保持不变，correctness gates 全部通过。
5. 默认关闭 diagnostic logging，避免把 instrumentation 开销带入 release path。

## 证据来源

- `reports/final_audit_report.md`
- `reports/final_cpu_logging_gate_gpu_feasibility_report.md`
- `momentum_compressor_openmp_simd_final.c`
- `benchmark_threads.sh`
- `run_benchmark_threads.sh`
