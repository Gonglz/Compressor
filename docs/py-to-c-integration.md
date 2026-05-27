# PY 到 C 集成详解

## 目标

PY 这一部分解决的是“APPFL/Python 训练框架如何调用 C compressor”。它不是单独实现一个 Python 压缩算法，而是在 Python wrapper 中完成输入规范化、ctypes 参数构造、调用 C `.so`、接收 payload，并提供 profiling hook。

报告里可以把这一部分写成：

> Python 层负责接入 APPFL 生态和数据类型适配，C 层负责实际压缩计算。profiling 显示 ctypes 桥接本身不是主要瓶颈，主要耗时来自 C compression 和必要的数据搬运。

## 关键文件

| 文件 | 作用 |
|---|---|
| `EB-FaLCom/src/appfl/compressor/FalCom.py` | 原始 Python-facing compressor path |
| `EB-FaLCom/src/appfl/compressor/FalComC.py` | Python 到 C 的 ctypes bridge，包含 profiling hooks |
| `EB-FaLCom/C_INTEGRATION.md` | C 集成边界说明 |
| `appfl_data_residency.csv` | ResNet18 APPFL data residency profiling |
| `appfl_data_residency_resnet50_cuda.csv` | ResNet50 CUDA tensor profiling |

## 调用链

```text
APPFL / Python model update
  -> FalComC.py receives layer tensors or numpy arrays
  -> torch Tensor path: detach().cpu().numpy() when needed
  -> ctypes structures are built
  -> libmomentum_compressor.so is called
  -> compressed payload is copied back to Python-owned objects
```

关键边界：

- Python wrapper 不改变 C ABI。
- Python wrapper 不改变 CPU wire format。
- CUDA tensor 进入当前 host-pointer C path 前，需要先转成 CPU numpy。
- profiling 通过环境变量开启，不影响默认路径。

## Profiling 开关

```bash
export FALCOM_APPFL_PROFILE=1
export FALCOM_APPFL_PROFILE_CSV=$PWD/appfl_data_residency.csv
```

profiling 记录字段包括：

- tensor/array 类型、device、dtype、shape、numel。
- `.detach().cpu().numpy()` 时间。
- ctypes 参数构造时间。
- C compression 时间。
- payload copy 时间。
- layer total 和 model total。

## 性能证据

| 模型/输入 | 层数 | 原始大小 | model total | GPU->CPU numpy | ctypes | C compress | payload copy |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 numpy CPU | 101 | 42.66 MB | 323.868 ms | 0.000 ms | 1.195 ms | 310.631 ms | 3.761 ms |
| ResNet18 torch CPU | 101 | 42.66 MB | 303.459 ms | 1.140 ms | 1.493 ms | 290.474 ms | 3.031 ms |
| ResNet18 torch CUDA | 101 | 42.66 MB | 323.091 ms | 15.083 ms | 1.836 ms | 294.274 ms | 3.335 ms |
| ResNet50 torch CUDA | 266 | 89.93 MB | 739.525 ms | 38.502 ms | 4.619 ms | 663.696 ms | 6.732 ms |

## 分析

ctypes 构造只占总时间的一小部分。以 ResNet50 CUDA tensor 为例，`ctypes_build_ms=4.619 ms`，而 `c_compress_ms=663.696 ms`。这说明 Python 调用 C 的桥接不是最主要瓶颈。

CUDA tensor path 会发生 GPU->CPU copy。ResNet50 上这一项是 `38.502 ms`，可测但仍小于 C compression。这个结果支持后续 GPU 方向的判断：如果只是在 C host-pointer wrapper 内部加 CUDA kernel，但每层仍然 H2D/D2H，整体收益会被数据搬运吞掉；真正值得做的是 device-resident fused CUDA path。

## 报告写法建议

可以这样描述：

1. Python 层完成框架集成，C 层完成高性能压缩。
2. profiling 证明 ctypes 不是瓶颈。
3. CUDA tensor 在当前 CPU compressor path 中会回落到 CPU numpy，因此 GPU 优化必须避免频繁 host-device 往返。
4. 这一部分为 C baseline、CPU grouped、CUDA v4 三个后续方向提供了统一入口和测量工具。

## 证据来源

- `MIGRATION_EXPERIMENT_SUMMARY.md`
- `README_GPU_TRANSFER.md`
- `appfl_data_residency.csv`
- `appfl_data_residency_resnet50_cuda.csv`
- `EB-FaLCom/src/appfl/compressor/FalComC.py`
