# C Baseline 详解

## 目标

C baseline 的目标是提供一个可审计、可回退、语义清楚的 compressor 实现。它是后续 CPU parallel 和 GPU experimental 的对照基线，也是保证 ABI、wire format、状态语义不乱的锚点。

报告里可以把它写成：

> C baseline 将核心压缩逻辑从 Python 运行时中剥离出来，用稳定 ABI 暴露给 APPFL wrapper。它优先保证正确性、状态语义和可回退性，然后作为 CPU/GPU 加速的对照。

## 关键文件

| 文件 | 作用 |
|---|---|
| `momentum_compressor_final.c` | C baseline compressor |
| `momentum_compressor.h` | 公共结构和函数声明 |
| `build_lib.sh` | baseline C library build script |
| `test_runtime_contracts.c` | runtime contract tests |
| `test_reference_state_oracle.c` | 状态语义 oracle |
| `test_static_contracts.py` | 静态 contract checks |
| `EB-FaLCom/src/appfl/compressor/FalComC.py` | Python wrapper 调用 C `.so` |

## 设计边界

C baseline 保持以下边界：

- public C API 不变。
- Python ctypes ABI 不变。
- CPU wire format 不变。
- 默认路径不依赖 CUDA。
- 遇到 CUDA experimental 关闭或不可用时，仍可回到 CPU fallback。

这些边界让后续优化可以独立实验，不会破坏 production/default path。

## 状态语义

compressor 有两个核心状态：

| 状态 | key | 作用 |
|---|---|---|
| history | `client_id:layer_name` | 保存上一轮 layer gradient/history |
| prediction memory | `client_id -> layer_name|dtype|shape` | momentum prediction memory |

audit 中修复了两个重要问题：

1. prediction memory 原先只按 shape 建 key，同 shape 不同 layer 可能共享 momentum state。
2. 同一个 history key 如果遇到不同 shape/dtype 的 tensor，旧 history 可能误入 momentum path。

修复后：

- prediction memory key 包含 `layer_name`、`dtype`、`shape`。
- momentum path 进入前检查 layout 是否一致。
- grouped path 的 commit 仍按原 batch 顺序提交，避免半提交。

## Correctness gates

现有报告记录的 gate：

```text
python3 test_static_contracts.py                         PASS
gcc -fsyntax-only momentum_compressor_openmp_simd_final.c PASS
./test_runtime_contracts_openmp                          PASS
./test_batch_state_equivalence                           PASS
./test_reference_state_oracle                            PASS
ctypes load local .so                                    PASS
ctypes load installed .so                                PASS
ASAN/UBSAN runtime contracts                             PASS
```

虽然这里的 `gcc` 文件名是 OpenMP/SIMD final 版本，但 gate 同时覆盖了 C ABI、runtime contract、reference state oracle 和 ctypes load，是 C baseline 继续演进的安全网。

## Baseline 性能口径

和 CPU grouped 直接可比的 C baseline 来自 `reports/final_audit_report.md` 中的 safe serial scan。

| 模型 | threads | safe serial compress | safe serial role |
|---|---:|---:|---|
| ResNet50 | 8 | 797.62 ms | CPU grouped speedup baseline |
| ResNet18 | 8 | 337.84 ms | CPU grouped speedup baseline |

注意：safe serial 不是“纯 Python baseline”。它是 C compressor 的 serial/safe 对照路径，用来衡量后续 CPU grouped parallel 的收益。

## 和 PY / CPU / GPU 的关系

- PY 层通过 `FalComC.py` 调用 C baseline 或最终 CPU `.so`。
- CPU 加速是在 C baseline 的状态语义上引入 grouped batch、OpenMP 和 SIMD。
- GPU 加速没有替换 C baseline，而是作为 experimental closed-loop codec 独立验证可行性。

## 报告写法建议

可以这样写：

1. C baseline 先解决“能不能稳定从 Python 进入 native compressor”的问题。
2. 它定义 ABI、wire format、history 和 prediction memory 语义。
3. 后续 CPU 和 GPU 优化都不能破坏这个 baseline 的 correctness gates。
4. safe serial C baseline 是 CPU grouped speedup 的对照，不等同于 Python baseline。

## 证据来源

- `reports/final_audit_report.md`
- `reports/final_cpu_logging_gate_gpu_feasibility_report.md`
- `momentum_compressor_final.c`
- `momentum_compressor.h`
- `test_reference_state_oracle.c`
- `test_batch_state_equivalence.c`
