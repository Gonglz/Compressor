# GPU CUDA v4 加速详解

## 目标

GPU 部分的目标是验证 compressor 是否能从 CUDA 中获得实际收益。最终保留的是 CUDA v4 q8 guarded all-CUDA experimental closed-loop codec。

报告里可以把它写成：

> GPU 实验先用 microbench 判断哪些 hot kernels 值得搬到 CUDA，再用 v0 到 v4 多代 wrapper 验证端到端 closed-loop codec。最终 CUDA v4 guarded all-CUDA 在 ResNet50 上达到 `44.865 ms` closed-loop median，并通过质量检查。

## 关键文件

| 文件 | 作用 |
|---|---|
| `cuda_feasibility/falcom_cuda_microbench.cu` | standalone CUDA hot-kernel microbench |
| `cuda_feasibility/falcom_cuda_v0/` | early CUDA wrapper and feasibility tests |
| `cuda_feasibility/falcom_cuda_v2/` | serialized envelope / state tests |
| `cuda_feasibility/falcom_cuda_v3/` | training smoke and mixed fallback experiments |
| `cuda_feasibility/falcom_cuda_v4/` | final q8 CUDA codec and guarded all-CUDA path |
| `cuda_feasibility/falcom_cuda_v4/run_cuda_v4_guarded_all_cuda.py` | final official experiment driver |
| `logs/cuda_v4_guarded_all_cuda/` | final timing, quality, backend evidence |
| `reports/final_cuda_v4_selected_path.md` | final selected path summary |
| `cuda_microbench_results.csv` | raw microbench results |

## CUDA microbench 结论

最大代表层 `numel=2359296`：

| Kernel | CPU ms | GPU-only ms | H2D+GPU+D2H ms | GPU-only speedup | Transfer-inclusive speedup |
|---|---:|---:|---:|---:|---:|
| momentum_update | 1.6150 | 0.0320 | 6.4951 | 50.5x | 0.25x |
| residual_predmem_update | 2.6756 | 0.0421 | 6.4983 | 63.6x | 0.41x |
| threshold_scan | 2.0723 | 0.0193 | 2.6728 | 107.3x | 0.78x |
| bitmap_pack | 6.5086 | 0.0053 | 0.6446 | 1218.8x | 10.10x |
| dominant_sign_pack | 6.5496 | 0.0209 | 2.1081 | 313.2x | 3.11x |
| decompress_reconstruct | 1.5613 | 0.0317 | 6.4878 | 49.2x | 0.24x |

解释：

- GPU-only kernel 很快，说明 CUDA 有潜力。
- 如果每层都 H2D/D2H，momentum/update/reconstruct 会被 transfer cost 吃掉。
- packing/sign packing 即使含 transfer 仍有明显收益。
- 因此最终方向不是简单 host-pointer CUDA wrapper，而是更 device-resident 的 closed-loop CUDA codec。

## CUDA v4 final selected path

最终配置：

```text
codec = cuda_v4_q8
mode = guarded_all_cuda
activation = FALCOM_CUDA_EXPERIMENTAL=1
optional explicit mode = FALCOM_CUDA_V4_GUARDED_ALL_CUDA=1
```

核心结果：

| 模型 | closed-loop median | compress median | decompress median | rel L2 | finite | ratio retention | CUDA/fallback layers | payload objects | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet50 | 44.865 ms | 22.550 ms | 22.635 ms | 0.019808 | 1.000 | 1.063 | 266 / 0 | 1 | accepted final selected path |
| ResNet18 | 20.808 ms | 10.237 ms | 10.455 ms | 0.020187 | 1.000 | 1.047 | 101 / 0 | 1 | supporting pass |

ResNet50 的官方 threshold 2048 对照有 197 个 fallback layer；guarded all-CUDA 把这些 fallback 转成 CUDA 后，fallback layer 数变为 0。

## 质量边界

CUDA v4 通过：

- extension import。
- q8 quality。
- mixed fallback。
- experimental gate。
- backend ownership。
- transaction rollback。
- separate encoder/decoder state。
- packed envelope。
- guarded all-valid-CUDA gate。

质量指标：

- finite rate 最小值为 `1.000`。
- ResNet50 relative L2 median 为 `0.019808`。
- ResNet18 relative L2 median 为 `0.020187`。
- ratio retention 高于 `1.0`，说明相对 CPU wrapper compression ratio 没有恶化。

## 和 CPU path 的关系

CUDA v4 不是 production/default CPU path 的替代品：

- 不修改 installed CPU `.so`。
- 不修改 CPU ABI。
- 不修改 CPU wire format。
- 使用自己的 payload envelope 和 decode path。
- 需要 `FALCOM_CUDA_EXPERIMENTAL=1` 显式启用。

默认未启用实验开关时，系统应保持 CPU fallback。

## 报告写法建议

可以这样写：

1. microbench 证明 GPU kernel 级别有显著潜力，但 host-device transfer 是核心约束。
2. v4 选择 closed-loop CUDA codec，而不是直接替换 CPU wire format。
3. guarded all-CUDA 让 ResNet50/ResNet18 全部 layer 走 CUDA，fallback layer 为 0。
4. ResNet50 closed-loop median `44.865 ms`，质量通过，是最终 selected experimental path。
5. CUDA path 仍是 experimental，production/default 仍是 CPU safe grouped。

## 证据来源

- `MIGRATION_EXPERIMENT_SUMMARY.md`
- `cuda_microbench_results.csv`
- `reports/final_cuda_v4_selected_path.md`
- `logs/cuda_v4_guarded_all_cuda/final_guarded_all_cuda_report.md`
- `logs/cuda_v4_guarded_all_cuda/quality.csv`
- `logs/cuda_v4_guarded_all_cuda/layer_backend_diff.csv`
