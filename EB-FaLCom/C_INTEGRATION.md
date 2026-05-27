# FalCom C 集成说明

本实验通过 `appfl.compressor.FalComC` 调用本地 C compressor。CPU 路径作为 production/default baseline 保留，CUDA v4 路径作为独立 experimental closed-loop codec 使用。

关键边界：

- 不修改 CPU C ABI。
- 不修改 CPU wire format。
- 不替换 installed CPU `.so`。
- `FALCOM_CUDA_EXPERIMENTAL=0/unset` 时保持 CPU fallback。
- CUDA v4 使用自己的 payload envelope 和 decode path。

最终 GPU 实验路径为：

```text
cuda_v4_q8 + guarded_all_cuda
```

详见 `../reports/final_cuda_v4_selected_path.md`。
