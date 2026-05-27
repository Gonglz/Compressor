# appfl.compressor 裁剪模块

这个目录保留本实验需要的 compressor 相关代码，重点是：

- `FalComC.py`：Python 到 C compressor 的 wrapper。
- `FalCom.py`：原 Python compressor 参考实现。
- `base_compressor.py`、`compressor.py`：APPFL compressor 基类和兼容层。
- SZ/ZFP/QSGD 相关 wrapper：保留为原 APPFL compressor 依赖上下文。

CUDA v4 experimental codec 不修改这里的 CPU ABI 或 wire format。最终 GPU 路径在 `cuda_feasibility/falcom_cuda_v4/` 中实现，通过 `FALCOM_CUDA_EXPERIMENTAL=1` 显式启用。
