# EB-FaLCom 裁剪说明

这个目录不是完整 APPFL 发布包，而是本实验使用的最小 APPFL/FalCom compressor 集成快照。

保留内容：

- `src/appfl/compressor/`：FalCom/FalComC 及相关 compressor wrapper。
- `src/appfl/misc/deprecation.py`：`appfl.compressor.compressor` 的最小依赖。
- `setup.py`、`pyproject.toml`、`LICENSE`：保留用于说明来源和最小 packaging 结构。

已删除内容：

- APPFL 示例、notebook、测试、通信后端、登录管理、服务端入口等与本 compressor HPC 实验无关的模块。
- 原始数据和编译产物。

本仓库的重点是 FalCom CPU baseline 与 CUDA v4 experimental closed-loop compressor 的 HPC 结果，而不是完整 APPFL 框架复刻。
