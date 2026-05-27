# CUDA v4 最终选定实验路径

## 决策

最终选定的 GPU experimental config：

- codec：`cuda_v4_q8`
- mode：`guarded_all_cuda`
- threshold：`N/A / all valid CUDA`
- activation：`FALCOM_CUDA_EXPERIMENTAL=1`
- optional explicit mode：`FALCOM_CUDA_V4_GUARDED_ALL_CUDA=1` 或 `--guarded-all-cuda`

这是 CUDA v4 experimental closed-loop 的最终结果。不建议继续 v5、新 codec、kernel math 优化或 compressor 内部 envelope patch。

## 主要结果

ResNet50 guarded official：

- closed-loop median：`44.865 ms`
- compress/decompress median：`22.550 / 22.635 ms`
- rel L2：`0.019808`
- finite rate：`1.000`
- ratio retention：`1.063`
- CUDA/fallback layers：`266 / 0`
- payload objects：`1`

ResNet18 supporting result：

- closed-loop median：`20.808 ms`
- compress/decompress median：`10.237 / 10.455 ms`
- rel L2：`0.020187`
- finite rate：`1.000`
- ratio retention：`1.047`
- CUDA/fallback layers：`101 / 0`
- payload objects：`1`

## 边界

- CPU safe grouped OpenMP/SIMD 仍然是 production/default path。
- installed CPU `.so`、CPU ABI 和 CPU wire format 均未改变。
- CUDA v4 是独立 experimental closed-loop codec，使用自己的 payload envelope 和 decode path。
- `FALCOM_CUDA_EXPERIMENTAL=0/unset` 仍然是纯 CPU fallback。

## 关键产物

- `logs/cuda_v4_guarded_all_cuda/final_guarded_all_cuda_report.md`
- `logs/cuda_v4_guarded_all_cuda/official_timing.csv`
- `logs/cuda_v4_guarded_all_cuda/quality.csv`
- `logs/cuda_v4_guarded_all_cuda/layer_backend_diff.csv`
