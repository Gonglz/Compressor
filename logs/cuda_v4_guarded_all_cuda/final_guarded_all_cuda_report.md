# CUDA v4 Guarded All-Valid-CUDA 报告

这是 CUDA v4 q8 的最后一个 narrow guard-validation patch。它没有创建 v5，没有修改 CPU compressor ABI、CPU wire format、installed `.so`、v2 或 v3。

## 官方最小 instrumentation timing

| model | threshold | guarded | closed-loop median ms | IQR | compress ms | decompress ms | CUDA layers | fallback layers | payload objects | rel L2 | finite | ratio retention | quality | acceptance |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| resnet18 | 1024 | no | 24.824 | 0.308 | 13.287 | 11.537 | 21 | 80 | 2 | 0.020185 | 1.000 | 1.041 | yes | comparison_quality_pass |
| resnet18 | 2048 | no | 26.464 | 0.333 | 14.292 | 12.330 | 20 | 81 | 2 | 0.020185 | 1.000 | 1.040 | yes | comparison_quality_pass |
| resnet18 | 512 | no | 22.575 | 0.771 | 11.787 | 10.534 | 41 | 60 | 2 | 0.020186 | 1.000 | 1.044 | yes | comparison_quality_pass |
| resnet18 | guarded_all_cuda | yes | 20.808 | 0.692 | 10.237 | 10.455 | 101 | 0 | 1 | 0.020187 | 1.000 | 1.047 | yes | supporting_pass |
| resnet50 | 1024 | no | 54.728 | 0.265 | 28.992 | 25.552 | 98 | 168 | 2 | 0.019807 | 1.000 | 1.056 | yes | comparison_quality_pass |
| resnet50 | 2048 | no | 60.828 | 1.586 | 32.512 | 28.067 | 69 | 197 | 2 | 0.019806 | 1.000 | 1.052 | yes | comparison_quality_pass |
| resnet50 | 512 | no | 51.525 | 1.489 | 26.967 | 24.691 | 142 | 124 | 2 | 0.019808 | 1.000 | 1.059 | yes | comparison_quality_pass |
| resnet50 | guarded_all_cuda | yes | 44.865 | 1.151 | 22.550 | 22.635 | 266 | 0 | 1 | 0.019808 | 1.000 | 1.063 | yes | accepted_v4_final_selected_path |

## 最终选定 GPU 实验配置

- codec：`cuda_v4_q8`
- mode：`guarded_all_cuda`
- threshold：`N/A / all valid CUDA`
- ResNet50 closed-loop median：`44.865 ms`
- ResNet50 compress/decompress median：`22.550 / 22.635 ms`
- CUDA/fallback layers：`266 / 0`
- payload objects：`1`
- 数值 threshold 只作为对照，不是最终推荐路径。

## Layer backend diff

- ResNet50 在 official threshold 2048 下有 197 个 fallback layer 被 guarded mode 转为 CUDA。
- ResNet50 在 guarded mode 下仍 blocked 的 official threshold 2048 fallback layer 数量为 0。
- 被转换的 fallback reason 全部是 `numel_below_threshold=197`。

## 决策

- `guarded_all_cuda` 被接受为 CUDA v4 final selected experimental path。ResNet50 closed-loop median 为 `44.865 ms`，满足 `<=55.5 ms` 目标，并且质量指标通过。
- 不建议继续 v5、新 codec、kernel math 优化或 compressor 内部 envelope writer/parser patch。
- 只有在未来端到端 APPFL/pipeline profiling 证明 envelope 是系统瓶颈时，才重新考虑 envelope 工作。

## Regression notes

- `libmomentum_compressor_openmp_simd_final.so` SHA256：`fba66a70ea2f4fb0f6bbfdd4c9eedd3fa94ebd0d2704e1087fe4fb97b4df4b23`
- `EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so` SHA256：`fba66a70ea2f4fb0f6bbfdd4c9eedd3fa94ebd0d2704e1087fe4fb97b4df4b23`
- CPU gates：PASS（`test_static_contracts.py`、gcc syntax-only with SZ3 include、batch state equivalence、reference state oracle、两个 `.so` 的 ctypes load）。
- CUDA v4 gates：PASS（extension import、q8 quality、mixed fallback、experimental gate、backend ownership、transaction rollback、separate encoder/decoder state、packed envelope、guarded all-valid-CUDA gate）。
- CPU `.so` hash 在 guarded official run 前后未变化。

## 产物

- `official_timing.csv`
- `quality.csv`
- `layer_backend_diff.csv`
