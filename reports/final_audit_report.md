# Compressor Final Audit Report

Task: `audit-state-semantics-before-more-optimization`

Workspace: `/home/exouser/compressor/final`

Date: 2026-05-25 UTC

## Executive Summary

The audit found two real state-semantics bugs in the current safe path and fixed both without changing the public C API or Python ctypes ABI:

1. Prediction memory was keyed only by tensor shape, so different same-shape layers could share momentum state.
2. Momentum compression could use a previous gradient with incompatible shape or dtype.

After fixes, all correctness gates pass, ASAN/UBSAN runtime contracts pass, and the new safe grouped batch path remains within the target envelope:

- ResNet50 best safe parallel hot-round median: `211.36 ms` at 8 threads.
- ResNet18 best safe parallel hot-round median: `128.66 ms` at 8 threads.
- Compression ratio delta vs safe serial: `0.00%` for the safe grouped path in the measured scan.
- Decompress regression vs safe serial at the recommended 8-thread setting: no regression for ResNet50; ResNet18 is faster than serial at 8 threads.

Recommendation: install/keep the new safe grouped path as the default for the actual tree at:

```text
/home/exouser/compressor/final/EB-FaLCom/src/appfl/compressor/libmomentum_compressor.so
```

The plan path `/home/exouser/EB-FaLCom/...` does not exist on this machine.

## Files Changed

- `momentum_compressor_openmp_simd_final.c`
- `test_static_contracts.py`
- `test_batch_state_equivalence.c`
- `test_reference_state_oracle.c` added
- audit/perf logs under `logs/`

Public API unchanged:

- `momentum_compressor_compress_batch`
- `BatchItem`
- `CompressedLayerData`

## Bugs Found

### Bug 1: Prediction Memory Key Collision

Original behavior:

```text
prediction_memory_key = conv_(shape...)
```

Trigger:

```text
convA shape = [64, 64, 3, 3]
convB shape = [64, 64, 3, 3]
```

Impact:

`convA` and `convB` could share prediction memory under the same client, contaminating momentum state.

Fix:

```text
prediction_memory_key = layer_name + "|" + dtype + "|conv_(shape...)"
```

Key code locations:

- `make_prediction_memory_key`: `momentum_compressor_openmp_simd_final.c:1470`
- keyed prediction-memory lookup/set: `momentum_compressor_openmp_simd_final.c:1631`
- named wrappers: `momentum_compressor_openmp_simd_final.c:1669`
- batch prepass key generation: `momentum_compressor_openmp_simd_final.c:2571`

Covered by:

- `test_reference_state_oracle.c:test_same_shape_different_layers_are_independent`

### Bug 2: Incompatible History Entering Momentum Path

Original risk:

The same `client_id:layer_name` history could later see a tensor with changed shape/dtype, and momentum code could still consume the old `prev_grad`.

Fix:

Added `ndarray_layout_matches` and required layout equality before momentum compression in both serial and grouped paths.

Key code locations:

- layout guard helper: `momentum_compressor_openmp_simd_final.c:1495`
- grouped momentum guard: `momentum_compressor_openmp_simd_final.c:2724`
- serial momentum guard: `momentum_compressor_openmp_simd_final.c:3043`

Covered by:

- `test_reference_state_oracle.c:test_shape_change_does_not_use_incompatible_history`

### Bug 3: Batch Breakdown Was Not Env-Gated

`[BATCH_BREAKDOWN]` output is now gated behind:

```text
FALCOM_BATCH_BREAKDOWN=1
```

Fallback priority:

```text
FALCOM_BATCH_SERIAL=1            highest priority
FALCOM_BATCH_PARALLEL=0          force serial
default                          safe grouped parallel
FALCOM_BATCH_BREAKDOWN=1         print grouped batch timing
```

Covered by:

- `test_static_contracts.py:test_batch_breakdown_is_env_gated`

## Final State Semantics

### Step

Serial semantics are:

```text
increment_step_count(history_key)
current_step = get_step_count(history_key)
CompressedLayerData.step = current_step
```

Grouped path:

- prepass snapshots initial global step
- group compute uses a local step chain
- commit increments global step once per item in original batch order

For repeated same-layer items, expected step chain is `1, 2, 3, ...`.

### History

History key:

```text
history_key = client_id + ":" + layer_name
```

`add_gradient_to_history` deep-copies input data. Grouped compute stores private result buffers and history copies, then serial commit transfers ownership into global history. Commit processes all items, not only the final item per group.

Known semantic limitation:

The original serial path may store a simulated reconstructed gradient after lossy compression. The grouped contract currently verifies output/decompressed equivalence within lossy tolerance and preserves the local state chain, but exact reconstructed-history bit equivalence is not asserted. If future work requires exact reconstructed-history matching, grouped compute should route its `history_gradient` through the same reconstruction function before commit.

### Prediction Memory

Prediction memory key:

```text
client_id -> layer_name|dtype|shape
```

Lifecycle:

- missing memory initializes as zeros in momentum path
- momentum path reads/writes prediction memory
- direct/generic paths do not use prediction memory
- stored prediction memory is deep-copied
- grouped path uses a group-local prediction-memory chain and commits final group state serially

### Key Collision Risk

Resolved:

- same-shape different layers no longer share prediction memory
- same layer with different dtype no longer shares prediction memory

Remaining:

- `history_key` still aliases only by `client_id:layer_name`.
- Reusing the same `client_id` and layer names across different model graphs can intentionally or accidentally share history/step state.
- No model identity exists in the public ABI.

Recommended future debug-only guard:

```text
FALCOM_DEBUG_KEY_COLLISIONS=1
```

Track last `{dtype, ndim, shape, total_size}` per `history_key` and warn on changes.

## Old v22 Upper Bound Audit

Source: `ompv22sz3.c`

Unsafe findings:

- prediction memory key is shape-only: `ompv22sz3.c:1607`
- pure compute increments global step in an OpenMP region: `ompv22sz3.c:2427`
- batch compute calls pure function under `#pragma omp parallel for`: `ompv22sz3.c:4025`
- commit has TODO/FIXME prediction-memory persistence with `NULL`: `ompv22sz3.c:4073`

Conclusion:

v22 remains an unsafe performance ceiling reference only. It should not be merged into the default path.

## Correctness Gates

Logs:

- `logs/audit_after_fix/static_contracts.log`
- `logs/audit_after_fix/fsyntax.log`
- `logs/audit_after_fix/runtime_contracts.log`
- `logs/audit_after_fix/batch_state_equivalence.log`
- `logs/audit_after_fix/reference_state_oracle.log`
- `logs/audit_after_fix/so_load.log`
- `logs/sanitizer/runtime_asan.log`

Results:

```text
python3 test_static_contracts.py                         PASS, 16 checks
gcc -fsyntax-only momentum_compressor_openmp_simd_final.c PASS, no warnings
./test_runtime_contracts_openmp                          PASS
./test_batch_state_equivalence                           PASS
./test_reference_state_oracle                            PASS
ctypes load local .so                                    PASS
ctypes load installed .so                                PASS
ASAN/UBSAN runtime contracts                             PASS
```

The runtime contract intentionally prints `[ERROR] Size mismatch: got 4, expected 8` as part of a negative test; the executable exits successfully.

## Failure Path

Added fault injection:

```text
FALCOM_FAULT_INJECT_COMPRESS_FAIL_AT=N
```

Covered case:

- first item succeeds privately
- second item fails during grouped compute
- public `items[i].compressed` remains uncommitted
- temporary payloads are released through pipeline cleanup

Test:

```text
test_batch_state_equivalence.c:test_grouped_failure_does_not_half_commit_outputs
```

ASAN/UBSAN did not report use-after-free, double-free, or leak failures in runtime contracts.

## Performance Method

Compiled explicit benchmark binaries to avoid stale executable ambiguity:

```text
test_current_safe_perf   = test_v21_full.c + momentum_compressor_openmp_simd_final.c
test_unsafe_v22_perf     = test_v21_full.c + ompv22sz3.c
```

Environment:

```text
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_NUM_THREADS=1,2,4,8,16
```

Each config:

- 5 runs for median
- hot-round median uses rounds 1 and 2 only
- extra `FALCOM_BATCH_BREAKDOWN=1` run is excluded from timing median and only used for breakdown fields

Detailed CSV:

- `logs/perf_after_audit/resnet50.csv`
- `logs/perf_after_audit/resnet18.csv`
- `logs/perf_after_audit/summary.md`
- raw logs under `logs/perf_after_audit/raw/`

## ResNet50 Thread Scan

| threads | safe serial ms | safe grouped ms | unsafe v22 ms | speedup vs serial | grouped/v22 gap | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 936.56 | 1098.56 | 907.36 | 0.85x | 1.21x | 62.44 |
| 2 | 913.18 | 548.04 | 472.43 | 1.67x | 1.16x | 62.44 |
| 4 | 803.11 | 304.25 | 262.88 | 2.64x | 1.16x | 62.44 |
| 8 | 797.62 | 211.36 | 178.27 | 3.77x | 1.19x | 62.44 |
| 16 | 901.40 | 265.33 | 238.53 | 3.40x | 1.11x | 62.44 |

Best safe grouped setting:

```text
threads=8
compress median = 211.36 ms
decompress median = 273.43 ms
compression ratio = 62.44:1
speedup vs safe serial = 3.77x
gap vs unsafe v22 = 1.19x slower
```

## ResNet18 Thread Scan

| threads | safe serial ms | safe grouped ms | unsafe v22 ms | speedup vs serial | grouped/v22 gap | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 323.64 | 395.77 | 299.90 | 0.82x | 1.32x | 220.71 |
| 2 | 324.39 | 221.24 | 181.71 | 1.47x | 1.22x | 220.71 |
| 4 | 311.53 | 132.54 | 110.62 | 2.35x | 1.20x | 220.71 |
| 8 | 337.84 | 128.66 | 108.06 | 2.63x | 1.19x | 220.71 |
| 16 | 355.88 | 133.58 | 108.89 | 2.66x | 1.23x | 220.71 |

Best safe grouped setting:

```text
threads=8
compress median = 128.66 ms
decompress median = 118.81 ms
compression ratio = 220.71:1
speedup vs safe serial = 2.63x
gap vs unsafe v22 = 1.19x slower
```

## Breakdown Snapshot

Representative safe grouped breakdown values are recorded in the CSV from the excluded breakdown sample. For ResNet50 at 8 threads:

```text
prepass_ms ~= 1.1
compute_ms ~= 162.8
commit_ms ~= 16.1
sum_group_compute_ms ~= 993.5
max_group_compute_ms ~= 123.4
groups = 266
```

Interpretation:

- Batch parallelism is working; wall compute is much lower than sum of group compute.
- Max group compute remains the dominant lower bound.
- At 16 threads, overhead and imbalance reduce gains, so 8 threads is the recommended default benchmark point.

## Follow-up Experiments

No additional performance experiment was merged in this audit patch.

Reason:

- The audit itself fixed correctness bugs.
- The safe grouped path already meets the main performance targets.
- Mixing per-thread scratch/ZSTD context reuse into the same audit patch would blur correctness review.

Experiment status:

- per-thread scratch / `ZSTD_CCtx` reuse: not run; still the next safe experiment.
- dominant-group fine breakdown: not run.
- packing small allocation merge: not run.
- unsafe fast mode: not implemented and not recommended as default.

Previously known negative experiments remain excluded:

- group cost sorting: rolled back before this audit because it was a negative optimization.
- nested OpenMP for ResNet50: not adopted because it was a negative optimization.

## Runtime Verify Switch Note

`FALCOM_VERIFY_PARALLEL` is present as a diagnostic switch, but the full dual-run serial-vs-grouped verifier is implemented as an external C contract (`test_batch_state_equivalence`) rather than running inside the hot public API. This avoids needing to clone mutable compressor state inside production calls.

If always-on debug dual-run verification is still desired, it should be implemented as a separate patch with a deep compressor-state clone and strict limits on batch size.

## Merge Judgment

Can merge as default path:

- correctness contracts pass
- reference oracle passes
- sanitizer runtime pass
- ResNet50 safe grouped median is below `250 ms`
- ResNet18 safe grouped median is below `140 ms`
- compression ratio delta is within `1%`
- decompress regression is within `15%`
- `FALCOM_BATCH_SERIAL=1` fallback works
- `[BATCH_BREAKDOWN]` default output is gated
- unsafe v22 remains only a reference binary/source

Recommended default benchmark setting:

```text
OMP_NUM_THREADS=8
OMP_PROC_BIND=close
OMP_PLACES=cores
```
