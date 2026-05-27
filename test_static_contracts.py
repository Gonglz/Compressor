#!/usr/bin/env python3
"""Static contract checks for the C compressor and Python ctypes wrapper."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
C_FINAL = (ROOT / "momentum_compressor_openmp_simd_final.c").read_text()
C_BASE = (ROOT / "momentum_compressor_final.c").read_text()
HEADER = (ROOT / "momentum_compressor.h").read_text()
WRAPPER = (ROOT / "EB-FaLCom/src/appfl/compressor/FalComC.py").read_text()
BUILD_LIB = (ROOT / "build_lib.sh").read_text()


def test_ndarray_ctypes_layout_matches_c_header() -> None:
    c_layout = re.search(r"typedef struct \{\s*void \*data;.*?DataType dtype;.*?\} NDArray;", HEADER, re.S)
    assert c_layout, "C NDArray layout not found"

    wrapper_fields = re.search(r"class NDArrayC.*?_fields_ = \[(.*?)\]", WRAPPER, re.S)
    assert wrapper_fields, "Python NDArrayC fields not found"
    fields = re.findall(r'\("([^"]+)"', wrapper_fields.group(1))

    assert fields == ["data", "shape", "ndim", "total_size", "dtype"]


def test_wrapper_uses_struct_returning_layer_api() -> None:
    assert "class CompressedLayerDataC" in WRAPPER
    assert "momentum_compressor_compress_layer.restype = ctypes.POINTER(CompressedLayerDataC)" in WRAPPER
    assert "ctypes.byref(out_size)" not in WRAPPER
    assert "momentum_compressor_decompress_layer.argtypes = [" in WRAPPER
    assert "ctypes.POINTER(CompressedLayerDataC)" in WRAPPER


def test_compressed_layer_shape_supports_max_ndim() -> None:
    assert "size_t shape[MAX_NDIM]" in HEADER


def test_sz3_memcpy_codec_is_decompressible() -> None:
    assert 'strcmp(codec, "sz3_memcpy") == 0' in C_FINAL
    assert 'strcmp(codec, "sz3_memcpy") == 0' in C_BASE
    assert 'strcmp(compressed->codec, "sz3") == 0 ||\n            strcmp(compressed->codec, "sz3_memcpy") == 0' in C_FINAL
    assert 'strcmp(compressed->codec, "sz3") == 0 ||\n            strcmp(compressed->codec, "sz3_memcpy") == 0' in C_BASE


def test_direct_decompression_failures_do_not_return_zero_arrays() -> None:
    expected_null_guard = '''if (!decompressed) {
            fprintf(stderr, "[ERROR] Decompression failed for codec: %s\\n", compressed->codec);
            ndarray_destroy(result);
            return NULL;
        }'''
    expected_size_guard = '''if (decompressed_size!= expected_size) {
            fprintf(stderr, "[ERROR] Size mismatch: got %zu, expected %zu\\n",
                    decompressed_size, expected_size);
            free(decompressed);
            ndarray_destroy(result);
            return NULL;
        }'''
    assert expected_null_guard in C_FINAL
    assert expected_null_guard in C_BASE
    assert expected_size_guard in C_FINAL
    assert expected_size_guard in C_BASE


def test_direct_decompression_maps_all_supported_dtypes() -> None:
    assert 'else if (strcmp(compressed->original_dtype, "uint8") == 0) dtype = DTYPE_UINT8;' in C_FINAL
    assert 'else if (strcmp(compressed->original_dtype, "uint8") == 0) dtype = DTYPE_UINT8;' in C_BASE
    assert 'Unknown original dtype: %s' in C_FINAL
    assert 'Unknown original dtype: %s' in C_BASE


def test_dominant_sign_matches_python_positive_vs_negative_rule() -> None:
    assert "info.dominant_sign = (positives >= negatives)? 1: -1;" in C_FINAL
    assert "return (positives >= negatives)? 1: -1;" in C_BASE


def test_openmp_memory_creation_does_not_leak_temporary_ndarrays() -> None:
    assert "ndarray_destroy(layer_memory);" in C_FINAL
    assert "layer_memory_array = get_prediction_memory_for_named_layer" in C_FINAL
    assert "layer_memory_array already points to the stored hash entry" in C_FINAL


def test_history_key_is_layer_identity_not_shape_only() -> None:
    body = _function_body(C_FINAL, "make_history_key")
    assert "client_id" in body
    assert "layer_name" in body
    assert "shape" not in body
    assert "conv_(" not in body
    assert "make_prediction_memory_key" not in body


def test_model_compress_sets_client_context_and_checks_failures() -> None:
    assert "momentum_compressor_set_client(compressor, client_id);" in C_FINAL
    assert "momentum_compressor_set_client(compressor, client_id);" in C_BASE
    assert "if (!compressed_layers[i])" in C_FINAL
    assert "if (!compressed_layers[i])" in C_BASE


def test_model_decompress_initializes_stack_layer_metadata() -> None:
    assert "CompressedLayerData layer_data;\n        memset(&layer_data, 0, sizeof(layer_data));" in C_FINAL
    assert "CompressedLayerData layer_data;\n        memset(&layer_data, 0, sizeof(layer_data));" in C_BASE
    assert "strncpy(layer_data.stored_dtype, dtype_str, sizeof(layer_data.stored_dtype) - 1);" in C_FINAL
    assert "strncpy(layer_data.stored_dtype, dtype_str, sizeof(layer_data.stored_dtype) - 1);" in C_BASE


def test_default_build_embeds_sz3_runtime_search_path() -> None:
    assert '-Wl,-rpath,$SZ3_LIB_PATH' in BUILD_LIB


def test_openmp_batch_does_not_expose_unsafe_experimental_parallel_path() -> None:
    assert "MOMENTUM_COMPRESSOR_EXPERIMENTAL_BATCH_PARALLEL" not in C_FINAL
    assert "NULL  // [FIXME] notecompressednoteprediction_memory" not in C_FINAL
    assert "momentum_compressor_compress_layer_pure" not in C_FINAL
    assert "LayerResult" not in C_FINAL


def test_openmp_batch_uses_safe_grouped_parallel_pipeline() -> None:
    assert "FALCOM_BATCH_SERIAL" in C_FINAL
    assert "FALCOM_BATCH_PARALLEL" in C_FINAL
    assert "FALCOM_BATCH_BREAKDOWN" in C_FINAL
    assert "FALCOM_VERIFY_PARALLEL" in C_FINAL
    assert "BatchGroup" in C_FINAL
    assert "BatchItemMeta" in C_FINAL
    assert "BatchComputeResult" in C_FINAL
    assert "process_batch_group" in C_FINAL
    has_parallel_for = "#pragma omp parallel for schedule(dynamic, 1)" in C_FINAL
    has_split_parallel_for = (
        "#pragma omp parallel" in C_FINAL and
        "#pragma omp for schedule(dynamic, 1)" in C_FINAL
    )
    assert has_parallel_for or has_split_parallel_for
    assert "local_prediction_memory" in C_FINAL
    assert "commit_grouped_batch_results" in C_FINAL


def _function_body(source: str, name: str) -> str:
    start = source.index(f" {name}(")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:index + 1]
    raise AssertionError(f"function body not found: {name}")


def test_grouped_compute_stage_does_not_mutate_global_state() -> None:
    body = _function_body(C_FINAL, "process_batch_group")
    forbidden = [
        "increment_step_count",
        "add_gradient_to_history",
        "set_prediction_memory_for_layer",
        "get_prediction_memory_for_layer",
        "HASH_ADD",
        "HASH_DEL",
        "items[item_index].compressed",
        "items[i].compressed",
        "stats.total_compressions",
    ]
    for token in forbidden:
        assert token not in body


def test_batch_breakdown_is_env_gated() -> None:
    body = _function_body(C_FINAL, "momentum_compressor_compress_batch")
    assert 'env_flag_enabled("FALCOM_BATCH_BREAKDOWN")' in body
    assert 'printf("[BATCH_BREAKDOWN]' in body


def test_release_logging_and_breakdown_default_off() -> None:
    assert "static int g_enable_breakdown = 0;" in C_FINAL
    assert "static volatile int g_wall_timing_enabled = 0;" in C_FINAL
    create_body = _function_body(C_FINAL, "momentum_compressor_create")
    assert 'getenv("FALCOM_LOG_LEVEL")' in create_body
    assert "LOG_LEVEL_WARNING" in create_body
    assert "mc->log_level = 1" not in create_body


def test_legacy_and_batch_breakdown_envs_are_distinct() -> None:
    body = _function_body(C_FINAL, "momentum_compressor_compress_batch")
    assert 'env_flag_enabled("FALCOM_LEGACY_BREAKDOWN")' in body
    assert 'env_flag_enabled("FALCOM_BATCH_BREAKDOWN")' in body
    assert "g_wall_timing_enabled = (legacy_breakdown_enabled || batch_breakdown_enabled)? 1: 0;" in body
    assert 'printf("[BREAKDOWN]' in body
    assert 'printf("[BATCH_BREAKDOWN]' in body
    legacy_index = body.index('env_flag_enabled("FALCOM_LEGACY_BREAKDOWN")')
    batch_index = body.index('env_flag_enabled("FALCOM_BATCH_BREAKDOWN")')
    assert legacy_index!= batch_index


def test_batch_info_logging_is_log_level_gated() -> None:
    body = _function_body(C_FINAL, "momentum_compressor_compress_batch")
    for marker in [
        'printf("[INFO] Batch compress: processing',
        'printf("[INFO] Batch compress: %zu/%zu layers succeeded',
    ]:
        pos = body.index(marker)
        guard_window = body[max(0, pos - 220):pos]
        assert "compressor->log_level <= LOG_LEVEL_INFO" in guard_window


def test_appfl_profile_is_env_gated_and_buffered() -> None:
    assert "FALCOM_APPFL_PROFILE" in WRAPPER
    assert "FALCOM_APPFL_PROFILE_CSV" in WRAPPER
    assert "torch.cuda.synchronize()" in WRAPPER
    assert "writer.writerows(rows)" in WRAPPER
    assert "writer.writerow(row)" not in WRAPPER


def test_openmp_final_has_no_unconditional_debug_logging() -> None:
    assert "[DEBUG" not in C_FINAL


if __name__ == "__main__":
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
    print(f"ok - {len(tests)} static contract checks passed")
