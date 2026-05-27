#define _POSIX_C_SOURCE 200809L

#include "momentum_compressor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

static MomentumCompressor *create_oracle_compressor(void) {
    CompressorConfig config = momentum_compressor_default_config();
    config.param_count_threshold = 4;
    config.error_bound = 1.0f;
    strcpy(config.lossless_compressor, "zstd");
    strcpy(config.error_bounding_mode, "REL");
    strcpy(config.sz3_lib_path, "/home/exouser/.appfl/.compressor/SZ3/lib/libSZ3c.so");

    MomentumCompressor *compressor = momentum_compressor_create(&config);
    if (compressor) {
        momentum_compressor_set_client(compressor, "client_0");
    }
    return compressor;
}

static NDArray *make_array(const size_t *shape, size_t ndim, DataType dtype, int seed) {
    NDArray *array = ndarray_create(shape, ndim, dtype);
    if (!array) return NULL;

    if (dtype == DTYPE_FLOAT32) {
        float *data = (float *)array->data;
        for (size_t i = 0; i < array->total_size; i++) {
            data[i] = sinf((float)(i + 1) * 0.13f + (float)seed * 0.17f) +
                      cosf((float)(i + 3) * 0.05f - (float)seed * 0.11f);
        }
    } else if (dtype == DTYPE_UINT8) {
        uint8_t *data = (uint8_t *)array->data;
        for (size_t i = 0; i < array->total_size; i++) {
            data[i] = (uint8_t)((seed + (int)i * 7) & 0xff);
        }
    }
    return array;
}

static void free_batch(BatchItem *items, size_t count) {
    if (!items) return;
    for (size_t i = 0; i < count; i++) {
        compressed_layer_data_free(items[i].compressed);
        ndarray_destroy((NDArray *)items[i].gradient);
    }
    free(items);
}

static int run_serial_batch(MomentumCompressor *compressor, BatchItem *items, size_t count) {
    setenv("FALCOM_BATCH_SERIAL", "1", 1);
    unsetenv("FALCOM_BATCH_PARALLEL");
    return momentum_compressor_compress_batch(compressor, items, count, "client_0");
}

static int decoded_finite(MomentumCompressor *compressor,
                          const CompressedLayerData *compressed,
                          const char *layer_name) {
    NDArray *decoded = momentum_compressor_decompress_layer(
        compressor, compressed, "client_0", layer_name);
    if (!decoded) return fail("decompression returned NULL");
    if (decoded->dtype == DTYPE_FLOAT32) {
        const float *data = (const float *)decoded->data;
        for (size_t i = 0; i < decoded->total_size; i++) {
            if (!isfinite(data[i])) {
                ndarray_destroy(decoded);
                return fail("decompression produced non-finite value");
            }
        }
    }
    ndarray_destroy(decoded);
    return 0;
}

static int same_compressed_shape_dtype(const CompressedLayerData *a,
                                       const CompressedLayerData *b) {
    if (!a || !b) return fail("compressed result is NULL");
    if (a->ndim != b->ndim) return fail("ndim mismatch");
    if (strcmp(a->stored_dtype, b->stored_dtype) != 0) return fail("stored_dtype mismatch");
    if (strcmp(a->type, b->type) != 0) return fail("type mismatch");
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return fail("shape mismatch");
    }
    return 0;
}

static int test_repeated_layer_step_chain(void) {
    MomentumCompressor *compressor = create_oracle_compressor();
    if (!compressor) return fail("failed to create compressor");

    size_t shape[4] = {2, 2, 2, 2};
    BatchItem *items = (BatchItem *)calloc(3, sizeof(BatchItem));
    if (!items) return fail("failed to allocate items");
    for (size_t i = 0; i < 3; i++) {
        items[i].layer_name = "convA.weight";
        items[i].gradient = make_array(shape, 4, DTYPE_FLOAT32, (int)i + 1);
    }

    if (run_serial_batch(compressor, items, 3) != 0) return fail("serial repeated layer batch failed");
    for (int i = 0; i < 3; i++) {
        if (!items[i].compressed) return fail("missing compressed item");
        if (items[i].compressed->step != i + 1) return fail("step did not increment through repeated layer");
        if (decoded_finite(compressor, items[i].compressed, items[i].layer_name) != 0) return 1;
    }

    free_batch(items, 3);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_interleaved_layer_independent_steps(void) {
    MomentumCompressor *compressor = create_oracle_compressor();
    if (!compressor) return fail("failed to create compressor");

    const char *names[] = {
        "convA.weight", "fcB.bias", "convA.weight", "bnC.bias", "fcB.bias", "convA.weight"
    };
    size_t shape4[4] = {2, 2, 2, 2};
    size_t shape1[1] = {8};
    BatchItem *items = (BatchItem *)calloc(6, sizeof(BatchItem));
    if (!items) return fail("failed to allocate items");
    for (size_t i = 0; i < 6; i++) {
        items[i].layer_name = names[i];
        if (strstr(names[i], "conv")) {
            items[i].gradient = make_array(shape4, 4, DTYPE_FLOAT32, (int)i + 10);
        } else {
            items[i].gradient = make_array(shape1, 1, DTYPE_FLOAT32, (int)i + 10);
        }
    }

    if (run_serial_batch(compressor, items, 6) != 0) return fail("serial interleaved batch failed");
    int expected_steps[] = {1, 1, 2, 1, 2, 3};
    for (size_t i = 0; i < 6; i++) {
        if (items[i].compressed->step != expected_steps[i]) {
            return fail("interleaved layer step chain mismatch");
        }
    }

    free_batch(items, 6);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_same_shape_different_layers_are_independent(void) {
    size_t shape[4] = {2, 2, 2, 2};
    MomentumCompressor *shared = create_oracle_compressor();
    MomentumCompressor *only_a = create_oracle_compressor();
    MomentumCompressor *only_b = create_oracle_compressor();
    if (!shared || !only_a || !only_b) return fail("failed to create compressors");

    for (int round = 0; round < 3; round++) {
        BatchItem *shared_items = (BatchItem *)calloc(2, sizeof(BatchItem));
        BatchItem *a_item = (BatchItem *)calloc(1, sizeof(BatchItem));
        BatchItem *b_item = (BatchItem *)calloc(1, sizeof(BatchItem));
        if (!shared_items || !a_item || !b_item) return fail("failed to allocate items");

        shared_items[0].layer_name = "convA.weight";
        shared_items[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 100 + round);
        shared_items[1].layer_name = "convB.weight";
        shared_items[1].gradient = make_array(shape, 4, DTYPE_FLOAT32, 200 + round);

        a_item[0].layer_name = "convA.weight";
        a_item[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 100 + round);
        b_item[0].layer_name = "convB.weight";
        b_item[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 200 + round);

        if (run_serial_batch(shared, shared_items, 2) != 0) return fail("shared compressor failed");
        if (run_serial_batch(only_a, a_item, 1) != 0) return fail("A-only compressor failed");
        if (run_serial_batch(only_b, b_item, 1) != 0) return fail("B-only compressor failed");

        if (same_compressed_shape_dtype(shared_items[0].compressed, a_item[0].compressed) != 0) return 1;
        if (same_compressed_shape_dtype(shared_items[1].compressed, b_item[0].compressed) != 0) return 1;
        if (shared_items[0].compressed->data_size != a_item[0].compressed->data_size) {
            return fail("convA same-shape reference data_size mismatch");
        }
        if (shared_items[1].compressed->data_size != b_item[0].compressed->data_size) {
            return fail("convB appears to share prediction memory with convA");
        }

        free_batch(shared_items, 2);
        free_batch(a_item, 1);
        free_batch(b_item, 1);
    }

    momentum_compressor_destroy(shared);
    momentum_compressor_destroy(only_a);
    momentum_compressor_destroy(only_b);
    return 0;
}

static int test_same_shape_history_prev_grad_is_layer_isolated(void) {
    size_t shape[4] = {2, 2, 2, 2};
    MomentumCompressor *shared = create_oracle_compressor();
    MomentumCompressor *only_a = create_oracle_compressor();
    MomentumCompressor *only_b = create_oracle_compressor();
    if (!shared || !only_a || !only_b) return fail("failed to create compressors");

    for (int round = 0; round < 3; round++) {
        BatchItem *shared_items = (BatchItem *)calloc(2, sizeof(BatchItem));
        BatchItem *a_item = (BatchItem *)calloc(1, sizeof(BatchItem));
        BatchItem *b_item = (BatchItem *)calloc(1, sizeof(BatchItem));
        if (!shared_items || !a_item || !b_item) return fail("failed to allocate history isolation items");

        shared_items[0].layer_name = "same_shape_A.weight";
        shared_items[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 700 + round);
        shared_items[1].layer_name = "same_shape_B.weight";
        shared_items[1].gradient = make_array(shape, 4, DTYPE_FLOAT32, 800 + round);

        a_item[0].layer_name = "same_shape_A.weight";
        a_item[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 700 + round);
        b_item[0].layer_name = "same_shape_B.weight";
        b_item[0].gradient = make_array(shape, 4, DTYPE_FLOAT32, 800 + round);

        if (run_serial_batch(shared, shared_items, 2) != 0) return fail("shared history isolation batch failed");
        if (run_serial_batch(only_a, a_item, 1) != 0) return fail("A-only history isolation batch failed");
        if (run_serial_batch(only_b, b_item, 1) != 0) return fail("B-only history isolation batch failed");

        if (shared_items[0].compressed->step != a_item[0].compressed->step) {
            return fail("convA history step was affected by same-shape convB");
        }
        if (shared_items[1].compressed->step != b_item[0].compressed->step) {
            return fail("convB history step was affected by same-shape convA");
        }
        if (strcmp(shared_items[0].compressed->type, a_item[0].compressed->type) != 0 ||
            strcmp(shared_items[1].compressed->type, b_item[0].compressed->type) != 0) {
            return fail("same-shape layer history changed compression mode");
        }
        if (shared_items[0].compressed->data_size != a_item[0].compressed->data_size ||
            shared_items[1].compressed->data_size != b_item[0].compressed->data_size) {
            return fail("same-shape layer history appears to share prev_grad");
        }

        free_batch(shared_items, 2);
        free_batch(a_item, 1);
        free_batch(b_item, 1);
    }

    momentum_compressor_destroy(shared);
    momentum_compressor_destroy(only_a);
    momentum_compressor_destroy(only_b);
    return 0;
}

static int test_shape_change_does_not_use_incompatible_history(void) {
    MomentumCompressor *compressor = create_oracle_compressor();
    if (!compressor) return fail("failed to create compressor");
    size_t shape1[4] = {2, 2, 2, 2};
    size_t shape2[4] = {3, 2, 2, 2};

    BatchItem *items = (BatchItem *)calloc(3, sizeof(BatchItem));
    if (!items) return fail("failed to allocate items");
    for (size_t i = 0; i < 3; i++) {
        items[i].layer_name = "layerX.weight";
    }
    items[0].gradient = make_array(shape1, 4, DTYPE_FLOAT32, 301);
    items[1].gradient = make_array(shape2, 4, DTYPE_FLOAT32, 302);
    items[2].gradient = make_array(shape1, 4, DTYPE_FLOAT32, 303);

    if (run_serial_batch(compressor, items, 3) != 0) return fail("shape-change batch failed");
    if (items[0].compressed->step != 1 || items[1].compressed->step != 2 || items[2].compressed->step != 3) {
        return fail("shape-change steps mismatch");
    }
    if (strcmp(items[1].compressed->type, "momentum_predicted") == 0 ||
        strcmp(items[2].compressed->type, "momentum_predicted") == 0) {
        return fail("shape-change reused incompatible momentum history");
    }

    free_batch(items, 3);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_dtype_metadata_roundtrip(void) {
    MomentumCompressor *compressor = create_oracle_compressor();
    if (!compressor) return fail("failed to create compressor");
    size_t shape[1] = {8};
    NDArray *array = make_array(shape, 1, DTYPE_UINT8, 401);
    if (!array) return fail("failed to create uint8 array");

    CompressedLayerData *compressed = momentum_compressor_compress_layer(compressor, "metadata", array);
    if (!compressed) return fail("uint8 compression failed");
    if (strcmp(compressed->stored_dtype, "uint8") != 0) return fail("uint8 stored_dtype mismatch");
    NDArray *decoded = momentum_compressor_decompress_layer(compressor, compressed, "client_0", "metadata");
    if (!decoded) return fail("uint8 decode failed");
    if (decoded->dtype != DTYPE_UINT8) return fail("uint8 decoded dtype mismatch");

    ndarray_destroy(decoded);
    compressed_layer_data_free(compressed);
    ndarray_destroy(array);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_non_finite_inputs_are_documented_unsupported(void) {
    MomentumCompressor *compressor = create_oracle_compressor();
    if (!compressor) return fail("failed to create compressor");
    size_t shape[1] = {8};
    NDArray *array = ndarray_create(shape, 1, DTYPE_FLOAT32);
    if (!array) return fail("failed to create float array");
    float values[8] = {0.0f, -0.0f, NAN, INFINITY, -INFINITY, 1e-30f, 1e30f, -3.0f};
    memcpy(array->data, values, sizeof(values));

    CompressedLayerData *compressed = momentum_compressor_compress_layer(compressor, "nan_inf.weight", array);
    if (!compressed) {
        ndarray_destroy(array);
        momentum_compressor_destroy(compressor);
        return 0;
    }

    NDArray *decoded = momentum_compressor_decompress_layer(compressor, compressed, "client_0", "nan_inf.weight");
    if (decoded) {
        ndarray_destroy(decoded);
    }
    compressed_layer_data_free(compressed);
    ndarray_destroy(array);
    momentum_compressor_destroy(compressor);
    return 0;
}

int main(void) {
    if (test_repeated_layer_step_chain() != 0) return 1;
    if (test_interleaved_layer_independent_steps() != 0) return 1;
    if (test_same_shape_different_layers_are_independent() != 0) return 1;
    if (test_same_shape_history_prev_grad_is_layer_isolated() != 0) return 1;
    if (test_shape_change_does_not_use_incompatible_history() != 0) return 1;
    if (test_dtype_metadata_roundtrip() != 0) return 1;
    if (test_non_finite_inputs_are_documented_unsupported() != 0) return 1;
    unsetenv("FALCOM_BATCH_SERIAL");
    unsetenv("FALCOM_BATCH_PARALLEL");
    printf("ok - reference state oracle passed\n");
    return 0;
}
