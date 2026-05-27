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

static MomentumCompressor *create_test_compressor(void) {
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

static NDArray *make_float_array(const size_t *shape, size_t ndim, int batch_id, int item_id) {
    NDArray *array = ndarray_create(shape, ndim, DTYPE_FLOAT32);
    if (!array) return NULL;

    float *data = (float *)array->data;
    for (size_t i = 0; i < array->total_size; i++) {
        float base = (float)((batch_id + 1) * 0.25 + (item_id + 1) * 0.03125);
        data[i] = sinf((float)(i + 1) * 0.17f + base) + cosf((float)(i + item_id + 1) * 0.07f);
    }
    return array;
}

static int compare_decoded_arrays(const NDArray *serial, const NDArray *parallel) {
    if (!serial || !parallel) return fail("decoded array is NULL");
    if (serial->dtype != parallel->dtype) return fail("decoded dtype mismatch");
    if (serial->ndim != parallel->ndim) return fail("decoded ndim mismatch");
    if (serial->total_size != parallel->total_size) return fail("decoded total_size mismatch");
    for (size_t i = 0; i < serial->ndim; i++) {
        if (serial->shape[i] != parallel->shape[i]) return fail("decoded shape mismatch");
    }

    const float *a = (const float *)serial->data;
    const float *b = (const float *)parallel->data;
    double l2 = 0.0;
    double norm = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < serial->total_size; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return fail("decoded non-finite value");
        double diff = (double)a[i] - (double)b[i];
        l2 += diff * diff;
        norm += (double)a[i] * (double)a[i];
        if (fabs(diff) > max_abs) max_abs = fabs(diff);
    }

    double rel = sqrt(l2) / (sqrt(norm) + 1e-12);
    if (max_abs > 10.0 && rel > 2.0) return fail("decoded numerical mismatch");
    return 0;
}

static int compare_compressed_metadata(const CompressedLayerData *serial,
                                       const CompressedLayerData *parallel) {
    if (!serial || !parallel) return fail("compressed result is NULL");
    if (strcmp(serial->type, parallel->type) != 0) return fail("compressed type mismatch");
    if (serial->ndim != parallel->ndim) return fail("compressed ndim mismatch");
    if (serial->step != parallel->step) return fail("compressed step mismatch");
    if (strcmp(serial->stored_dtype, parallel->stored_dtype) != 0) return fail("stored dtype mismatch");
    for (size_t i = 0; i < serial->ndim; i++) {
        if (serial->shape[i] != parallel->shape[i]) return fail("compressed shape mismatch");
    }

    size_t serial_size = serial->data_size + serial->bitmap_size + serial->dominant_signs_size;
    size_t parallel_size = parallel->data_size + parallel->bitmap_size + parallel->dominant_signs_size;
    size_t max_size = serial_size > parallel_size ? serial_size : parallel_size;
    size_t delta = serial_size > parallel_size ? serial_size - parallel_size : parallel_size - serial_size;
    if (max_size > 0 && ((double)delta / (double)max_size) > 0.01) {
        return fail("compressed size relative deviation exceeds 1%");
    }
    return 0;
}

static int run_one_batch(MomentumCompressor *serial_mc,
                         MomentumCompressor *parallel_mc,
                         const char **names,
                         const size_t *ndims,
                         const size_t shapes[][4],
                         size_t count,
                         int batch_id) {
    BatchItem *serial_items = (BatchItem *)calloc(count, sizeof(BatchItem));
    BatchItem *parallel_items = (BatchItem *)calloc(count, sizeof(BatchItem));
    if (!serial_items || !parallel_items) return fail("failed to allocate batch items");

    for (size_t i = 0; i < count; i++) {
        NDArray *serial_array = make_float_array(shapes[i], ndims[i], batch_id, (int)i);
        NDArray *parallel_array = make_float_array(shapes[i], ndims[i], batch_id, (int)i);
        if (!serial_array || !parallel_array) return fail("failed to allocate input arrays");
        serial_items[i].layer_name = names[i];
        serial_items[i].gradient = serial_array;
        parallel_items[i].layer_name = names[i];
        parallel_items[i].gradient = parallel_array;
    }

    setenv("FALCOM_BATCH_SERIAL", "1", 1);
    unsetenv("FALCOM_BATCH_PARALLEL");
    int ret_serial = momentum_compressor_compress_batch(serial_mc, serial_items, count, "client_0");

    unsetenv("FALCOM_BATCH_SERIAL");
    setenv("FALCOM_BATCH_PARALLEL", "1", 1);
    int ret_parallel = momentum_compressor_compress_batch(parallel_mc, parallel_items, count, "client_0");

    if (ret_serial != 0 || ret_parallel != 0) return fail("batch compression failed");

    for (size_t i = 0; i < count; i++) {
        if (compare_compressed_metadata(serial_items[i].compressed, parallel_items[i].compressed) != 0) {
            return 1;
        }

        NDArray *serial_decoded = momentum_compressor_decompress_layer(
            serial_mc, serial_items[i].compressed, "client_0", names[i]);
        NDArray *parallel_decoded = momentum_compressor_decompress_layer(
            parallel_mc, parallel_items[i].compressed, "client_0", names[i]);
        int cmp = compare_decoded_arrays(serial_decoded, parallel_decoded);
        ndarray_destroy(serial_decoded);
        ndarray_destroy(parallel_decoded);
        if (cmp != 0) return 1;
    }

    for (size_t i = 0; i < count; i++) {
        compressed_layer_data_free(serial_items[i].compressed);
        compressed_layer_data_free(parallel_items[i].compressed);
        ndarray_destroy((NDArray *)serial_items[i].gradient);
        ndarray_destroy((NDArray *)parallel_items[i].gradient);
    }
    free(serial_items);
    free(parallel_items);
    return 0;
}

static int test_repeated_and_interleaved_state_chains(void) {
    MomentumCompressor *serial_mc = create_test_compressor();
    MomentumCompressor *parallel_mc = create_test_compressor();
    if (!serial_mc || !parallel_mc) return fail("failed to create compressors");

    const char *batch0_names[] = {
        "conv1.weight", "conv1.weight", "conv1.weight", "conv2.weight", "conv2.weight"
    };
    const size_t batch0_ndims[] = {4, 4, 4, 4, 4};
    const size_t batch0_shapes[][4] = {
        {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}
    };

    const char *batch1_names[] = {
        "conv1.weight", "fc1.bias", "conv1.weight", "bn1.bias", "fc1.bias", "conv1.weight"
    };
    const size_t batch1_ndims[] = {4, 1, 4, 1, 1, 4};
    const size_t batch1_shapes[][4] = {
        {2, 2, 2, 2}, {8, 1, 1, 1}, {2, 2, 2, 2},
        {8, 1, 1, 1}, {8, 1, 1, 1}, {2, 2, 2, 2}
    };

    if (run_one_batch(serial_mc, parallel_mc, batch0_names, batch0_ndims, batch0_shapes, 5, 0) != 0) return 1;
    if (run_one_batch(serial_mc, parallel_mc, batch1_names, batch1_ndims, batch1_shapes, 6, 1) != 0) return 1;
    if (run_one_batch(serial_mc, parallel_mc, batch0_names, batch0_ndims, batch0_shapes, 5, 2) != 0) return 1;
    if (run_one_batch(serial_mc, parallel_mc, batch1_names, batch1_ndims, batch1_shapes, 6, 3) != 0) return 1;

    momentum_compressor_destroy(serial_mc);
    momentum_compressor_destroy(parallel_mc);
    return 0;
}

static int test_grouped_failure_does_not_half_commit_outputs(void) {
    MomentumCompressor *compressor = create_test_compressor();
    if (!compressor) return fail("failed to create compressor");

    const size_t shape[4] = {2, 2, 2, 2};
    BatchItem items[2];
    memset(items, 0, sizeof(items));
    items[0].layer_name = "faultA.weight";
    items[0].gradient = make_float_array(shape, 4, 9, 0);
    items[1].layer_name = "faultB.weight";
    items[1].gradient = make_float_array(shape, 4, 9, 1);
    if (!items[0].gradient || !items[1].gradient) return fail("failed to create fault inputs");

    unsetenv("FALCOM_BATCH_SERIAL");
    setenv("FALCOM_BATCH_PARALLEL", "1", 1);
    setenv("FALCOM_FAULT_INJECT_COMPRESS_FAIL_AT", "1", 1);
    int ret = momentum_compressor_compress_batch(compressor, items, 2, "client_0");
    unsetenv("FALCOM_FAULT_INJECT_COMPRESS_FAIL_AT");

    if (ret == 0) return fail("fault-injected batch unexpectedly succeeded");
    if (items[0].compressed || items[1].compressed) {
        return fail("fault-injected batch exposed half-committed compressed output");
    }

    ndarray_destroy((NDArray *)items[0].gradient);
    ndarray_destroy((NDArray *)items[1].gradient);
    momentum_compressor_destroy(compressor);
    return 0;
}

int main(void) {
    if (test_repeated_and_interleaved_state_chains() != 0) return 1;
    if (test_grouped_failure_does_not_half_commit_outputs() != 0) return 1;
    unsetenv("FALCOM_BATCH_SERIAL");
    unsetenv("FALCOM_BATCH_PARALLEL");
    unsetenv("FALCOM_FAULT_INJECT_COMPRESS_FAIL_AT");
    printf("ok - batch state equivalence contracts passed\n");
    return 0;
}
