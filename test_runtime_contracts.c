#include "momentum_compressor.h"

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
    config.param_count_threshold = 1024 * 1024;
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

static int test_uint8_direct_roundtrip(void) {
    MomentumCompressor *compressor = create_test_compressor();
    if (!compressor) return fail("failed to create compressor");

    size_t shape[1] = {4};
    NDArray *array = ndarray_create(shape, 1, DTYPE_UINT8);
    if (!array) return fail("failed to create uint8 array");
    uint8_t values[4] = {1, 2, 3, 4};
    memcpy(array->data, values, sizeof(values));

    CompressedLayerData *compressed = momentum_compressor_compress_layer(
        compressor, "metadata", array);
    if (!compressed) return fail("uint8 compression failed");

    NDArray *decoded = momentum_compressor_decompress_layer(
        compressor, compressed, "client_0", "metadata");
    if (!decoded) return fail("uint8 decompression failed");
    if (decoded->dtype != DTYPE_UINT8) return fail("uint8 dtype was not preserved");
    if (decoded->total_size != 4) return fail("uint8 total_size changed");
    if (memcmp(decoded->data, values, sizeof(values)) != 0) return fail("uint8 payload changed");

    ndarray_destroy(decoded);
    compressed_layer_data_free(compressed);
    ndarray_destroy(array);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_sz3_memcpy_direct_decompress(void) {
    MomentumCompressor *compressor = create_test_compressor();
    if (!compressor) return fail("failed to create compressor");

    float values[2] = {1.25f, -2.5f};
    CompressedLayerData layer;
    memset(&layer, 0, sizeof(layer));
    strcpy(layer.type, "direct");
    strcpy(layer.codec, "sz3_memcpy");
    layer.data = (uint8_t *)values;
    layer.data_size = sizeof(values);
    layer.shape[0] = 2;
    layer.ndim = 1;
    strcpy(layer.original_dtype, "float32");
    strcpy(layer.stored_dtype, "float32");

    NDArray *decoded = momentum_compressor_decompress_layer(
        compressor, &layer, "client_0", "manual_sz3_memcpy");
    if (!decoded) return fail("sz3_memcpy decompression failed");
    if (decoded->dtype != DTYPE_FLOAT32) return fail("sz3_memcpy dtype mismatch");
    if (memcmp(decoded->data, values, sizeof(values)) != 0) return fail("sz3_memcpy payload changed");

    ndarray_destroy(decoded);
    momentum_compressor_destroy(compressor);
    return 0;
}

static int test_direct_decompress_size_mismatch_returns_null(void) {
    MomentumCompressor *compressor = create_test_compressor();
    if (!compressor) return fail("failed to create compressor");

    float value = 1.0f;
    CompressedLayerData layer;
    memset(&layer, 0, sizeof(layer));
    strcpy(layer.type, "direct");
    strcpy(layer.codec, "sz3_memcpy");
    layer.data = (uint8_t *)&value;
    layer.data_size = sizeof(value);
    layer.shape[0] = 2;
    layer.ndim = 1;
    strcpy(layer.original_dtype, "float32");
    strcpy(layer.stored_dtype, "float32");

    NDArray *decoded = momentum_compressor_decompress_layer(
        compressor, &layer, "client_0", "bad_sz3_memcpy");
    if (decoded) {
        ndarray_destroy(decoded);
        return fail("size mismatch returned a decoded array");
    }

    momentum_compressor_destroy(compressor);
    return 0;
}

int main(void) {
    if (test_uint8_direct_roundtrip() != 0) return 1;
    if (test_sz3_memcpy_direct_decompress() != 0) return 1;
    if (test_direct_decompress_size_mismatch_returns_null() != 0) return 1;
    printf("ok - runtime contracts passed\n");
    return 0;
}
