/**
 * ===========================================================================
 * MomentumPredictorCompressor 
 * 
 * 对齐目标: src/appfl/compressor/momentum_predictor_compressor.py
 * 
 * Momentum-based predictor compressor for federated learning gradients.
 * 
 * 核心特性:
 * - 仅当层名含 'weight'、元素数 > param_cutoff、且 dtype 为 float32/64 时，才做有损（SZ3/动量预测）
 * - 其他一律无损（pickle），避免 int64 等 dtype 触发 SZ3 解压类型错误
 * - 记录 codec（'sz3' | 'pickle'）与 stored_dtype，解压严格按记录执行
 * - direct/generic 解压后写入历史，确保动量预测链不断
 * - zstd 兼容导入；conv key 统一；全局 min/max 预计算以降低开销
 * 
 * ===========================================================================
 */

#include "momentum_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <zstd.h>
#include "uthash.h"

// SZ3 真实集成 (条件编译)
#ifdef USE_REAL_SZ3
#include <SZ3c/sz3c.h>
#define SZ3_ENABLED 1
#else
#define SZ3_ENABLED 0
#endif

// Blosc 集成 (条件编译)
#ifdef USE_BLOSC
#include <blosc.h>
#define BLOSC_ENABLED 1
#else
#define BLOSC_ENABLED 0
#endif

// -------------------- helpers --------------------
// 数据类型辅助函数 - 对齐Python代码结构

/**
 * 获取数据类型的字节大小
 * 对齐Python: 无对应函数，但用于实现numpy数组的itemsize
 */
static size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32: return sizeof(int32_t);
        case DTYPE_INT64: return sizeof(int64_t);
        case DTYPE_UINT8: return sizeof(uint8_t);
        default: return 0;
    }
}

/**
 * 判断是否为浮点类型
 * 对齐Python: FLOAT_DTYPES = (np.float32, np.float64)
 */
static bool is_float_dtype(DataType dtype) {
    return (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64);
}

static const char* dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return "float32";
        case DTYPE_FLOAT64: return "float64";
        case DTYPE_INT32: return "int32";
        case DTYPE_INT64: return "int64";
        case DTYPE_UINT8: return "uint8";
        default: return "unknown";
    }
}

// -------------------- helpers --------------------

/**
 * 判断是否应该进行有损压缩
 * 对齐Python: def _should_lossy(self, layer_name: str, arr: np.ndarray) -> bool
 * 
 * 判断条件:
 * - 层名包含 'weight'
 * - 元素数 > param_count_threshold
 * - dtype 为 float32/64
 * 
 * 返回: true 表示应该使用有损压缩
 */
static bool should_use_lossy_compression(
    const char *layer_name,
    size_t param_count,
    DataType dtype,
    size_t param_count_threshold
) {
    // 对齐Python逻辑:
    // return ("weight" in layer_name) and 
    //        (arr.size > self.param_count_threshold) and
    //        (arr.dtype in FLOAT_DTYPES)
    
    bool has_weight = (strstr(layer_name, "weight") != NULL);
    bool exceeds_threshold = (param_count > param_count_threshold);
    bool is_float = is_float_dtype(dtype);
    
    return has_weight && exceeds_threshold && is_float;
}

// ===========================================================================
// 内部数据结构
// ===========================================================================
typedef struct GradientNode {
    NDArray *gradient;
    struct GradientNode *next;
} GradientNode;

typedef struct {
    char key[512];
    GradientNode *gradients_head;
    int gradient_count;
    UT_hash_handle hh;
} LayerHistory;

// 二级结构: 第一级按layer_key(仅基于shape)
typedef struct {
    char layer_key[256];      // e.g., "conv_(64,32,3,3)"
    NDArray *memory;
    UT_hash_handle hh;
} LayerMemoryEntry;

// 第二级: 按client_id
typedef struct {
    char client_id[256];
    LayerMemoryEntry *layer_memories;  // 哈希表,按layer_key索引
    UT_hash_handle hh;
} PredictionMemory;

typedef struct {
    char key[512];            // "client_id:layer_name" 格式
    int step;
    UT_hash_handle hh;
} StepCount;

struct MomentumCompressor {
    CompressorConfig config;
    
    LayerHistory *layer_histories;
    PredictionMemory *prediction_memories;
    StepCount *step_counts;
    
    char current_client_id[256];
    int log_level;  // 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
    
    struct {
        size_t total_compressions;
        float *prediction_ratios;
        size_t prediction_ratio_count;
        size_t prediction_ratio_capacity;
        float *sign_mismatch_ratios;
        size_t sign_mismatch_ratio_count;
        size_t sign_mismatch_ratio_capacity;
    } stats;
};

// ===========================================================================
// NDArray 函数实现（支持多dtype）                      
// ===========================================================================

NDArray* ndarray_create(const size_t *shape, size_t ndim, DataType dtype) {
    // 输入验证
    if (!shape) {
        fprintf(stderr, "[ERROR] ndarray_create: shape is NULL\n");
        return NULL;
    }
    
    if (ndim < MIN_NDIM || ndim > MAX_NDIM) {
        fprintf(stderr, "[ERROR] ndarray_create: invalid ndim=%zu (valid range: %d-%d)\n", 
                ndim, MIN_NDIM, MAX_NDIM);
        return NULL;
    }
    
    // 检查shape有效性
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) {
            fprintf(stderr, "[ERROR] ndarray_create: shape[%zu]=0 is invalid\n", i);
            return NULL;
        }
    }
    
    if (dtype == DTYPE_UNKNOWN) {
        fprintf(stderr, "[ERROR] ndarray_create: unknown dtype\n");
        return NULL;
    }
    
    NDArray *arr = (NDArray*)malloc(sizeof(NDArray));
    if (!arr) {
        fprintf(stderr, "[ERROR] ndarray_create: failed to allocate NDArray struct\n");
        return NULL;
    }
    
    arr->ndim = ndim;
    arr->dtype = dtype;
    
    arr->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->shape) {
        fprintf(stderr, "[ERROR] ndarray_create: failed to allocate shape array\n");
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(size_t));
    
    arr->total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->total_size *= shape[i];
    }
    
    size_t element_size = dtype_size(dtype);
    arr->data = calloc(arr->total_size, element_size);
    if (!arr->data) {
        fprintf(stderr, "[ERROR] ndarray_create: failed to allocate data (%zu elements, %zu bytes each)\n",
                arr->total_size, element_size);
        free(arr->shape);
        free(arr);
        return NULL;
    }
    
    return arr;
}

void ndarray_destroy(NDArray *array) {
    if (!array) return;
    free(array->data);
    free(array->shape);
    free(array);
}

NDArray* ndarray_copy(const NDArray *src) {
    if (!src) return NULL;
    
    NDArray *dst = ndarray_create(src->shape, src->ndim, src->dtype);
    if (!dst) return NULL;
    
    size_t byte_size = src->total_size * dtype_size(src->dtype);
    memcpy(dst->data, src->data, byte_size);
    return dst;
}

void compressed_layer_data_free(CompressedLayerData *data) {
    if (!data) return;
    free(data->data);
    free(data->bitmap);
    free(data->dominant_signs);
    free(data);
}

// -------------------- helpers --------------------
// 压缩器辅助函数 - 对齐Python _lossless_compress() 和 _compress_with_sz3()

/**
 * ZSTD压缩
 * 对齐Python: zstd.compress(data_bytes, 10)
 * 
 * 参数:
 *   data: 输入数据
 *   size: 数据大小（字节）
 *   out_size: 输出压缩后大小（字节）
 * 
 * 返回: 压缩后的数据指针，失败返回NULL
 */
static uint8_t* zstd_compress(const void *data, size_t size, size_t *out_size) {
    static ZSTD_CCtx* cctx = NULL;
    if (!cctx) cctx = ZSTD_createCCtx();
    
    size_t max_size = ZSTD_compressBound(size);
    uint8_t *compressed = (uint8_t*)malloc(max_size);
    if (!compressed) return NULL;
    
    size_t rc = ZSTD_compressCCtx(cctx, compressed, max_size, data, size, 10);
    if (ZSTD_isError(rc)) {
        fprintf(stderr, "[ERROR] ZSTD compression failed: %s\n", ZSTD_getErrorName(rc));
        free(compressed);
        return NULL;
    }
    
    *out_size = rc;
    compressed = (uint8_t*)realloc(compressed, *out_size);
    return compressed;
}

/**
 * ZSTD解压
 * 对齐Python: zstd.decompress(compressed_data)
 * 
 * 参数:
 *   compressed: 压缩数据
 *   compressed_size: 压缩数据大小
 *   out_size: 输出解压后大小
 * 
 * 返回: 解压后的数据指针，失败返回NULL
 */
static void* zstd_decompress(const uint8_t *compressed, size_t compressed_size, size_t *out_size) {
    static ZSTD_DCtx* dctx = NULL;
    if (!dctx) dctx = ZSTD_createDCtx();
    
    unsigned long long bound = ZSTD_getFrameContentSize(compressed, compressed_size);
    if (bound == ZSTD_CONTENTSIZE_ERROR || bound == ZSTD_CONTENTSIZE_UNKNOWN) {
        fprintf(stderr, "[ERROR] Cannot determine decompression size\n");
        return NULL;
    }
    
    void *decompressed = malloc(bound);
    if (!decompressed) return NULL;
    
    size_t rc = ZSTD_decompressDCtx(dctx, decompressed, bound, compressed, compressed_size);
    if (ZSTD_isError(rc)) {
        fprintf(stderr, "[ERROR] ZSTD decompression failed: %s\n", ZSTD_getErrorName(rc));
        free(decompressed);
        return NULL;
    }
    
    *out_size = rc;
    return decompressed;
}

/**
 * Pickle压缩模拟
 * 对齐Python: pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
 * 
 * 注意: 这是简化版pickle实现，实际Python pickle会添加协议头
 * 这里简化处理，仅用于数据序列化
 */
static uint8_t* pickle_compress(const void *data, size_t size, size_t *out_size) {
    // 简化版pickle: 直接拷贝数据，添加简单header
    uint8_t *result = (uint8_t*)malloc(size + 16);
    if (!result) return NULL;
    
    // 添加简单header: "PICKLE\0\0" + size
    memcpy(result, "PICKLE\0\0", 8);
    *((size_t*)(result + 8)) = size;
    memcpy(result + 16, data, size);
    
    *out_size = size + 16;
    return result;
}

static void* pickle_decompress(const uint8_t *compressed, size_t compressed_size, size_t *out_size) {
    if (compressed_size < 16) return NULL;
    
    // 检查header
    if (memcmp(compressed, "PICKLE\0\0", 8) != 0) {
        fprintf(stderr, "[ERROR] Invalid pickle header\n");
        return NULL;
    }
    
    size_t data_size = *((size_t*)(compressed + 8));
    if (data_size + 16 != compressed_size) {
        fprintf(stderr, "[ERROR] Pickle size mismatch\n");
        return NULL;
    }
    
    void *result = malloc(data_size);
    if (!result) return NULL;
    
    memcpy(result, compressed + 16, data_size);
    *out_size = data_size;
    return result;
}

// ===========================================================================
// Blosc 压缩/解压实现
// ===========================================================================

#ifdef USE_BLOSC
/**
 * Blosc 压缩 - 使用 Blosc C API
 * Blosc 是快速的无损压缩器,特别适合数值数据
 */
static uint8_t* blosc_compress_real(
    const void *data,
    size_t size,
    size_t *out_size
) {
    // Blosc 需要初始化一次
    static int blosc_initialized = 0;
    if (!blosc_initialized) {
        blosc_init();
        blosc_initialized = 1;
    }
    
    // 分配输出缓冲区 (最坏情况: size + BLOSC_MAX_OVERHEAD)
    size_t max_size = size + BLOSC_MAX_OVERHEAD;
    uint8_t *compressed = (uint8_t*)malloc(max_size);
    if (!compressed) {
        fprintf(stderr, "[ERROR] blosc_compress: failed to allocate buffer\n");
        return NULL;
    }
    
    // 调用 Blosc 压缩
    // blosc_compress(clevel, doshuffle, typesize, nbytes, src, dest, destsize)
    int compressed_size = blosc_compress(
        5,          // clevel: 压缩级别 (1-9, 5是平衡)
        1,          // shuffle: 启用字节shuffle (提高压缩率)
        1,          // typesize: 元素大小 (1 byte,自动检测)
        size,       // nbytes: 输入大小
        data,       // src: 输入数据
        compressed, // dest: 输出缓冲区
        max_size    // destsize: 输出缓冲区大小
    );
    
    if (compressed_size <= 0) {
        fprintf(stderr, "[ERROR] Blosc compression failed (code: %d)\n", compressed_size);
        free(compressed);
        return NULL;
    }
    
    *out_size = compressed_size;
    
    // 可选: 调整大小以节省内存
    compressed = (uint8_t*)realloc(compressed, compressed_size);
    
    fprintf(stderr, "[INFO] Blosc compression: %zu → %d bytes (%.2f%%)\n",
            size, compressed_size, 100.0 * compressed_size / size);
    
    return compressed;
}

/**
 * Blosc 解压
 */
static void* blosc_decompress_real(
    const uint8_t *compressed,
    size_t compressed_size,
    size_t *out_size
) {
    // 获取解压后大小
    size_t nbytes, cbytes, blocksize;
    blosc_cbuffer_sizes(compressed, &nbytes, &cbytes, &blocksize);
    
    if (nbytes == 0) {
        fprintf(stderr, "[ERROR] Blosc: invalid compressed data\n");
        return NULL;
    }
    
    // 分配输出缓冲区
    void *decompressed = malloc(nbytes);
    if (!decompressed) {
        fprintf(stderr, "[ERROR] blosc_decompress: failed to allocate buffer\n");
        return NULL;
    }
    
    // 调用 Blosc 解压
    int result = blosc_decompress(compressed, decompressed, nbytes);
    
    if (result <= 0) {
        fprintf(stderr, "[ERROR] Blosc decompression failed (code: %d)\n", result);
        free(decompressed);
        return NULL;
    }
    
    *out_size = nbytes;
    return decompressed;
}
#endif // USE_BLOSC

// -------------------- helpers --------------------

/**
 * 无损压缩
 * 对齐Python: def _lossless_compress(self, arr: np.ndarray) -> Tuple[bytes, str]
 * 
 * 支持多种压缩器:
 * - zstd: 使用ZSTD压缩
 * - blosc: 使用Blosc压缩（如果可用）
 * - pickle: 使用Pickle序列化
 * 
 * 参数:
 *   data: 输入数据
 *   size: 数据大小
 *   compressor_type: 压缩器类型 ("zstd", "blosc", "pickle")
 *   codec_used: 输出实际使用的codec
 *   out_size: 输出压缩后大小
 * 
 * 返回: 压缩后的数据指针，失败返回NULL
 */
static uint8_t* lossless_compress(
    const void *data,
    size_t size,
    const char *compressor_type,
    const char **codec_used,
    size_t *out_size
) {
    uint8_t *result = NULL;
    
    if (strcmp(compressor_type, "zstd") == 0) {
        result = zstd_compress(data, size, out_size);
        *codec_used = "zstd";
    }
    else if (strcmp(compressor_type, "blosc") == 0) {
#ifdef USE_BLOSC
        result = blosc_compress_real(data, size, out_size);
        *codec_used = "blosc";
#else
        fprintf(stderr, "[WARNING] Blosc not available, using zstd fallback\n");
        result = zstd_compress(data, size, out_size);
        *codec_used = "zstd";
#endif
    }
    else if (strcmp(compressor_type, "pickle") == 0) {
        result = pickle_compress(data, size, out_size);
        *codec_used = "pickle";
    }
    else {
        // 默认使用pickle
        result = pickle_compress(data, size, out_size);
        *codec_used = "pickle";
    }
    
    if (!result) {
        fprintf(stderr, "[WARNING] %s compression failed, fallback to pickle\n", compressor_type);
        result = pickle_compress(data, size, out_size);
        *codec_used = "pickle";
    }
    
    return result;
}

/**
 * 无损解压
 * 对齐Python: def _lossless_decompress(self, compressed_data: bytes, codec: str) -> np.ndarray
 * 
 * 根据codec选择解压方法:
 * - 'zstd': 使用ZSTD解压
 * - 'blosc': 使用Blosc解压（如果可用）
 * - 'pickle': 使用Pickle反序列化
 * 
 * 注意: 此函数只处理无损压缩器，不处理 'sz3'（有损压缩）
 * SZ3 解压应通过 lossy_decompress_with_shape() 调用
 * 
 * 参数:
 *   compressed: 压缩数据
 *   compressed_size: 压缩数据大小
 *   codec: 压缩器类型 ("zstd", "blosc", "pickle")
 *   out_size: 输出解压后大小
 * 
 * 返回: 解压后的数据指针，失败返回NULL
 */
static void* lossless_decompress(
    const uint8_t *compressed,
    size_t compressed_size,
    const char *codec,
    size_t *out_size
) {
    if (strcmp(codec, "zstd") == 0) {
        return zstd_decompress(compressed, compressed_size, out_size);
    }
    else if (strcmp(codec, "blosc") == 0) {
#ifdef USE_BLOSC
        return blosc_decompress_real(compressed, compressed_size, out_size);
#else
        fprintf(stderr, "[WARNING] Blosc not available, using zstd fallback\n");
        return zstd_decompress(compressed, compressed_size, out_size);
#endif
    }
    else if (strcmp(codec, "pickle") == 0) {
        return pickle_decompress(compressed, compressed_size, out_size);
    }
    else {
        // 对齐Python: _lossless_decompress 不处理 "sz3"
        // 如果传入 "sz3"，应该由上层调用 lossy_decompress_with_shape()
        fprintf(stderr, "[ERROR] Unknown codec in lossless_decompress: %s (only supports zstd/blosc/pickle)\n", codec);
        return NULL;
    }
}

// ===========================================================================
// SZ3 前向声明
// ===========================================================================

#ifdef USE_REAL_SZ3
static uint8_t* sz3_compress_real(
    const void *data,
    size_t total_size,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    const char *error_mode,
    float abs_bound,
    float rel_bound,
    size_t *out_size
);

static void* sz3_decompress_real(
    const uint8_t *compressed,
    size_t compressed_size,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    size_t *out_size
);
#endif

// 有损解压(带形状信息)
static void* lossy_decompress_with_shape(
    const uint8_t *compressed,
    size_t compressed_size,
    const char *codec,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    size_t *out_size
) {
    if (strcmp(codec, "sz3_memcpy") == 0) {
        void *result = malloc(compressed_size);
        if (!result) {
            return NULL;
        }
        memcpy(result, compressed, compressed_size);
        *out_size = compressed_size;
        return result;
    }

    if (strcmp(codec, "sz3") == 0) {
#ifdef USE_REAL_SZ3
        return sz3_decompress_real(compressed, compressed_size, shape, ndim, dtype, out_size);
#else
        // 避免未使用参数警告
        (void)shape;
        (void)ndim;
        (void)dtype;
        fprintf(stderr, "[WARNING] SZ3 not available, using zstd fallback\n");
        return zstd_decompress(compressed, compressed_size, out_size);
#endif
    }
    else {
        // 其他codec降级到无损解压
        return lossless_decompress(compressed, compressed_size, codec, out_size);
    }
}

// ===========================================================================
// SZ3 真实压缩/解压实现
// ===========================================================================

#ifdef USE_REAL_SZ3
/**
 * SZ3 真实压缩 - 使用 SZ3 C API
 * 对齐Python: pysz.SZ.compress() 支持 float32 和 float64
 * 
 * 参数:
 *   data: 原始数据 (float32 或 float64)
 *   total_size: 元素个数
 *   shape: 数组维度
 *   ndim: 维度数
 *   dtype: 数据类型 (DTYPE_FLOAT32 或 DTYPE_FLOAT64)
 *   error_mode: "ABS" or "REL"
 *   abs_bound: 绝对误差界
 *   rel_bound: 相对误差界 (0.01 = 1%)
 *   out_size: 输出压缩后大小
 * 返回: 压缩数据 (需要free)
 */
static uint8_t* sz3_compress_real(
    const void *data,
    size_t total_size,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    const char *error_mode,
    float abs_bound,
    float rel_bound,
    size_t *out_size
) {
    // 1. 转换维度 (SZ3使用反序: r1=最内层)
    size_t r1=1, r2=0, r3=0, r4=0, r5=0;
    
    if (ndim >= 1) r1 = shape[ndim-1];
    if (ndim >= 2) r2 = shape[ndim-2];
    if (ndim >= 3) r3 = shape[ndim-3];
    if (ndim >= 4) r4 = shape[ndim-4];
    if (ndim >= 5) r5 = shape[ndim-5];
    
    // 2. 选择误差模式
    int mode = REL;  // 默认相对误差
    if (error_mode && strcmp(error_mode, "ABS") == 0) {
        mode = ABS;
    }
    
    // 3. 选择数据类型 (对齐Python: 支持 float32 和 float64)
    int sz_datatype;
    size_t element_size;
    if (dtype == DTYPE_FLOAT64) {
        sz_datatype = SZ_DOUBLE;  // float64
        element_size = sizeof(double);
    } else {
        sz_datatype = SZ_FLOAT;   // float32 (默认)
        element_size = sizeof(float);
    }
    
    // 4. 调用 SZ3 压缩
    unsigned char *sz3_buffer = SZ_compress_args(
        sz_datatype,              // 数据类型 (SZ_FLOAT 或 SZ_DOUBLE)
        (void*)data,              // 数据指针
        out_size,                 // 输出: 压缩后大小
        mode,                     // 误差模式
        (double)abs_bound,        // 绝对误差
        (double)rel_bound,        // 相对误差
        0.0,                      // pwrBound (不使用)
        r5, r4, r3, r2, r1        // 维度 (5D支持)
    );
    
    if (!sz3_buffer) {
        fprintf(stderr, "[ERROR] SZ3 compression failed (mode=%s, rel=%.3f)\n",
                error_mode, rel_bound);
        return NULL;
    }
    
    // 4. 拷贝到我们的内存 (SZ3使用自己的分配器) 先拷贝数据再释放
    uint8_t *result = (uint8_t*)malloc(*out_size);
    if (!result) {
        fprintf(stderr, "[ERROR] sz3_compress: failed to allocate result buffer\n");
        free_buf(sz3_buffer);
        return NULL;
    }
    
    memcpy(result, sz3_buffer, *out_size);
    free_buf(sz3_buffer);  // 释放 SZ3 内存
    
    fprintf(stderr, "[INFO] SZ3 real compression: %zu → %zu bytes (%.2f%%, dtype=%s)\n",
            total_size * element_size, *out_size, 
            100.0 * (*out_size) / (total_size * element_size),
            dtype_to_string(dtype));
    
    return result;
}

/**
 * SZ3 真实解压
 * 对齐Python: pysz.SZ.decompress() 支持 float32 和 float64
 * 
 * 参数:
 *   compressed: 压缩数据
 *   compressed_size: 压缩数据大小
 *   shape: 数组维度
 *   ndim: 维度数
 *   dtype: 数据类型 (DTYPE_FLOAT32 或 DTYPE_FLOAT64)
 *   out_size: 输出解压后大小
 * 返回: 解压数据 (需要free)
 */
static void* sz3_decompress_real(
    const uint8_t *compressed,
    size_t compressed_size,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    size_t *out_size
) {
    // 1. 转换维度
    size_t r1=1, r2=0, r3=0, r4=0, r5=0;
    
    if (ndim >= 1) r1 = shape[ndim-1];
    if (ndim >= 2) r2 = shape[ndim-2];
    if (ndim >= 3) r3 = shape[ndim-3];
    if (ndim >= 4) r4 = shape[ndim-4];
    if (ndim >= 5) r5 = shape[ndim-5];
    
    // 2. 选择数据类型 
    int sz_datatype;
    size_t element_size;
    if (dtype == DTYPE_FLOAT64) {
        sz_datatype = SZ_DOUBLE;  // float64
        element_size = sizeof(double);
    } else {
        sz_datatype = SZ_FLOAT;   // float32 (默认)
        element_size = sizeof(float);
    }
    
    // 3. 调用 SZ3 解压
    void *sz3_buffer = SZ_decompress(
        sz_datatype,              // 数据类型 (SZ_FLOAT 或 SZ_DOUBLE)
        (unsigned char*)compressed,
        compressed_size,
        r5, r4, r3, r2, r1        // 维度
    );
    
    if (!sz3_buffer) {
        fprintf(stderr, "[ERROR] SZ3 decompression failed (dtype=%s)\n", dtype_to_string(dtype));
        return NULL;
    }
    
    // 4. 计算输出大小
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    *out_size = total_size * element_size;
    
    // 4. 拷贝到我们的内存
    void *result = malloc(*out_size);
    if (!result) {
        fprintf(stderr, "[ERROR] sz3_decompress: failed to allocate result buffer\n");
        free_buf(sz3_buffer);
        return NULL;
    }
    
    memcpy(result, sz3_buffer, *out_size);
    free_buf(sz3_buffer);  // 释放 SZ3 内存
    
    return result;
}
#endif // USE_REAL_SZ3

// -------------------- SZ3 wrappers --------------------

/**
 * SZ3压缩（统一接口，支持真实/模拟）
 * 对齐Python: def _compress_with_sz3(self, data: np.ndarray, error_mode, abs_bound, rel_bound) -> Tuple[bytes, float]
 * 
 * 返回: (compressed_data, compression_ratio, codec)
 * 
 * 功能:
 * - 如果定义了USE_REAL_SZ3且dtype为float32，尝试使用真实SZ3压缩
 * - 如果SZ3不可用或失败，回退到ZSTD无损压缩
 * - 记录实际使用的codec（"sz3"或"zstd"）
 * 
 * 参数:
 *   data: 输入数据
 *   size: 元素个数
 *   dtype: 数据类型
 *   shape: 数组形状（用于真实SZ3）
 *   ndim: 维度数
 *   error_mode: 误差模式 ("ABS" 或 "REL")
 *   abs_bound: 绝对误差界
 *   rel_bound: 相对误差界
 *   compression_ratio: 输出压缩率
 *   codec_used: 输出实际使用的codec
 *   out_size: 输出压缩后大小
 * 
 * 返回: 压缩后的数据指针，失败返回NULL
 */
static uint8_t* sz3_compress_simulate(
    const void *data,
    size_t size,
    DataType dtype,
    const size_t *shape,      // 形状信息
    size_t ndim,              // 维度数
    const char *error_mode,
    float abs_bound,
    float rel_bound,
    float *compression_ratio,
    const char **codec_used,
    size_t *out_size
) {
#ifdef USE_REAL_SZ3
    // 使用真实 SZ3 
    if ((dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64) && shape && ndim > 0) {
        uint8_t *result = sz3_compress_real(
            data,          // 使用 void* 支持 float32 和 float64
            size,
            shape,         // 传递形状
            ndim,          // 传递维度
            dtype,         // 传递数据类型
            error_mode,
            abs_bound,
            rel_bound,
            out_size
        );
        
        if (result) {
            size_t element_size = dtype_size(dtype);
            *compression_ratio = (float)(size * element_size) / (float)(*out_size);
            *codec_used = "sz3";  // 真实 SZ3
            return result;
        }
        // 失败则降级到 zstd
    }
#else
    // 避免未使用参数警告
    (void)shape;
    (void)ndim;
#endif
    
    // 降级: 使用ZSTD模拟SZ3
    fprintf(stderr, "[INFO] SZ3 not available or unsupported dtype, using ZSTD fallback (error_mode=%s, abs=%.3f, rel=%.3f)\n",
            error_mode, abs_bound, rel_bound);
    
    uint8_t *result = zstd_compress(data, size * dtype_size(dtype), out_size);
    if (result) {
        *compression_ratio = (float)(size * dtype_size(dtype)) / (float)(*out_size);
        *codec_used = "zstd";  // 降级
    } else {
        *compression_ratio = 1.0f;
        *codec_used = "pickle";
    }
    
    return result;
}

// ===========================================================================
// 统计函数（支持多dtype）
// ===========================================================================

static float compute_mean_float32(const float *data, size_t size) {
    if (size == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) sum += data[i];
    return (float)(sum / size);
}

static float compute_std_float32(const float *data, size_t size, float mean) {
    if (size == 0) return 0.0f;
    double variance = 0.0;
    for (size_t i = 0; i < size; i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    return sqrtf((float)(variance / size));
}

static void compute_min_max_float32(const float *data, size_t size, float *min, float *max) {
    if (size == 0) return;
    *min = *max = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] < *min) *min = data[i];
        if (data[i] > *max) *max = data[i];
    }
}

// ===========================================================================
// 符号一致性计算 - 对齐Python _compute_normalized_sign_consistency()
// ===========================================================================

static float compute_sign_consistency(const float *signs, size_t size) {
    if (size == 0) return 0.0f;
    
    size_t positives = 0, negatives = 0, zeros = 0;
    
    for (size_t i = 0; i < size; i++) {
        if (signs[i] > 0.0f) positives++;
        else if (signs[i] < 0.0f) negatives++;
        else zeros++;
    }
    
    size_t majority = (positives >= negatives) ? (positives + zeros) : (negatives + zeros);
    return ((float)majority / (float)size - 0.5f) * 2.0f;
}

/**
 * 获取主导符号
 * 对齐Python: def _get_dominant_sign(self, kernel_signs: np.ndarray) -> int
 * 
 * 返回: 1 如果正数 >= 负数，否则 -1
 */
static int get_dominant_sign(const float *signs, size_t size) {
    size_t positives = 0, negatives = 0;
    for (size_t i = 0; i < size; i++) {
        if (signs[i] > 0.0f) positives++;
        else if (signs[i] < 0.0f) negatives++;
    }
    return (positives >= negatives) ? 1 : -1;
}

// ===========================================================================
// 状态管理函数
// ===========================================================================

static void make_history_key(char *key, const char *client_id, const char *layer_name) {
    snprintf(key, 512, "%s:%s", client_id, layer_name);
}

// 生成仅基于shape的layer_key (对齐Python: "conv_(64,32,3,3)")
static void make_layer_key_from_shape(char *layer_key, const size_t *shape, size_t ndim) {
    int offset = snprintf(layer_key, 256, "conv_(");
    for (size_t i = 0; i < ndim && offset < 250; i++) {
        offset += snprintf(layer_key + offset, 256 - offset, "%zu%s", 
                          shape[i], (i < ndim - 1) ? ", " : "");
    }
    snprintf(layer_key + offset, 256 - offset, ")");
}

static LayerHistory* get_layer_history(MomentumCompressor *mc, const char *key) {
    LayerHistory *entry = NULL;
    HASH_FIND_STR(mc->layer_histories, key, entry);
    return entry;
}

static void add_gradient_to_history(MomentumCompressor *mc, const char *key, const NDArray *gradient) {
    LayerHistory *entry = get_layer_history(mc, key);
    
    if (!entry) {
        entry = (LayerHistory*)calloc(1, sizeof(LayerHistory));
        strncpy(entry->key, key, 511);
        entry->gradients_head = NULL;
        entry->gradient_count = 0;
        HASH_ADD_STR(mc->layer_histories, key, entry);
    }
    
    GradientNode *node = (GradientNode*)malloc(sizeof(GradientNode));
    node->gradient = ndarray_copy(gradient);
    node->next = entry->gradients_head;
    entry->gradients_head = node;
    entry->gradient_count++;
    
    // 限制历史长度（对齐Python）
    if (entry->gradient_count > mc->config.max_history_length) {
        GradientNode *curr = entry->gradients_head;
        GradientNode *prev = NULL;
        while (curr->next) {
            prev = curr;
            curr = curr->next;
        }
        if (prev) {
            prev->next = NULL;
            ndarray_destroy(curr->gradient);
            free(curr);
            entry->gradient_count--;
        }
    }
}

static NDArray* get_latest_gradient(MomentumCompressor *mc, const char *key) {
    LayerHistory *entry = get_layer_history(mc, key);
    if (!entry || !entry->gradients_head) return NULL;
    return entry->gradients_head->gradient;
}

static int get_step_count(MomentumCompressor *mc, const char *key) {
    StepCount *entry = NULL;
    HASH_FIND_STR(mc->step_counts, key, entry);
    return entry ? entry->step : 0;
}

static void increment_step_count(MomentumCompressor *mc, const char *key) {
    StepCount *entry = NULL;
    HASH_FIND_STR(mc->step_counts, key, entry);
    
    if (!entry) {
        entry = (StepCount*)calloc(1, sizeof(StepCount));
        strncpy(entry->key, key, 511);
        entry->step = 0;
        HASH_ADD_STR(mc->step_counts, key, entry);
    }
    
    entry->step++;
}

static PredictionMemory* get_or_create_client_memory(MomentumCompressor *mc, const char *client_id) {
    PredictionMemory *client_entry = NULL;
    HASH_FIND_STR(mc->prediction_memories, client_id, client_entry);
    
    if (!client_entry) {
        client_entry = (PredictionMemory*)calloc(1, sizeof(PredictionMemory));
        strncpy(client_entry->client_id, client_id, sizeof(client_entry->client_id) - 1);
        client_entry->layer_memories = NULL;
        HASH_ADD_STR(mc->prediction_memories, client_id, client_entry);
    }
    
    return client_entry;
}

static NDArray* get_prediction_memory_for_layer(MomentumCompressor *mc, const char *client_id, 
                                                const size_t *shape, size_t ndim) {
    // 获取或创建client级别的entry
    PredictionMemory *client_entry = get_or_create_client_memory(mc, client_id);
    
    // 生成layer_key (仅基于shape)
    char layer_key[256];
    make_layer_key_from_shape(layer_key, shape, ndim);
    
    // 查找layer memory
    LayerMemoryEntry *layer_entry = NULL;
    HASH_FIND_STR(client_entry->layer_memories, layer_key, layer_entry);
    
    return layer_entry ? layer_entry->memory : NULL;
}

static void set_prediction_memory_for_layer(MomentumCompressor *mc, const char *client_id,
                                           const size_t *shape, size_t ndim, const NDArray *memory) {
    // 获取或创建client级别的entry
    PredictionMemory *client_entry = get_or_create_client_memory(mc, client_id);
    
    // 生成layer_key
    char layer_key[256];
    make_layer_key_from_shape(layer_key, shape, ndim);
    
    // 查找或创建layer memory entry
    LayerMemoryEntry *layer_entry = NULL;
    HASH_FIND_STR(client_entry->layer_memories, layer_key, layer_entry);
    
    if (!layer_entry) {
        layer_entry = (LayerMemoryEntry*)calloc(1, sizeof(LayerMemoryEntry));
        strncpy(layer_entry->layer_key, layer_key, sizeof(layer_entry->layer_key) - 1);
        layer_entry->memory = NULL;
        HASH_ADD_STR(client_entry->layer_memories, layer_key, layer_entry);
    }
    
    // ✅ 修复use-after-free: 先复制新的，再释放旧的
    // 检查是否传入的是同一个指针（避免自我复制后释放）
    if (layer_entry->memory != memory) {
        // 先复制新的memory
        NDArray *new_memory = ndarray_copy(memory);
        // 再释放旧的
        if (layer_entry->memory) {
            ndarray_destroy(layer_entry->memory);
        }
        // 设置新的
        layer_entry->memory = new_memory;
    }
    // 如果是同一个指针，不需要任何操作
}


// ===========================================================================
// 配置函数
// ===========================================================================

CompressorConfig momentum_compressor_default_config(void) {
    CompressorConfig config;
    
    // 对齐Python默认值
    config.momentum_lr = 0.07f;
    config.consistency_threshold = 0.5f;
    strcpy(config.lossless_compressor, "zstd");
    strcpy(config.error_bounding_mode, "REL");
    config.error_bound = 1.0f;
    // 对齐Python硬编码路径
    strcpy(config.sz3_lib_path, "/eagle/lc-mpi/ZhijingYe/FLComp/SZ_NP/lib64/libSZ3c.so");
    config.param_count_threshold = 1024;  // 对齐Python
    config.max_history_length = 3;
    
    return config;
}

// ===========================================================================
// 主压缩器创建/销毁
// ===========================================================================

MomentumCompressor* momentum_compressor_create(const CompressorConfig *config) {
    MomentumCompressor *mc = (MomentumCompressor*)calloc(1, sizeof(MomentumCompressor));
    if (!mc) return NULL;
    
    memcpy(&mc->config, config, sizeof(CompressorConfig));
    
    mc->layer_histories = NULL;
    mc->prediction_memories = NULL;
    mc->step_counts = NULL;
    mc->log_level = 1;  // 默认INFO级别
    
    mc->stats.total_compressions = 0;
    mc->stats.prediction_ratios = NULL;
    mc->stats.prediction_ratio_count = 0;
    mc->stats.prediction_ratio_capacity = 0;
    
    printf("✓ MomentumCompressor created\n");
    printf("  - momentum_lr: %.3f\n", config->momentum_lr);
    printf("  - consistency_threshold: %.3f\n", config->consistency_threshold);
    printf("  - param_count_threshold: %zu\n", config->param_count_threshold);
    printf("  - lossless_compressor: %s\n", config->lossless_compressor);
    
    return mc;
}

void momentum_compressor_destroy(MomentumCompressor *compressor) {
    if (!compressor) return;
    
    // 清理梯度历史
    LayerHistory *lh, *lh_tmp;
    HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
        HASH_DEL(compressor->layer_histories, lh);
        GradientNode *node = lh->gradients_head;
        while (node) {
            GradientNode *next = node->next;
            ndarray_destroy(node->gradient);
            free(node);
            node = next;
        }
        free(lh);
    }
    
    // 清理预测记忆（二级结构）
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        HASH_DEL(compressor->prediction_memories, pm);
        
        // 清理每个client的layer memories
        LayerMemoryEntry *lme, *lme_tmp;
        HASH_ITER(hh, pm->layer_memories, lme, lme_tmp) {
            HASH_DEL(pm->layer_memories, lme);
            if (lme->memory) {
                ndarray_destroy(lme->memory);
            }
            free(lme);
        }
        free(pm);
    }
    
    // 清理步数计数
    StepCount *sc, *sc_tmp;
    HASH_ITER(hh, compressor->step_counts, sc, sc_tmp) {
        HASH_DEL(compressor->step_counts, sc);
        free(sc);
    }
    
    free(compressor->stats.prediction_ratios);
    free(compressor->stats.sign_mismatch_ratios);
    free(compressor);
    
    printf("✓ MomentumCompressor destroyed\n");
}

void momentum_compressor_set_client(MomentumCompressor *compressor, const char *client_id) {
    if (!compressor) return;
    strncpy(compressor->current_client_id, client_id, 255);
    compressor->current_client_id[255] = '\0';
}

// -------------------- core per-layer compression --------------------

/**
 * 卷积层压缩（基于动量预测）
 * 对齐Python: def _compress_conv_layer(self, current_grad: np.ndarray, prev_grad: np.ndarray, client_id: str, current_step: int) -> Dict[str, Any]
 * 
 * 压缩流程:
 * 1. 计算统计信息（均值、标准差、min/max）
 * 2. 归一化前一梯度
 * 3. 获取/创建预测记忆
 * 4. 计算符号一致性，生成预测位图
 * 5. 使用动量预测更新记忆，计算预测kernel
 * 6. 计算残差并压缩
 * 7. 压缩位图和主导符号
 * 
 * 参数:
 *   mc: 压缩器实例
 *   current_grad: 当前梯度
 *   prev_grad: 前一梯度（已重建）
 *   client_id: 客户端ID
 *   layer_name: 层名称
 *   current_step: 当前步数
 * 
 * 返回: 压缩层数据，失败返回NULL
 */
static CompressedLayerData* compress_conv_layer_momentum(
    MomentumCompressor *mc,
    const NDArray *current_grad,
    const NDArray *prev_grad,
    const char *client_id,
    const char *layer_name __attribute__((unused)),
    int current_step
) {
    // 对齐Python: _compress_conv_layer 传入的是4D数组

    const size_t *shape = current_grad->shape;
    size_t out_ch = shape[0];
    size_t in_ch = shape[1];
    size_t h = shape[2];
    size_t w = shape[3];
    size_t kernel_size = h * w;
    size_t total_size = current_grad->total_size;
    
    float *current_data = (float*)current_grad->data;
    float *prev_data = (float*)prev_grad->data;
    
    // 分配结果
    CompressedLayerData *result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
    strcpy(result->type, "momentum_predicted");
    strcpy(result->codec, "zstd");  // 默认，后续会更新
    memcpy(result->shape, shape, 4 * sizeof(size_t));
    result->ndim = 4;
    result->step = current_step;
    strcpy(result->original_dtype, "float32");
    strcpy(result->stored_dtype, "float32");
    
    // 1. 计算统计信息（对齐Python）
    float *abs_current = (float*)malloc(total_size * sizeof(float));
    float *abs_prev = (float*)malloc(total_size * sizeof(float));
    if (!abs_current || !abs_prev) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate abs arrays\n");
        free(abs_current);
        free(abs_prev);
        free(result);
        return NULL;
    }
    
    for (size_t i = 0; i < total_size; i++) {
        abs_current[i] = fabsf(current_data[i]);
        abs_prev[i] = fabsf(prev_data[i]);
    }
    
    result->current_mean = compute_mean_float32(abs_current, total_size);
    result->current_std = compute_std_float32(abs_current, total_size, result->current_mean);
    result->prev_mean = compute_mean_float32(abs_prev, total_size);
    result->prev_std = compute_std_float32(abs_prev, total_size, result->prev_mean);
    compute_min_max_float32(current_data, total_size, &result->global_min, &result->global_max);
    
    // 2. 归一化前一梯度
    float *prev_normalized = (float*)malloc(total_size * sizeof(float));
    if (!prev_normalized) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate prev_normalized\n");
        free(abs_current);
        free(abs_prev);
        free(result);
        return NULL;
    }
    
    for (size_t i = 0; i < total_size; i++) {
        prev_normalized[i] = abs_prev[i] - result->prev_mean;
        if (result->prev_std > 1e-8f) {
            prev_normalized[i] /= result->prev_std;
        }
    }
    
    // 3. 获取/创建预测记忆（使用新的二级结构API）
    NDArray *layer_memory = get_prediction_memory_for_layer(mc, client_id, shape, 4);
    
    if (!layer_memory) {
        layer_memory = ndarray_create(shape, 4, DTYPE_FLOAT32);
        // 初始化为0
        memset(layer_memory->data, 0, layer_memory->total_size * sizeof(float));
        set_prediction_memory_for_layer(mc, client_id, shape, 4, layer_memory);
        // 现在重新获取,因为set会clone
        ndarray_destroy(layer_memory);
        layer_memory = get_prediction_memory_for_layer(mc, client_id, shape, 4);
    }
    
    if (!layer_memory) {
        fprintf(stderr, "[ERROR] compress_conv: failed to get/create prediction memory\n");
        free(abs_current);
        free(abs_prev);
        free(prev_normalized);
        free(result);
        return NULL;
    }
    
    float *memory_data = (float*)layer_memory->data;
    
    // 4. 计算符号一致性
    bool *prediction_bitmap = (bool*)malloc(out_ch * in_ch * sizeof(bool));
    int *dominant_signs = (int*)malloc(out_ch * in_ch * sizeof(int));
    if (!prediction_bitmap || !dominant_signs) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate prediction arrays\n");
        free(prediction_bitmap);
        free(dominant_signs);
        free(abs_current);
        free(abs_prev);
        free(prev_normalized);
        free(result);
        return NULL;
    }
    
    size_t predicted_count = 0;
    
    for (size_t oc = 0; oc < out_ch; oc++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            size_t kernel_idx = oc * in_ch + ic;
            size_t kernel_offset = kernel_idx * kernel_size;
            
            float consistency = compute_sign_consistency(
                current_data + kernel_offset, kernel_size);
            
            prediction_bitmap[kernel_idx] = (consistency >= mc->config.consistency_threshold);
            dominant_signs[kernel_idx] = get_dominant_sign(
                current_data + kernel_offset, kernel_size);
            
            if (prediction_bitmap[kernel_idx]) {
                predicted_count++;
            }
        }
    }
    
    result->num_predicted_kernels = (int)predicted_count;
    result->prediction_ratio = (float)predicted_count / (float)(out_ch * in_ch);
    
    // 5. 生成残差并应用动量预测
    float *residual_data = (float*)malloc(total_size * sizeof(float));
    if (!residual_data) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate residual_data\n");
        free(prediction_bitmap);
        free(dominant_signs);
        free(abs_current);
        free(abs_prev);
        free(prev_normalized);
        free(result);
        return NULL;
    }
    
    memcpy(residual_data, current_data, total_size * sizeof(float));
    
    size_t sign_mismatch_count = 0;
    size_t total_predicted_elements = 0;
    
    if (predicted_count > 0) {
        for (size_t oc = 0; oc < out_ch; oc++) {
            for (size_t ic = 0; ic < in_ch; ic++) {
                size_t kernel_idx = oc * in_ch + ic;
                
                if (!prediction_bitmap[kernel_idx]) {
                    continue;
                }
                
                size_t kernel_offset = kernel_idx * kernel_size;
                
                // 更新动量记忆: memory = (1-lr)*old + lr*prev_norm
                for (size_t i = 0; i < kernel_size; i++) {
                    size_t idx = kernel_offset + i;
                    float old_mem = memory_data[idx];
                    float new_mem = (1.0f - mc->config.momentum_lr) * old_mem + 
                                   mc->config.momentum_lr * prev_normalized[idx];
                    memory_data[idx] = new_mem;
                }
                
                // 反归一化预测值
                float *abs_predicted = (float*)malloc(kernel_size * sizeof(float));
                if (!abs_predicted) {
                    fprintf(stderr, "[ERROR] compress_conv: failed to allocate abs_predicted\n");
                    continue;  // 跳过此kernel,继续处理其他kernel
                }
                
                for (size_t i = 0; i < kernel_size; i++) {
                    size_t idx = kernel_offset + i;
                    float pred_norm = memory_data[idx];
                    
                    if (result->current_std > 1e-8f) {
                        abs_predicted[i] = pred_norm * result->current_std + result->current_mean;
                    } else {
                        abs_predicted[i] = pred_norm + result->current_mean;
                    }
                    abs_predicted[i] = fabsf(abs_predicted[i]);
                }
                
                // 应用主导符号
                int dom_sign = dominant_signs[kernel_idx];
                float dom_sign_f = (float)dom_sign;
                
                float *predicted_kernel = (float*)malloc(kernel_size * sizeof(float));
                if (!predicted_kernel) {
                    fprintf(stderr, "[ERROR] compress_conv: failed to allocate predicted_kernel\n");
                    free(abs_predicted);
                    continue;  // 跳过此kernel
                }
                
                for (size_t i = 0; i < kernel_size; i++) {
                    predicted_kernel[i] = dom_sign_f * abs_predicted[i];
                }
                
                // 计算符号不匹配
                for (size_t i = 0; i < kernel_size; i++) {
                    size_t idx = kernel_offset + i;
                    float pred_sign = (predicted_kernel[i] > 0) ? 1.0f : -1.0f;
                    float actual_sign = (current_data[idx] > 0) ? 1.0f : -1.0f;
                    
                    if (pred_sign * actual_sign < 0) {
                        sign_mismatch_count++;
                    }
                    total_predicted_elements++;
                }
                
                // 计算残差: residual = current - predicted
                for (size_t i = 0; i < kernel_size; i++) {
                    size_t idx = kernel_offset + i;
                    residual_data[idx] = current_data[idx] - predicted_kernel[i];
                    
                    // 裁剪到全局范围
                    if (residual_data[idx] < result->global_min) {
                        residual_data[idx] = result->global_min;
                    }
                    if (residual_data[idx] > result->global_max) {
                        residual_data[idx] = result->global_max;
                    }
                }
                
                free(abs_predicted);
                free(predicted_kernel);
            }
        }
        
        // 更新预测记忆（使用新的二级结构API）
        NDArray *new_memory = ndarray_create(shape, 4, DTYPE_FLOAT32);
        memcpy(new_memory->data, memory_data, total_size * sizeof(float));
        set_prediction_memory_for_layer(mc, client_id, shape, 4, new_memory);
        ndarray_destroy(new_memory);
    }
    
    result->sign_mismatch_ratio = (total_predicted_elements > 0) ?
        (float)sign_mismatch_count / (float)total_predicted_elements : 0.0f;
    
    // 6. 压缩残差（对齐Python: 使用SZ3 ABS模式）
    float abs_err = mc->config.error_bound * (result->global_max - result->global_min);
    float compression_ratio;
    const char *codec_used;
    
    result->data = sz3_compress_simulate(
        residual_data,
        total_size,
        DTYPE_FLOAT32,
        shape,              // 传递形状
        4,                  // 4D卷积
        "ABS",
        abs_err,
        0.0f,
        &compression_ratio,
        &codec_used,
        &result->data_size
    );
    
    strcpy(result->codec, codec_used);
    
    // 7. 压缩位图
    size_t bitmap_bytes = (out_ch * in_ch + 7) / 8;
    uint8_t *packed_bitmap = (uint8_t*)calloc(bitmap_bytes, 1);
    for (size_t i = 0; i < out_ch * in_ch; i++) {
        if (prediction_bitmap[i]) {
            packed_bitmap[i / 8] |= (1 << (i % 8));
        }
    }
    result->bitmap = zstd_compress(packed_bitmap, bitmap_bytes, &result->bitmap_size);
    
    // 8. 压缩主导符号
    if (predicted_count > 0) {
        size_t dom_bytes = (predicted_count + 7) / 8;
        uint8_t *packed_dom = (uint8_t*)calloc(dom_bytes, 1);
        size_t pred_idx = 0;
        for (size_t i = 0; i < out_ch * in_ch; i++) {
            if (prediction_bitmap[i]) {
                if (dominant_signs[i] > 0) {
                    packed_dom[pred_idx / 8] |= (1 << (pred_idx % 8));
                }
                pred_idx++;
            }
        }
        result->dominant_signs = zstd_compress(packed_dom, dom_bytes, 
                                               &result->dominant_signs_size);
        free(packed_dom);
    }
    
    // 清理
    free(abs_current);
    free(abs_prev);
    free(prev_normalized);
    free(prediction_bitmap);
    free(dominant_signs);
    free(residual_data);
    free(packed_bitmap);
    
    return result;
}

// -------------------- core per-layer compression --------------------

/**
 * 通用层压缩（直接压缩，无动量预测）
 * 对齐Python: def _compress_generic_layer(self, current_grad: np.ndarray, client_id: str, current_step: int, layer_name: str) -> Dict[str, Any]
 * 
 * 压缩策略:
 * - 如果满足有损条件（_should_lossy），使用SZ3有损压缩
 * - 否则使用无损压缩（zstd/blosc/pickle）
 * 
 * 参数:
 *   mc: 压缩器实例
 *   current_grad: 当前梯度
 *   client_id: 客户端ID（未使用，保留接口一致性）
 *   layer_name: 层名称
 *   current_step: 当前步数
 * 
 * 返回: 压缩层数据，失败返回NULL
 */
static CompressedLayerData* compress_generic_layer(
    MomentumCompressor *mc,
    const NDArray *current_grad,
    const char *client_id __attribute__((unused)),  // 保留以保持接口一致性
    const char *layer_name,
    int current_step
) {
    CompressedLayerData *result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
    strcpy(result->type, "direct_generic");
    memcpy(result->shape, current_grad->shape, current_grad->ndim * sizeof(size_t));
    result->ndim = current_grad->ndim;
    result->step = current_step;
    strcpy(result->original_dtype, dtype_to_string(current_grad->dtype));
    
    // 智能判断是否使用有损压缩（对齐Python _should_lossy）
    bool use_lossy = should_use_lossy_compression(
        layer_name,
        current_grad->total_size,
        current_grad->dtype,
        mc->config.param_count_threshold
    );
    
    const char *codec_used;
    size_t byte_size = current_grad->total_size * dtype_size(current_grad->dtype);
    
    if (use_lossy && is_float_dtype(current_grad->dtype)) {
        // 使用SZ3有损压缩
        float compression_ratio;
        result->data = sz3_compress_simulate(
            current_grad->data,
            current_grad->total_size,
            current_grad->dtype,
            current_grad->shape,        // 传递形状
            current_grad->ndim,         // 传递维度
            "REL",                      // 硬编码REL，对齐Python
            0.0f,
            mc->config.error_bound,
            &compression_ratio,
            &codec_used,
            &result->data_size
        );
        strcpy(result->stored_dtype, dtype_to_string(current_grad->dtype));
    } else {
        // 使用无损压缩
        result->data = lossless_compress(
            current_grad->data,
            byte_size,
            mc->config.lossless_compressor,
            &codec_used,
            &result->data_size
        );
        strcpy(result->stored_dtype, dtype_to_string(current_grad->dtype));
    }
    
    strcpy(result->codec, codec_used);
    
    return result;
}

// -------------------- core per-layer compression --------------------

/**
 * 单层压缩（主接口）
 * 对齐Python: def _create_compressed_data(self, gradient: np.ndarray, client_id: str, layer_name: str) -> Dict[str, Any]
 * 
 * 压缩策略:
 * - Step 1: 直接压缩（按should_lossy判断使用SZ3或无损压缩）
 * - Step >= 2: 
 *   - 如果是4D卷积层且满足有损条件，使用动量预测压缩
 *   - 否则使用通用层压缩
 * 
 * 参数:
 *   compressor: 压缩器实例
 *   layer_name: 层名称
 *   gradient: 梯度数据
 * 
 * 返回: 压缩层数据，失败返回NULL
 */
CompressedLayerData* momentum_compressor_compress_layer(
    MomentumCompressor *compressor,
    const char *layer_name,
    const NDArray *gradient
) {
    // 输入验证
    if (!compressor) {
        fprintf(stderr, "[ERROR] compress_layer: compressor is NULL\n");
        return NULL;
    }
    
    if (!layer_name || strlen(layer_name) == 0) {
        fprintf(stderr, "[ERROR] compress_layer: invalid layer_name\n");
        return NULL;
    }
    
    if (!gradient) {
        fprintf(stderr, "[ERROR] compress_layer: gradient is NULL\n");
        return NULL;
    }
    
    // 使用compressor内部保存的client_id
    const char *client_id = compressor->current_client_id;
    if (!client_id || strlen(client_id) == 0) {
        fprintf(stderr, "[ERROR] compress_layer: client_id not set (call set_client first)\n");
        return NULL;
    }
    
    // 放宽层名长度限制,允许带.的长层名如layer1.0.bn1.running_mean
    if (strlen(layer_name) >= MAX_LAYER_NAME_LENGTH) {
        fprintf(stderr, "[ERROR] compress_layer: layer_name too long (%zu >= %d)\n",
                strlen(layer_name), MAX_LAYER_NAME_LENGTH);
        return NULL;
    }
    
    char key[MAX_KEY_LENGTH];
    make_history_key(key, client_id, layer_name);
    
    // 增加步数
    increment_step_count(compressor, key);
    int current_step = get_step_count(compressor, key);
    
    CompressedLayerData *result = NULL;
    
    if (current_step == 1) {
        // 步骤1: 直接压缩（对齐Python）
        result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
        if (!result) {
            fprintf(stderr, "[ERROR] compress_layer: failed to allocate CompressedLayerData\n");
            return NULL;
        }
        strcpy(result->type, "direct");
        memcpy(result->shape, gradient->shape, gradient->ndim * sizeof(size_t));
        result->ndim = gradient->ndim;
        result->step = current_step;
        strcpy(result->original_dtype, dtype_to_string(gradient->dtype));
        
        // 智能判断压缩方式
        bool use_lossy = should_use_lossy_compression(
            layer_name,
            gradient->total_size,
            gradient->dtype,
            compressor->config.param_count_threshold
        );
        
        const char *codec_used;
        size_t byte_size = gradient->total_size * dtype_size(gradient->dtype);
        
        if (use_lossy && is_float_dtype(gradient->dtype)) {
            float compression_ratio;
            result->data = sz3_compress_simulate(
                gradient->data,
                gradient->total_size,
                gradient->dtype,
                gradient->shape,            // 传递形状
                gradient->ndim,             // 传递维度
                "REL",
                0.0f,
                compressor->config.error_bound,
                &compression_ratio,
                &codec_used,
                &result->data_size
            );
        } else {
            result->data = lossless_compress(
                gradient->data,
                byte_size,
                compressor->config.lossless_compressor,
                &codec_used,
                &result->data_size
            );
        }
        
        strcpy(result->codec, codec_used);
        strcpy(result->stored_dtype, dtype_to_string(gradient->dtype));
        
        add_gradient_to_history(compressor, key, gradient);
    } else {
        // 步骤>=2: 动量预测或通用压缩
        NDArray *prev_grad = get_latest_gradient(compressor, key);
        
        if (prev_grad && gradient->ndim == 4 && gradient->dtype == DTYPE_FLOAT32 &&
            should_use_lossy_compression(layer_name, gradient->total_size, gradient->dtype, 
                                        compressor->config.param_count_threshold)) {
            // 4D卷积层 - 动量预测
            result = compress_conv_layer_momentum(compressor, gradient, prev_grad,
                                                  client_id, layer_name, current_step);
        } else {
            // 其他层 - 通用压缩
            result = compress_generic_layer(compressor, gradient, client_id, 
                                           layer_name, current_step);
        }
        
        add_gradient_to_history(compressor, key, gradient);
    }
    
    // 更新统计
    compressor->stats.total_compressions++;
    
    if (result && strcmp(result->type, "momentum_predicted") == 0) {
        // 记录预测比例
        if (compressor->stats.prediction_ratio_count >= compressor->stats.prediction_ratio_capacity) {
            size_t new_capacity = (compressor->stats.prediction_ratio_capacity == 0) ? 
                                 16 : compressor->stats.prediction_ratio_capacity * 2;
            compressor->stats.prediction_ratios = (float*)realloc(
                compressor->stats.prediction_ratios, new_capacity * sizeof(float));
            compressor->stats.prediction_ratio_capacity = new_capacity;
        }
        compressor->stats.prediction_ratios[compressor->stats.prediction_ratio_count++] = 
            result->prediction_ratio;
        
        if (compressor->stats.sign_mismatch_ratio_count >= compressor->stats.sign_mismatch_ratio_capacity) {
            size_t new_capacity = (compressor->stats.sign_mismatch_ratio_capacity == 0) ? 
                                 16 : compressor->stats.sign_mismatch_ratio_capacity * 2;
            compressor->stats.sign_mismatch_ratios = (float*)realloc(
                compressor->stats.sign_mismatch_ratios, new_capacity * sizeof(float));
            compressor->stats.sign_mismatch_ratio_capacity = new_capacity;
        }
        compressor->stats.sign_mismatch_ratios[compressor->stats.sign_mismatch_ratio_count++] = 
            result->sign_mismatch_ratio;
    }
    
    return result;
}

// -------------------- decompression --------------------

/**
 * 解压momentum_predicted类型的层（完整实现）
 * 对齐Python: def _decompress_conv_layer() 和 def _simulate_reconstruction()
 * 
 * 解压流程:
 * 1. 解压残差数据（支持SZ3有损或ZSTD无损）
 * 2. 解压位图（指示哪些kernel被预测）
 * 3. 解压主导符号（预测kernel的符号）
 * 4. 获取前一梯度历史和统计信息
 * 5. 归一化前一梯度
 * 6. 获取预测记忆
 * 7. 重建预测kernel并合并到残差
 * 
 * 参数:
 *   compressor: 压缩器实例
 *   compressed: 压缩层数据
 *   client_id: 客户端ID
 *   layer_name: 层名称
 * 
 * 返回: 重建的梯度数组，失败返回NULL
 */
static NDArray* decompress_momentum_predicted_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
) {
    // 1. 创建结果数组并解压残差
    DataType dtype = DTYPE_FLOAT32;
    if (strcmp(compressed->original_dtype, "float32") == 0) dtype = DTYPE_FLOAT32;
    else if (strcmp(compressed->original_dtype, "float64") == 0) dtype = DTYPE_FLOAT64;
    
    NDArray *residual = ndarray_create(compressed->shape, compressed->ndim, dtype);
    if (!residual) return NULL;
    
    // 解压残差数据（支持SZ3）
    size_t decompressed_size;
    void *residual_data = lossy_decompress_with_shape(
        compressed->data, 
        compressed->data_size,
        compressed->codec,
        compressed->shape,
        compressed->ndim,
        dtype,  // 传递数据类型
        &decompressed_size
    );
    if (!residual_data) {
        ndarray_destroy(residual);
        return NULL;
    }
    
    size_t expected_size = residual->total_size * dtype_size(dtype);
    if (decompressed_size != expected_size) {
        fprintf(stderr, "[WARNING] Residual size mismatch: got %zu, expected %zu\n",
                decompressed_size, expected_size);
        free(residual_data);
        ndarray_destroy(residual);
        return NULL;
    }
    memcpy(residual->data, residual_data, expected_size);
    free(residual_data);
    
    // 2. 解压位图
    if (!compressed->bitmap || compressed->bitmap_size == 0) {
        fprintf(stderr, "[WARNING] No bitmap data\n");
        return residual;  // 无位图,返回残差
    }
    
    size_t bitmap_decompressed_size;
    uint8_t *bitmap_packed = (uint8_t*)lossless_decompress(
        compressed->bitmap, compressed->bitmap_size, "zstd", &bitmap_decompressed_size);
    
    if (!bitmap_packed) {
        fprintf(stderr, "[WARNING] Bitmap decompression failed\n");
        return residual;
    }
    
    // 位图应该是 out_ch × in_ch (卷积核数量),而非 total_size
    size_t out_ch = residual->shape[0];
    size_t in_ch = residual->shape[1];
    size_t bitmap_bits = out_ch * in_ch;  // 每个卷积核一个位
    
    // 验证位图大小
    size_t expected_bitmap_bytes = (bitmap_bits + 7) / 8;
    if (bitmap_decompressed_size != expected_bitmap_bytes) {
        fprintf(stderr, "[WARNING] Bitmap size mismatch: got %zu bytes, expected %zu bytes for %zu kernels\n",
                bitmap_decompressed_size, expected_bitmap_bytes, bitmap_bits);
    }
    
    // 解包位图
    bool *bitmap = (bool*)calloc(bitmap_bits, sizeof(bool));
    size_t safe_bits = (bitmap_decompressed_size * 8 < bitmap_bits) ? 
                        bitmap_decompressed_size * 8 : bitmap_bits;
    for (size_t i = 0; i < safe_bits; i++) {
        bitmap[i] = (bool)((bitmap_packed[i / 8] >> (i % 8)) & 1);
    }
    free(bitmap_packed);
    
    // 3. 计算预测核数量
    size_t num_predicted = 0;
    for (size_t i = 0; i < bitmap_bits; i++) {
        if (bitmap[i]) num_predicted++;
    }
    
    if (num_predicted == 0) {
        free(bitmap);
        return residual;  // 无预测,返回残差
    }
    
    // 4. 解压主导符号
    int8_t *dominant_signs = NULL;
    if (compressed->dominant_signs && compressed->dominant_signs_size > 0) {
        size_t signs_decompressed_size;
        uint8_t *signs_packed = (uint8_t*)lossless_decompress(
            compressed->dominant_signs, compressed->dominant_signs_size, 
            "zstd", &signs_decompressed_size);
        
        if (signs_packed) {
            // 创建稀疏到稠密的映射
            // dominant_signs[kernel_idx] 对应 bitmap[kernel_idx]
            dominant_signs = (int8_t*)calloc(bitmap_bits, sizeof(int8_t));
            if (!dominant_signs) {
                fprintf(stderr, "[ERROR] decompress_momentum: failed to allocate dominant_signs\n");
                free(signs_packed);
                free(bitmap);
                return residual;
            }
            
            // 将压缩的稠密索引映射回稀疏索引
            size_t pred_idx = 0;
            for (size_t i = 0; i < bitmap_bits; i++) {
                if (bitmap[i]) {
                    if (pred_idx / 8 < signs_decompressed_size) {
                        int bit = (signs_packed[pred_idx / 8] >> (pred_idx % 8)) & 1;
                        dominant_signs[i] = bit ? 1 : -1;
                    } else {
                        dominant_signs[i] = 1;  // 默认正号
                    }
                    pred_idx++;
                }
            }
            free(signs_packed);
            
            if (pred_idx != num_predicted) {
                fprintf(stderr, "[WARNING] Predicted count mismatch: bitmap=%zu, signs=%zu\n",
                        pred_idx, num_predicted);
            }
        }
    }
    
    // 5. 获取前一梯度历史
    char history_key[512];
    make_history_key(history_key, client_id, layer_name);
    
    LayerHistory *lh = NULL;
    HASH_FIND_STR(compressor->layer_histories, history_key, lh);
    
    if (!lh || !lh->gradients_head) {
        fprintf(stderr, "[WARNING] No gradient history for %s, using residual only\n", layer_name);
        free(bitmap);
        if (dominant_signs) free(dominant_signs);
        return residual;
    }
    
    NDArray *prev_grad = lh->gradients_head->gradient;
    
    // 6. 获取统计信息
    float current_mean = compressed->current_mean;
    float current_std = compressed->current_std;
    float prev_mean = compressed->prev_mean;
    float prev_std = compressed->prev_std;
    
    // 7. 归一化前一梯度
    float *abs_prev = (float*)malloc(prev_grad->total_size * sizeof(float));
    float *prev_normalized = (float*)malloc(prev_grad->total_size * sizeof(float));
    if (!abs_prev || !prev_normalized) {
        fprintf(stderr, "[ERROR] decompress_momentum: failed to allocate normalization arrays\n");
        free(abs_prev);
        free(prev_normalized);
        free(bitmap);
        if (dominant_signs) free(dominant_signs);
        return residual;
    }
    
    float *prev_data = (float*)prev_grad->data;
    for (size_t i = 0; i < prev_grad->total_size; i++) {
        abs_prev[i] = fabsf(prev_data[i]);
        prev_normalized[i] = abs_prev[i] - prev_mean;
        if (prev_std > 1e-8f) {
            prev_normalized[i] /= prev_std;
        }
    }
    
    // 8. 获取或创建预测记忆（使用新的二级结构API）
    NDArray *layer_memory_array = get_prediction_memory_for_layer(compressor, client_id, 
                                                                    compressed->shape, compressed->ndim);
    
    if (!layer_memory_array) {
        layer_memory_array = ndarray_create(compressed->shape, compressed->ndim, DTYPE_FLOAT32);
        memset(layer_memory_array->data, 0, layer_memory_array->total_size * sizeof(float));
        set_prediction_memory_for_layer(compressor, client_id, compressed->shape, 
                                       compressed->ndim, layer_memory_array);
        ndarray_destroy(layer_memory_array);
        layer_memory_array = get_prediction_memory_for_layer(compressor, client_id,
                                                             compressed->shape, compressed->ndim);
        if (!layer_memory_array) {
            fprintf(stderr, "[ERROR] decompress_momentum: failed to create prediction memory\n");
            free(bitmap);
            if (dominant_signs) free(dominant_signs);
            free(abs_prev);
            free(prev_normalized);
            return residual;
        }
    }
    
    float *layer_memory = (float*)layer_memory_array->data;
    
    // 9. 应用动量预测重建（按卷积核处理）
    float *reconstructed = (float*)malloc(residual->total_size * sizeof(float));
    float *residual_floats = (float*)residual->data;
    memcpy(reconstructed, residual_floats, residual->total_size * sizeof(float));
    
    // 按卷积核索引处理，与压缩逻辑对齐
    size_t kernel_size = (compressed->ndim >= 3) ? 
                         compressed->shape[2] * ((compressed->ndim >= 4) ? compressed->shape[3] : 1) : 1;
    
    for (size_t oc = 0; oc < out_ch; oc++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            size_t kernel_idx = oc * in_ch + ic;
            
            if (!bitmap[kernel_idx]) {
                continue;  // 未预测的核，保持残差不变
            }
            
            size_t kernel_offset = kernel_idx * kernel_size;
            
            // 更新记忆并预测 (每个kernel的每个元素)
            for (size_t k = 0; k < kernel_size; k++) {
                size_t idx = kernel_offset + k;
                
                // 动量更新: memory = (1 - lr) * old + lr * prev_norm
                float old_memory = layer_memory[idx];
                float prev_norm = prev_normalized[idx];
                float new_memory = (1.0f - compressor->config.momentum_lr) * old_memory + 
                                   compressor->config.momentum_lr * prev_norm;
                layer_memory[idx] = new_memory;
                
                // 反归一化预测值
                float abs_pred;
                if (current_std > 1e-8f) {
                    abs_pred = new_memory * current_std + current_mean;
                } else {
                    abs_pred = new_memory + current_mean;
                }
                abs_pred = fabsf(abs_pred);
                
                // 应用主导符号 (每个kernel一个符号)
                float sign = 1.0f;
                if (dominant_signs) {
                    sign = (float)dominant_signs[kernel_idx];  // 使用kernel_idx
                }
                
                // 重建: reconstructed = residual + predicted
                float predicted_val = sign * abs_pred;
                reconstructed[idx] = residual_floats[idx] + predicted_val;
            }
        }
    }
    
    // 10. 创建最终结果
    NDArray *result = ndarray_create(compressed->shape, compressed->ndim, dtype);
    memcpy(result->data, reconstructed, residual->total_size * sizeof(float));
    
    // 10.5 layer_memory_array already points to the stored hash entry, so the
    // reconstruction loop updated prediction memory in place.
    
    // 11. 添加到历史记录
    add_gradient_to_history(compressor, history_key, result);
    
    // 清理
    free(bitmap);
    if (dominant_signs) free(dominant_signs);
    free(abs_prev);
    free(prev_normalized);
    free(reconstructed);
    ndarray_destroy(residual);
    
    return result;
}

// -------------------- decompression --------------------

/**
 * 单层解压（主接口）
 * 对齐Python: 解压逻辑（在compress/decompress方法中）
 * 
 * 解压策略:
 * - "direct" 或 "direct_generic": 直接解压（根据codec选择SZ3有损或无损解压）
 * - "momentum_predicted": 动量预测解压（完整重建流程）
 * 
 * 参数:
 *   compressor: 压缩器实例
 *   compressed: 压缩层数据
 *   client_id: 客户端ID
 *   layer_name: 层名称
 * 
 * 返回: 重建的梯度数组，失败返回NULL
 */
NDArray* momentum_compressor_decompress_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
) {
    // 输入验证
    if (!compressor) {
        fprintf(stderr, "[ERROR] decompress_layer: compressor is NULL\n");
        return NULL;
    }
    
    if (!compressed) {
        fprintf(stderr, "[ERROR] decompress_layer: compressed data is NULL\n");
        return NULL;
    }
    
    if (!client_id || strlen(client_id) == 0) {
        fprintf(stderr, "[ERROR] decompress_layer: invalid client_id\n");
        return NULL;
    }
    
    if (!layer_name || strlen(layer_name) == 0) {
        fprintf(stderr, "[ERROR] decompress_layer: invalid layer_name\n");
        return NULL;
    }
    
    // 验证压缩数据结构
    if (strlen(compressed->type) == 0) {
        fprintf(stderr, "[ERROR] decompress_layer: compressed type is empty\n");
        return NULL;
    }
    
    if (compressed->ndim < MIN_NDIM || compressed->ndim > MAX_NDIM) {
        fprintf(stderr, "[ERROR] decompress_layer: invalid ndim %zu (valid: %d-%d)\n",
                compressed->ndim, MIN_NDIM, MAX_NDIM);
        return NULL;
    }
    
    if (!compressed->data || compressed->data_size == 0) {
        fprintf(stderr, "[ERROR] decompress_layer: compressed data is empty\n");
        return NULL;
    }
    
    // 根据原始dtype创建数组
    DataType dtype = DTYPE_FLOAT32;  // 默认
    if (strcmp(compressed->original_dtype, "float32") == 0) dtype = DTYPE_FLOAT32;
    else if (strcmp(compressed->original_dtype, "float64") == 0) dtype = DTYPE_FLOAT64;
    else if (strcmp(compressed->original_dtype, "int32") == 0) dtype = DTYPE_INT32;
    else if (strcmp(compressed->original_dtype, "int64") == 0) dtype = DTYPE_INT64;
    else if (strcmp(compressed->original_dtype, "uint8") == 0) dtype = DTYPE_UINT8;
    else {
        fprintf(stderr, "[ERROR] Unknown original dtype: %s\n", compressed->original_dtype);
        return NULL;
    }
    
    if (strcmp(compressed->type, "direct") == 0 || 
        strcmp(compressed->type, "direct_generic") == 0) {
        // 直接解压（根据codec，支持有损和无损）
        NDArray *result = ndarray_create(compressed->shape, compressed->ndim, dtype);
        if (!result) return NULL;
        
        size_t decompressed_size;
        void *decompressed = NULL;
        
        // 根据codec选择解压方式
        if (strcmp(compressed->codec, "sz3") == 0 ||
            strcmp(compressed->codec, "sz3_memcpy") == 0) {
            // SZ3有损解压（需要形状信息和数据类型）
            decompressed = lossy_decompress_with_shape(
                compressed->data, 
                compressed->data_size,
                compressed->codec, 
                compressed->shape, 
                compressed->ndim,
                dtype,  // 传递数据类型
                &decompressed_size);
        } else {
            // 无损解压（zstd/blosc）
            decompressed = lossless_decompress(
                compressed->data, 
                compressed->data_size,
                compressed->codec,
                &decompressed_size);
        }
        
        if (!decompressed) {
            fprintf(stderr, "[ERROR] Decompression failed for codec: %s\n", compressed->codec);
            ndarray_destroy(result);
            return NULL;
        }

        size_t expected_size = result->total_size * dtype_size(dtype);
        if (decompressed_size != expected_size) {
            fprintf(stderr, "[ERROR] Size mismatch: got %zu, expected %zu\n",
                    decompressed_size, expected_size);
            free(decompressed);
            ndarray_destroy(result);
            return NULL;
        }
        memcpy(result->data, decompressed, expected_size);
        free(decompressed);
        
        // 添加到历史（对齐Python）
        char key[512];
        make_history_key(key, client_id, layer_name);
        add_gradient_to_history(compressor, key, result);
        
        return result;
    }
    else if (strcmp(compressed->type, "momentum_predicted") == 0) {
        // 调用完整的动量预测解压
        return decompress_momentum_predicted_layer(compressor, compressed, client_id, layer_name);
    }
    
    // 未知类型
    fprintf(stderr, "[ERROR] Unknown compression type: %s\n", compressed->type);
    return NULL;
}

// ===========================================================================
// 模型级压缩 - 对齐Python compress_model()
// ===========================================================================

uint8_t* momentum_compressor_compress_model(
    MomentumCompressor *compressor,
    const NDArray **gradients,
    const char **layer_names,
    size_t num_layers,
    const char *client_id,
    size_t *out_size
) {
    printf("[INFO] Compressing model with %zu layers for client %s\n", num_layers, client_id);

    if (client_id && strlen(client_id) > 0) {
        momentum_compressor_set_client(compressor, client_id);
    }
    
    // 1. 压缩所有层
    CompressedLayerData **compressed_layers = 
        (CompressedLayerData**)malloc(num_layers * sizeof(CompressedLayerData*));
    
    for (size_t i = 0; i < num_layers; i++) {
        compressed_layers[i] = momentum_compressor_compress_layer(
            compressor, layer_names[i], gradients[i]);
        if (!compressed_layers[i]) {
            fprintf(stderr, "[ERROR] Failed to compress layer %zu: %s\n", i, layer_names[i]);
            for (size_t j = 0; j < i; j++) {
                compressed_layer_data_free(compressed_layers[j]);
            }
            free(compressed_layers);
            *out_size = 0;
            return NULL;
        }
    }
    
    // 2. 序列化所有压缩层（简化序列化格式）
    // 格式: [magic_number(4)][num_layers(4)][metadata_size(4)][metadata][layer1][layer2]...
    // 每层: [name_len(4)][name][type_len(4)][type][data_size(8)][data]...
    
    // 计算总大小
    size_t total_size = 12;  // magic + num_layers + metadata_size
    
    // 元数据（简化版JSON格式字符串）
    char metadata[1024];
    snprintf(metadata, sizeof(metadata),
             "{\"compressor_type\":\"MomentumPredictorCompressor\","
             "\"client_id\":\"%s\","
             "\"layer_count\":%zu,"
             "\"momentum_lr\":%.6f,"
             "\"param_cutoff\":%zu}",
             client_id, num_layers, compressor->config.momentum_lr, 
             compressor->config.param_count_threshold);
    size_t metadata_len = strlen(metadata);
    total_size += metadata_len;
    
    // 计算每层大小
    for (size_t i = 0; i < num_layers; i++) {
        total_size += 4 + strlen(layer_names[i]);  // name_len + name
        total_size += 4 + strlen(compressed_layers[i]->type);  // type_len + type
        total_size += 8 + compressed_layers[i]->data_size;  // data_size + data
        // 添加其他字段大小
        total_size += 4 + strlen(compressed_layers[i]->codec);
        total_size += 4 + strlen(compressed_layers[i]->original_dtype);
        total_size += 4 + compressed_layers[i]->ndim * sizeof(size_t);  // shape
        total_size += 8 + compressed_layers[i]->bitmap_size;  // bitmap
        total_size += 8 + compressed_layers[i]->dominant_signs_size;  // dominant_signs
        total_size += 32;  // stats (4 floats * 4 bytes + counters)
    }
    
    // 分配缓冲区
    uint8_t *buffer = (uint8_t*)malloc(total_size);
    uint8_t *ptr = buffer;
    
    // 写magic number
    uint32_t magic = 0x4D4F4D43;  // "MOMC"
    memcpy(ptr, &magic, 4); ptr += 4;
    
    // 写层数
    uint32_t num_layers_32 = (uint32_t)num_layers;
    memcpy(ptr, &num_layers_32, 4); ptr += 4;
    
    // 写元数据
    uint32_t metadata_len_32 = (uint32_t)metadata_len;
    memcpy(ptr, &metadata_len_32, 4); ptr += 4;
    memcpy(ptr, metadata, metadata_len); ptr += metadata_len;
    
    // 写每层数据
    for (size_t i = 0; i < num_layers; i++) {
        CompressedLayerData *layer = compressed_layers[i];
        
        // 层名
        uint32_t name_len = (uint32_t)strlen(layer_names[i]);
        memcpy(ptr, &name_len, 4); ptr += 4;
        memcpy(ptr, layer_names[i], name_len); ptr += name_len;
        
        // 类型
        uint32_t type_len = (uint32_t)strlen(layer->type);
        memcpy(ptr, &type_len, 4); ptr += 4;
        memcpy(ptr, layer->type, type_len); ptr += type_len;
        
        // Codec
        uint32_t codec_len = (uint32_t)strlen(layer->codec);
        memcpy(ptr, &codec_len, 4); ptr += 4;
        memcpy(ptr, layer->codec, codec_len); ptr += codec_len;
        
        // Dtype
        uint32_t dtype_len = (uint32_t)strlen(layer->original_dtype);
        memcpy(ptr, &dtype_len, 4); ptr += 4;
        memcpy(ptr, layer->original_dtype, dtype_len); ptr += dtype_len;
        
        // Shape
        uint32_t ndim = (uint32_t)layer->ndim;
        memcpy(ptr, &ndim, 4); ptr += 4;
        memcpy(ptr, layer->shape, ndim * sizeof(size_t)); ptr += ndim * sizeof(size_t);
        
        // 主数据
        uint64_t data_size = (uint64_t)layer->data_size;
        memcpy(ptr, &data_size, 8); ptr += 8;
        memcpy(ptr, layer->data, layer->data_size); ptr += layer->data_size;
        
        // 位图
        uint64_t bitmap_size = (uint64_t)layer->bitmap_size;
        memcpy(ptr, &bitmap_size, 8); ptr += 8;
        if (bitmap_size > 0) {
            memcpy(ptr, layer->bitmap, layer->bitmap_size); ptr += layer->bitmap_size;
        }
        
        // 主导符号
        uint64_t signs_size = (uint64_t)layer->dominant_signs_size;
        memcpy(ptr, &signs_size, 8); ptr += 8;
        if (signs_size > 0) {
            memcpy(ptr, layer->dominant_signs, layer->dominant_signs_size);
            ptr += layer->dominant_signs_size;
        }
        
        // 统计信息
        memcpy(ptr, &layer->current_mean, 4); ptr += 4;
        memcpy(ptr, &layer->current_std, 4); ptr += 4;
        memcpy(ptr, &layer->prev_mean, 4); ptr += 4;
        memcpy(ptr, &layer->prev_std, 4); ptr += 4;
        
        uint64_t num_pred = (uint64_t)layer->num_predicted_kernels;
        memcpy(ptr, &num_pred, 8); ptr += 8;
    }
    
    *out_size = (size_t)(ptr - buffer);
    
    // 清理
    for (size_t i = 0; i < num_layers; i++) {
        compressed_layer_data_free(compressed_layers[i]);
    }
    free(compressed_layers);
    
    return buffer;
}

/**
 * 解压整个模型（多层）- 对齐Python decompress_model()
 */
NDArray** momentum_compressor_decompress_model(
    MomentumCompressor *compressor,
    const uint8_t *compressed_data,
    size_t compressed_size,
    char ***layer_names_out,
    size_t *num_layers_out
) {
    if (!compressor || !compressed_data || compressed_size < 12) {
        fprintf(stderr, "[ERROR] Invalid input to decompress_model\n");
        return NULL;
    }
    
    const uint8_t *ptr = compressed_data;
    
    // 1. 读取magic number
    uint32_t magic;
    memcpy(&magic, ptr, 4); ptr += 4;
    if (magic != 0x4D4F4D43) {  // "MOMC"
        fprintf(stderr, "[ERROR] Invalid magic number: 0x%X\n", magic);
        return NULL;
    }
    
    // 2. 读取层数
    uint32_t num_layers_32;
    memcpy(&num_layers_32, ptr, 4); ptr += 4;
    size_t num_layers = (size_t)num_layers_32;
    
    // 3. 读取元数据
    uint32_t metadata_len;
    memcpy(&metadata_len, ptr, 4); ptr += 4;
    char *metadata = (char*)malloc(metadata_len + 1);
    memcpy(metadata, ptr, metadata_len);
    metadata[metadata_len] = '\0';
    ptr += metadata_len;
    
    printf("[INFO] Decompressing model: %zu layers\n", num_layers);
    printf("[INFO] Metadata: %s\n", metadata);
    
    // 提取client_id (简单解析)
    char client_id[256] = "unknown";
    char *cid_start = strstr(metadata, "\"client_id\":\"");
    if (cid_start) {
        cid_start += 13;
        char *cid_end = strchr(cid_start, '"');
        if (cid_end) {
            size_t cid_len = cid_end - cid_start;
            if (cid_len < sizeof(client_id)) {
                strncpy(client_id, cid_start, cid_len);
                client_id[cid_len] = '\0';
            }
        }
    }
    free(metadata);
    
    // 4. 分配输出数组
    NDArray **gradients = (NDArray**)malloc(num_layers * sizeof(NDArray*));
    char **layer_names = (char**)malloc(num_layers * sizeof(char*));
    
    // 5. 读取并解压每层
    for (size_t i = 0; i < num_layers; i++) {
        // 读取层名
        uint32_t name_len;
        memcpy(&name_len, ptr, 4); ptr += 4;
        layer_names[i] = (char*)malloc(name_len + 1);
        memcpy(layer_names[i], ptr, name_len);
        layer_names[i][name_len] = '\0';
        ptr += name_len;
        
        // 读取类型
        uint32_t type_len;
        memcpy(&type_len, ptr, 4); ptr += 4;
        char *type = (char*)malloc(type_len + 1);
        memcpy(type, ptr, type_len);
        type[type_len] = '\0';
        ptr += type_len;
        
        // 读取codec
        uint32_t codec_len;
        memcpy(&codec_len, ptr, 4); ptr += 4;
        char *codec = (char*)malloc(codec_len + 1);
        memcpy(codec, ptr, codec_len);
        codec[codec_len] = '\0';
        ptr += codec_len;
        
        // 读取dtype
        uint32_t dtype_len;
        memcpy(&dtype_len, ptr, 4); ptr += 4;
        char *dtype_str = (char*)malloc(dtype_len + 1);
        memcpy(dtype_str, ptr, dtype_len);
        dtype_str[dtype_len] = '\0';
        ptr += dtype_len;
        
        // 读取shape
        uint32_t ndim;
        memcpy(&ndim, ptr, 4); ptr += 4;
        size_t *shape = (size_t*)malloc(ndim * sizeof(size_t));
        memcpy(shape, ptr, ndim * sizeof(size_t));
        ptr += ndim * sizeof(size_t);
        
        // 读取主数据
        uint64_t data_size;
        memcpy(&data_size, ptr, 8); ptr += 8;
        uint8_t *data = (uint8_t*)malloc(data_size);
        memcpy(data, ptr, data_size);
        ptr += data_size;
        
        // 读取位图
        uint64_t bitmap_size;
        memcpy(&bitmap_size, ptr, 8); ptr += 8;
        uint8_t *bitmap = NULL;
        if (bitmap_size > 0) {
            bitmap = (uint8_t*)malloc(bitmap_size);
            memcpy(bitmap, ptr, bitmap_size);
            ptr += bitmap_size;
        }
        
        // 读取主导符号
        uint64_t signs_size;
        memcpy(&signs_size, ptr, 8); ptr += 8;
        uint8_t *signs = NULL;
        if (signs_size > 0) {
            signs = (uint8_t*)malloc(signs_size);
            memcpy(signs, ptr, signs_size);
            ptr += signs_size;
        }
        
        // 读取统计信息
        float current_mean, current_std, prev_mean, prev_std;
        memcpy(&current_mean, ptr, 4); ptr += 4;
        memcpy(&current_std, ptr, 4); ptr += 4;
        memcpy(&prev_mean, ptr, 4); ptr += 4;
        memcpy(&prev_std, ptr, 4); ptr += 4;
        
        uint64_t num_pred;
        memcpy(&num_pred, ptr, 8); ptr += 8;
        
        // 构建CompressedLayerData
        CompressedLayerData layer_data;
        memset(&layer_data, 0, sizeof(layer_data));
        strncpy(layer_data.type, type, sizeof(layer_data.type) - 1);
        strncpy(layer_data.codec, codec, sizeof(layer_data.codec) - 1);
        strncpy(layer_data.original_dtype, dtype_str, sizeof(layer_data.original_dtype) - 1);
        strncpy(layer_data.stored_dtype, dtype_str, sizeof(layer_data.stored_dtype) - 1);
        layer_data.ndim = (size_t)ndim;
        memcpy(layer_data.shape, shape, ndim * sizeof(size_t));
        layer_data.data = data;
        layer_data.data_size = (size_t)data_size;
        layer_data.bitmap = bitmap;
        layer_data.bitmap_size = (size_t)bitmap_size;
        layer_data.dominant_signs = signs;
        layer_data.dominant_signs_size = (size_t)signs_size;
        layer_data.current_mean = current_mean;
        layer_data.current_std = current_std;
        layer_data.prev_mean = prev_mean;
        layer_data.prev_std = prev_std;
        layer_data.num_predicted_kernels = (size_t)num_pred;
        
        // 解压该层
        gradients[i] = momentum_compressor_decompress_layer(
            compressor, &layer_data, client_id, layer_names[i]);
        
        if (!gradients[i]) {
            fprintf(stderr, "[ERROR] Failed to decompress layer %zu: %s\n", i, layer_names[i]);
        }
        
        // 清理临时数据
        free(type);
        free(codec);
        free(dtype_str);
        free(shape);
        free(data);
        if (bitmap) free(bitmap);
        if (signs) free(signs);
    }
    
    *layer_names_out = layer_names;
    *num_layers_out = num_layers;
    
    return gradients;
}

// ===========================================================================
// 重置函数
// ===========================================================================

void momentum_compressor_reset_client(MomentumCompressor *compressor, const char *client_id) {
    if (!compressor || !client_id) return;
    
    printf("[INFO] Resetting state for client: %s\n", client_id);
    
    size_t client_id_len = strlen(client_id);
    int deleted_count = 0;
    
    // 1. 清理gradient_history中匹配的条目
    LayerHistory *lh, *lh_tmp;
    HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
        // 检查key是否以"client_id:"开头
        if (strncmp(lh->key, client_id, client_id_len) == 0 && 
            lh->key[client_id_len] == ':') {
            HASH_DEL(compressor->layer_histories, lh);
            
            // 清理梯度链表
            GradientNode *node = lh->gradients_head;
            while (node) {
                GradientNode *next = node->next;
                ndarray_destroy(node->gradient);
                free(node);
                node = next;
            }
            free(lh);
            deleted_count++;
        }
    }
    
    // 2. 清理prediction_memory中匹配的client条目（二级结构）
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        if (strcmp(pm->client_id, client_id) == 0) {
            HASH_DEL(compressor->prediction_memories, pm);
            
            // 清理该client的所有layer memories
            LayerMemoryEntry *lme, *lme_tmp;
            HASH_ITER(hh, pm->layer_memories, lme, lme_tmp) {
                HASH_DEL(pm->layer_memories, lme);
                if (lme->memory) {
                    ndarray_destroy(lme->memory);
                }
                free(lme);
            }
            free(pm);
            deleted_count++;
        }
    }
    
    // 3. 清理step_count中匹配的条目
    StepCount *sc, *sc_tmp;
    HASH_ITER(hh, compressor->step_counts, sc, sc_tmp) {
        if (strncmp(sc->key, client_id, client_id_len) == 0 && 
            sc->key[client_id_len] == ':') {
            HASH_DEL(compressor->step_counts, sc);
            free(sc);
            deleted_count++;
        }
    }
    
    printf("✓ Deleted %d entries for client %s\n", deleted_count, client_id);
}

void momentum_compressor_reset_all(MomentumCompressor *compressor) {
    if (!compressor) return;
    
    // 清理所有哈希表
    LayerHistory *lh, *lh_tmp;
    HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
        HASH_DEL(compressor->layer_histories, lh);
        GradientNode *node = lh->gradients_head;
        while (node) {
            GradientNode *next = node->next;
            ndarray_destroy(node->gradient);
            free(node);
            node = next;
        }
        free(lh);
    }
    compressor->layer_histories = NULL;
    
    // 清理二级prediction_memories结构
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        HASH_DEL(compressor->prediction_memories, pm);
        
        // 清理每个client的layer memories
        LayerMemoryEntry *lme, *lme_tmp;
        HASH_ITER(hh, pm->layer_memories, lme, lme_tmp) {
            HASH_DEL(pm->layer_memories, lme);
            if (lme->memory) {
                ndarray_destroy(lme->memory);
            }
            free(lme);
        }
        free(pm);
    }
    compressor->prediction_memories = NULL;
    
    StepCount *sc, *sc_tmp;
    HASH_ITER(hh, compressor->step_counts, sc, sc_tmp) {
        HASH_DEL(compressor->step_counts, sc);
        free(sc);
    }
    compressor->step_counts = NULL;
    
    printf("✓ All states reset\n");
}

// ===========================================================================
// 统计信息
// ===========================================================================

void momentum_compressor_print_stats(const MomentumCompressor *compressor) {
    if (!compressor) return;
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Compression Statistics                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Total compressions: %-42zu║\n", compressor->stats.total_compressions);
    printf("║ History entries: %-47u║\n", HASH_COUNT(compressor->layer_histories));
    printf("║ Prediction memories: %-43u║\n", HASH_COUNT(compressor->prediction_memories));
    printf("║ Step counters: %-49u║\n", HASH_COUNT(compressor->step_counts));
    
    if (compressor->stats.prediction_ratio_count > 0) {
        float sum = 0.0f;
        for (size_t i = 0; i < compressor->stats.prediction_ratio_count; i++) {
            sum += compressor->stats.prediction_ratios[i];
        }
        float avg = sum / compressor->stats.prediction_ratio_count;
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Avg prediction ratio: %.2f%% (from %zu predictions)%*s║\n", 
               avg * 100, compressor->stats.prediction_ratio_count,
               (int)(19 - (compressor->stats.prediction_ratio_count >= 10 ? 2 : 1)), "");
    }
    
    if (compressor->stats.sign_mismatch_ratio_count > 0) {
        float sum = 0.0f;
        for (size_t i = 0; i < compressor->stats.sign_mismatch_ratio_count; i++) {
            sum += compressor->stats.sign_mismatch_ratios[i];
        }
        float avg = sum / compressor->stats.sign_mismatch_ratio_count;
        printf("║ Avg sign mismatch: %.2f%%%*s║\n", avg * 100, 42, "");
    }
    
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

// ===========================================================================
// 辅助API实现 - 对齐Python辅助函数
// ===========================================================================

CompressionStats* momentum_compressor_get_stats(const MomentumCompressor *compressor) {
    if (!compressor) return NULL;
    
    CompressionStats *stats = (CompressionStats*)malloc(sizeof(CompressionStats));
    if (!stats) return NULL;
    
    stats->total_compressions = compressor->stats.total_compressions;
    stats->num_predictions = compressor->stats.prediction_ratio_count;
    
    // 计算平均预测比例
    if (compressor->stats.prediction_ratio_count > 0) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (size_t i = 0; i < compressor->stats.prediction_ratio_count; i++) {
            float val = compressor->stats.prediction_ratios[i];
            sum += val;
            sum_sq += val * val;
        }
        stats->avg_prediction_ratio = sum / compressor->stats.prediction_ratio_count;
        
        // 计算标准差: sqrt(E[X^2] - (E[X])^2)
        float mean_sq = sum_sq / compressor->stats.prediction_ratio_count;
        float variance = mean_sq - (stats->avg_prediction_ratio * stats->avg_prediction_ratio);
        stats->std_prediction_ratio = sqrtf(variance > 0 ? variance : 0);
    } else {
        stats->avg_prediction_ratio = 0.0f;
        stats->std_prediction_ratio = 0.0f;
    }
    
    // 计算平均符号不匹配比例
    if (compressor->stats.sign_mismatch_ratio_count > 0) {
        float sum = 0.0f;
        for (size_t i = 0; i < compressor->stats.sign_mismatch_ratio_count; i++) {
            sum += compressor->stats.sign_mismatch_ratios[i];
        }
        stats->avg_sign_mismatch_ratio = sum / compressor->stats.sign_mismatch_ratio_count;
    } else {
        stats->avg_sign_mismatch_ratio = 0.0f;
    }
    
    return stats;
}

DetailedStats* momentum_compressor_get_detailed_stats(const MomentumCompressor *compressor) {
    if (!compressor) return NULL;
    
    DetailedStats *detailed = (DetailedStats*)malloc(sizeof(DetailedStats));
    if (!detailed) return NULL;
    
    // 复制压缩统计
    CompressionStats *comp_stats = momentum_compressor_get_stats(compressor);
    if (!comp_stats) {
        free(detailed);
        return NULL;
    }
    detailed->compression = *comp_stats;
    free(comp_stats);
    
    // 收集唯一的客户端ID (从step_counts中提取)
    // 临时数组存储唯一客户端ID
    char unique_clients[256][256];
    size_t num_unique = 0;
    
    StepCount *sc, *tmp;
    HASH_ITER(hh, compressor->step_counts, sc, tmp) {
        // 提取客户端ID (格式: "ClientN:layer_name")
        const char *colon = strchr(sc->key, ':');
        if (!colon) continue;
        
        size_t id_len = colon - sc->key;
        if (id_len == 0 || id_len >= 256) continue;  // 跳过无效长度
        
        // 检查是否已存在
        bool found = false;
        for (size_t i = 0; i < num_unique; i++) {
            if (strncmp(unique_clients[i], sc->key, id_len) == 0 && 
                unique_clients[i][id_len] == '\0') {
                found = true;
                break;
            }
        }
        
        if (!found && num_unique < 256) {
            strncpy(unique_clients[num_unique], sc->key, id_len);
            unique_clients[num_unique][id_len] = '\0';
            num_unique++;
        }
    }
    
    detailed->num_clients = num_unique;
    
    if (num_unique == 0) {
        detailed->clients = NULL;
        return detailed;
    }
    
    detailed->clients = (ClientStats*)calloc(num_unique, sizeof(ClientStats));
    if (!detailed->clients) {
        free(detailed);
        return NULL;
    }
    
    // 为每个客户端收集统计
    for (size_t i = 0; i < num_unique; i++) {
        // unique_clients[i]已经保证<256且null终止
        size_t len = strlen(unique_clients[i]);
        memcpy(detailed->clients[i].client_id, unique_clients[i], len + 1);  // +1 for null terminator
        
        size_t id_len = len;
        
        // 统计步数 (取该客户端所有层的最大步数)
        int max_step = 0;
        HASH_ITER(hh, compressor->step_counts, sc, tmp) {
            if (strncmp(sc->key, unique_clients[i], id_len) == 0 && 
                sc->key[id_len] == ':') {
                if (sc->step > max_step) max_step = sc->step;
            }
        }
        detailed->clients[i].step_count = max_step;
        
        // 统计历史记录数
        size_t hist_count = 0;
        LayerHistory *lh, *lh_tmp;
        HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
            if (strncmp(lh->key, unique_clients[i], id_len) == 0 &&
                lh->key[id_len] == ':') {
                hist_count++;
            }
        }
        detailed->clients[i].history_length = hist_count;
        
        // 统计预测记忆层数（使用二级结构）
        size_t mem_count = 0;
        PredictionMemory *pm;
        HASH_FIND_STR(compressor->prediction_memories, unique_clients[i], pm);
        if (pm) {
            // 统计该client有多少个layer memories
            LayerMemoryEntry *lme, *lme_tmp;
            HASH_ITER(hh, pm->layer_memories, lme, lme_tmp) {
                mem_count++;
            }
        }
        detailed->clients[i].num_memory_layers = mem_count;
    }
    
    return detailed;
}

void compression_stats_free(CompressionStats *stats) {
    if (stats) free(stats);
}

void detailed_stats_free(DetailedStats *stats) {
    if (!stats) return;
    if (stats->clients) free(stats->clients);
    free(stats);
}

void momentum_compressor_set_log_level(MomentumCompressor *compressor, const char *level) {
    if (!compressor || !level) return;
    
    // 对齐Python的日志级别映射
    if (strcmp(level, "DEBUG") == 0) {
        compressor->log_level = 0;
    } else if (strcmp(level, "INFO") == 0) {
        compressor->log_level = 1;
    } else if (strcmp(level, "WARNING") == 0) {
        compressor->log_level = 2;
    } else if (strcmp(level, "ERROR") == 0) {
        compressor->log_level = 3;
    } else {
        fprintf(stderr, "Unknown log level: %s (using INFO)\n", level);
        compressor->log_level = 1;
    }
}

void momentum_compressor_set_client_context(MomentumCompressor *compressor, const char *client_id) {
    if (!compressor || !client_id) return;
    
    // 规范化客户端ID为"ClientN"格式
    // 如果已经是"ClientN"格式，直接使用
    if (strncmp(client_id, "Client", 6) == 0 && isdigit(client_id[6])) {
        strncpy(compressor->current_client_id, client_id, sizeof(compressor->current_client_id) - 1);
        compressor->current_client_id[sizeof(compressor->current_client_id) - 1] = '\0';
    } else {
        // 尝试从字符串中提取数字
        const char *p = client_id;
        while (*p && !isdigit(*p)) p++;
        
        if (*p) {
            int num = atoi(p);
            snprintf(compressor->current_client_id, sizeof(compressor->current_client_id), 
                    "Client%d", num);
        } else {
            // 无法提取数字，使用原始ID
            strncpy(compressor->current_client_id, client_id, sizeof(compressor->current_client_id) - 1);
            compressor->current_client_id[sizeof(compressor->current_client_id) - 1] = '\0';
        }
    }
    
    if (compressor->log_level <= 1) {  // INFO
        printf("[INFO] Client context set to: %s\n", compressor->current_client_id);
    }
}

// ===========================================================================
// 测试辅助函数
// ===========================================================================

int momentum_compressor_get_log_level(const MomentumCompressor *compressor) {
    return compressor ? compressor->log_level : -1;
}

const char* momentum_compressor_get_current_client_id(const MomentumCompressor *compressor) {
    return compressor ? compressor->current_client_id : NULL;
}

// ===========================================================================
// 批处理API实现 - 对齐Python批量操作
// ===========================================================================

/**
 * 批量压缩多层梯度
 * 
 * 功能: 一次性压缩多个层的梯度,减少函数调用开销
 * 优势: 
 *   1. 减少重复的客户端设置操作
 *   2. 可选并行处理 (OpenMP)
 *   3. 统一错误处理和资源管理
 * 
 * @param compressor 压缩器实例
 * @param items 批处理项数组 (输入gradient, 输出compressed)
 * @param num_items 批处理项数量
 * @param client_id 客户端ID
 * @return 成功返回0, 失败返回负数错误码
 */
int momentum_compressor_compress_batch(
    MomentumCompressor *compressor,
    BatchItem *items,
    size_t num_items,
    const char *client_id
) {
    // 参数验证
    if (!compressor) {
        fprintf(stderr, "[ERROR] Batch compress: compressor is NULL\n");
        return ERROR_INVALID_PARAM;
    }
    
    if (!items || num_items == 0) {
        fprintf(stderr, "[ERROR] Batch compress: invalid items (ptr=%p, count=%zu)\n", 
                (void*)items, num_items);
        return ERROR_INVALID_PARAM;
    }
    
    if (!client_id || strlen(client_id) == 0) {
        fprintf(stderr, "[ERROR] Batch compress: invalid client_id\n");
        return ERROR_INVALID_PARAM;
    }
    
    // 检查批量大小限制
    if (num_items > MAX_BATCH_SIZE) {
        fprintf(stderr, "[ERROR] Batch compress: num_items (%zu) exceeds MAX_BATCH_SIZE (%d)\n",
                num_items, MAX_BATCH_SIZE);
        return ERROR_INVALID_PARAM;
    }
    
    // 日志记录
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch compress: processing %zu layers for client '%s'\n", 
               num_items, client_id);
    }
    
    // 设置客户端上下文 (仅一次)
    momentum_compressor_set_client(compressor, client_id);
    
    // 统计信息
    size_t success_count = 0;
    size_t total_compressed_size = 0;
    size_t total_original_size = 0;
    
    // 顺序压缩各层
    // TODO: 可选OpenMP并行化 (#pragma omp parallel for)
    for (size_t i = 0; i < num_items; i++) {
        // 初始化输出
        items[i].compressed = NULL;
        items[i].error = ERROR_NONE;
        
        // 验证单项
        if (!items[i].gradient || !items[i].layer_name) {
            fprintf(stderr, "[ERROR] Batch compress: item[%zu] has NULL gradient or layer_name\n", i);
            items[i].error = ERROR_INVALID_PARAM;
            continue;
        }
        
        // 压缩单层
        CompressedLayerData *compressed = momentum_compressor_compress_layer(
            compressor,
            items[i].layer_name,
            items[i].gradient
        );
        
        if (!compressed) {
            fprintf(stderr, "[ERROR] Batch compress: failed to compress item[%zu] layer='%s'\n",
                    i, items[i].layer_name);
            items[i].error = ERROR_COMPRESSION;
            continue;
        }
        
        // 成功
        items[i].compressed = compressed;
        items[i].error = ERROR_NONE;
        success_count++;
        
        // 统计压缩率
        total_compressed_size += compressed->data_size;
        if (compressed->bitmap_size > 0) {
            total_compressed_size += compressed->bitmap_size;
        }
        if (compressed->dominant_signs_size > 0) {
            total_compressed_size += compressed->dominant_signs_size;
        }
        
        total_original_size += items[i].gradient->total_size * 
                               (items[i].gradient->dtype == DTYPE_FLOAT32 ? 4 : 8);
    }
    
    // 日志记录结果
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        float compression_ratio = total_original_size > 0 ? 
            (float)total_compressed_size / total_original_size : 0.0f;
        
        printf("[INFO] Batch compress: %zu/%zu layers succeeded, "
               "compression ratio: %.2f%% (%.2f KB → %.2f KB)\n",
               success_count, num_items,
               compression_ratio * 100.0f,
               total_original_size / 1024.0f,
               total_compressed_size / 1024.0f);
    }
    
    // 返回结果
    if (success_count == 0) {
        return ERROR_COMPRESSION;  // 全部失败
    } else if (success_count < num_items) {
        return -((int)num_items - (int)success_count);  // 部分失败,返回失败数量
    } else {
        return ERROR_NONE;  // 全部成功
    }
}

/**
 * 批量解压多层梯度
 * 
 * 功能: 一次性解压多个层的梯度
 * 
 * @param compressor 压缩器实例
 * @param items 批处理项数组 (输入compressed)
 * @param num_items 批处理项数量
 * @param client_id 客户端ID
 * @param out_gradients 输出梯度数组 (调用者需提供空间)
 * @return 成功返回0, 失败返回负数错误码
 */
int momentum_compressor_decompress_batch(
    MomentumCompressor *compressor,
    const BatchItem *items,
    size_t num_items,
    const char *client_id,
    NDArray **out_gradients
) {
    // 参数验证
    if (!compressor) {
        fprintf(stderr, "[ERROR] Batch decompress: compressor is NULL\n");
        return ERROR_INVALID_PARAM;
    }
    
    if (!items || num_items == 0) {
        fprintf(stderr, "[ERROR] Batch decompress: invalid items (ptr=%p, count=%zu)\n",
                (void*)items, num_items);
        return ERROR_INVALID_PARAM;
    }
    
    if (!client_id || strlen(client_id) == 0) {
        fprintf(stderr, "[ERROR] Batch decompress: invalid client_id\n");
        return ERROR_INVALID_PARAM;
    }
    
    if (!out_gradients) {
        fprintf(stderr, "[ERROR] Batch decompress: out_gradients is NULL\n");
        return ERROR_INVALID_PARAM;
    }
    
    // 检查批量大小限制
    if (num_items > MAX_BATCH_SIZE) {
        fprintf(stderr, "[ERROR] Batch decompress: num_items (%zu) exceeds MAX_BATCH_SIZE (%d)\n",
                num_items, MAX_BATCH_SIZE);
        return ERROR_INVALID_PARAM;
    }
    
    // 日志记录
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch decompress: processing %zu layers for client '%s'\n",
               num_items, client_id);
    }
    
    // 设置客户端上下文 (仅一次)
    momentum_compressor_set_client(compressor, client_id);
    
    // 统计信息
    size_t success_count = 0;
    
    // 顺序解压各层
    // TODO: 可选OpenMP并行化 (#pragma omp parallel for)
    for (size_t i = 0; i < num_items; i++) {
        // 初始化输出
        out_gradients[i] = NULL;
        
        // 验证单项
        if (!items[i].compressed || !items[i].layer_name) {
            fprintf(stderr, "[ERROR] Batch decompress: item[%zu] has NULL compressed or layer_name\n", i);
            continue;
        }
        
        // 解压单层
        NDArray *gradient = momentum_compressor_decompress_layer(
            compressor,
            items[i].compressed,
            client_id,
            items[i].layer_name
        );
        
        if (!gradient) {
            fprintf(stderr, "[ERROR] Batch decompress: failed to decompress item[%zu] layer='%s'\n",
                    i, items[i].layer_name);
            continue;
        }
        
        // 成功
        out_gradients[i] = gradient;
        success_count++;
    }
    
    // 日志记录结果
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch decompress: %zu/%zu layers succeeded\n",
               success_count, num_items);
    }
    
    // 返回结果
    if (success_count == 0) {
        return ERROR_DECOMPRESSION;  // 全部失败
    } else if (success_count < num_items) {
        return -((int)num_items - (int)success_count);  // 部分失败,返回失败数量
    } else {
        return ERROR_NONE;  // 全部成功
    }
}
