/**
 * ===========================================================================
 * MomentumPredictorCompressor
 * 
 * 原始Python代码: momentum_predictor_compressor.py
 * 
 * 主要功能:
 * - 基于动量的梯度预测
 * - 符号一致性分析
 * - 混合压缩策略 (SZ3有损 + Zstd无损)
 * - 多客户端状态管理
 * ===========================================================================
 */

#ifndef MOMENTUM_COMPRESSOR_H
#define MOMENTUM_COMPRESSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// 常量定义 
// ===========================================================================

// 缓冲区大小限制
#define MAX_KEY_LENGTH 512
#define MAX_CLIENT_ID_LENGTH 256
#define MAX_LAYER_NAME_LENGTH 256
#define MAX_COMPRESSOR_NAME_LENGTH 64
#define MAX_PATH_LENGTH 512
#define MAX_ERROR_MESSAGE_LENGTH 256

// 数据维度限制
#define MAX_NDIM 8
#define MIN_NDIM 1

// 压缩参数默认值
#define DEFAULT_MOMENTUM_LR 0.07f  // 对齐Python默认值
#define DEFAULT_CONSISTENCY_THRESHOLD 0.5f
#define DEFAULT_PARAM_COUNT_THRESHOLD 1024
#define DEFAULT_MAX_HISTORY_LENGTH 3
#define DEFAULT_ERROR_BOUND 1.0f

// 性能相关
#define MIN_COMPRESSION_SIZE 1024
#define MAX_BATCH_SIZE 1024
#define STATS_INITIAL_CAPACITY 100

// 日志级别
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR 3

// ===========================================================================
// 数据类型定义
// ===========================================================================

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UNKNOWN
} DataType;

typedef enum {
    ERROR_NONE = 0,
    ERROR_ALLOCATION = -1,
    ERROR_COMPRESSION = -2,
    ERROR_DECOMPRESSION = -3,
    ERROR_INVALID_PARAM = -4,
    ERROR_NOT_FOUND = -5,
    ERROR_SZ3_FAILED = -6
} ErrorCode;

// 多维数组结构 (替代 numpy.ndarray)
typedef struct {
    void *data;           // 数据指针（支持多种类型）
    size_t *shape;        // 各维度大小 [dim0, dim1, ...]
    size_t ndim;          // 维度数量
    size_t total_size;    // 总元素数量
    DataType dtype;       // 数据类型
} NDArray;

// ===========================================================================
// 配置结构
// ===========================================================================

typedef struct {
    float momentum_lr;                    // 动量学习率, 默认 0.07
    float consistency_threshold;          // 一致性阈值, 默认 0.5
    char lossless_compressor[32];         // "zstd" 或 "blosc"
    
    // SZ3 配置
    char error_bounding_mode[16];         // "ABS", "REL" 等
    float error_bound;                    // 误差界, 默认 1.0
    char sz3_lib_path[512];               // SZ3库路径
    
    size_t param_count_threshold;         // 参数阈值, 默认 1024
    int max_history_length;               // 历史记录最大长度, 默认 3
} CompressorConfig;

// ===========================================================================
// 压缩数据结构
// ===========================================================================

typedef struct {
    char type[32];                        // "direct", "momentum_predicted", "direct_generic"
    char codec[16];                       // "sz3", "zstd", "pickle"
    
    uint8_t *data;                        // 压缩后的主数据
    size_t data_size;
    
    uint8_t *bitmap;                      // 预测位图 (仅momentum_predicted)
    size_t bitmap_size;
    
    uint8_t *dominant_signs;              // 主导符号 (仅momentum_predicted)
    size_t dominant_signs_size;
    
    size_t shape[MAX_NDIM];               // 原始形状
    size_t ndim;                          // 维度数
    
    char original_dtype[16];              // 原始数据类型
    char stored_dtype[16];                // 存储数据类型
    
    int step;                             // 步数
    int num_predicted_kernels;            // 预测的卷积核数量
    
    // 统计信息
    float prediction_ratio;
    float sign_mismatch_ratio;
    float current_mean, current_std;
    float prev_mean, prev_std;
    float global_min, global_max;
    
    // Breakdown 时间统计（单位：秒）
    double breakdown_stats_time;
    double breakdown_normalize_time;
    double breakdown_consistency_time;
    double breakdown_prediction_time;
    double breakdown_residual_compress_time;
    double breakdown_bitmap_compress_time;
    double breakdown_metadata_time;
    double breakdown_total_time;
} CompressedLayerData;

// ===========================================================================
// 前向声明 (避免循环依赖)
// ===========================================================================

typedef struct MomentumCompressor MomentumCompressor;

// ===========================================================================
// 公共API函数
// ===========================================================================

/**
 * 创建压缩器实例
 * @param config 配置参数
 * @return 压缩器指针, 失败返回NULL
 */
MomentumCompressor* momentum_compressor_create(const CompressorConfig *config);

/**
 * 销毁压缩器实例
 * @param compressor 压缩器指针
 */
void momentum_compressor_destroy(MomentumCompressor *compressor);

/**
 * 设置当前客户端上下文
 * @param compressor 压缩器指针
 * @param client_id 客户端ID
 */
void momentum_compressor_set_client(MomentumCompressor *compressor, const char *client_id);

/**
 * 压缩单层梯度
 * @param compressor 压缩器指针
 * @param layer_name 层名称
 * @param gradient 梯度数组
 * @return 压缩结果, 失败返回NULL
 * @note client_id 通过 momentum_compressor_set_client() 预先设置
 */
CompressedLayerData* momentum_compressor_compress_layer(
    MomentumCompressor *compressor,
    const char *layer_name,
    const NDArray *gradient
);

/**
 * 解压单层梯度
 * @param compressor 压缩器指针
 * @param compressed 压缩数据
 * @param client_id 客户端ID
 * @param layer_name 层名称
 * @return 解压后的数组, 失败返回NULL
 */
NDArray* momentum_compressor_decompress_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
);

// ===========================================================================
// P1批处理API - 高性能批量压缩/解压 
// ===========================================================================

/**
 * 批处理压缩项 - 用于批量操作
 */
typedef struct {
    const char *layer_name;              // 层名称
    const NDArray *gradient;             // 输入梯度 (压缩时)
    CompressedLayerData *compressed;     // 压缩结果 (输出/输入)
    ErrorCode error;                     // 错误码
} BatchItem;

/**
 * 批量压缩多层梯度 - 对齐Python批量操作
 * @param compressor 压缩器指针
 * @param items 批处理项数组
 * @param num_items 项数量
 * @param client_id 客户端ID
 * @return 成功返回0, 失败返回错误码
 * 
 * 示例:
 *   BatchItem items[3];
 *   items[0].layer_name = "layer1";
 *   items[0].gradient = &grad1;
 *   items[1].layer_name = "layer2";
 *   items[1].gradient = &grad2;
 *   
 *   int ret = momentum_compressor_compress_batch(comp, items, 3, "Client1");
 *   if (ret == 0) {
 *       // items[0].compressed, items[1].compressed 已填充
 *   }
 */
int momentum_compressor_compress_batch(
    MomentumCompressor *compressor,
    BatchItem *items,
    size_t num_items,
    const char *client_id
);

/**
 * 批量解压多层梯度 - 对齐Python批量操作
 * @param compressor 压缩器指针
 * @param items 批处理项数组 (compressed字段已填充)
 * @param num_items 项数量
 * @param client_id 客户端ID
 * @param out_gradients 输出梯度数组 (调用者需提供数组空间)
 * @return 成功返回0, 失败返回错误码
 * 
 * 示例:
 *   BatchItem items[3];
 *   items[0].layer_name = "layer1";
 *   items[0].compressed = compressed1;
 *   items[1].layer_name = "layer2";
 *   items[1].compressed = compressed2;
 *   
 *   NDArray *gradients[3];
 *   int ret = momentum_compressor_decompress_batch(comp, items, 3, "Client1", gradients);
 *   if (ret == 0) {
 *       // gradients[0], gradients[1] 已填充
 *   }
 */
int momentum_compressor_decompress_batch(
    MomentumCompressor *compressor,
    const BatchItem *items,
    size_t num_items,
    const char *client_id,
    NDArray **out_gradients
);

/**
 * 重置客户端状态
 * @param compressor 压缩器指针
 * @param client_id 客户端ID
 */
void momentum_compressor_reset_client(MomentumCompressor *compressor, const char *client_id);

/**
 * 重置所有状态
 * @param compressor 压缩器指针
 */
void momentum_compressor_reset_all(MomentumCompressor *compressor);

/**
 * 获取压缩统计信息 (打印版)
 * @param compressor 压缩器指针
 */
void momentum_compressor_print_stats(const MomentumCompressor *compressor);

// ===========================================================================
// P2辅助API - 对齐Python辅助函数
// ===========================================================================

/**
 * 压缩统计结构 - 对齐Python get_compression_stats()
 */
typedef struct {
    size_t total_compressions;
    float avg_prediction_ratio;
    float std_prediction_ratio;
    float avg_sign_mismatch_ratio;
    size_t num_predictions;        // 有多少次使用了动量预测
} CompressionStats;

/**
 * 客户端详细统计 - 对齐Python get_detailed_stats()
 */
typedef struct {
    char client_id[256];
    size_t history_length;         // 历史记录数量
    int step_count;                // 步数计数
    size_t num_memory_layers;      // 预测记忆层数
} ClientStats;

/**
 * 详细统计结构
 */
typedef struct {
    CompressionStats compression;
    ClientStats *clients;
    size_t num_clients;
} DetailedStats;

/**
 * 获取压缩统计信息 (结构化) - 对齐Python get_compression_stats()
 * @param compressor 压缩器指针
 * @return 统计结构指针，需要调用者释放
 */
CompressionStats* momentum_compressor_get_stats(const MomentumCompressor *compressor);

/**
 * 获取详细统计信息 - 对齐Python get_detailed_stats()
 * @param compressor 压缩器指针
 * @return 详细统计结构指针，需要调用者释放
 */
DetailedStats* momentum_compressor_get_detailed_stats(const MomentumCompressor *compressor);

/**
 * 释放统计结构
 */
void compression_stats_free(CompressionStats *stats);
void detailed_stats_free(DetailedStats *stats);

/**
 * 设置日志级别 - 对齐Python set_log_level()
 * @param compressor 压缩器指针
 * @param level "DEBUG", "INFO", "WARNING", "ERROR"
 */
void momentum_compressor_set_log_level(MomentumCompressor *compressor, const char *level);

/**
 * 设置客户端上下文 - 对齐Python set_client_context()
 * @param compressor 压缩器指针
 * @param client_id 客户端ID (会自动规范化为"ClientN"格式)
 */
void momentum_compressor_set_client_context(MomentumCompressor *compressor, const char *client_id);

/**
 * 测试辅助函数 - 获取log_level (仅用于测试)
 */
int momentum_compressor_get_log_level(const MomentumCompressor *compressor);

/**
 * 测试辅助函数 - 获取current_client_id (仅用于测试)
 */
const char* momentum_compressor_get_current_client_id(const MomentumCompressor *compressor);

/**
 * 压缩整个模型（多层）- 对齐Python compress_model()
 * @param compressor 压缩器指针
 * @param gradients 梯度数组
 * @param layer_names 层名称数组
 * @param num_layers 层数量
 * @param client_id 客户端ID
 * @param out_size 输出数据大小
 * @return 压缩后的字节数据, 失败返回NULL
 */
uint8_t* momentum_compressor_compress_model(
    MomentumCompressor *compressor,
    const NDArray **gradients,
    const char **layer_names,
    size_t num_layers,
    const char *client_id,
    size_t *out_size
);

/**
 * 解压整个模型（多层）- 对齐Python decompress_model()
 * @param compressor 压缩器指针
 * @param compressed_data 压缩数据
 * @param compressed_size 压缩数据大小
 * @param layer_names 层名称数组（输出）
 * @param num_layers 层数量（输出）
 * @return 解压后的梯度数组, 失败返回NULL
 */
NDArray** momentum_compressor_decompress_model(
    MomentumCompressor *compressor,
    const uint8_t *compressed_data,
    size_t compressed_size,
    char ***layer_names,
    size_t *num_layers
);

// ===========================================================================
// 辅助函数
// ===========================================================================

/**
 * 创建NDArray
 */
NDArray* ndarray_create(const size_t *shape, size_t ndim, DataType dtype);

/**
 * 销毁NDArray
 */
void ndarray_destroy(NDArray *array);

/**
 * 拷贝NDArray
 */
NDArray* ndarray_copy(const NDArray *src);

/**
 * 释放CompressedLayerData
 */
void compressed_layer_data_free(CompressedLayerData *data);

/**
 * 创建默认配置
 */
CompressorConfig momentum_compressor_default_config(void);

// ===========================================================================
// Breakdown 性能分析函数
// ===========================================================================

/**
 * 启用/禁用 breakdown 性能分析
 * @param enable 1=启用, 0=禁用
 */
void momentum_compressor_enable_breakdown(int enable);

/**
 * 打印最后一次 batch 压缩的 breakdown 结果
 * 包含每层的详细时间分解和汇总统计
 */
void momentum_compressor_print_breakdown(void);

#ifdef __cplusplus
}
#endif

#endif // MOMENTUM_COMPRESSOR_H
