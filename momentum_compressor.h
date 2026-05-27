/**
 * ===========================================================================
 * MomentumPredictorCompressor
 *
 * notePythonnote: momentum_predictor_compressor.py
 *
 * note:
 * - note
 * - note
 * - note (SZ3note + Zstdnote)
 * - note
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
// note
// ===========================================================================

// note
#define MAX_KEY_LENGTH 512
#define MAX_CLIENT_ID_LENGTH 256
#define MAX_LAYER_NAME_LENGTH 256
#define MAX_COMPRESSOR_NAME_LENGTH 64
#define MAX_PATH_LENGTH 512
#define MAX_ERROR_MESSAGE_LENGTH 256

// datanote
#define MAX_NDIM 8
#define MIN_NDIM 1

// notedefaultnote
#define DEFAULT_MOMENTUM_LR 0.07f  // notePythondefaultnote
#define DEFAULT_CONSISTENCY_THRESHOLD 0.5f
#define DEFAULT_PARAM_COUNT_THRESHOLD 1024
#define DEFAULT_MAX_HISTORY_LENGTH 3
#define DEFAULT_ERROR_BOUND 1.0f

// note
#define MIN_COMPRESSION_SIZE 1024
#define MAX_BATCH_SIZE 1024
#define STATS_INITIAL_CAPACITY 100

// note
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR 3

// ===========================================================================
// dataclassnote
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

// note (note numpy.ndarray)
typedef struct {
    void *data;           // datanote(noteclassnote)
    size_t *shape;        // note [dim0, dim1,...]
    size_t ndim;          // note
    size_t total_size;    // note
    DataType dtype;       // dataclassnote
} NDArray;

// ===========================================================================
// configurationnote
// ===========================================================================

typedef struct {
    float momentum_lr;                    // note, default 0.07
    float consistency_threshold;          // note, default 0.5
    char lossless_compressor[32];         // "zstd" note "blosc"

    // SZ3 configuration
    char error_bounding_mode[16];         // "ABS", "REL" note
    float error_bound;                    // note, default 1.0
    char sz3_lib_path[512];               // SZ3notepath

    size_t param_count_threshold;         // note, default 1024
    int max_history_length;               // note, default 3
} CompressorConfig;

// ===========================================================================
// notedatanote
// ===========================================================================

typedef struct {
    char type[32];                        // "direct", "momentum_predicted", "direct_generic"
    char codec[16];                       // "sz3", "zstd", "pickle"

    uint8_t *data;                        // notedata
    size_t data_size;

    uint8_t *bitmap;                      // note (notemomentum_predicted)
    size_t bitmap_size;

    uint8_t *dominant_signs;              // note (notemomentum_predicted)
    size_t dominant_signs_size;

    size_t shape[MAX_NDIM];               // note
    size_t ndim;                          // note

    char original_dtype[16];              // notedataclassnote
    char stored_dtype[16];                // notedataclassnote

    int step;                             // note
    int num_predicted_kernels;            // note

    // note
    float prediction_ratio;
    float sign_mismatch_ratio;
    float current_mean, current_std;
    float prev_mean, prev_std;
    float global_min, global_max;

    // Breakdown note(note: note)
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
// firstnote (note)
// ===========================================================================

typedef struct MomentumCompressor MomentumCompressor;

// ===========================================================================
// noteAPIfunction
// ===========================================================================

/**
 * create compressornote
 * @param config configurationnote
 * @return note, failednoteNULL
 */
MomentumCompressor* momentum_compressor_create(const CompressorConfig *config);

/**
 * note
 * @param compressor note
 */
void momentum_compressor_destroy(MomentumCompressor *compressor);

/**
 * notecurrentnote
 * @param compressor note
 * @param client_id noteID
 */
void momentum_compressor_set_client(MomentumCompressor *compressor, const char *client_id);

/**
 * note
 * @param compressor note
 * @param layer_name note
 * @param gradient note
 * @return noteresult, failednoteNULL
 * @note client_id note momentum_compressor_set_client() note
 */
CompressedLayerData* momentum_compressor_compress_layer(
    MomentumCompressor *compressor,
    const char *layer_name,
    const NDArray *gradient
);

/**
 * note
 * @param compressor note
 * @param compressed notedata
 * @param client_id noteID
 * @param layer_name note
 * @return note, failednoteNULL
 */
NDArray* momentum_compressor_decompress_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
);

// ===========================================================================
// P1noteAPI - note/note
// ===========================================================================

/**
 * note - note
 */
typedef struct {
    const char *layer_name;              // note
    const NDArray *gradient;             // inputnote (note)
    CompressedLayerData *compressed;     // noteresult (output/input)
    ErrorCode error;                     // note
} BatchItem;

/**
 * note - notePythonnote
 * @param compressor note
 * @param items note
 * @param num_items note
 * @param client_id noteID
 * @return succeedednote0, failednote
 *
 * note:
 *   BatchItem items[3];
 *   items[0].layer_name = "layer1";
 *   items[0].gradient = &grad1;
 *   items[1].layer_name = "layer2";
 *   items[1].gradient = &grad2;
 *
 *   int ret = momentum_compressor_compress_batch(comp, items, 3, "Client1");
 *   if (ret == 0) {
 *       // items[0].compressed, items[1].compressed note
 *   }
 */
int momentum_compressor_compress_batch(
    MomentumCompressor *compressor,
    BatchItem *items,
    size_t num_items,
    const char *client_id
);

/**
 * note - notePythonnote
 * @param compressor note
 * @param items note (compressednote)
 * @param num_items note
 * @param client_id noteID
 * @param out_gradients outputnote (note)
 * @return succeedednote0, failednote
 *
 * note:
 *   BatchItem items[3];
 *   items[0].layer_name = "layer1";
 *   items[0].compressed = compressed1;
 *   items[1].layer_name = "layer2";
 *   items[1].compressed = compressed2;
 *
 *   NDArray *gradients[3];
 *   int ret = momentum_compressor_decompress_batch(comp, items, 3, "Client1", gradients);
 *   if (ret == 0) {
 *       // gradients[0], gradients[1] note
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
 * note
 * @param compressor note
 * @param client_id noteID
 */
void momentum_compressor_reset_client(MomentumCompressor *compressor, const char *client_id);

/**
 * note
 * @param compressor note
 */
void momentum_compressor_reset_all(MomentumCompressor *compressor);

/**
 * note (note)
 * @param compressor note
 */
void momentum_compressor_print_stats(const MomentumCompressor *compressor);

// ===========================================================================
// P2noteAPI - notePythonnotefunction
// ===========================================================================

/**
 * note - notePython get_compression_stats()
 */
typedef struct {
    size_t total_compressions;
    float avg_prediction_ratio;
    float std_prediction_ratio;
    float avg_sign_mismatch_ratio;
    size_t num_predictions;        // note
} CompressionStats;

/**
 * note - notePython get_detailed_stats()
 */
typedef struct {
    char client_id[256];
    size_t history_length;         // note
    int step_count;                // note
    size_t num_memory_layers;      // note
} ClientStats;

/**
 * note
 */
typedef struct {
    CompressionStats compression;
    ClientStats *clients;
    size_t num_clients;
} DetailedStats;

/**
 * note (note) - notePython get_compression_stats()
 * @param compressor note
 * @return note, note
 */
CompressionStats* momentum_compressor_get_stats(const MomentumCompressor *compressor);

/**
 * note - notePython get_detailed_stats()
 * @param compressor note
 * @return note, note
 */
DetailedStats* momentum_compressor_get_detailed_stats(const MomentumCompressor *compressor);

/**
 * note
 */
void compression_stats_free(CompressionStats *stats);
void detailed_stats_free(DetailedStats *stats);

/**
 * note - notePython set_log_level()
 * @param compressor note
 * @param level "DEBUG", "INFO", "WARNING", "ERROR"
 */
void momentum_compressor_set_log_level(MomentumCompressor *compressor, const char *level);

/**
 * note - notePython set_client_context()
 * @param compressor note
 * @param client_id noteID (note"ClientN"note)
 */
void momentum_compressor_set_client_context(MomentumCompressor *compressor, const char *client_id);

/**
 * notefunction - notelog_level (note)
 */
int momentum_compressor_get_log_level(const MomentumCompressor *compressor);

/**
 * notefunction - notecurrent_client_id (note)
 */
const char* momentum_compressor_get_current_client_id(const MomentumCompressor *compressor);

/**
 * notemodel(note)- notePython compress_model()
 * @param compressor note
 * @param gradients note
 * @param layer_names note
 * @param num_layers note
 * @param client_id noteID
 * @param out_size outputdatanote
 * @return notedata, failednoteNULL
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
 * notemodel(note)- notePython decompress_model()
 * @param compressor note
 * @param compressed_data notedata
 * @param compressed_size notedatanote
 * @param layer_names note(output)
 * @param num_layers note(output)
 * @return note, failednoteNULL
 */
NDArray** momentum_compressor_decompress_model(
    MomentumCompressor *compressor,
    const uint8_t *compressed_data,
    size_t compressed_size,
    char ***layer_names,
    size_t *num_layers
);

// ===========================================================================
// notefunction
// ===========================================================================

/**
 * noteNDArray
 */
NDArray* ndarray_create(const size_t *shape, size_t ndim, DataType dtype);

/**
 * noteNDArray
 */
void ndarray_destroy(NDArray *array);

/**
 * noteNDArray
 */
NDArray* ndarray_copy(const NDArray *src);

/**
 * noteCompressedLayerData
 */
void compressed_layer_data_free(CompressedLayerData *data);

/**
 * notedefaultconfiguration
 */
CompressorConfig momentum_compressor_default_config(void);

// ===========================================================================
// Breakdown notefunction
// ===========================================================================

/**
 * note/note breakdown note
 * @param enable 1=note, 0=note
 */
void momentum_compressor_enable_breakdown(int enable);

/**
 * note batch note breakdown result
 * note
 */
void momentum_compressor_print_breakdown(void);

#ifdef __cplusplus
}
#endif

#endif // MOMENTUM_COMPRESSOR_H
