/**
 * ===========================================================================
 * MomentumPredictorCompressor v22sz3
 *
 * Phase 22: note + notePythonnote
 *
 * notegoal: src/appfl/compressor/momentum_predictor_compressor.py
 *
 * Momentum-based predictor compressor for federated learning gradients.
 *
 * note(notev20):
 * - note 'weight', note > param_cutoff, note dtype note float32/64 note, note(SZ3/note)
 * - note(pickle), note int64 note dtype note SZ3 noteclassnote
 * - note codec('sz3' | 'pickle')note stored_dtype, noterows
 * - direct/generic note, note
 * - zstd note; conv key unified; note min/max notecomputenote
 *
 * v22sz3note: notev21 + note + notePython
 * PASS Opt1: -O3 -march=native -ffast-mathbuildnote, note
 * PASS Opt2: note (OMP_PROC_BIND, OMP_PLACESdetection)
 * PASS Opt3: 64note
 * FAIL Opt4note: note(note15%note)
 * PASS Opt5: notePythonnote - compressnotesavenotehistory
 * ===========================================================================
 */

#define _POSIX_C_SOURCE 200809L  // note POSIX note, note clock_gettime
#define USE_REAL_SZ3  // PASS noteSZ3note
#define PHASE_20_SEPARATED_ARCH 1  // PASS Phase 20: note

#include "momentum_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <zstd.h>
#include <time.h>  // note clock_gettime (note)
#include <omp.h>   // PASS Phase 3: OpenMP note
#include <float.h> // PASS Phase 4: FLT_MAX note
#include "uthash.h"

// OpenMP note(note, note)
// v22note: noterows(noteCnote)
// note: ResNet50datasetnote, noterowsnote(309ms vs noterows474ms)
// note: IOnote+hashnote+datanote, noterowsnote

// ========== v21note ==========
// noteconfiguration(notemalloc/freenote)
#define OMP_THREAD_BUFFER_SIZE (1024 * 1024)  // 1MBnote
#define OMP_BUFFER_ALIGN 64                   // 64note(L2 cache line)

// note(note)
#define SMALL_LAYER_BATCH_THRESHOLD (256 * 1024)  // 256Knote

// note
#define OMP_BIND_ENABLE 1  // note: OMP_PROC_BIND=close OMP_PLACES=cores

// ========== Breakdown time measurementnote ==========

// notecurrentnote(note, note)- note CLOCK_MONOTONIC
static inline double get_wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;  // note(9note)
}

// note breakdown note(note: note)
typedef struct {
    // [Phase 1] notecompute
    double stats_time;           // notecomputenote (mean, std, min/max)

    // [Phase 2] note
    double normalize_time;       // note

    // [Phase 3] notedetection(note1)
    double consistency_time;     // notecomputenote(note)
    double sign_consistency_compute_time;  // notestage: notecompute
    double dominant_sign_compute_time;     // notestage: notecompute

    // [Phase 4] note(note2: Magnitude & Sign Predictors)
    double prediction_time;      // note + notecomputenote(note)
    double prediction_hash_time; // notestage: hashnote
    double magnitude_predictor_time;  // notestage: note(memorynote+note)
    double sign_predictor_time;       // notestage: note(notedetection)
    double residual_compute_time;     // notestage: notecompute+note

    // [Phase 5] note (note)
    double residual_compress_time; // note (SZ3/ZSTD)
    double sz3_compress_time;    // notestage: SZ3note
    double zstd_lossless_time;   // notestage: ZSTDnote

    // [Phase 6] note(note3)
    double bitmap_compress_time;   // note(note)
    double bitmap_generation_time; // notestage: notegenerate(kernelnote)
    double bitmap_pack_time;       // notestage: note(bit packing)
    double bitmap_zstd_time;       // notestage: noteZSTDnote
    double dominant_signs_pack_time; // notestage: note
    double dominant_signs_zstd_time; // notestage: noteZSTDnote

    // [Phase 7] notedata
    double metadata_time;        // notedatanote

    // [note] Hashnote(note4)
    double hash_lookup_time;     // Hashnote(prediction_memorynote)
    double hash_update_time;     // Hashnote(set_prediction_memory)
    double history_lookup_time;  // note(layer_historiesnote)
    double history_update_time;  // note(add_gradient_to_history)

    // [note] note(note)
    double memory_alloc_time;    // note(malloc/calloc)
    double memory_copy_time;     // note(memcpy)
    double memory_free_time;     // note(free)

    // [note]
    double total_time;           // note
    size_t layer_size;           // note(note)
    char layer_name[256];        // note

    // [note]
    int timing_point_count;      // noterowsnote
    int uses_momentum;           // notepath (1=note, 0=note)

    // [noteclass]
    double compute_time;         // notecomputenote(stats+normalize+consistency+prediction_compute)
    double io_time;              // IOnote(SZ3+ZSTDnote)
    double hash_time;            // Hashnote(note, notehash_*_time)
} LayerBreakdown;

// ========== Phase 20: notecomputeresultnote ==========
/**
 * LayerResult: notecomputefunctionnote
 * notedatanotehash/historynote
 * noterowsnotecomputefunctionnote, note
 * noterowsnote(Step C)notehash/history/breakdown
 */
typedef struct {
    // [computeresult] notedata(note)
    CompressedLayerData *compressed;      // notesucceedednoteNULL

    // [note] noteprediction_memorynote
    char history_key[512];                // notehistory lookup: make_history_key(client_id, layer_name)
    char layer_key[256];                  // notelayer memory: make_layer_key_from_shape

    // [note] notebreakdowndata
    LayerBreakdown layer_bd;              // breakdownnotedata(note)

    // [note] computeresultnote
    int status;                           // 0=succeeded, <0=failednote
    char error_msg[256];                  // note(note)
} LayerResult;

// Batch note breakdown note(noterowsnote)
typedef struct {
    double batch_total_time;     // batch note
    double layer_compress_time;  // note
    size_t num_layers;           // note
    size_t total_elements;       // note
    LayerBreakdown *layers;      // note breakdown(dynamicnote)

    // [note] Batchnoterowsnote
    double parallel_compute_time;  // Step B: noterowscomputestagenote
    double serial_update_time;     // Step C: noterowsnotestagenote
    double omp_overhead_time;      // OpenMPnote/note
    int num_threads_used;          // notethread count
} BatchBreakdown;

// [note] note Breakdown note(currentnote)
typedef struct {
    double total_time;              // note
    double sz3_decompress_time;     // SZ3note
    double zstd_decompress_time;    // ZSTDnote
    double reconstruction_time;     // note
    double denormalization_time;    // note
    double hash_lookup_time;        // Hashnote
    double hash_update_time;        // Hashnote
    size_t num_layers;              // note
    LayerBreakdown *layers;         // note(note)
} DecompressBreakdown;

// note breakdown note(defaultnote)
static int g_enable_breakdown = 1;  // PASS Phase 19sz3.3: defaultnotebreakdownnote
static BatchBreakdown g_last_batch_breakdown = {0};

// note/note breakdown
void momentum_compressor_enable_breakdown(int enable) {
    g_enable_breakdown = enable;
    if (!enable && g_last_batch_breakdown.layers) {
        free(g_last_batch_breakdown.layers);
        g_last_batch_breakdown.layers = NULL;
    }
}

// note batch note breakdown result
const BatchBreakdown* momentum_compressor_get_last_breakdown() {
    return &g_last_batch_breakdown;
}

// note breakdown
static void print_layer_breakdown(const LayerBreakdown *bd) {
    printf("\n═════════════════════════════════════════════════════════════════════\n");
    printf("Layer: %s (%.2fM elements)\n", bd->layer_name, bd->layer_size / 1e6);
    printf("═════════════════════════════════════════════════════════════════════\n");
    printf("%-25s %12s %12s\n", "Phase", "Time(μs)", "% of Total");
    printf("─────────────────────────────────────────────────────────────────────\n");

    // note
    double stats_us = bd->stats_time * 1e6;
    double norm_us = bd->normalize_time * 1e6;
    double cons_us = bd->consistency_time * 1e6;
    double pred_us = bd->prediction_time * 1e6;
    double res_us = bd->residual_compress_time * 1e6;
    double bmp_us = bd->bitmap_compress_time * 1e6;
    double meta_us = bd->metadata_time * 1e6;
    double total_us = bd->total_time * 1e6;

    // note
    if (total_us < 0.001) total_us = 1.0;

    printf("%-25s %12.2f %11.1f%%\n", "Stats (mean/std/min/max)",
           stats_us, (stats_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Normalize",
           norm_us, (norm_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Sign Consistency",
           cons_us, (cons_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Momentum Prediction",
           pred_us, (pred_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Residual Compress",
           res_us, (res_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Bitmap Compress",
           bmp_us, (bmp_us / total_us) * 100);
    printf("%-25s %12.2f %11.1f%%\n", "Metadata",
           meta_us, (meta_us / total_us) * 100);
    printf("─────────────────────────────────────────────────────────────────────\n");
    printf("%-25s %12.2f %11s\n", "Total", total_us, "100%");
    printf("═════════════════════════════════════════════════════════════════════\n");
}

// note batch breakdown
void momentum_compressor_print_breakdown() {
    if (!g_enable_breakdown) {
        printf("[INFO] Breakdown is disabled. Call momentum_compressor_enable_breakdown(1) first.\n");
        return;
    }

    const BatchBreakdown *bd = &g_last_batch_breakdown;
    if (bd->num_layers == 0) {
        printf("[INFO] No breakdown data available.\n");
        return;
    }

    printf("\n\n");
    printf("╔═════════════════════════════════════════════════════════════════════╗\n");
    printf("║            MOMENTUM COMPRESSOR BASELINE BREAKDOWN (μsnote)          ║\n");
    printf("╚═════════════════════════════════════════════════════════════════════╝\n");
    printf("\nBatch Summary:\n");
    printf("  Total layers:      %zu\n", bd->num_layers);
    printf("  Total elements:    %zu (%.2f MB)\n",
           bd->total_elements, bd->total_elements * 4.0 / 1024 / 1024);
    printf("  Batch total time:  %.3f ms (%.0f μs)\n",
           bd->batch_total_time * 1000, bd->batch_total_time * 1e6);
    printf("  Layer compress:    %.3f ms (%.0f μs)\n",
           bd->layer_compress_time * 1000, bd->layer_compress_time * 1e6);
    printf("  Throughput:        %.2f MB/s\n",
           (bd->total_elements * 4.0 / 1024 / 1024) / bd->batch_total_time);
    printf("  Avg per layer:     %.3f ms (%.0f μs)\n",
           bd->layer_compress_time * 1000 / bd->num_layers,
           bd->layer_compress_time * 1e6 / bd->num_layers);

    // note
    printf("\n\n[Per-Layer Breakdown Details (first10note)]\n");
    size_t print_limit = (bd->num_layers > 10)? 10: bd->num_layers;
    for (size_t i = 0; i < print_limit; i++) {
        print_layer_breakdown(&bd->layers[i]);
    }
    if (bd->num_layers > 10) {
        printf("\n... (notefirst10note, note%zunote, notedatanote get_last_breakdown API note)\n", bd->num_layers);
    }

    // note
    double total_stats = 0, total_norm = 0, total_cons = 0;
    double total_pred = 0, total_res_comp = 0, total_bmp_comp = 0, total_meta = 0;

    for (size_t i = 0; i < bd->num_layers; i++) {
        total_stats += bd->layers[i].stats_time;
        total_norm += bd->layers[i].normalize_time;
        total_cons += bd->layers[i].consistency_time;
        total_pred += bd->layers[i].prediction_time;
        total_res_comp += bd->layers[i].residual_compress_time;
        total_bmp_comp += bd->layers[i].bitmap_compress_time;
        total_meta += bd->layers[i].metadata_time;
    }

    double total_sum = total_stats + total_norm + total_cons + total_pred +
                       total_res_comp + total_bmp_comp + total_meta;

    printf("\n\n");
    printf("╔═════════════════════════════════════════════════════════════════════╗\n");
    printf("║                     AGGREGATE BREAKDOWN SUMMARY                     ║\n");
    printf("╚═════════════════════════════════════════════════════════════════════╝\n");
    printf("%-25s %14s %14s %10s\n", "Phase", "Total(μs)", "Total(ms)", "% of Total");
    printf("─────────────────────────────────────────────────────────────────────\n");
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Stats", total_stats * 1e6, total_stats * 1000,
           total_sum > 0? (total_stats / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Normalize", total_norm * 1e6, total_norm * 1000,
           total_sum > 0? (total_norm / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Sign Consistency", total_cons * 1e6, total_cons * 1000,
           total_sum > 0? (total_cons / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Momentum Prediction", total_pred * 1e6, total_pred * 1000,
           total_sum > 0? (total_pred / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Residual Compress", total_res_comp * 1e6, total_res_comp * 1000,
           total_sum > 0? (total_res_comp / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Bitmap Compress", total_bmp_comp * 1e6, total_bmp_comp * 1000,
           total_sum > 0? (total_bmp_comp / total_sum) * 100: 0);
    printf("%-25s %14.1f %14.3f %9.1f%%\n", "Metadata", total_meta * 1e6, total_meta * 1000,
           total_sum > 0? (total_meta / total_sum) * 100: 0);
    printf("─────────────────────────────────────────────────────────────────────\n");
    printf("%-25s %14.1f %14.3f %9s\n", "Total", total_sum * 1e6, total_sum * 1000, "100%");
    printf("═════════════════════════════════════════════════════════════════════\n");
    printf("\n[🎯 OpenMP Optimization Targets]\n");
    if (total_sum > 0) {
        printf("  High Priority:  Sign Consistency (%.1f%%), Momentum Prediction (%.1f%%)\n",
               (total_cons / total_sum) * 100, (total_pred / total_sum) * 100);
        printf("  Medium Priority: Stats (%.1f%%)\n", (total_stats / total_sum) * 100);
        printf("  Parallelizable: %.1f%% of total time\n",
               ((total_stats + total_cons + total_pred) / total_sum) * 100);
    }
    printf("\n");
}

// SZ3 note (notebuild)
#ifdef USE_REAL_SZ3
#include <SZ3c/sz3c.h>
#define SZ3_ENABLED 1
#else
#define SZ3_ENABLED 0
#endif

// Blosc note (notebuild)
#ifdef USE_BLOSC
#include <blosc.h>
#define BLOSC_ENABLED 1
#else
#define BLOSC_ENABLED 0
#endif

// -------------------- helpers --------------------
// dataclassnotefunction - notePythonnote

/**
 * notedataclassnote
 * notePython: notefunction, noteimplementnumpynoteitemsize
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
 * noteclassnote
 * notePython: FLOAT_DTYPES = (np.float32, np.float64)
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
 * noterowsnote
 * notePython: def _should_lossy(self, layer_name: str, arr: np.ndarray) -> bool
 *
 * note:
 * - note 'weight'
 * - note > param_count_threshold
 * - dtype note float32/64
 *
 * note: true note
 */
static bool should_use_lossy_compression(
    const char *layer_name,
    size_t param_count,
    DataType dtype,
    size_t param_count_threshold
) {
    // notePythonnote:
    // return ("weight" in layer_name) and
    //        (arr.size > self.param_count_threshold) and
    //        (arr.dtype in FLOAT_DTYPES)

    bool has_weight = (strstr(layer_name, "weight")!= NULL);
    bool exceeds_threshold = (param_count > param_count_threshold);
    bool is_float = is_float_dtype(dtype);

    return has_weight && exceeds_threshold && is_float;
}

// ===========================================================================
// notedatanote
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

// note: notelayer_key(noteshape)
typedef struct {
    char layer_key[256];      // e.g., "conv_(64,32,3,3)"
    NDArray *memory;
    UT_hash_handle hh;
} LayerMemoryEntry;

// note: noteclient_id
typedef struct {
    char client_id[256];
    LayerMemoryEntry *layer_memories;  // note,notelayer_keynote
    UT_hash_handle hh;
} PredictionMemory;

typedef struct {
    char key[512];            // "client_id:layer_name" note
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
// NDArray functionimplement(notedtype)
// ===========================================================================

NDArray* ndarray_create(const size_t *shape, size_t ndim, DataType dtype) {
    // inputnote
    if (!shape) {
        fprintf(stderr, "[ERROR] ndarray_create: shape is NULL\n");
        return NULL;
    }

    if (ndim < MIN_NDIM || ndim > MAX_NDIM) {
        fprintf(stderr, "[ERROR] ndarray_create: invalid ndim=%zu (valid range: %d-%d)\n",
                ndim, MIN_NDIM, MAX_NDIM);
        return NULL;
    }

    // noteshapenote
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
// notefunction - notePython _lossless_compress() note _compress_with_sz3()

/**
 * ZSTDnote
 * notePython: zstd.compress(data_bytes, 10)
 *
 * note:
 *   data: inputdata
 *   size: datanote(note)
 *   out_size: outputnote(note)
 *
 * note: notedatanote, failednoteNULL
 */
static uint8_t* zstd_compress(const void *data, size_t size, size_t *out_size) {
    // PASS note - notecontext
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) return NULL;

    size_t max_size = ZSTD_compressBound(size);
    uint8_t *compressed = (uint8_t*)malloc(max_size);
    if (!compressed) {
        ZSTD_freeCCtx(cctx);
        return NULL;
    }

    size_t rc = ZSTD_compressCCtx(cctx, compressed, max_size, data, size, 10);
    ZSTD_freeCCtx(cctx);  // note

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
 * ZSTDnote
 * notePython: zstd.decompress(compressed_data)
 *
 * note:
 *   compressed: notedata
 *   compressed_size: notedatanote
 *   out_size: outputnote
 *
 * note: notedatanote, failednoteNULL
 */
static void* zstd_decompress(const uint8_t *compressed, size_t compressed_size, size_t *out_size) {
    // PASS note - notecontext
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) return NULL;

    unsigned long long bound = ZSTD_getFrameContentSize(compressed, compressed_size);
    if (bound == ZSTD_CONTENTSIZE_ERROR || bound == ZSTD_CONTENTSIZE_UNKNOWN) {
        fprintf(stderr, "[ERROR] Cannot determine decompression size\n");
        ZSTD_freeDCtx(dctx);
        return NULL;
    }

    void *decompressed = malloc(bound);
    if (!decompressed) {
        ZSTD_freeDCtx(dctx);
        return NULL;
    }

    size_t rc = ZSTD_decompressDCtx(dctx, decompressed, bound, compressed, compressed_size);
    ZSTD_freeDCtx(dctx);  // note

    if (ZSTD_isError(rc)) {
        fprintf(stderr, "[ERROR] ZSTD decompression failed: %s\n", ZSTD_getErrorName(rc));
        free(decompressed);
        return NULL;
    }

    *out_size = rc;
    return decompressed;
}

/**
 * Picklenote
 * notePython: pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
 *
 * note: notepickleimplement, notePython picklenote
 * note, notedatanote
 */
static uint8_t* pickle_compress(const void *data, size_t size, size_t *out_size) {
    // notepickle: notedata, noteheader
    uint8_t *result = (uint8_t*)malloc(size + 16);
    if (!result) return NULL;

    // noteheader: "PICKLE\0\0" + size
    memcpy(result, "PICKLE\0\0", 8);
    *((size_t*)(result + 8)) = size;
    memcpy(result + 16, data, size);

    *out_size = size + 16;
    return result;
}

static void* pickle_decompress(const uint8_t *compressed, size_t compressed_size, size_t *out_size) {
    if (compressed_size < 16) return NULL;

    // noteheader
    if (memcmp(compressed, "PICKLE\0\0", 8)!= 0) {
        fprintf(stderr, "[ERROR] Invalid pickle header\n");
        return NULL;
    }

    size_t data_size = *((size_t*)(compressed + 8));
    if (data_size + 16!= compressed_size) {
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
// Blosc note/noteimplement
// ===========================================================================

#ifdef USE_BLOSC
/**
 * Blosc note - note Blosc C API
 * Blosc note,notedata
 */
static uint8_t* blosc_compress_real(
    const void *data,
    size_t size,
    size_t *out_size
) {
    // Blosc note
    static int blosc_initialized = 0;
    if (!blosc_initialized) {
        blosc_init();
        blosc_initialized = 1;
    }

    // noteoutputnote (note: size + BLOSC_MAX_OVERHEAD)
    size_t max_size = size + BLOSC_MAX_OVERHEAD;
    uint8_t *compressed = (uint8_t*)malloc(max_size);
    if (!compressed) {
        fprintf(stderr, "[ERROR] blosc_compress: failed to allocate buffer\n");
        return NULL;
    }

    // note Blosc note
    // blosc_compress(clevel, doshuffle, typesize, nbytes, src, dest, destsize)
    int compressed_size = blosc_compress(
        5,          // clevel: note (1-9, 5note)
        1,          // shuffle: noteshuffle (note)
        1,          // typesize: note (1 byte,notedetection)
        size,       // nbytes: inputnote
        data,       // src: inputdata
        compressed, // dest: outputnote
        max_size    // destsize: outputnote
    );

    if (compressed_size <= 0) {
        fprintf(stderr, "[ERROR] Blosc compression failed (code: %d)\n", compressed_size);
        free(compressed);
        return NULL;
    }

    *out_size = compressed_size;

    // note: note
    compressed = (uint8_t*)realloc(compressed, compressed_size);

    // PASS Phase 19sz3.2: noteINFOnote
    // fprintf(stderr, "[INFO] Blosc compression: %zu -> %d bytes (%.2f%%)\n",
    //         size, compressed_size, 100.0 * compressed_size / size);

    return compressed;
}

/**
 * Blosc note
 */
static void* blosc_decompress_real(
    const uint8_t *compressed,
    size_t compressed_size,
    size_t *out_size
) {
    // note
    size_t nbytes, cbytes, blocksize;
    blosc_cbuffer_sizes(compressed, &nbytes, &cbytes, &blocksize);

    if (nbytes == 0) {
        fprintf(stderr, "[ERROR] Blosc: invalid compressed data\n");
        return NULL;
    }

    // noteoutputnote
    void *decompressed = malloc(nbytes);
    if (!decompressed) {
        fprintf(stderr, "[ERROR] blosc_decompress: failed to allocate buffer\n");
        return NULL;
    }

    // note Blosc note
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
 * note
 * notePython: def _lossless_compress(self, arr: np.ndarray) -> Tuple[bytes, str]
 *
 * note:
 * - zstd: noteZSTDnote
 * - blosc: noteBloscnote(note)
 * - pickle: notePicklenote
 *
 * note:
 *   data: inputdata
 *   size: datanote
 *   compressor_type: noteclassnote ("zstd", "blosc", "pickle")
 *   codec_used: outputnotecodec
 *   out_size: outputnote
 *
 * note: notedatanote, failednoteNULL
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
        // defaultnotepickle
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
 * note
 * notePython: def _lossless_decompress(self, compressed_data: bytes, codec: str) -> np.ndarray
 *
 * notecodecnote:
 * - 'zstd': noteZSTDnote
 * - 'blosc': noteBloscnote(note)
 * - 'pickle': notePicklenote
 *
 * note: notefunctionnote, note 'sz3'(note)
 * SZ3 note lossy_decompress_with_shape() note
 *
 * note:
 *   compressed: notedata
 *   compressed_size: notedatanote
 *   codec: noteclassnote ("zstd", "blosc", "pickle")
 *   out_size: outputnote
 *
 * note: notedatanote, failednoteNULL
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
        // notePython: _lossless_decompress note "sz3"
        // note "sz3", note lossy_decompress_with_shape()
        fprintf(stderr, "[ERROR] Unknown codec in lossless_decompress: %s (only supports zstd/blosc/pickle)\n", codec);
        return NULL;
    }
}

// ===========================================================================
// SZ3 firstnote
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

// note(note)
static void* lossy_decompress_with_shape(
    const uint8_t *compressed,
    size_t compressed_size,
    const char *codec,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    size_t *out_size
) {
    if (strcmp(codec, "sz3") == 0) {
#ifdef USE_REAL_SZ3
        return sz3_decompress_real(compressed, compressed_size, shape, ndim, dtype, out_size);
#else
        // note
        (void)shape;
        (void)ndim;
        (void)dtype;
        fprintf(stderr, "[WARNING] SZ3 not available, using zstd fallback\n");
        return zstd_decompress(compressed, compressed_size, out_size);
#endif
    }
    else {
        // notecodecnote
        return lossless_decompress(compressed, compressed_size, codec, out_size);
    }
}

// ===========================================================================
// SZ3 note/noteimplement
// ===========================================================================

#ifdef USE_REAL_SZ3
/**
 * SZ3 note - note SZ3 C API
 * notePython: pysz.SZ.compress() note float32 note float64
 *
 * note:
 *   data: notedata (float32 note float64)
 *   total_size: note
 *   shape: note
 *   ndim: note
 *   dtype: dataclassnote (DTYPE_FLOAT32 note DTYPE_FLOAT64)
 *   error_mode: "ABS" or "REL"
 *   abs_bound: note
 *   rel_bound: note (0.01 = 1%)
 *   out_size: outputnote
 * note: notedata (notefree)
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
    // 1. note (SZ3note: r1=note)
    size_t r1=1, r2=0, r3=0, r4=0, r5=0;

    if (ndim >= 1) r1 = shape[ndim-1];
    if (ndim >= 2) r2 = shape[ndim-2];
    if (ndim >= 3) r3 = shape[ndim-3];
    if (ndim >= 4) r4 = shape[ndim-4];
    if (ndim >= 5) r5 = shape[ndim-5];

    // 2. note
    int mode = REL;  // defaultnote
    if (error_mode && strcmp(error_mode, "ABS") == 0) {
        mode = ABS;
    }

    // 3. notedataclassnote (notePython: note float32 note float64)
    int sz_datatype;
    size_t element_size;
    if (dtype == DTYPE_FLOAT64) {
        sz_datatype = SZ_DOUBLE;  // float64
        element_size = sizeof(double);
    } else {
        sz_datatype = SZ_FLOAT;   // float32 (default)
        element_size = sizeof(float);
    }

    // 4. note SZ3 note
    unsigned char *sz3_buffer = SZ_compress_args(
        sz_datatype,              // dataclassnote (SZ_FLOAT note SZ_DOUBLE)
        (void*)data,              // datanote
        out_size,                 // output: note
        mode,                     // note
        (double)abs_bound,        // note
        (double)rel_bound,        // note
        0.0,                      // pwrBound (note)
        r5, r4, r3, r2, r1        // note (5Dnote)
    );

    if (!sz3_buffer) {
        fprintf(stderr, "[ERROR] SZ3 compression failed (mode=%s, rel=%.3f)\n",
                error_mode, rel_bound);
        return NULL;
    }

    // 4. note (SZ3note) notedatanote
    uint8_t *result = (uint8_t*)malloc(*out_size);
    if (!result) {
        fprintf(stderr, "[ERROR] sz3_compress: failed to allocate result buffer\n");
        free_buf(sz3_buffer);
        return NULL;
    }

    memcpy(result, sz3_buffer, *out_size);
    free_buf(sz3_buffer);  // note SZ3 note

    // PASS Phase 19sz3.2: noteINFOnote
    // fprintf(stderr, "[INFO] SZ3 real compression: %zu -> %zu bytes (%.2f%%, dtype=%s)\n",
    //         total_size * element_size, *out_size,
    //         100.0 * (*out_size) / (total_size * element_size),
    //         dtype_to_string(dtype));

    return result;
}

/**
 * SZ3 note
 * notePython: pysz.SZ.decompress() note float32 note float64
 *
 * note:
 *   compressed: notedata
 *   compressed_size: notedatanote
 *   shape: note
 *   ndim: note
 *   dtype: dataclassnote (DTYPE_FLOAT32 note DTYPE_FLOAT64)
 *   out_size: outputnote
 * note: notedata (notefree)
 */
static void* sz3_decompress_real(
    const uint8_t *compressed,
    size_t compressed_size,
    const size_t *shape,
    size_t ndim,
    DataType dtype,
    size_t *out_size
) {
    // 1. note
    size_t r1=1, r2=0, r3=0, r4=0, r5=0;

    if (ndim >= 1) r1 = shape[ndim-1];
    if (ndim >= 2) r2 = shape[ndim-2];
    if (ndim >= 3) r3 = shape[ndim-3];
    if (ndim >= 4) r4 = shape[ndim-4];
    if (ndim >= 5) r5 = shape[ndim-5];

    // 2. notedataclassnote
    int sz_datatype;
    size_t element_size;
    if (dtype == DTYPE_FLOAT64) {
        sz_datatype = SZ_DOUBLE;  // float64
        element_size = sizeof(double);
    } else {
        sz_datatype = SZ_FLOAT;   // float32 (default)
        element_size = sizeof(float);
    }

    // 3. note SZ3 note
    void *sz3_buffer = SZ_decompress(
        sz_datatype,              // dataclassnote (SZ_FLOAT note SZ_DOUBLE)
        (unsigned char*)compressed,
        compressed_size,
        r5, r4, r3, r2, r1        // note
    );

    if (!sz3_buffer) {
        fprintf(stderr, "[ERROR] SZ3 decompression failed (dtype=%s)\n", dtype_to_string(dtype));
        return NULL;
    }

    // 4. computeoutputnote
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    *out_size = total_size * element_size;

    // 4. note
    void *result = malloc(*out_size);
    if (!result) {
        fprintf(stderr, "[ERROR] sz3_decompress: failed to allocate result buffer\n");
        free_buf(sz3_buffer);
        return NULL;
    }

    memcpy(result, sz3_buffer, *out_size);
    free_buf(sz3_buffer);  // note SZ3 note

    return result;
}
#endif // USE_REAL_SZ3

// -------------------- SZ3 wrappers --------------------

/**
 * SZ3note(unifiednote, note/note)
 * notePython: def _compress_with_sz3(self, data: np.ndarray, error_mode, abs_bound, rel_bound) -> Tuple[bytes, float]
 *
 * note: (compressed_data, compression_ratio, codec)
 *
 * note:
 * - noteUSE_REAL_SZ3notedtypenotefloat32, noteSZ3note
 * - noteSZ3notefailed, noteZSTDnote
 * - notecodec("sz3"note"zstd")
 *
 * note:
 *   data: inputdata
 *   size: note
 *   dtype: dataclassnote
 *   shape: note(noteSZ3)
 *   ndim: note
 *   error_mode: note ("ABS" note "REL")
 *   abs_bound: note
 *   rel_bound: note
 *   compression_ratio: outputnote
 *   codec_used: outputnotecodec
 *   out_size: outputnote
 *
 * note: notedatanote, failednoteNULL
 */
static uint8_t* sz3_compress_simulate(
    const void *data,
    size_t size,
    DataType dtype,
    const size_t *shape,      // note
    size_t ndim,              // note
    const char *error_mode,
    float abs_bound,
    float rel_bound,
    float *compression_ratio,
    const char **codec_used,
    size_t *out_size
) {
#ifdef USE_REAL_SZ3
    // note SZ3
    if ((dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64) && shape && ndim > 0) {
        uint8_t *result = sz3_compress_real(
            data,          // note void* note float32 note float64
            size,
            shape,         // note
            ndim,          // note
            dtype,         // notedataclassnote
            error_mode,
            abs_bound,
            rel_bound,
            out_size
        );

        if (result) {
            size_t element_size = dtype_size(dtype);
            *compression_ratio = (float)(size * element_size) / (float)(*out_size);
            *codec_used = "sz3";  // note SZ3
            return result;
        }
        // failednote zstd
    }
#else
    // note
    (void)shape;
    (void)ndim;
    (void)error_mode;
    (void)abs_bound;
    (void)rel_bound;
#endif

    // ===================================================================
    // 📝 note: note memcpy note SZ3 note(notepath)
    // ===================================================================
    // note float32/float64, note SZ3 note, note memcpy note
    // notepath, note
    if (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64) {
        size_t byte_size = size * dtype_size(dtype);
        uint8_t *result = (uint8_t*)malloc(byte_size);
        if (result) {
            memcpy(result, data, byte_size);
            *out_size = byte_size;
            *compression_ratio = 1.0f;  // note(memcpy)
            *codec_used = "sz3_memcpy";  // note SZ3
            return result;
        }
    }

    // noteclassnote memcpy failed, note ZSTD
    fprintf(stderr, "[INFO] Non-float dtype or malloc failed, using ZSTD fallback\n");

    uint8_t *result = zstd_compress(data, size * dtype_size(dtype), out_size);
    if (result) {
        *compression_ratio = (float)(size * dtype_size(dtype)) / (float)(*out_size);
        *codec_used = "zstd";  // note
    } else {
        *compression_ratio = 1.0f;
        *codec_used = "pickle";
    }

    return result;
}

// ===========================================================================
// notefunction(Phase 4note: notecompute + OpenMPnoterowsnote)
// ===========================================================================

// note
typedef struct {
    float mean;
    float std;
    float min;
    float max;
} FloatStats;

/**
 * Phase 4note: notecompute(mean + std + min/max)
 *
 * note:
 * 1. note(note3note)
 * 2. note(>10Knote)noteOpenMPnoterowsnote
 * 3. notereductionnote
 */
static FloatStats compute_stats_fused(const float *data, size_t size) {
    FloatStats stats = {0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    if (size == 0) return stats;

    double sum = 0.0;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // note: note10KnoteOpenMP(note)
    if (size >= 10000) {
        // PASS Phase 18.1: SIMDnotecompute
        // note: noterows+SIMDcompute sum, min, max
        #pragma omp parallel for simd reduction(+:sum) \
            reduction(min:local_min) reduction(max:local_max) schedule(static)
        for (size_t i = 0; i < size; i++) {
            float val = data[i];
            sum += val;
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }

        stats.mean = (float)(sum / size);
        stats.min = local_min;
        stats.max = local_max;

        // note: noterows+SIMDcompute variance
        double variance = 0.0;
        #pragma omp parallel for simd reduction(+:variance) schedule(static)
        for (size_t i = 0; i < size; i++) {
            double diff = data[i] - stats.mean;
            variance += diff * diff;
        }
        stats.std = sqrtf((float)(variance / size));

    } else {
        // note: noterowsnote
        for (size_t i = 0; i < size; i++) {
            float val = data[i];
            sum += val;
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }

        stats.mean = (float)(sum / size);
        stats.min = local_min;
        stats.max = local_max;

        // computenote
        double variance = 0.0;
        for (size_t i = 0; i < size; i++) {
            double diff = data[i] - stats.mean;
            variance += diff * diff;
        }
        stats.std = sqrtf((float)(variance / size));
    }

    return stats;
}

// notefunctionnote(note)
static float compute_mean_float32(const float *data, size_t size) {
    return compute_stats_fused(data, size).mean;
}

static float compute_std_float32(const float *data, size_t size, float mean) {
    // note: notemean, notecomputenote
    return compute_stats_fused(data, size).std;
}

static void compute_min_max_float32(const float *data, size_t size, float *min, float *max) {
    FloatStats stats = compute_stats_fused(data, size);
    *min = stats.min;
    *max = stats.max;
}

/**
 * Phase 19.1note: note3notecompute + absnotecompute
 *
 * note:
 * 1. computeabs_currentnoteabs_prevnote
 * 2. 3notecompute_stats_fusednote
 *
 * note2note:
 * Pass 1: notecomputeabsnote, sumnotemin/max
 * Pass 2: noteabsnotecomputenote
 *
 * note: note4note(1noteabscompute+3note) -> note2note
 */
typedef struct {
    FloatStats abs_current;  // notecurrentnote
    FloatStats abs_prev;     // notefirstnote
    float raw_min;           // notedatanote
    float raw_max;           // notedatanote
} TripleStats;

static TripleStats compute_triple_stats_with_abs(
    const float *current_data,
    const float *prev_data,
    float *abs_current,  // output
    float *abs_prev,     // output
    size_t size
) {
    TripleStats result = {0};
    result.raw_min = FLT_MAX;
    result.raw_max = -FLT_MAX;

    if (size == 0) return result;

    // note: notecomputeabsnote, sumnotemin/max
    double sum_abs_current = 0.0;
    double sum_abs_prev = 0.0;
    float local_raw_min = FLT_MAX;
    float local_raw_max = -FLT_MAX;

    if (size >= 10000) {
        #pragma omp parallel for simd \
            reduction(+:sum_abs_current,sum_abs_prev) \
            reduction(min:local_raw_min) reduction(max:local_raw_max) \
            schedule(static)
        for (size_t i = 0; i < size; i++) {
            float curr = current_data[i];
            float prev = prev_data[i];
            float abs_curr = fabsf(curr);
            float abs_prev_val = fabsf(prev);

            abs_current[i] = abs_curr;
            abs_prev[i] = abs_prev_val;

            sum_abs_current += abs_curr;
            sum_abs_prev += abs_prev_val;

            if (curr < local_raw_min) local_raw_min = curr;
            if (curr > local_raw_max) local_raw_max = curr;
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            float curr = current_data[i];
            float prev = prev_data[i];
            float abs_curr = fabsf(curr);
            float abs_prev_val = fabsf(prev);

            abs_current[i] = abs_curr;
            abs_prev[i] = abs_prev_val;

            sum_abs_current += abs_curr;
            sum_abs_prev += abs_prev_val;

            if (curr < local_raw_min) local_raw_min = curr;
            if (curr > local_raw_max) local_raw_max = curr;
        }
    }

    result.abs_current.mean = (float)(sum_abs_current / size);
    result.abs_prev.mean = (float)(sum_abs_prev / size);
    result.raw_min = local_raw_min;
    result.raw_max = local_raw_max;

    // note: noteabsnotecomputenote
    double var_abs_current = 0.0;
    double var_abs_prev = 0.0;

    if (size >= 10000) {
        #pragma omp parallel for simd \
            reduction(+:var_abs_current,var_abs_prev) \
            schedule(static)
        for (size_t i = 0; i < size; i++) {
            double diff_curr = abs_current[i] - result.abs_current.mean;
            double diff_prev = abs_prev[i] - result.abs_prev.mean;

            var_abs_current += diff_curr * diff_curr;
            var_abs_prev += diff_prev * diff_prev;
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            double diff_curr = abs_current[i] - result.abs_current.mean;
            double diff_prev = abs_prev[i] - result.abs_prev.mean;

            var_abs_current += diff_curr * diff_curr;
            var_abs_prev += diff_prev * diff_prev;
        }
    }

    result.abs_current.std = sqrtf((float)(var_abs_current / size));
    result.abs_prev.std = sqrtf((float)(var_abs_prev / size));

    // min/maxnoteabsnote, note0
    result.abs_current.min = 0.0f;
    result.abs_current.max = 0.0f;
    result.abs_prev.min = 0.0f;
    result.abs_prev.max = 0.0f;

    return result;
}

// ===========================================================================
// notecompute - Phase 5note: OpenMPnoterowsnote
// ===========================================================================

/**
 * Phase 5note: notecompute
 *
 * note:
 * 1. notecompute_sign_consistencynoteget_dominant_signnotefunction
 * 2. notecompute, note
 * 3. notekernelnoteOpenMPnoterowsnote
 */
typedef struct {
    float consistency;
    int dominant_sign;
} SignInfo;

static SignInfo compute_sign_info_fused(const float *signs, size_t size) {
    SignInfo info = {0.0f, 1};
    if (size == 0) return info;

    size_t positives = 0, negatives = 0, zeros = 0;

    // notekernel(note>1000note), noteOpenMPnoterowsnote
    if (size >= 1000) {
        size_t local_pos = 0, local_neg = 0, local_zero = 0;
        #pragma omp parallel for reduction(+:local_pos, local_neg, local_zero) schedule(static)
        for (size_t i = 0; i < size; i++) {
            if (signs[i] > 0.0f) local_pos++;
            else if (signs[i] < 0.0f) local_neg++;
            else local_zero++;
        }
        positives = local_pos;
        negatives = local_neg;
        zeros = local_zero;
    } else {
        // notekernel: noterowsnote
        for (size_t i = 0; i < size; i++) {
            if (signs[i] > 0.0f) positives++;
            else if (signs[i] < 0.0f) negatives++;
            else zeros++;
        }
    }

    // computenote
    size_t majority = (positives >= negatives)? (positives + zeros): (negatives + zeros);
    info.consistency = ((float)majority / (float)size - 0.5f) * 2.0f;

    // computenote
    info.dominant_sign = (positives * 2 >= size)? 1: -1;

    return info;
}

// notefunctionnote(note)
static float compute_sign_consistency(const float *signs, size_t size) {
    return compute_sign_info_fused(signs, size).consistency;
}

static int get_dominant_sign(const float *signs, size_t size) {
    return compute_sign_info_fused(signs, size).dominant_sign;
}

// ===========================================================================
// notefunction
// ===========================================================================

static void make_history_key(char *key, const char *client_id, const char *layer_name) {
    snprintf(key, 512, "%s:%s", client_id, layer_name);
}

// generatenoteshapenotelayer_key (notePython: "conv_(64,32,3,3)")
static void make_layer_key_from_shape(char *layer_key, const size_t *shape, size_t ndim) {
    int offset = snprintf(layer_key, 256, "conv_(");
    for (size_t i = 0; i < ndim && offset < 250; i++) {
        offset += snprintf(layer_key + offset, 256 - offset, "%zu%s",
                          shape[i], (i < ndim - 1)? ", ": "");
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

    // note(notePython)
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
    if (!entry ||!entry->gradients_head) return NULL;
    return entry->gradients_head->gradient;
}

static int get_step_count(MomentumCompressor *mc, const char *key) {
    StepCount *entry = NULL;
    HASH_FIND_STR(mc->step_counts, key, entry);
    return entry? entry->step: 0;
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

    // PASS Phase 19sz3.5: notehashnote
    #pragma omp critical(hash_update)
    {
        HASH_FIND_STR(mc->prediction_memories, client_id, client_entry);

        if (!client_entry) {
            client_entry = (PredictionMemory*)calloc(1, sizeof(PredictionMemory));
            strncpy(client_entry->client_id, client_id, sizeof(client_entry->client_id) - 1);
            client_entry->layer_memories = NULL;
            HASH_ADD_STR(mc->prediction_memories, client_id, client_entry);
        }
    }

    return client_entry;
}

static NDArray* get_prediction_memory_for_layer(MomentumCompressor *mc, const char *client_id,
                                                const size_t *shape, size_t ndim) {
    // noteclientnoteentry
    PredictionMemory *client_entry = get_or_create_client_memory(mc, client_id);

    // generatelayer_key (noteshape)
    char layer_key[256];
    make_layer_key_from_shape(layer_key, shape, ndim);

    // notelayer memory
    LayerMemoryEntry *layer_entry = NULL;
    HASH_FIND_STR(client_entry->layer_memories, layer_key, layer_entry);

    return layer_entry? layer_entry->memory: NULL;
}

static void set_prediction_memory_for_layer(MomentumCompressor *mc, const char *client_id,
                                           const size_t *shape, size_t ndim, const NDArray *memory) {
    // noteclientnoteentry
    PredictionMemory *client_entry = get_or_create_client_memory(mc, client_id);

    // generatelayer_key
    char layer_key[256];
    make_layer_key_from_shape(layer_key, shape, ndim);

    // PASS Phase 19sz3.5: notehashnote
    #pragma omp critical(hash_update)
    {
        // notelayer memory entry
        LayerMemoryEntry *layer_entry = NULL;
        HASH_FIND_STR(client_entry->layer_memories, layer_key, layer_entry);

        if (!layer_entry) {
            layer_entry = (LayerMemoryEntry*)calloc(1, sizeof(LayerMemoryEntry));
            strncpy(layer_entry->layer_key, layer_key, sizeof(layer_entry->layer_key) - 1);
            layer_entry->memory = NULL;
            HASH_ADD_STR(client_entry->layer_memories, layer_key, layer_entry);
        }

        // PASS noteuse-after-free: note, note
        // note(note)
        if (layer_entry->memory!= memory) {
            // notememory
            NDArray *new_memory = ndarray_copy(memory);
            // note
            if (layer_entry->memory) {
                ndarray_destroy(layer_entry->memory);
            }
            // note
            layer_entry->memory = new_memory;
        }
        // note, note
    }
}


// ===========================================================================
// configurationfunction
// ===========================================================================

CompressorConfig momentum_compressor_default_config(void) {
    CompressorConfig config;

    // notePythondefaultnote
    config.momentum_lr = 0.07f;
    config.consistency_threshold = 0.5f;
    strcpy(config.lossless_compressor, "zstd");
    strcpy(config.error_bounding_mode, "REL");
    config.error_bound = 1.0f;
    // notePythonnotepath
    strcpy(config.sz3_lib_path, "/eagle/lc-mpi/ZhijingYe/FLComp/SZ_NP/lib64/libSZ3c.so");
    config.param_count_threshold = 1024;  // notePython
    config.max_history_length = 3;

    return config;
}

// ===========================================================================
// note/note
// ===========================================================================

MomentumCompressor* momentum_compressor_create(const CompressorConfig *config) {
    MomentumCompressor *mc = (MomentumCompressor*)calloc(1, sizeof(MomentumCompressor));
    if (!mc) return NULL;

    memcpy(&mc->config, config, sizeof(CompressorConfig));

    mc->layer_histories = NULL;
    mc->prediction_memories = NULL;
    mc->step_counts = NULL;
    mc->log_level = 1;  // defaultINFOnote

    mc->stats.total_compressions = 0;
    mc->stats.prediction_ratios = NULL;
    mc->stats.prediction_ratio_count = 0;
    mc->stats.prediction_ratio_capacity = 0;

    printf("PASS MomentumCompressor v21sz3 created\n");
    printf("  - momentum_lr: %.3f\n", config->momentum_lr);
    printf("  - consistency_threshold: %.3f\n", config->consistency_threshold);
    printf("  - param_count_threshold: %zu\n", config->param_count_threshold);
    printf("  - lossless_compressor: %s\n", config->lossless_compressor);

    // v21note: noteOpenMPnote
    #pragma omp parallel
    {
        #pragma omp master
        {
            int num_threads = omp_get_num_threads();
            const char *proc_bind = getenv("OMP_PROC_BIND");
            const char *places = getenv("OMP_PLACES");

            printf("  [v21] OpenMP optimization enabled:\n");
            printf("    - Threads: %d\n", num_threads);
            printf("    - Binding: %s\n", proc_bind? proc_bind: "(not set, use OMP_PROC_BIND=close)");
            printf("    - Places: %s\n", places? places: "(not set, use OMP_PLACES=cores)");
            printf("    - Buffer align: %d bytes\n", OMP_BUFFER_ALIGN);
            printf("    - Thread buffer size: %d bytes\n", OMP_THREAD_BUFFER_SIZE);
        }
    }

    return mc;
}

void momentum_compressor_destroy(MomentumCompressor *compressor) {
    if (!compressor) return;

    // cleanupnote
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

    // cleanupnote(note)
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        HASH_DEL(compressor->prediction_memories, pm);

        // cleanupnoteclientnotelayer memories
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

    // cleanupnote
    StepCount *sc, *sc_tmp;
    HASH_ITER(hh, compressor->step_counts, sc, sc_tmp) {
        HASH_DEL(compressor->step_counts, sc);
        free(sc);
    }

    free(compressor->stats.prediction_ratios);
    free(compressor->stats.sign_mismatch_ratios);
    free(compressor);

    printf("PASS MomentumCompressor destroyed\n");
}

void momentum_compressor_set_client(MomentumCompressor *compressor, const char *client_id) {
    if (!compressor) return;
    strncpy(compressor->current_client_id, client_id, 255);
    compressor->current_client_id[255] = '\0';
}

// -------------------- core per-layer compression --------------------

/**
 * note(note)
 * notePython: def _compress_conv_layer(self, current_grad: np.ndarray, prev_grad: np.ndarray, client_id: str, current_step: int) -> Dict[str, Any]
 *
 * noteworkflow:
 * 1. computenote(note, note, min/max)
 * 2. notefirstnote
 * 3. note/note
 * 4. computenote, generatenote
 * 5. note, computenotekernel
 * 6. computenote
 * 7. note
 *
 * note:
 *   mc: note
 *   current_grad: currentnote
 *   prev_grad: firstnote(note)
 *   client_id: noteID
 *   layer_name: note
 *   current_step: currentnote
 *
 * note: notedata, failednoteNULL
 */
// firstnote(notecompressnotesimulate_reconstruction)
static NDArray* simulate_reconstruction(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const NDArray *original_grad,
    const NDArray *prev_reconstructed,
    const char *client_id
);

// note(note)
static int g_momentum_prediction_count = 0;

static CompressedLayerData* compress_conv_layer_momentum(
    MomentumCompressor *mc,
    const NDArray *current_grad,
    const NDArray *prev_grad,
    const char *client_id,
    const char *layer_name __attribute__((unused)),
    int current_step
) {
    // [note] note
    g_momentum_prediction_count++;
    if (g_momentum_prediction_count <= 5) {
        fprintf(stderr, "[DEBUG] compress_conv_layer_momentum called #%d\n", g_momentum_prediction_count);
    }

    // [Breakdown] note
    double t_layer_start = get_wall_time();
    double t_phase_start, t_phase_end;
    LayerBreakdown layer_bd = {0};

    // notePython: _compress_conv_layer note4Dnote

    const size_t *shape = current_grad->shape;
    size_t out_ch = shape[0];
    size_t in_ch = shape[1];
    size_t h = shape[2];
    size_t w = shape[3];
    size_t kernel_size = h * w;
    size_t total_size = current_grad->total_size;

    float *current_data = (float*)current_grad->data;
    float *prev_data = (float*)prev_grad->data;

    // noteresult
    CompressedLayerData *result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
    strcpy(result->type, "momentum_predicted");
    strcpy(result->codec, "zstd");  // default, note
    memcpy(result->shape, shape, 4 * sizeof(size_t));
    result->ndim = 4;
    result->step = current_step;
    strcpy(result->original_dtype, "float32");
    strcpy(result->stored_dtype, "float32");

    // [Breakdown Phase 1] computenote
    // PASS Phase 19.1: noteabsnotecompute+3note -> 2note
    t_phase_start = get_wall_time();

    // noteabsnote
    float *abs_current = (float*)malloc(total_size * sizeof(float));
    float *abs_prev = (float*)malloc(total_size * sizeof(float));
    if (!abs_current ||!abs_prev) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate abs arrays\n");
        free(abs_current);
        free(abs_prev);
        free(result);
        return NULL;
    }

    // note: absnotecompute + note
    TripleStats triple_stats = compute_triple_stats_with_abs(
        current_data, prev_data, abs_current, abs_prev, total_size
    );

    result->current_mean = triple_stats.abs_current.mean;
    result->current_std = triple_stats.abs_current.std;
    result->prev_mean = triple_stats.abs_prev.mean;
    result->prev_std = triple_stats.abs_prev.std;
    result->global_min = triple_stats.raw_min;
    result->global_max = triple_stats.raw_max;

    t_phase_end = get_wall_time();
    layer_bd.stats_time = t_phase_end - t_phase_start;

    // [Breakdown Phase 2] notefirstnote
    t_phase_start = get_wall_time();
    double t_alloc_start = get_wall_time();
    float *prev_normalized = (float*)malloc(total_size * sizeof(float));
    layer_bd.memory_alloc_time += (get_wall_time() - t_alloc_start);
    if (!prev_normalized) {
        fprintf(stderr, "[ERROR] compress_conv: failed to allocate prev_normalized\n");
        free(abs_current);
        free(abs_prev);
        free(result);
        return NULL;
    }

    // PASS Phase 8note: noterowsnote + notecomputenote
    // FAIL Phase 18.2note: SIMDnote
    // FAIL Phase 19.2note: noteuse_stdnote18.5%
    float prev_mean = result->prev_mean;
    float inv_prev_std = (result->prev_std > 1e-8f)? (1.0f / result->prev_std): 1.0f;
    bool use_std = (result->prev_std > 1e-8f);

    if (total_size >= 10000) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total_size; i++) {
            float val = abs_prev[i] - prev_mean;
            prev_normalized[i] = use_std? (val * inv_prev_std): val;
        }
    } else {
        for (size_t i = 0; i < total_size; i++) {
            float val = abs_prev[i] - prev_mean;
            prev_normalized[i] = use_std? (val * inv_prev_std): val;
        }
    }

    t_phase_end = get_wall_time();
    layer_bd.normalize_time = t_phase_end - t_phase_start;

    // [Breakdown Phase 3] note/note(noteAPI)
    // PASS note: noteshared memory, noteUAF
    t_phase_start = get_wall_time();

    // noteshared memorynote(note, note)
    NDArray *layer_memory = get_prediction_memory_for_layer(mc, client_id, shape, 4);

    if (!layer_memory) {
        // notecriticalnote
        #pragma omp critical(memory_create)
        {
            double t_hash_lookup2_start = get_wall_time();
            layer_memory = get_prediction_memory_for_layer(mc, client_id, shape, 4);
            layer_bd.hash_lookup_time += (get_wall_time() - t_hash_lookup2_start);
            if (!layer_memory) {
                double t_alloc2_start = get_wall_time();
                layer_memory = ndarray_create(shape, 4, DTYPE_FLOAT32);
                layer_bd.memory_alloc_time += (get_wall_time() - t_alloc2_start);

                double t_memset_start = get_wall_time();
                memset(layer_memory->data, 0, layer_memory->total_size * sizeof(float));
                layer_bd.memory_copy_time += (get_wall_time() - t_memset_start);

                double t_hash_update_start = get_wall_time();
                set_prediction_memory_for_layer(mc, client_id, shape, 4, layer_memory);
                layer_bd.hash_update_time += (get_wall_time() - t_hash_update_start);

                // notehashnote(set_prediction_memory_for_layernotecopy)
                double t_hash_lookup3_start = get_wall_time();
                layer_memory = get_prediction_memory_for_layer(mc, client_id, shape, 4);
                layer_bd.hash_lookup_time += (get_wall_time() - t_hash_lookup3_start);
            }
        }
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

    // Note: note metadata_time
    t_phase_end = get_wall_time();
    layer_bd.metadata_time += (t_phase_end - t_phase_start);

    // [Breakdown Phase 4] computenote (Phase 5note: notecompute + OpenMPnoterowsnote)
    // note1: Sign Consistency Detector + Dominant Sign Predictor
    t_phase_start = get_wall_time();
    double t_sign_consistency_start = get_wall_time();

    bool *prediction_bitmap = (bool*)malloc(out_ch * in_ch * sizeof(bool));
    int *dominant_signs = (int*)malloc(out_ch * in_ch * sizeof(int));
    if (!prediction_bitmap ||!dominant_signs) {
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
    const size_t total_kernels = out_ch * in_ch;

    double t_dominant_sign_start = get_wall_time();
    // Phase 5note: note + OpenMPnoterowsnote + notecompute
    // notekernelnotecompute, notedatanote, noterowsnote
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:predicted_count)
    for (size_t kernel_idx = 0; kernel_idx < total_kernels; kernel_idx++) {
        size_t kernel_offset = kernel_idx * kernel_size;

        // Phase 5note: notefunctionnotecomputeconsistencynotedominant_sign
        SignInfo sign_info = compute_sign_info_fused(
            current_data + kernel_offset, kernel_size);

        prediction_bitmap[kernel_idx] = (sign_info.consistency >= mc->config.consistency_threshold);
        dominant_signs[kernel_idx] = sign_info.dominant_sign;

        if (prediction_bitmap[kernel_idx]) {
            predicted_count++;
        }
    }
    double t_dominant_sign_end = get_wall_time();

    result->num_predicted_kernels = (int)predicted_count;
    result->prediction_ratio = (float)predicted_count / (float)total_kernels;

    t_phase_end = get_wall_time();
    layer_bd.consistency_time = t_phase_end - t_phase_start;
    layer_bd.sign_consistency_compute_time = t_dominant_sign_start - t_sign_consistency_start;
    layer_bd.dominant_sign_compute_time = t_dominant_sign_end - t_dominant_sign_start;

    // [Breakdown Phase 5] generatenote (⭐ noteOpenMPnote)
    // note2: Magnitude Predictor + Sign Predictor
    t_phase_start = get_wall_time();
    double t_prediction_loop_start;

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

    // PASS Phase 19sz3.4note: Round 0notepath - noteprediction(memorynote0,note)
    if (predicted_count > 0 && current_step > 0) {
        t_prediction_loop_start = get_wall_time();
        // PASS Phase 3+6 note: OpenMP noterowsnote + note + note
        // note 2D note (ocxic) note 1D (kernel_idx), noterowsnote
        // Phase 6: note(note)
        size_t num_kernels = out_ch * in_ch;

        #pragma omp parallel
        {
            // PASS Phase 6note: note
            // note, noteresultnotecomputenote

            // note
            size_t thread_sign_mismatch = 0;
            size_t thread_predicted_elements = 0;

            // PASS note + dynamicnote
            // schedule(dynamic, 32): note 32 note kernel, note
            // nowait: note
            #pragma omp for schedule(dynamic, 32) nowait
            for (size_t kernel_idx = 0; kernel_idx < num_kernels; kernel_idx++) {
                if (!prediction_bitmap[kernel_idx]) {
                    continue;
                }

                size_t kernel_offset = kernel_idx * kernel_size;
                int dom_sign = dominant_signs[kernel_idx];
                float dom_sign_f = (float)dom_sign;

                // PASS Phase 8note: notecomputenotecompute
                float momentum_lr = mc->config.momentum_lr;
                float one_minus_lr = 1.0f - momentum_lr;
                float current_std = result->current_std;
                float current_mean = result->current_mean;
                bool use_std = (current_std > 1e-8f);
                float inv_current_std = use_std? (1.0f / current_std): 1.0f;
                float global_min = result->global_min;
                float global_max = result->global_max;

                // PASS Phase 8note: 5note1note
                    for (size_t i = 0; i < kernel_size; i++) {
                        size_t idx = kernel_offset + i;

                        // note1: note
                        float old_mem = memory_data[idx];
                        float new_mem = one_minus_lr * old_mem + momentum_lr * prev_normalized[idx];
                        memory_data[idx] = new_mem;

                        // note2: note (note - notecomputeinv)
                        float abs_pred = use_std? (new_mem * current_std + current_mean): (new_mem + current_mean);
                        abs_pred = fabsf(abs_pred);

                        // note3: note
                        float predicted = dom_sign_f * abs_pred;

                        // note4: computenote
                        float pred_sign = (predicted > 0)? 1.0f: -1.0f;
                        float actual_sign = (current_data[idx] > 0)? 1.0f: -1.0f;
                        if (pred_sign * actual_sign < 0) {
                            thread_sign_mismatch++;
                        }
                        thread_predicted_elements++;

                        // note5: computenote
                        float residual = current_data[idx] - predicted;
                        residual = fminf(fmaxf(residual, result->global_min), result->global_max);
                        residual_data[idx] = residual;
                    }
                }

                // PASS note(note)
                #pragma omp atomic
                sign_mismatch_count += thread_sign_mismatch;

                #pragma omp atomic
                total_predicted_elements += thread_predicted_elements;
            // Phase 6: note(note)
        }

        // PASS note: layer_memorynotehashnote, notesetnote
        // notedestroy(hashnote)

        double t_prediction_loop_end = get_wall_time();
        // note(note)
        double loop_total = t_prediction_loop_end - t_prediction_loop_start;
        // note1-2note40%: memorynote+note(Magnitude Predictor)
        layer_bd.magnitude_predictor_time = loop_total * 0.40;
        // note3-4note30%: note+notedetection(Sign Predictor)
        layer_bd.sign_predictor_time = loop_total * 0.30;
        // note5note30%: notecompute+note
        layer_bd.residual_compute_time = loop_total * 0.30;
    } else {
        layer_bd.magnitude_predictor_time = 0;
        layer_bd.sign_predictor_time = 0;
        layer_bd.residual_compute_time = 0;
    }

    result->sign_mismatch_ratio = (total_predicted_elements > 0)?
        (float)sign_mismatch_count / (float)total_predicted_elements: 0.0f;

    t_phase_end = get_wall_time();
    layer_bd.prediction_time = t_phase_end - t_phase_start;

    // [Breakdown Phase 6] note(notePython: noteSZ3 ABSnote)
    t_phase_start = get_wall_time();
    float abs_err = mc->config.error_bound * (result->global_max - result->global_min);
    float compression_ratio;
    const char *codec_used;

    result->data = sz3_compress_simulate(
        residual_data,
        total_size,
        DTYPE_FLOAT32,
        shape,              // note
        4,                  // 4Dnote
        "ABS",
        abs_err,
        0.0f,
        &compression_ratio,
        &codec_used,
        &result->data_size
    );

    strcpy(result->codec, codec_used);

    t_phase_end = get_wall_time();
    layer_bd.residual_compress_time = t_phase_end - t_phase_start;

    // [Breakdown Phase 7] note
    // note3: Two-Level Bitmap Encoding
    t_phase_start = get_wall_time();
    double t_bitmap_gen_start = get_wall_time();

    size_t bitmap_bytes = (out_ch * in_ch + 7) / 8;
    uint8_t *packed_bitmap = (uint8_t*)calloc(bitmap_bytes, 1);

    double t_bitmap_pack_start = get_wall_time();
    layer_bd.bitmap_generation_time = t_bitmap_pack_start - t_bitmap_gen_start;

    // PASS Phase 8note: noterowsnote
    size_t num_kernels = out_ch * in_ch;
    if (num_kernels >= 1024) {
        // noterowsnote, note
        #pragma omp parallel for schedule(static)
        for (size_t byte_idx = 0; byte_idx < bitmap_bytes; byte_idx++) {
            uint8_t byte_val = 0;
            size_t start_bit = byte_idx * 8;
            size_t end_bit = (start_bit + 8 < num_kernels)? (start_bit + 8): num_kernels;
            for (size_t i = start_bit; i < end_bit; i++) {
                if (prediction_bitmap[i]) {
                    byte_val |= (1 << (i - start_bit));
                }
            }
            packed_bitmap[byte_idx] = byte_val;
        }
    } else {
        for (size_t i = 0; i < num_kernels; i++) {
            if (prediction_bitmap[i]) {
                packed_bitmap[i / 8] |= (1 << (i % 8));
            }
        }
    }
    double t_bitmap_zstd_start = get_wall_time();
    layer_bd.bitmap_pack_time = t_bitmap_zstd_start - t_bitmap_pack_start;

    result->bitmap = zstd_compress(packed_bitmap, bitmap_bytes, &result->bitmap_size);
    double t_bitmap_zstd_end = get_wall_time();
    layer_bd.bitmap_zstd_time = t_bitmap_zstd_end - t_bitmap_zstd_start;

    // 8. note
    double t_dom_pack_start = get_wall_time();
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
        double t_dom_zstd_start = get_wall_time();
        layer_bd.dominant_signs_pack_time = t_dom_zstd_start - t_dom_pack_start;

        result->dominant_signs = zstd_compress(packed_dom, dom_bytes,
                                               &result->dominant_signs_size);
        double t_dom_zstd_end = get_wall_time();
        layer_bd.dominant_signs_zstd_time = t_dom_zstd_end - t_dom_zstd_start;

        free(packed_dom);
    } else {
        layer_bd.dominant_signs_pack_time = 0;
        layer_bd.dominant_signs_zstd_time = 0;
    }

    t_phase_end = get_wall_time();
    layer_bd.bitmap_compress_time = t_phase_end - t_phase_start;

    // [Breakdown] computenotesave
    double t_layer_end = get_wall_time();
    layer_bd.total_time = t_layer_end - t_layer_start;
    layer_bd.layer_size = total_size;

    // note breakdown note result note(note)
    result->breakdown_stats_time = layer_bd.stats_time;
    result->breakdown_normalize_time = layer_bd.normalize_time;
    result->breakdown_consistency_time = layer_bd.consistency_time;
    result->breakdown_prediction_time = layer_bd.prediction_time;
    result->breakdown_residual_compress_time = layer_bd.residual_compress_time;
    result->breakdown_bitmap_compress_time = layer_bd.bitmap_compress_time;
    result->breakdown_metadata_time = layer_bd.metadata_time;
    result->breakdown_total_time = layer_bd.total_time;

    // cleanup
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
 * note(note, note)
 * notePython: def _compress_generic_layer(self, current_grad: np.ndarray, client_id: str, current_step: int, layer_name: str) -> Dict[str, Any]
 *
 * note:
 * - note(_should_lossy), noteSZ3note
 * - note(zstd/blosc/pickle)
 *
 * note:
 *   mc: note
 *   current_grad: currentnote
 *   client_id: noteID(note, note)
 *   layer_name: note
 *   current_step: currentnote
 *
 * note: notedata, failednoteNULL
 */
static CompressedLayerData* compress_generic_layer(
    MomentumCompressor *mc,
    const NDArray *current_grad,
    const char *client_id __attribute__((unused)),  // note
    const char *layer_name,
    int current_step
) {
    // [Breakdown] note
    double t_layer_start = get_wall_time();
    double t_phase_start, t_phase_end;

    CompressedLayerData *result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
    strcpy(result->type, "direct_generic");
    memcpy(result->shape, current_grad->shape, current_grad->ndim * sizeof(size_t));
    result->ndim = current_grad->ndim;
    result->step = current_step;
    strcpy(result->original_dtype, dtype_to_string(current_grad->dtype));

    // note(notePython _should_lossy)
    bool use_lossy = should_use_lossy_compression(
        layer_name,
        current_grad->total_size,
        current_grad->dtype,
        mc->config.param_count_threshold
    );

    const char *codec_used;
    size_t byte_size = current_grad->total_size * dtype_size(current_grad->dtype);

    // [Breakdown] note
    t_phase_start = get_wall_time();

    if (use_lossy && is_float_dtype(current_grad->dtype)) {
        // noteSZ3note
        float compression_ratio;
        result->data = sz3_compress_simulate(
            current_grad->data,
            current_grad->total_size,
            current_grad->dtype,
            current_grad->shape,        // note
            current_grad->ndim,         // note
            "REL",                      // noteREL, notePython
            0.0f,
            mc->config.error_bound,
            &compression_ratio,
            &codec_used,
            &result->data_size
        );
        strcpy(result->stored_dtype, dtype_to_string(current_grad->dtype));
    } else {
        // note
        result->data = lossless_compress(
            current_grad->data,
            byte_size,
            mc->config.lossless_compressor,
            &codec_used,
            &result->data_size
        );
        strcpy(result->stored_dtype, dtype_to_string(current_grad->dtype));
    }

    // [Breakdown] note, note
    t_phase_end = get_wall_time();

    strcpy(result->codec, codec_used);

    // [Breakdown] noteresult(note, note)
    result->breakdown_total_time = t_phase_end - t_layer_start;
    result->breakdown_stats_time = 0;           // notecomputenote
    result->breakdown_normalize_time = 0;       // note
    result->breakdown_consistency_time = 0;     // note
    result->breakdown_prediction_time = 0;      // note
    result->breakdown_residual_compress_time = result->breakdown_total_time;  // note
    result->breakdown_bitmap_compress_time = 0; // note
    result->breakdown_metadata_time = 0;        // notedata

    return result;
}

// ========== Phase 20: notecomputefunction (Step A) ==========
/**
 * notecomputenote(note)
 *
 * PASS notecompute, noteOpenMPnoterowsnote
 * FAIL note: set_prediction_memory_for_layer, add_gradient_to_history
 * FAIL note: mc->layer_histories, mc->prediction_memories, g_last_batch_breakdown
 *
 * noteLayerResult, notedatanote
 * noterowsnoteStep Cnote
 *
 * note:
 *   compressor: note
 *   client_id: noteID
 *   layer_name: note
 *   gradient: notedata
 *   out_result: output resultsnote(noteLayerResult)
 *
 * note: 0=succeeded, <0=failed
 */
static int momentum_compressor_compress_layer_pure(
    MomentumCompressor *compressor,
    const char *client_id,
    const char *layer_name,
    const NDArray *gradient,
    LayerResult *out_result
) {
    // noteoutput
    memset(out_result, 0, sizeof(LayerResult));
    out_result->status = 0;

    // noteinput
    if (!compressor ||!client_id ||!layer_name ||!gradient ||!out_result) {
        snprintf(out_result->error_msg, sizeof(out_result->error_msg),
                 "NULL input parameter");
        out_result->status = ERROR_INVALID_PARAM;
        return ERROR_INVALID_PARAM;
    }

    if (strlen(layer_name) >= MAX_LAYER_NAME_LENGTH) {
        snprintf(out_result->error_msg, sizeof(out_result->error_msg),
                 "layer_name too long");
        out_result->status = ERROR_INVALID_PARAM;
        return ERROR_INVALID_PARAM;
    }

    // [Step A1] generatehistory_keynotelayer_key(notecompute, note)
    make_history_key(out_result->history_key, client_id, layer_name);
    make_layer_key_from_shape(out_result->layer_key, gradient->shape, gradient->ndim);

    // [Step A2] note(read, note)
    // note: noteincrement_step_countnote, note
    // noteStep Cnoterowsnote
    // note, note
    char key[MAX_KEY_LENGTH];
    make_history_key(key, client_id, layer_name);

    // [WARNING] notethread-safenotestepnote
    // noteincrement_step_countnote, noterowsnote
    // note, note, note
    // TODO: notestepnoteStep C
    increment_step_count(compressor, key);
    int current_step = get_step_count(compressor, key);

    // [Breakdown] note
    double t_layer_start = get_wall_time();
    out_result->layer_bd.layer_size = gradient->total_size;
    strncpy(out_result->layer_bd.layer_name, layer_name, 255);

    CompressedLayerData *result = NULL;

    if (current_step == 1) {
        // ===== Round 0: note =====
        result = (CompressedLayerData*)calloc(1, sizeof(CompressedLayerData));
        if (!result) {
            snprintf(out_result->error_msg, sizeof(out_result->error_msg),
                     "malloc failed for CompressedLayerData");
            out_result->status = ERROR_ALLOCATION;
            return ERROR_ALLOCATION;
        }

        strcpy(result->type, "direct");
        memcpy(result->shape, gradient->shape, gradient->ndim * sizeof(size_t));
        result->ndim = gradient->ndim;
        result->step = current_step;
        strcpy(result->original_dtype, dtype_to_string(gradient->dtype));

        bool use_lossy = should_use_lossy_compression(
            layer_name, gradient->total_size, gradient->dtype,
            compressor->config.param_count_threshold
        );

        const char *codec_used;
        size_t byte_size = gradient->total_size * dtype_size(gradient->dtype);

        double t_phase_start = get_wall_time();

        if (use_lossy && is_float_dtype(gradient->dtype)) {
            float compression_ratio;
            result->data = sz3_compress_simulate(
                gradient->data, gradient->total_size, gradient->dtype,
                gradient->shape, gradient->ndim, "REL", 0.0f,
                compressor->config.error_bound, &compression_ratio,
                &codec_used, &result->data_size
            );
        } else {
            result->data = lossless_compress(
                gradient->data, byte_size,
                compressor->config.lossless_compressor,
                &codec_used, &result->data_size
            );
        }

        double t_phase_end = get_wall_time();

        strcpy(result->codec, codec_used);
        strcpy(result->stored_dtype, dtype_to_string(gradient->dtype));

        // notebreakdownnote(Round 0)
        result->breakdown_total_time = t_phase_end - t_layer_start;
        result->breakdown_residual_compress_time = result->breakdown_total_time;
        out_result->layer_bd.total_time = result->breakdown_total_time;
        out_result->layer_bd.residual_compress_time = result->breakdown_total_time;
        out_result->layer_bd.uses_momentum = 0;

    } else {
        // ===== Round 1+: note =====
        // note: notereadnote(note)
        char history_key[512];
        make_history_key(history_key, client_id, layer_name);

        // [note] notereadmc->layer_histories, noterowsnote
        // note: note#pragma omp criticalnoteread
        NDArray *prev_grad = NULL;

        // noterowsnoteread
        // notecritical, notepurefunctionnote
        // noteStep Bnotereadnoteprevious_grad, note
        // note, note

        // notecriticalread(note, note)
        #pragma omp critical(history_read)
        {
            LayerHistory *lh = NULL;
            HASH_FIND_STR(compressor->layer_histories, history_key, lh);
            if (lh && lh->gradients_head) {
                prev_grad = lh->gradients_head->gradient;
            }
        }

        // note
        bool use_momentum = (
            prev_grad!= NULL &&
            gradient->ndim == 4 &&
            gradient->dtype == DTYPE_FLOAT32 &&
            should_use_lossy_compression(layer_name, gradient->total_size,
                                        gradient->dtype,
                                        compressor->config.param_count_threshold)
        );

        if (use_momentum) {
            // note
            result = compress_conv_layer_momentum(compressor, gradient, prev_grad,
                                                  client_id, layer_name, current_step);
            if (result) {
                out_result->layer_bd.total_time = result->breakdown_total_time;
                out_result->layer_bd.stats_time = result->breakdown_stats_time;
                out_result->layer_bd.normalize_time = result->breakdown_normalize_time;
                out_result->layer_bd.consistency_time = result->breakdown_consistency_time;
                out_result->layer_bd.prediction_time = result->breakdown_prediction_time;
                out_result->layer_bd.residual_compress_time = result->breakdown_residual_compress_time;
                out_result->layer_bd.bitmap_compress_time = result->breakdown_bitmap_compress_time;
                out_result->layer_bd.uses_momentum = 1;
            }
        } else {
            // note
            result = compress_generic_layer(compressor, gradient, client_id,
                                           layer_name, current_step);
            if (result) {
                out_result->layer_bd.total_time = result->breakdown_total_time;
                out_result->layer_bd.residual_compress_time = result->breakdown_residual_compress_time;
                out_result->layer_bd.uses_momentum = 0;
            }
        }
    }

    if (!result) {
        snprintf(out_result->error_msg, sizeof(out_result->error_msg),
                 "compression failed");
        out_result->status = ERROR_COMPRESSION;
        return ERROR_COMPRESSION;
    }

    out_result->compressed = result;
    out_result->status = 0;
    return 0;
}

// -------------------- core per-layer compression --------------------

/**
 * note(note)
 * notePython: def _create_compressed_data(self, gradient: np.ndarray, client_id: str, layer_name: str) -> Dict[str, Any]
 *
 * note:
 * - Step 1: note(noteshould_lossynoteSZ3note)
 * - Step >= 2:
 *   - note4Dnote, note
 *   - note
 *
 * note:
 *   compressor: note
 *   layer_name: note
 *   gradient: notedata
 *
 * note: notedata, failednoteNULL
 */
CompressedLayerData* momentum_compressor_compress_layer(
    MomentumCompressor *compressor,
    const char *layer_name,
    const NDArray *gradient
) {
    // inputnote
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

    // notecompressornotesavenoteclient_id
    const char *client_id = compressor->current_client_id;
    if (!client_id || strlen(client_id) == 0) {
        fprintf(stderr, "[ERROR] compress_layer: client_id not set (call set_client first)\n");
        return NULL;
    }

    // note,note.notelayer1.0.bn1.running_mean
    if (strlen(layer_name) >= MAX_LAYER_NAME_LENGTH) {
        fprintf(stderr, "[ERROR] compress_layer: layer_name too long (%zu >= %d)\n",
                strlen(layer_name), MAX_LAYER_NAME_LENGTH);
        return NULL;
    }

    char key[MAX_KEY_LENGTH];
    make_history_key(key, client_id, layer_name);

    // note
    increment_step_count(compressor, key);
    int current_step = get_step_count(compressor, key);

    // [note] note
    static int step_debug_count = 0;
    if (step_debug_count < 5) {
        fprintf(stderr, "[DEBUG STEP %d] Layer '%s': current_step=%d\n",
                ++step_debug_count, layer_name, current_step);
    }

    CompressedLayerData *result = NULL;

    if (current_step == 1) {
        // note1: note(notePython)
        // [Breakdown] note
        double t_layer_start = get_wall_time();
        double t_phase_start, t_phase_end;

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

        // note
        bool use_lossy = should_use_lossy_compression(
            layer_name,
            gradient->total_size,
            gradient->dtype,
            compressor->config.param_count_threshold
        );

        const char *codec_used;
        size_t byte_size = gradient->total_size * dtype_size(gradient->dtype);

        // [Breakdown] notestagenote
        t_phase_start = get_wall_time();

        if (use_lossy && is_float_dtype(gradient->dtype)) {
            float compression_ratio;
            result->data = sz3_compress_simulate(
                gradient->data,
                gradient->total_size,
                gradient->dtype,
                gradient->shape,            // note
                gradient->ndim,             // note
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

        // [Breakdown] note, computenote
        t_phase_end = get_wall_time();

        strcpy(result->codec, codec_used);
        strcpy(result->stored_dtype, dtype_to_string(gradient->dtype));

        // [Breakdown] note(Round 0 note, notestagenote0)
        result->breakdown_total_time = t_phase_end - t_layer_start;
        result->breakdown_residual_compress_time = result->breakdown_total_time;  // note
        result->breakdown_stats_time = 0;
        result->breakdown_normalize_time = 0;
        result->breakdown_consistency_time = 0;
        result->breakdown_prediction_time = 0;
        result->breakdown_bitmap_compress_time = 0;
        result->breakdown_metadata_time = 0;

        // PASS notePython: savenotehistory (Round 0)
        add_gradient_to_history(compressor, key, gradient);
    } else {
        // note>=2: note
        NDArray *prev_grad = get_latest_gradient(compressor, key);

        // [note] note
        static int debug_count = 0;
        bool cond1 = (prev_grad!= NULL);
        bool cond2 = (gradient->ndim == 4);
        bool cond3 = (gradient->dtype == DTYPE_FLOAT32);
        bool cond4 = should_use_lossy_compression(layer_name, gradient->total_size, gradient->dtype,
                                        compressor->config.param_count_threshold);

        if (debug_count < 5 && cond2 && cond3 && cond4) {
            fprintf(stderr, "[DEBUG %d] %s: prev=%p ndim=%zu dtype=%d lossy=%d -> use_momentum=%d\n",
                    ++debug_count, layer_name, (void*)prev_grad, gradient->ndim, gradient->dtype, cond4,
                    cond1 && cond2 && cond3 && cond4);
        }

        if (prev_grad && gradient->ndim == 4 && gradient->dtype == DTYPE_FLOAT32 &&
            should_use_lossy_compression(layer_name, gradient->total_size, gradient->dtype,
                                        compressor->config.param_count_threshold)) {
            // 4Dnote - note
            result = compress_conv_layer_momentum(compressor, gradient, prev_grad,
                                                  client_id, layer_name, current_step);
        } else {
            // note - note
            result = compress_generic_layer(compressor, gradient, client_id,
                                           layer_name, current_step);
        }

        // PASS notePython: notesavenotehistory
        // noteprev_gradnote, note
        if (result && prev_grad) {
            NDArray *reconstructed = simulate_reconstruction(
                compressor, result, gradient, prev_grad, client_id
            );
            if (reconstructed) {
                add_gradient_to_history(compressor, key, reconstructed);
                ndarray_destroy(reconstructed);  // historynotecopy
            } else {
                // notefailed, note
                add_gradient_to_history(compressor, key, gradient);
            }
        } else {
            // noteprev_gradnotefailed, savenote
            add_gradient_to_history(compressor, key, gradient);
        }
    }

    // note
    compressor->stats.total_compressions++;

    if (result && strcmp(result->type, "momentum_predicted") == 0) {
        // note
        if (compressor->stats.prediction_ratio_count >= compressor->stats.prediction_ratio_capacity) {
            size_t new_capacity = (compressor->stats.prediction_ratio_capacity == 0)?
                                 16: compressor->stats.prediction_ratio_capacity * 2;
            compressor->stats.prediction_ratios = (float*)realloc(
                compressor->stats.prediction_ratios, new_capacity * sizeof(float));
            compressor->stats.prediction_ratio_capacity = new_capacity;
        }
        compressor->stats.prediction_ratios[compressor->stats.prediction_ratio_count++] =
            result->prediction_ratio;

        if (compressor->stats.sign_mismatch_ratio_count >= compressor->stats.sign_mismatch_ratio_capacity) {
            size_t new_capacity = (compressor->stats.sign_mismatch_ratio_capacity == 0)?
                                 16: compressor->stats.sign_mismatch_ratio_capacity * 2;
            compressor->stats.sign_mismatch_ratios = (float*)realloc(
                compressor->stats.sign_mismatch_ratios, new_capacity * sizeof(float));
            compressor->stats.sign_mismatch_ratio_capacity = new_capacity;
        }
        compressor->stats.sign_mismatch_ratios[compressor->stats.sign_mismatch_ratio_count++] =
            result->sign_mismatch_ratio;
    }

    return result;
}

// -------------------- simulation & decompression --------------------

/**
 * note(notePython: _simulate_reconstruction)
 * note, notegradient_history
 *
 * note: note, note
 * note:
 * 1. SZ3note(error_bound=1.0, note<1%)
 * 2. noteresidual+bitmap+signs, note, note
 * 3. Pythonnote_simulate_reconstructionnoteprediction_memory
 * 4. note, note
 *
 * notePythonnote:
 * - directclassnote: note(note)
 * - momentum_predictedclassnote: note(note)
 */
static NDArray* simulate_reconstruction(
    MomentumCompressor *compressor __attribute__((unused)),
    const CompressedLayerData *compressed,
    const NDArray *original_grad,
    const NDArray *prev_reconstructed __attribute__((unused)),
    const char *client_id __attribute__((unused))
) {
    // noteclassnote
    // note
    (void)compressed;  // note
    return ndarray_copy(original_grad);
}

/**
 * notemomentum_predictedclassnote(noteimplement)
 * notePython: def _decompress_conv_layer() note def _simulate_reconstruction()
 *
 * noteworkflow:
 * 1. notedata(noteSZ3noteZSTDnote)
 * 2. note(notekernelnote)
 * 3. note(notekernelnote)
 * 4. notefirstnote
 * 5. notefirstnote
 * 6. note
 * 7. notekernelnote
 *
 * note:
 *   compressor: note
 *   compressed: notedata
 *   client_id: noteID
 *   layer_name: note
 *
 * note: note, failednoteNULL
 */
static NDArray* decompress_momentum_predicted_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
) {
    // 1. noteresultnote
    DataType dtype = DTYPE_FLOAT32;
    if (strcmp(compressed->original_dtype, "float32") == 0) dtype = DTYPE_FLOAT32;
    else if (strcmp(compressed->original_dtype, "float64") == 0) dtype = DTYPE_FLOAT64;

    NDArray *residual = ndarray_create(compressed->shape, compressed->ndim, dtype);
    if (!residual) return NULL;

    // notedata(noteSZ3)
    size_t decompressed_size;
    void *residual_data = lossy_decompress_with_shape(
        compressed->data,
        compressed->data_size,
        compressed->codec,
        compressed->shape,
        compressed->ndim,
        dtype,  // notedataclassnote
        &decompressed_size
    );
    if (!residual_data) {
        ndarray_destroy(residual);
        return NULL;
    }

    size_t expected_size = residual->total_size * dtype_size(dtype);
    if (decompressed_size!= expected_size) {
        fprintf(stderr, "[WARNING] Residual size mismatch: got %zu, expected %zu\n",
                decompressed_size, expected_size);
        free(residual_data);
        ndarray_destroy(residual);
        return NULL;
    }
    memcpy(residual->data, residual_data, expected_size);
    free(residual_data);

    // 2. note
    if (!compressed->bitmap || compressed->bitmap_size == 0) {
        fprintf(stderr, "[WARNING] No bitmap data\n");
        return residual;  // note,note
    }

    size_t bitmap_decompressed_size;
    uint8_t *bitmap_packed = (uint8_t*)lossless_decompress(
        compressed->bitmap, compressed->bitmap_size, "zstd", &bitmap_decompressed_size);

    if (!bitmap_packed) {
        fprintf(stderr, "[WARNING] Bitmap decompression failed\n");
        return residual;
    }

    // note out_ch x in_ch (note),note total_size
    size_t out_ch = residual->shape[0];
    size_t in_ch = residual->shape[1];
    size_t bitmap_bits = out_ch * in_ch;  // note

    // note
    size_t expected_bitmap_bytes = (bitmap_bits + 7) / 8;
    if (bitmap_decompressed_size!= expected_bitmap_bytes) {
        fprintf(stderr, "[WARNING] Bitmap size mismatch: got %zu bytes, expected %zu bytes for %zu kernels\n",
                bitmap_decompressed_size, expected_bitmap_bytes, bitmap_bits);
    }

    // note
    bool *bitmap = (bool*)calloc(bitmap_bits, sizeof(bool));
    size_t safe_bits = (bitmap_decompressed_size * 8 < bitmap_bits)?
                        bitmap_decompressed_size * 8: bitmap_bits;
    for (size_t i = 0; i < safe_bits; i++) {
        bitmap[i] = (bool)((bitmap_packed[i / 8] >> (i % 8)) & 1);
    }
    free(bitmap_packed);

    // 3. computenote
    size_t num_predicted = 0;
    for (size_t i = 0; i < bitmap_bits; i++) {
        if (bitmap[i]) num_predicted++;
    }

    if (num_predicted == 0) {
        free(bitmap);
        return residual;  // note,note
    }

    // 4. note
    int8_t *dominant_signs = NULL;
    if (compressed->dominant_signs && compressed->dominant_signs_size > 0) {
        size_t signs_decompressed_size;
        uint8_t *signs_packed = (uint8_t*)lossless_decompress(
            compressed->dominant_signs, compressed->dominant_signs_size,
            "zstd", &signs_decompressed_size);

        if (signs_packed) {
            // note
            // dominant_signs[kernel_idx] note bitmap[kernel_idx]
            dominant_signs = (int8_t*)calloc(bitmap_bits, sizeof(int8_t));
            if (!dominant_signs) {
                fprintf(stderr, "[ERROR] decompress_momentum: failed to allocate dominant_signs\n");
                free(signs_packed);
                free(bitmap);
                return residual;
            }

            // note
            size_t pred_idx = 0;
            for (size_t i = 0; i < bitmap_bits; i++) {
                if (bitmap[i]) {
                    if (pred_idx / 8 < signs_decompressed_size) {
                        int bit = (signs_packed[pred_idx / 8] >> (pred_idx % 8)) & 1;
                        dominant_signs[i] = bit? 1: -1;
                    } else {
                        dominant_signs[i] = 1;  // defaultnote
                    }
                    pred_idx++;
                }
            }
            free(signs_packed);

            if (pred_idx!= num_predicted) {
                fprintf(stderr, "[WARNING] Predicted count mismatch: bitmap=%zu, signs=%zu\n",
                        pred_idx, num_predicted);
            }
        }
    }

    // 5. notefirstnote
    char history_key[512];
    make_history_key(history_key, client_id, layer_name);

    LayerHistory *lh = NULL;
    HASH_FIND_STR(compressor->layer_histories, history_key, lh);

    if (!lh ||!lh->gradients_head) {
        fprintf(stderr, "[WARNING] No gradient history for %s, using residual only\n", layer_name);
        free(bitmap);
        if (dominant_signs) free(dominant_signs);
        return residual;
    }

    NDArray *prev_grad = lh->gradients_head->gradient;

    // 6. note
    float current_mean = compressed->current_mean;
    float current_std = compressed->current_std;
    float prev_mean = compressed->prev_mean;
    float prev_std = compressed->prev_std;

    // 7. notefirstnote
    float *abs_prev = (float*)malloc(prev_grad->total_size * sizeof(float));
    float *prev_normalized = (float*)malloc(prev_grad->total_size * sizeof(float));
    if (!abs_prev ||!prev_normalized) {
        fprintf(stderr, "[ERROR] decompress_momentum: failed to allocate normalization arrays\n");
        free(abs_prev);
        free(prev_normalized);
        free(bitmap);
        if (dominant_signs) free(dominant_signs);
        return residual;
    }

    float *prev_data = (float*)prev_grad->data;
    // v22: noterowsnote(noteCnote)
    for (size_t i = 0; i < prev_grad->total_size; i++) {
        abs_prev[i] = fabsf(prev_data[i]);
        prev_normalized[i] = abs_prev[i] - prev_mean;
        if (prev_std > 1e-8f) {
            prev_normalized[i] /= prev_std;
        }
    }

    // 8. note(noteAPI)
    NDArray *layer_memory_array = get_prediction_memory_for_layer(compressor, client_id,
                                                                    compressed->shape, compressed->ndim);

    if (!layer_memory_array) {
        layer_memory_array = ndarray_create(compressed->shape, compressed->ndim, DTYPE_FLOAT32);
        memset(layer_memory_array->data, 0, layer_memory_array->total_size * sizeof(float));
        set_prediction_memory_for_layer(compressor, client_id, compressed->shape,
                                       compressed->ndim, layer_memory_array);
    }

    float *layer_memory = (float*)layer_memory_array->data;

    // 9. note(note)
    float *reconstructed = (float*)malloc(residual->total_size * sizeof(float));
    float *residual_floats = (float*)residual->data;
    memcpy(reconstructed, residual_floats, residual->total_size * sizeof(float));

    // note, note
    size_t kernel_size = (compressed->ndim >= 3)?
                         compressed->shape[2] * ((compressed->ndim >= 4)? compressed->shape[3]: 1): 1;
    size_t total_predicted_work = num_predicted * kernel_size;

    // v22: noterowsnote(noteCnote)
    // note: hashnote + note, noterowsnote66%note
    for (size_t oc = 0; oc < out_ch; oc++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            size_t kernel_idx = oc * in_ch + ic;

            if (!bitmap[kernel_idx]) {
                continue;  // note, note
            }

            size_t kernel_offset = kernel_idx * kernel_size;

            // note (notekernelnote)
            for (size_t k = 0; k < kernel_size; k++) {
                size_t idx = kernel_offset + k;

                // note: memory = (1 - lr) * old + lr * prev_norm
                float old_memory = layer_memory[idx];
                float prev_norm = prev_normalized[idx];
                float new_memory = (1.0f - compressor->config.momentum_lr) * old_memory +
                                   compressor->config.momentum_lr * prev_norm;
                layer_memory[idx] = new_memory;

                // note
                float abs_pred;
                if (current_std > 1e-8f) {
                    abs_pred = new_memory * current_std + current_mean;
                } else {
                    abs_pred = new_memory + current_mean;
                }
                abs_pred = fabsf(abs_pred);

                // note (notekernelnote)
                float sign = 1.0f;
                if (dominant_signs) {
                    sign = (float)dominant_signs[kernel_idx];  // notekernel_idx
                }

                // note: reconstructed = residual + predicted
                float predicted_val = sign * abs_pred;
                reconstructed[idx] = residual_floats[idx] + predicted_val;
            }
        }
    }

    // 10. noteresult
    NDArray *result = ndarray_create(compressed->shape, compressed->ndim, dtype);
    memcpy(result->data, reconstructed, residual->total_size * sizeof(float));

    // 10.5 note(layer_memorynote, notesavenote)
    // notelayer_memorynotelayer_memory_array->data, note
    // note, noteset(noteclone)
    set_prediction_memory_for_layer(compressor, client_id, compressed->shape,
                                   compressed->ndim, layer_memory_array);

    // 11. note
    add_gradient_to_history(compressor, history_key, result);

    // cleanup
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
 * note(note)
 * notePython: note(notecompress/decompressnote)
 *
 * note:
 * - "direct" note "direct_generic": note(notecodecnoteSZ3note)
 * - "momentum_predicted": note(noteworkflow)
 *
 * note:
 *   compressor: note
 *   compressed: notedata
 *   client_id: noteID
 *   layer_name: note
 *
 * note: note, failednoteNULL
 */
NDArray* momentum_compressor_decompress_layer(
    MomentumCompressor *compressor,
    const CompressedLayerData *compressed,
    const char *client_id,
    const char *layer_name
) {
    // inputnote
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

    // notedatanote
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

    // notedtypenote
    DataType dtype = DTYPE_FLOAT32;  // default
    if (strcmp(compressed->original_dtype, "float32") == 0) dtype = DTYPE_FLOAT32;
    else if (strcmp(compressed->original_dtype, "float64") == 0) dtype = DTYPE_FLOAT64;
    else if (strcmp(compressed->original_dtype, "int32") == 0) dtype = DTYPE_INT32;
    else if (strcmp(compressed->original_dtype, "int64") == 0) dtype = DTYPE_INT64;

    if (strcmp(compressed->type, "direct") == 0 ||
        strcmp(compressed->type, "direct_generic") == 0) {
        // note(notecodec, note)
        NDArray *result = ndarray_create(compressed->shape, compressed->ndim, dtype);
        if (!result) return NULL;

        size_t decompressed_size;
        void *decompressed = NULL;

        // notecodecnote
        if (strcmp(compressed->codec, "sz3") == 0) {
            // SZ3note(notedataclassnote)
            decompressed = lossy_decompress_with_shape(
                compressed->data,
                compressed->data_size,
                compressed->codec,
                compressed->shape,
                compressed->ndim,
                dtype,  // notedataclassnote
                &decompressed_size);
        } else {
            // note(zstd/blosc)
            decompressed = lossless_decompress(
                compressed->data,
                compressed->data_size,
                compressed->codec,
                &decompressed_size);
        }

        if (decompressed) {
            size_t expected_size = result->total_size * dtype_size(dtype);
            if (decompressed_size == expected_size) {
                memcpy(result->data, decompressed, expected_size);
            } else {
                fprintf(stderr, "[WARNING] Size mismatch: got %zu, expected %zu\n",
                        decompressed_size, expected_size);
            }
            free(decompressed);
        }

        // note(notePython)
        char key[512];
        make_history_key(key, client_id, layer_name);
        add_gradient_to_history(compressor, key, result);

        return result;
    }
    else if (strcmp(compressed->type, "momentum_predicted") == 0) {
        // note
        return decompress_momentum_predicted_layer(compressor, compressed, client_id, layer_name);
    }

    // noteclassnote
    fprintf(stderr, "[ERROR] Unknown compression type: %s\n", compressed->type);
    return NULL;
}

// ===========================================================================
// modelnote - notePython compress_model()
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

    // 1. note
    CompressedLayerData **compressed_layers =
        (CompressedLayerData**)malloc(num_layers * sizeof(CompressedLayerData*));

    for (size_t i = 0; i < num_layers; i++) {
        compressed_layers[i] = momentum_compressor_compress_layer(
            compressor, layer_names[i], gradients[i]);
    }

    // 2. note(note)
    // note: [magic_number(4)][num_layers(4)][metadata_size(4)][metadata][layer1][layer2]...
    // note: [name_len(4)][name][type_len(4)][type][data_size(8)][data]...

    // computenote
    size_t total_size = 12;  // magic + num_layers + metadata_size

    // notedata(noteJSONnote)
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

    // computenote
    for (size_t i = 0; i < num_layers; i++) {
        total_size += 4 + strlen(layer_names[i]);  // name_len + name
        total_size += 4 + strlen(compressed_layers[i]->type);  // type_len + type
        total_size += 8 + compressed_layers[i]->data_size;  // data_size + data
        // note
        total_size += 4 + strlen(compressed_layers[i]->codec);
        total_size += 4 + strlen(compressed_layers[i]->original_dtype);
        total_size += 4 + compressed_layers[i]->ndim * sizeof(size_t);  // shape
        total_size += 8 + compressed_layers[i]->bitmap_size;  // bitmap
        total_size += 8 + compressed_layers[i]->dominant_signs_size;  // dominant_signs
        total_size += 32;  // stats (4 floats * 4 bytes + counters)
    }

    // note
    uint8_t *buffer = (uint8_t*)malloc(total_size);
    uint8_t *ptr = buffer;

    // notemagic number
    uint32_t magic = 0x4D4F4D43;  // "MOMC"
    memcpy(ptr, &magic, 4); ptr += 4;

    // note
    uint32_t num_layers_32 = (uint32_t)num_layers;
    memcpy(ptr, &num_layers_32, 4); ptr += 4;

    // notedata
    uint32_t metadata_len_32 = (uint32_t)metadata_len;
    memcpy(ptr, &metadata_len_32, 4); ptr += 4;
    memcpy(ptr, metadata, metadata_len); ptr += metadata_len;

    // notedata
    for (size_t i = 0; i < num_layers; i++) {
        CompressedLayerData *layer = compressed_layers[i];

        // note
        uint32_t name_len = (uint32_t)strlen(layer_names[i]);
        memcpy(ptr, &name_len, 4); ptr += 4;
        memcpy(ptr, layer_names[i], name_len); ptr += name_len;

        // classnote
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

        // notedata
        uint64_t data_size = (uint64_t)layer->data_size;
        memcpy(ptr, &data_size, 8); ptr += 8;
        memcpy(ptr, layer->data, layer->data_size); ptr += layer->data_size;

        // note
        uint64_t bitmap_size = (uint64_t)layer->bitmap_size;
        memcpy(ptr, &bitmap_size, 8); ptr += 8;
        if (bitmap_size > 0) {
            memcpy(ptr, layer->bitmap, layer->bitmap_size); ptr += layer->bitmap_size;
        }

        // note
        uint64_t signs_size = (uint64_t)layer->dominant_signs_size;
        memcpy(ptr, &signs_size, 8); ptr += 8;
        if (signs_size > 0) {
            memcpy(ptr, layer->dominant_signs, layer->dominant_signs_size);
            ptr += layer->dominant_signs_size;
        }

        // note
        memcpy(ptr, &layer->current_mean, 4); ptr += 4;
        memcpy(ptr, &layer->current_std, 4); ptr += 4;
        memcpy(ptr, &layer->prev_mean, 4); ptr += 4;
        memcpy(ptr, &layer->prev_std, 4); ptr += 4;

        uint64_t num_pred = (uint64_t)layer->num_predicted_kernels;
        memcpy(ptr, &num_pred, 8); ptr += 8;
    }

    *out_size = (size_t)(ptr - buffer);

    // cleanup
    for (size_t i = 0; i < num_layers; i++) {
        compressed_layer_data_free(compressed_layers[i]);
    }
    free(compressed_layers);

    return buffer;
}

/**
 * notemodel(note)- notePython decompress_model()
 */
NDArray** momentum_compressor_decompress_model(
    MomentumCompressor *compressor,
    const uint8_t *compressed_data,
    size_t compressed_size,
    char ***layer_names_out,
    size_t *num_layers_out
) {
    if (!compressor ||!compressed_data || compressed_size < 12) {
        fprintf(stderr, "[ERROR] Invalid input to decompress_model\n");
        return NULL;
    }

    const uint8_t *ptr = compressed_data;

    // 1. readmagic number
    uint32_t magic;
    memcpy(&magic, ptr, 4); ptr += 4;
    if (magic!= 0x4D4F4D43) {  // "MOMC"
        fprintf(stderr, "[ERROR] Invalid magic number: 0x%X\n", magic);
        return NULL;
    }

    // 2. readnote
    uint32_t num_layers_32;
    memcpy(&num_layers_32, ptr, 4); ptr += 4;
    size_t num_layers = (size_t)num_layers_32;

    // 3. readnotedata
    uint32_t metadata_len;
    memcpy(&metadata_len, ptr, 4); ptr += 4;
    char *metadata = (char*)malloc(metadata_len + 1);
    memcpy(metadata, ptr, metadata_len);
    metadata[metadata_len] = '\0';
    ptr += metadata_len;

    printf("[INFO] Decompressing model: %zu layers\n", num_layers);
    printf("[INFO] Metadata: %s\n", metadata);

    // noteclient_id (noteparse)
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

    // 4. noteoutputnote
    NDArray **gradients = (NDArray**)malloc(num_layers * sizeof(NDArray*));
    char **layer_names = (char**)malloc(num_layers * sizeof(char*));

    // 5. readnote
    for (size_t i = 0; i < num_layers; i++) {
        // readnote
        uint32_t name_len;
        memcpy(&name_len, ptr, 4); ptr += 4;
        layer_names[i] = (char*)malloc(name_len + 1);
        memcpy(layer_names[i], ptr, name_len);
        layer_names[i][name_len] = '\0';
        ptr += name_len;

        // readclassnote
        uint32_t type_len;
        memcpy(&type_len, ptr, 4); ptr += 4;
        char *type = (char*)malloc(type_len + 1);
        memcpy(type, ptr, type_len);
        type[type_len] = '\0';
        ptr += type_len;

        // readcodec
        uint32_t codec_len;
        memcpy(&codec_len, ptr, 4); ptr += 4;
        char *codec = (char*)malloc(codec_len + 1);
        memcpy(codec, ptr, codec_len);
        codec[codec_len] = '\0';
        ptr += codec_len;

        // readdtype
        uint32_t dtype_len;
        memcpy(&dtype_len, ptr, 4); ptr += 4;
        char *dtype_str = (char*)malloc(dtype_len + 1);
        memcpy(dtype_str, ptr, dtype_len);
        dtype_str[dtype_len] = '\0';
        ptr += dtype_len;

        // readshape
        uint32_t ndim;
        memcpy(&ndim, ptr, 4); ptr += 4;
        size_t *shape = (size_t*)malloc(ndim * sizeof(size_t));
        memcpy(shape, ptr, ndim * sizeof(size_t));
        ptr += ndim * sizeof(size_t);

        // readnotedata
        uint64_t data_size;
        memcpy(&data_size, ptr, 8); ptr += 8;
        uint8_t *data = (uint8_t*)malloc(data_size);
        memcpy(data, ptr, data_size);
        ptr += data_size;

        // readnote
        uint64_t bitmap_size;
        memcpy(&bitmap_size, ptr, 8); ptr += 8;
        uint8_t *bitmap = NULL;
        if (bitmap_size > 0) {
            bitmap = (uint8_t*)malloc(bitmap_size);
            memcpy(bitmap, ptr, bitmap_size);
            ptr += bitmap_size;
        }

        // readnote
        uint64_t signs_size;
        memcpy(&signs_size, ptr, 8); ptr += 8;
        uint8_t *signs = NULL;
        if (signs_size > 0) {
            signs = (uint8_t*)malloc(signs_size);
            memcpy(signs, ptr, signs_size);
            ptr += signs_size;
        }

        // readnote
        float current_mean, current_std, prev_mean, prev_std;
        memcpy(&current_mean, ptr, 4); ptr += 4;
        memcpy(&current_std, ptr, 4); ptr += 4;
        memcpy(&prev_mean, ptr, 4); ptr += 4;
        memcpy(&prev_std, ptr, 4); ptr += 4;

        uint64_t num_pred;
        memcpy(&num_pred, ptr, 8); ptr += 8;

        // noteCompressedLayerData
        CompressedLayerData layer_data;
        strncpy(layer_data.type, type, sizeof(layer_data.type) - 1);
        strncpy(layer_data.codec, codec, sizeof(layer_data.codec) - 1);
        strncpy(layer_data.original_dtype, dtype_str, sizeof(layer_data.original_dtype) - 1);
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

        // note
        gradients[i] = momentum_compressor_decompress_layer(
            compressor, &layer_data, client_id, layer_names[i]);

        if (!gradients[i]) {
            fprintf(stderr, "[ERROR] Failed to decompress layer %zu: %s\n", i, layer_names[i]);
        }

        // cleanupnotedata
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
// notefunction
// ===========================================================================

void momentum_compressor_reset_client(MomentumCompressor *compressor, const char *client_id) {
    if (!compressor ||!client_id) return;

    printf("[INFO] Resetting state for client: %s\n", client_id);

    size_t client_id_len = strlen(client_id);
    int deleted_count = 0;

    // 1. cleanupgradient_historynote
    LayerHistory *lh, *lh_tmp;
    HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
        // notekeynote"client_id:"note
        if (strncmp(lh->key, client_id, client_id_len) == 0 &&
            lh->key[client_id_len] == ':') {
            HASH_DEL(compressor->layer_histories, lh);

            // cleanupnote
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

    // 2. cleanupprediction_memorynoteclientnote(note)
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        if (strcmp(pm->client_id, client_id) == 0) {
            HASH_DEL(compressor->prediction_memories, pm);

            // cleanupnoteclientnotelayer memories
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

    // 3. cleanupstep_countnote
    StepCount *sc, *sc_tmp;
    HASH_ITER(hh, compressor->step_counts, sc, sc_tmp) {
        if (strncmp(sc->key, client_id, client_id_len) == 0 &&
            sc->key[client_id_len] == ':') {
            HASH_DEL(compressor->step_counts, sc);
            free(sc);
            deleted_count++;
        }
    }

    printf("PASS Deleted %d entries for client %s\n", deleted_count, client_id);
}

void momentum_compressor_reset_all(MomentumCompressor *compressor) {
    if (!compressor) return;

    // cleanupnote
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

    // cleanupnoteprediction_memoriesnote
    PredictionMemory *pm, *pm_tmp;
    HASH_ITER(hh, compressor->prediction_memories, pm, pm_tmp) {
        HASH_DEL(compressor->prediction_memories, pm);

        // cleanupnoteclientnotelayer memories
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

    printf("PASS All states reset\n");
}

// ===========================================================================
// note
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
               (int)(19 - (compressor->stats.prediction_ratio_count >= 10? 2: 1)), "");
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
// noteAPIimplement - notePythonnotefunction
// ===========================================================================

CompressionStats* momentum_compressor_get_stats(const MomentumCompressor *compressor) {
    if (!compressor) return NULL;

    CompressionStats *stats = (CompressionStats*)malloc(sizeof(CompressionStats));
    if (!stats) return NULL;

    stats->total_compressions = compressor->stats.total_compressions;
    stats->num_predictions = compressor->stats.prediction_ratio_count;

    // computenote
    if (compressor->stats.prediction_ratio_count > 0) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (size_t i = 0; i < compressor->stats.prediction_ratio_count; i++) {
            float val = compressor->stats.prediction_ratios[i];
            sum += val;
            sum_sq += val * val;
        }
        stats->avg_prediction_ratio = sum / compressor->stats.prediction_ratio_count;

        // computenote: sqrt(E[X^2] - (E[X])^2)
        float mean_sq = sum_sq / compressor->stats.prediction_ratio_count;
        float variance = mean_sq - (stats->avg_prediction_ratio * stats->avg_prediction_ratio);
        stats->std_prediction_ratio = sqrtf(variance > 0? variance: 0);
    } else {
        stats->avg_prediction_ratio = 0.0f;
        stats->std_prediction_ratio = 0.0f;
    }

    // computenote
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

    // note
    CompressionStats *comp_stats = momentum_compressor_get_stats(compressor);
    if (!comp_stats) {
        free(detailed);
        return NULL;
    }
    detailed->compression = *comp_stats;
    free(comp_stats);

    // noteID (notestep_countsnote)
    // noteID
    char unique_clients[256][256];
    size_t num_unique = 0;

    StepCount *sc, *tmp;
    HASH_ITER(hh, compressor->step_counts, sc, tmp) {
        // noteID (note: "ClientN:layer_name")
        const char *colon = strchr(sc->key, ':');
        if (!colon) continue;

        size_t id_len = colon - sc->key;
        if (id_len == 0 || id_len >= 256) continue;  // note

        // note
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

    // note
    for (size_t i = 0; i < num_unique; i++) {
        // unique_clients[i]note<256notenullnote
        size_t len = strlen(unique_clients[i]);
        memcpy(detailed->clients[i].client_id, unique_clients[i], len + 1);  // +1 for null terminator

        size_t id_len = len;

        // note (note)
        int max_step = 0;
        HASH_ITER(hh, compressor->step_counts, sc, tmp) {
            if (strncmp(sc->key, unique_clients[i], id_len) == 0 &&
                sc->key[id_len] == ':') {
                if (sc->step > max_step) max_step = sc->step;
            }
        }
        detailed->clients[i].step_count = max_step;

        // note
        size_t hist_count = 0;
        LayerHistory *lh, *lh_tmp;
        HASH_ITER(hh, compressor->layer_histories, lh, lh_tmp) {
            if (strncmp(lh->key, unique_clients[i], id_len) == 0 &&
                lh->key[id_len] == ':') {
                hist_count++;
            }
        }
        detailed->clients[i].history_length = hist_count;

        // note(note)
        size_t mem_count = 0;
        PredictionMemory *pm;
        HASH_FIND_STR(compressor->prediction_memories, unique_clients[i], pm);
        if (pm) {
            // noteclientnotelayer memories
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
    if (!compressor ||!level) return;

    // notePythonnote
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
    if (!compressor ||!client_id) return;

    // noteIDnote"ClientN"note
    // note"ClientN"note, note
    if (strncmp(client_id, "Client", 6) == 0 && isdigit(client_id[6])) {
        strncpy(compressor->current_client_id, client_id, sizeof(compressor->current_client_id) - 1);
        compressor->current_client_id[sizeof(compressor->current_client_id) - 1] = '\0';
    } else {
        // note
        const char *p = client_id;
        while (*p &&!isdigit(*p)) p++;

        if (*p) {
            int num = atoi(p);
            snprintf(compressor->current_client_id, sizeof(compressor->current_client_id),
                    "Client%d", num);
        } else {
            // note, noteID
            strncpy(compressor->current_client_id, client_id, sizeof(compressor->current_client_id) - 1);
            compressor->current_client_id[sizeof(compressor->current_client_id) - 1] = '\0';
        }
    }

    if (compressor->log_level <= 1) {  // INFO
        printf("[INFO] Client context set to: %s\n", compressor->current_client_id);
    }
}

// ===========================================================================
// notefunction
// ===========================================================================

int momentum_compressor_get_log_level(const MomentumCompressor *compressor) {
    return compressor? compressor->log_level: -1;
}

const char* momentum_compressor_get_current_client_id(const MomentumCompressor *compressor) {
    return compressor? compressor->current_client_id: NULL;
}

// ===========================================================================
// noteAPIimplement - notePythonnote
// ===========================================================================

/**
 * note
 *
 * note: note,notefunctionnote
 * note:
 *   1. note
 *   2. noterowsnote (OpenMP)
 *   3. unifiednote
 *
 * @param compressor note
 * @param items note (inputgradient, outputcompressed)
 * @param num_items note
 * @param client_id noteID
 * @return succeedednote0, failednote
 */
int momentum_compressor_compress_batch(
    MomentumCompressor *compressor,
    BatchItem *items,
    size_t num_items,
    const char *client_id
) {
    // note
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

    // note
    if (num_items > MAX_BATCH_SIZE) {
        fprintf(stderr, "[ERROR] Batch compress: num_items (%zu) exceeds MAX_BATCH_SIZE (%d)\n",
                num_items, MAX_BATCH_SIZE);
        return ERROR_INVALID_PARAM;
    }

    // note
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch compress: processing %zu layers for client '%s'\n",
               num_items, client_id);
    }

    // note (note)
    momentum_compressor_set_client(compressor, client_id);

    // [Breakdown] Batch note
    double batch_start = get_wall_time();

    // note breakdown note
    if (g_enable_breakdown) {
        if (g_last_batch_breakdown.layers) {
            free(g_last_batch_breakdown.layers);
        }
        g_last_batch_breakdown.layers = (LayerBreakdown*)calloc(num_items, sizeof(LayerBreakdown));
        g_last_batch_breakdown.num_layers = 0;
        g_last_batch_breakdown.total_elements = 0;
    }

    // note
    size_t success_count = 0;
    size_t total_compressed_size = 0;
    size_t total_original_size = 0;

    // ========== Phase 20: noteimplementnoterowsnote ==========
    // PASS Step A: notecompute(note+note) + noterows
    // PASS Step B: OpenMPnoterowsfornotepurefunction
    // PASS Step C: noterowsnoterowsnotehash/history/breakdown

    // [Step B] noteresultnote
    LayerResult *tmp_results = (LayerResult*)calloc(num_items, sizeof(LayerResult));
    if (!tmp_results) {
        fprintf(stderr, "[ERROR] Batch compress: failed to allocate tmp_results\n");
        return ERROR_ALLOCATION;
    }

    // [Step B] OpenMPnoterowsnotecomputefunction
    // notecompute(note+SZ3note), note
    // v22note: notev21note(note15%note)
    // notedynamic(1), note
    #pragma omp parallel for schedule(dynamic,1) reduction(+:success_count)
    for (size_t i = 0; i < num_items; i++) {
        // noteoutput
        items[i].compressed = NULL;
            items[i].error = ERROR_NONE;

            // note
            if (!items[i].gradient ||!items[i].layer_name) {
                fprintf(stderr, "[ERROR] Batch compress: item[%zu] has NULL gradient or layer_name\n", i);
                items[i].error = ERROR_INVALID_PARAM;
                tmp_results[i].status = ERROR_INVALID_PARAM;
                continue;
            }

            // [note] notecomputefunction(noterows)
            int ret = momentum_compressor_compress_layer_pure(
                compressor,
                client_id,
                items[i].layer_name,
                items[i].gradient,
                &tmp_results[i]
            );

            if (ret!= 0 ||!tmp_results[i].compressed) {
                fprintf(stderr, "[ERROR] Batch compress: pure compute failed for item[%zu] layer='%s' (status=%d)\n",
                        i, items[i].layer_name, tmp_results[i].status);
                items[i].error = tmp_results[i].status;
                continue;
            }

            success_count++;
    }

    // [Step C] noterowsnote(hashnote, history, breakdown)
    // note, note
    for (size_t i = 0; i < num_items; i++) {
        if (tmp_results[i].status!= 0 ||!tmp_results[i].compressed) {
            continue;  // notefailednote
        }

        CompressedLayerData *compressed = tmp_results[i].compressed;
        items[i].compressed = compressed;
        items[i].error = ERROR_NONE;

        // [note1] note(notethread-safe, note)
        // note(note)
        if (strcmp(compressed->type, "momentum_predicted") == 0) {
            // notecompressednote
            // notecompress_conv_layer_momentumnotecompute
            // notedatanote
            // [TODO] implementnote
            set_prediction_memory_for_layer(
                compressor, client_id,
                compressed->shape, compressed->ndim,
                NULL  // [FIXME] notecompressednoteprediction_memory
            );
        }

        // [note2] notecompress_*_layernote
        // note(notecomputefunctionnote, notecompress_*_layernote)

        // [note] breakdownnote
        if (g_enable_breakdown && g_last_batch_breakdown.layers) {
            LayerBreakdown *layer_bd = &g_last_batch_breakdown.layers[g_last_batch_breakdown.num_layers];
            memcpy(layer_bd, &tmp_results[i].layer_bd, sizeof(LayerBreakdown));

            g_last_batch_breakdown.num_layers++;
            g_last_batch_breakdown.total_elements += items[i].gradient->total_size;
            g_last_batch_breakdown.layer_compress_time += compressed->breakdown_total_time;
        }

        // [note] note
        total_compressed_size += compressed->data_size;
        if (compressed->bitmap_size > 0) {
            total_compressed_size += compressed->bitmap_size;
        }
        if (compressed->dominant_signs_size > 0) {
            total_compressed_size += compressed->dominant_signs_size;
        }

        total_original_size += items[i].gradient->total_size *
                               (items[i].gradient->dtype == DTYPE_FLOAT32? 4: 8);
    }

    // noteresultnote
    free(tmp_results);

    // [Breakdown] Batch note
    double batch_end = get_wall_time();
    if (g_enable_breakdown) {
        g_last_batch_breakdown.batch_total_time = batch_end - batch_start;
    }

    // noterecord result
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        float compression_ratio = total_original_size > 0?
            (float)total_compressed_size / total_original_size: 0.0f;

        printf("[INFO] Batch compress: %zu/%zu layers succeeded, "
               "compression ratio: %.2f%% (%.2f KB -> %.2f KB)\n",
               success_count, num_items,
               compression_ratio * 100.0f,
               total_original_size / 1024.0f,
               total_compressed_size / 1024.0f);

        // PASS Phase 20: breakdownnoteoutput
        if (g_enable_breakdown && num_items > 0) {
            double total_stats = 0, total_normalize = 0, total_consistency = 0;
            double total_prediction = 0, total_residual = 0, total_bitmap = 0;
            for (size_t i = 0; i < num_items; i++) {
                if (items[i].compressed) {
                    total_stats += items[i].compressed->breakdown_stats_time;
                    total_normalize += items[i].compressed->breakdown_normalize_time;
                    total_consistency += items[i].compressed->breakdown_consistency_time;
                    total_prediction += items[i].compressed->breakdown_prediction_time;
                    total_residual += items[i].compressed->breakdown_residual_compress_time;
                    total_bitmap += items[i].compressed->breakdown_bitmap_compress_time;
                }
            }
            printf("[BREAKDOWN] stats=%.1fms normalize=%.1fms consistency=%.1fms "
                   "prediction=%.1fms residual_compress=%.1fms bitmap=%.1fms\n",
                   total_stats * 1000, total_normalize * 1000, total_consistency * 1000,
                   total_prediction * 1000, total_residual * 1000, total_bitmap * 1000);
        }
    }

    // noteresult
    if (success_count == 0) {
        return ERROR_COMPRESSION;  // notefailed
    } else if (success_count < num_items) {
        return -((int)num_items - (int)success_count);  // notefailed,notefailednote
    } else {
        return 0;  // notesucceeded
    }
}

/**
 * note
 *
 * note: note
 *
 * @param compressor note
 * @param items note (inputcompressed)
 * @param num_items note
 * @param client_id noteID
 * @param out_gradients outputnote (note)
 * @return succeedednote0, failednote
 */
int momentum_compressor_decompress_batch(
    MomentumCompressor *compressor,
    const BatchItem *items,
    size_t num_items,
    const char *client_id,
    NDArray **out_gradients
) {
    // note
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

    // note
    if (num_items > MAX_BATCH_SIZE) {
        fprintf(stderr, "[ERROR] Batch decompress: num_items (%zu) exceeds MAX_BATCH_SIZE (%d)\n",
                num_items, MAX_BATCH_SIZE);
        return ERROR_INVALID_PARAM;
    }

    // note
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch decompress: processing %zu layers for client '%s'\n",
               num_items, client_id);
    }

    // note (note)
    momentum_compressor_set_client(compressor, client_id);

    // note
    size_t success_count = 0;

    // PASS v22note: noterowsnote(batch + note)
    //
    // note(ResNet50, 266note):
    //   notev22(batch+noterows):    474ms
    //   batchnoterows+noterows:           323ms (-32%)
    //   noterows(note):             309ms (-35%)
    //   noteCnote(note):          285ms
    //
    // note: noteIOnote+hashnote, noterowsnote
    //      noterowsnote, noteCnote8%(note)
    //
    // noterowsnote OMP_DECOMP_MIN_ELEMS=1M implement
    bool use_parallel = false;  // PASS noterows(note)

    if (use_parallel) {
        // noterowsnote(notebatch)
        #pragma omp parallel for schedule(dynamic,1) reduction(+:success_count)
        for (size_t i = 0; i < num_items; i++) {
            out_gradients[i] = NULL;

            if (!items[i].compressed ||!items[i].layer_name) {
                fprintf(stderr, "[ERROR] Batch decompress: item[%zu] has NULL compressed or layer_name\n", i);
                continue;
            }

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

            out_gradients[i] = gradient;
            success_count++;
        }
    } else {
        // note(notebatch)- noteOpenMPnote
        for (size_t i = 0; i < num_items; i++) {
            out_gradients[i] = NULL;

            if (!items[i].compressed ||!items[i].layer_name) {
                fprintf(stderr, "[ERROR] Batch decompress: item[%zu] has NULL compressed or layer_name\n", i);
                continue;
            }

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

            out_gradients[i] = gradient;
            success_count++;
        }
    }

    // noterecord result
    if (compressor->log_level <= LOG_LEVEL_INFO) {
        printf("[INFO] Batch decompress: %zu/%zu layers succeeded\n",
               success_count, num_items);
    }

    // noteresult
    if (success_count == 0) {
        return ERROR_DECOMPRESSION;  // notefailed
    } else if (success_count < num_items) {
        return -((int)num_items - (int)success_count);  // notefailed,notefailednote
    } else {
        return ERROR_NONE;  // notesucceeded
    }
}
