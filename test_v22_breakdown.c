// test_v22_breakdown.c - v22sz3 note
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

// noteclassnotefunction(notefile)
typedef enum { DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32, DTYPE_INT64 } DataType;

typedef enum {
    ERROR_NONE = 0,
    ERROR_ALLOCATION = -1,
    ERROR_COMPRESSION = -2,
    ERROR_DECOMPRESSION = -3,
    ERROR_INVALID_PARAM = -4,
    ERROR_NOT_FOUND = -5,
    ERROR_SZ3_FAILED = -6
} ErrorCode;

typedef struct {
    void *data;
    size_t *shape;
    size_t ndim;
    DataType dtype;
    size_t total_size;
} NDArray;

// CompressedLayerDatanote(note)
typedef struct {
    char type[32];
    char codec[32];
    size_t shape[4];
    size_t ndim;
    int step;
    char original_dtype[16];
    char stored_dtype[16];
    float current_mean, current_std, prev_mean, prev_std;
    float global_min, global_max;
    int num_predicted_kernels;
    float prediction_ratio;
    void *data;
    size_t data_size;
    void *bitmap;
    size_t bitmap_size;
    void *dominant_signs;
    size_t dominant_signs_size;
    double breakdown_total_time;
    double breakdown_stats_time;
    double breakdown_normalize_time;
    double breakdown_consistency_time;
    double breakdown_prediction_time;
    double breakdown_residual_compress_time;
    double breakdown_bitmap_compress_time;
} CompressedLayerData;

typedef struct {
    double stats_time;
    double normalize_time;
    double consistency_time;
    double sign_consistency_compute_time;
    double dominant_sign_compute_time;
    double prediction_time;
    double prediction_hash_time;
    double magnitude_predictor_time;
    double sign_predictor_time;
    double residual_compute_time;
    double residual_compress_time;
    double sz3_compress_time;
    double zstd_lossless_time;
    double bitmap_compress_time;
    double bitmap_generation_time;
    double bitmap_pack_time;
    double bitmap_zstd_time;
    double dominant_signs_pack_time;
    double dominant_signs_zstd_time;
    double metadata_time;
    double total_time;
    size_t layer_size;
    char layer_name[256];
    int timing_point_count;
    int uses_momentum;
    double compute_time;
    double io_time;
    double hash_time;
} LayerBreakdown;

typedef struct {
    double batch_total_time;
    double layer_compress_time;
    size_t num_layers;
    size_t total_elements;
    LayerBreakdown *layers;
} BatchBreakdown;

typedef struct {
    float momentum_lr;
    float error_bound;
    float consistency_threshold;
    size_t param_count_threshold;
    const char *lossless_compressor;
    const char *error_bounding_mode;
} MomentumCompressorConfig;

typedef struct MomentumCompressor MomentumCompressor;

typedef struct {
    size_t num_layers;
    size_t total_compressed_size;
    void *data;  // note
} CompressedBatch;

// BatchItemnote
typedef struct {
    const char *layer_name;              // note
    const NDArray *gradient;             // inputnote (note)
    CompressedLayerData *compressed;     // noteresult (output/input)
    ErrorCode error;                     // note
} BatchItem;

// notefunctionnote
extern MomentumCompressor* momentum_compressor_create(const MomentumCompressorConfig *config);
extern void momentum_compressor_destroy(MomentumCompressor *mc);
extern void momentum_compressor_enable_breakdown(int enable);
extern const BatchBreakdown* momentum_compressor_get_last_breakdown();
extern int momentum_compressor_compress_batch(MomentumCompressor *mc, BatchItem *items, size_t num_items, const char *client_id);
extern NDArray* ndarray_create(const size_t *shape, size_t ndim, DataType dtype);
extern void ndarray_destroy(NDArray *arr);

// notetest_v21_full.cnotefunction
NDArray* load_npy(const char* filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filepath);
        return NULL;
    }

    char magic[6];
    fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6)!= 0) {
        fprintf(stderr, "Invalid NPY magic\n");
        fclose(f);
        return NULL;
    }

    unsigned char major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);

    unsigned short header_len;
    fread(&header_len, 2, 1, f);

    char *header = malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    char *descr_start = strstr(header, "'descr':");
    char dtype_char = 'f';
    int dtype_size = 4;
    if (descr_start) {
        char *dtype_str = strchr(descr_start, '\'');
        if (dtype_str) {
            dtype_str++;
            if (*dtype_str == '<' || *dtype_str == '>') dtype_str++;
            dtype_char = *dtype_str;
            dtype_size = atoi(dtype_str + 1);
        }
    }

    DataType dtype;
    if (dtype_char == 'f' && dtype_size == 4) dtype = DTYPE_FLOAT32;
    else if (dtype_char == 'f' && dtype_size == 8) dtype = DTYPE_FLOAT64;
    else if (dtype_char == 'i' && dtype_size == 4) dtype = DTYPE_INT32;
    else if (dtype_char == 'i' && dtype_size == 8) dtype = DTYPE_INT64;
    else {
        fprintf(stderr, "Unsupported dtype\n");
        free(header);
        fclose(f);
        return NULL;
    }

    char *shape_start = strstr(header, "'shape':");
    if (!shape_start) {
        fprintf(stderr, "No shape in header\n");
        free(header);
        fclose(f);
        return NULL;
    }

    size_t shape[8];
    size_t ndim = 0;
    size_t total_size = 1;
    char *p = strchr(shape_start, '(');
    if (p) {
        p++;
        while (*p && *p!= ')') {
            if (*p >= '0' && *p <= '9') {
                shape[ndim] = strtoul(p, &p, 10);
                total_size *= shape[ndim];
                ndim++;
            } else {
                p++;
            }
        }
    }

    free(header);

    size_t dtype_bytes = (dtype == DTYPE_FLOAT32 || dtype == DTYPE_INT32)? 4: 8;
    void *data = malloc(total_size * dtype_bytes);
    size_t read_count = fread(data, dtype_bytes, total_size, f);
    fclose(f);

    if (read_count!= total_size) {
        fprintf(stderr, "Read size mismatch\n");
        free(data);
        return NULL;
    }

    NDArray *arr = ndarray_create(shape, ndim, dtype);
    if (!arr) {
        free(data);
        return NULL;
    }
    memcpy(arr->data, data, total_size * dtype_bytes);
    free(data);
    return arr;
}

// note.binfile
NDArray* load_bin(const char* filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filepath);
        return NULL;
    }

    // notefilenote
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // notefloat32classnote
    size_t num_elements = file_size / sizeof(float);

    // readdata
    float *data = malloc(file_size);
    size_t read_count = fread(data, sizeof(float), num_elements, f);
    fclose(f);

    if (read_count!= num_elements) {
        fprintf(stderr, "Read size mismatch in %s\n", filepath);
        free(data);
        return NULL;
    }

    // note1Ddatanote4Dnote(note)
    // note [out_ch, in_ch, h, w] note
    // note
    size_t out_ch = 64;  // outputnote
    size_t in_ch = 64;   // inputnote
    size_t kernel_size = 3; // 3x3 note

    // note out_ch notedatanote
    size_t expected_size = out_ch * in_ch * kernel_size * kernel_size;
    if (expected_size > num_elements) {
        // datanote, note out_ch
        out_ch = num_elements / (in_ch * kernel_size * kernel_size);
        if (out_ch == 0) out_ch = 1;
    } else if (expected_size < num_elements) {
        // datanote, note out_ch
        out_ch = num_elements / (in_ch * kernel_size * kernel_size);
    }

    size_t shape[4] = {out_ch, in_ch, kernel_size, kernel_size};
    size_t actual_size = out_ch * in_ch * kernel_size * kernel_size;

    // note, note
    if (actual_size > num_elements) {
        // note1Dnote
        size_t shape_1d[1] = {num_elements};
        NDArray *arr = ndarray_create(shape_1d, 1, DTYPE_FLOAT32);
        if (!arr) {
            free(data);
            return NULL;
        }
        memcpy(arr->data, data, file_size);
        free(data);
        return arr;
    }

    // note4Dnote
    NDArray *arr = ndarray_create(shape, 4, DTYPE_FLOAT32);
    if (!arr) {
        free(data);
        return NULL;
    }

    // notedatanoteNDArray(note)
    size_t copy_size = actual_size * sizeof(float);
    if (copy_size > file_size) copy_size = file_size;
    memcpy(arr->data, data, copy_size);
    free(data);

    return arr;
}

typedef struct {
    char name[256];
    NDArray *data;
} LayerData;

int load_round_data(const char *dir, int round_num, LayerData **out_layers, size_t *out_count) {
    // notedirectorynote (round_0/, round_1/,...)
    char round_path[512];
    snprintf(round_path, sizeof(round_path), "%s/round_%d", dir, round_num);

    DIR *d = opendir(round_path);
    if (d) {
        // notedirectorynote, note.npynote
        size_t capacity = 300;
        size_t count = 0;
        LayerData *layers = malloc(capacity * sizeof(LayerData));

        struct dirent *entry;
        while ((entry = readdir(d))!= NULL) {
            if (strstr(entry->d_name, ".npy") == NULL) continue;
            if (strstr(entry->d_name, "._")!= NULL) continue;

            char filepath[768];
            snprintf(filepath, sizeof(filepath), "%s/%s", round_path, entry->d_name);

            NDArray *arr = load_npy(filepath);
            if (arr) {
                strncpy(layers[count].name, entry->d_name, 255);
                char *dot = strrchr(layers[count].name, '.');
                if (dot) *dot = '\0';
                layers[count].data = arr;
                count++;

                if (count >= capacity) {
                    capacity *= 2;
                    layers = realloc(layers, capacity * sizeof(LayerData));
                }
            }
        }
        closedir(d);

        *out_layers = layers;
        *out_count = count;
        return 0;
    }

    // notedirectorynote, note round_X_client_0.bin file
    char bin_path[512];
    snprintf(bin_path, sizeof(bin_path), "%s/round_%d_client_0.bin", dir, round_num);

    struct stat st;
    if (stat(bin_path, &st)!= 0) {
        fprintf(stderr, "Cannot find data for round %d (tried %s and %s)\n",
                round_num, round_path, bin_path);
        return -1;
    }

    // note.binfile
    NDArray *arr = load_bin(bin_path);
    if (!arr) {
        fprintf(stderr, "Failed to load %s\n", bin_path);
        return -1;
    }

    // note
    LayerData *layers = malloc(sizeof(LayerData));
    // note "weight" notepathnote breakdown
    snprintf(layers[0].name, sizeof(layers[0].name), "conv_weight_round_%d_client_0", round_num);
    layers[0].data = arr;

    *out_layers = layers;
    *out_count = 1;
    return 0;
}

void print_breakdown_summary(const BatchBreakdown *bd) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                         BREAKDOWN note                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // note
    double total_stats = 0, total_norm = 0, total_cons = 0;
    double total_pred = 0, total_residual = 0, total_bitmap = 0;
    double total_sz3 = 0, total_zstd = 0;
    int momentum_count = 0, direct_count = 0;

    for (size_t i = 0; i < bd->num_layers; i++) {
        const LayerBreakdown *lb = &bd->layers[i];
        total_stats += lb->stats_time;
        total_norm += lb->normalize_time;
        total_cons += lb->consistency_time;
        total_pred += lb->prediction_time;
        total_residual += lb->residual_compress_time;
        total_bitmap += lb->bitmap_compress_time;
        total_sz3 += lb->sz3_compress_time;
        total_zstd += lb->zstd_lossless_time;

        if (lb->uses_momentum) momentum_count++;
        else direct_count++;
    }

    double batch_total = bd->batch_total_time * 1000; // notems

    printf("note:\n");
    printf("  note: %.3f ms\n", batch_total);
    printf("  note: %zu (note: %d, note: %d)\n",
           bd->num_layers, momentum_count, direct_count);
    printf("  note: %.2fM\n\n", bd->total_elements / 1e6);

    printf("stagenote:\n");
    printf("┌────────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ stage                       │ note(ms) │ note(%%  ) │ note │\n");
    printf("├────────────────────────────┼──────────┼──────────┼──────────┤\n");

    #define PRINT_PHASE(name, time) \
        printf("│ %-26s │ %8.2f │ %8.1f │ %8.3f │\n", \
               name, (time)*1000, (time)*1000/batch_total*100, (time)*1000/bd->num_layers)

    PRINT_PHASE("1. notecompute", total_stats);
    PRINT_PHASE("2. note", total_norm);
    PRINT_PHASE("3. note", total_cons);
    PRINT_PHASE("4. note", total_pred);
    PRINT_PHASE("5. note(noteSZ3)", total_residual);
    PRINT_PHASE("6. note", total_bitmap);

    printf("├────────────────────────────┼──────────┼──────────┼──────────┤\n");

    double compute_time = total_stats + total_norm + total_cons + total_pred;
    double io_time = total_sz3 + total_zstd;

    PRINT_PHASE("notecomputenote", compute_time);
    PRINT_PHASE("IOnote(SZ3+ZSTD)", io_time);

    printf("└────────────────────────────┴──────────┴──────────┴──────────┘\n\n");

    // note
    double total_sign_cons = 0, total_dom_sign = 0;
    double total_mag_pred = 0, total_sign_pred = 0, total_res_comp = 0;
    double total_bmp_gen = 0, total_bmp_pack = 0, total_bmp_zstd = 0;
    double total_dom_pack = 0, total_dom_zstd = 0;

    for (size_t i = 0; i < bd->num_layers; i++) {
        const LayerBreakdown *lb = &bd->layers[i];
        total_sign_cons += lb->sign_consistency_compute_time;
        total_dom_sign += lb->dominant_sign_compute_time;
        total_mag_pred += lb->magnitude_predictor_time;
        total_sign_pred += lb->sign_predictor_time;
        total_res_comp += lb->residual_compute_time;
        total_bmp_gen += lb->bitmap_generation_time;
        total_bmp_pack += lb->bitmap_pack_time;
        total_bmp_zstd += lb->bitmap_zstd_time;
        total_dom_pack += lb->dominant_signs_pack_time;
        total_dom_zstd += lb->dominant_signs_zstd_time;
    }

    printf("note:\n");
    printf("┌────────────────────────────────────────┬──────────┬──────────┐\n");
    printf("│ note                                   │ note(ms) │ note(%%)  │\n");
    printf("├────────────────────────────────────────┼──────────┼──────────┤\n");
    printf("│ note1: Sign Consistency Detector       │          │          │\n");
    PRINT_PHASE("  - notecompute", total_sign_cons);
    PRINT_PHASE("  - notecompute", total_dom_sign);
    printf("├────────────────────────────────────────┼──────────┼──────────┤\n");
    printf("│ note2: Momentum-Based Predictors       │          │          │\n");
    PRINT_PHASE("  - Magnitude Predictor", total_mag_pred);
    PRINT_PHASE("  - Sign Predictor", total_sign_pred);
    PRINT_PHASE("  - Residual Compute", total_res_comp);
    printf("├────────────────────────────────────────┼──────────┼──────────┤\n");
    printf("│ note3: Two-Level Bitmap Encoding       │          │          │\n");
    PRINT_PHASE("  - Bitmapgenerate", total_bmp_gen);
    PRINT_PHASE("  - Bitmapnote", total_bmp_pack);
    PRINT_PHASE("  - Bitmap ZSTDnote", total_bmp_zstd);
    PRINT_PHASE("  - note", total_dom_pack);
    PRINT_PHASE("  - noteZSTDnote", total_dom_zstd);
    printf("└────────────────────────────────────────┴──────────┴──────────┘\n\n");

    // notestagenote
    printf("note(SZ3/ZSTDnote):\n");
    printf("  SZ3note:  %.2f ms (%.1f%%)\n", total_sz3*1000, total_sz3*1000/batch_total*100);
    printf("  ZSTDnote: %.2f ms (%.1f%%)\n\n", total_zstd*1000, total_zstd*1000/batch_total*100);

    // noteclass
    printf("noteclass:\n");
    printf("  CPUnote(note): %.2f ms (%.1f%%)\n",
           compute_time*1000, compute_time*1000/batch_total*100);
    printf("  IOnote(note):    %.2f ms (%.1f%%)\n\n",
           io_time*1000, io_time*1000/batch_total*100);

    // Top 5 note
    printf("Top 5 note:\n");
    printf("┌─────┬──────────────────────────────┬──────────┬──────────┐\n");
    printf("│ note│ note                       │ note(ms) │ note(M)  │\n");
    printf("├─────┼──────────────────────────────┼──────────┼──────────┤\n");

    // notetop 5
    LayerBreakdown sorted[5];
    int top_count = 0;
    for (size_t i = 0; i < bd->num_layers && top_count < 5; i++) {
        sorted[top_count++] = bd->layers[i];
    }
    for (int i = 0; i < top_count; i++) {
        for (int j = i+1; j < top_count; j++) {
            if (sorted[j].total_time > sorted[i].total_time) {
                LayerBreakdown tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }
    for (size_t i = top_count; i < bd->num_layers; i++) {
        if (bd->layers[i].total_time > sorted[top_count-1].total_time) {
            sorted[top_count-1] = bd->layers[i];
            for (int j = top_count-1; j > 0; j--) {
                if (sorted[j].total_time > sorted[j-1].total_time) {
                    LayerBreakdown tmp = sorted[j];
                    sorted[j] = sorted[j-1];
                    sorted[j-1] = tmp;
                }
            }
        }
    }

    for (int i = 0; i < top_count; i++) {
        char short_name[30];
        strncpy(short_name, sorted[i].layer_name, 26);
        short_name[26] = '\0';
        printf("│  %d  │ %-28s │ %8.2f │ %8.2f │\n",
               i+1, short_name, sorted[i].total_time*1000, sorted[i].layer_size/1e6);
    }
    printf("└─────┴──────────────────────────────┴──────────┴──────────┘\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dataset_dir> [rounds]\n", argv[0]);
        return 1;
    }

    const char *dataset_dir = argv[1];
    int num_rounds = (argc > 2)? atoi(argv[2]): 3;

    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  v22sz3 Breakdown note\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("dataset: %s\n", dataset_dir);
    printf("note: %d\n", num_rounds);
    printf("══════════════════════════════════════════════════════════════════\n\n");

    // create compressor
    MomentumCompressorConfig config = {.momentum_lr = 0.07f,.error_bound = 1.0f,.consistency_threshold = 0.5f,.param_count_threshold = 1024,.lossless_compressor = "zstd",.error_bounding_mode = "REL"
    };

    MomentumCompressor *compressor = momentum_compressor_create(&config);
    if (!compressor) {
        fprintf(stderr, "create compressorfailed\n");
        return 1;
    }

    // notebreakdown
    momentum_compressor_enable_breakdown(1);

    // note
    for (int round = 0; round < num_rounds; round++) {
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Round %d\n", round);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // notedata
        LayerData *layers = NULL;
        size_t num_layers = 0;
        if (load_round_data(dataset_dir, round, &layers, &num_layers)!= 0) {
            fprintf(stderr, "note Round %d datafailed\n", round);
            continue;
        }

        printf("note %zu notedata\n", num_layers);

        // noteBatchItemnote
        BatchItem *items = (BatchItem*)calloc(num_layers, sizeof(BatchItem));
        if (!items) {
            fprintf(stderr, "noteBatchItemfailed\n");
            for (size_t i = 0; i < num_layers; i++) {
                ndarray_destroy(layers[i].data);
            }
            free(layers);
            continue;
        }

        // noteBatchItemnote
        for (size_t i = 0; i < num_layers; i++) {
            items[i].layer_name = layers[i].name;
            items[i].gradient = layers[i].data;
            items[i].compressed = NULL;
            items[i].error = ERROR_NONE;
        }

        // note
        int compress_result = momentum_compressor_compress_batch(
            compressor, items, num_layers, "Client0"
        );

        if (compress_result!= 0) {
            fprintf(stderr, "notefailed, note: %d\n", compress_result);
            for (size_t i = 0; i < num_layers; i++) {
                ndarray_destroy(layers[i].data);
            }
            free(layers);
            free(items);
            continue;
        }

        // noteresult
        size_t success_count = 0;
        size_t total_compressed_size = 0;
        for (size_t i = 0; i < num_layers; i++) {
            if (items[i].compressed && items[i].error == ERROR_NONE) {
                success_count++;
                total_compressed_size += items[i].compressed->data_size;
                if (items[i].compressed->bitmap_size > 0) {
                    total_compressed_size += items[i].compressed->bitmap_size;
                }
                if (items[i].compressed->dominant_signs_size > 0) {
                    total_compressed_size += items[i].compressed->dominant_signs_size;
                }
            }
        }

        printf("note: %zu/%zu notesucceeded, note: %.2f MB\n",
               success_count, num_layers, total_compressed_size / 1e6);

        // notebreakdown
        const BatchBreakdown *bd = momentum_compressor_get_last_breakdown();
        if (bd && bd->layers) {
            print_breakdown_summary(bd);
        }

        // cleanup
        for (size_t i = 0; i < num_layers; i++) {
            if (items[i].compressed) {
                // note: noteCompressedLayerData, noteAPInote
                // notefunction
            }
            ndarray_destroy(layers[i].data);
        }
        free(items);
        free(layers);
    }

    momentum_compressor_destroy(compressor);

    printf("\ntest complete!\n");
    return 0;
}
