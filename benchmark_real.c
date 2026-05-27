/**
 * 性能基准测试程序 - 测试ompv8和ompv15在不同线程数下的表现
 * 使用ResNet50真实数据
 */

#include "momentum_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

// 时间测量
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// 简化的二进制加载（复用test_c_real的代码）
typedef struct {
    char layer_name[256];
    size_t shape[8];
    size_t ndim;
    DataType dtype;
    size_t data_size;
    void* data;
} TestLayer;

TestLayer* load_test_data(const char* filename, size_t* num_layers) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return NULL;
    }
    
    uint32_t round_num;
    char client_id[64];
    uint64_t layer_count;
    
    if (fread(&round_num, sizeof(uint32_t), 1, f) != 1 ||
        fread(client_id, 64, 1, f) != 1 ||
        fread(&layer_count, sizeof(uint64_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    *num_layers = layer_count;
    TestLayer* layers = (TestLayer*)malloc(layer_count * sizeof(TestLayer));
    
    for (size_t i = 0; i < layer_count; i++) {
        TestLayer* layer = &layers[i];
        
        char layer_name[256];
        uint64_t shape[8];
        uint64_t ndim;
        uint32_t dtype;
        uint64_t data_size;
        
        if (fread(layer_name, 256, 1, f) != 1 ||
            fread(shape, sizeof(uint64_t), 8, f) != 8 ||
            fread(&ndim, sizeof(uint64_t), 1, f) != 1 ||
            fread(&dtype, sizeof(uint32_t), 1, f) != 1 ||
            fread(&data_size, sizeof(uint64_t), 1, f) != 1) {
            free(layers);
            fclose(f);
            return NULL;
        }
        
        strncpy(layer->layer_name, layer_name, 255);
        for (size_t d = 0; d < ndim && d < 8; d++) {
            layer->shape[d] = shape[d];
        }
        layer->ndim = ndim;
        layer->dtype = (DataType)dtype;
        layer->data_size = data_size;
        
        layer->data = malloc(data_size);
        if (fread(layer->data, data_size, 1, f) != 1) {
            free(layer->data);
            free(layers);
            fclose(f);
            return NULL;
        }
    }
    
    fclose(f);
    return layers;
}

void free_test_data(TestLayer* layers, size_t num_layers) {
    for (size_t i = 0; i < num_layers; i++) {
        if (layers[i].data) free(layers[i].data);
    }
    free(layers);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <round1.bin> [round2.bin] [round3.bin]\n", argv[0]);
        return 1;
    }
    
    // 创建压缩器
    CompressorConfig config = {
        .momentum_lr = 0.07f,
        .consistency_threshold = 0.5f,
        .lossless_compressor = "zstd",
        .error_bounding_mode = "ABS",
        .error_bound = 1.0f,
        .param_count_threshold = 1024,
        .max_history_length = 3
    };
    strcpy(config.sz3_lib_path, "/home/exouser/.appfl/.compressor/SZ3");
    
    MomentumCompressor* compressor = momentum_compressor_create(&config);
    if (!compressor) {
        fprintf(stderr, "Failed to create compressor\n");
        return 1;
    }
    
    momentum_compressor_set_client(compressor, "client_0");
    
    double total_time = 0.0;
    int num_files = argc - 1;
    
    // 处理每一轮数据
    for (int file_idx = 1; file_idx < argc; file_idx++) {
        const char* filename = argv[file_idx];
        
        size_t num_layers;
        TestLayer* layers = load_test_data(filename, &num_layers);
        if (!layers) {
            fprintf(stderr, "Failed to load %s\n", filename);
            continue;
        }
        
        // 转换为NDArray格式
        NDArray** arrays = (NDArray**)malloc(num_layers * sizeof(NDArray*));
        char** layer_names = (char**)malloc(num_layers * sizeof(char*));
        
        for (size_t i = 0; i < num_layers; i++) {
            NDArray* arr = (NDArray*)malloc(sizeof(NDArray));
            arr->shape = (size_t*)malloc(layers[i].ndim * sizeof(size_t));
            memcpy(arr->shape, layers[i].shape, layers[i].ndim * sizeof(size_t));
            arr->ndim = layers[i].ndim;
            arr->dtype = layers[i].dtype;
            arr->total_size = layers[i].data_size / sizeof(float);
            arr->data = layers[i].data;
            arrays[i] = arr;
            
            layer_names[i] = strdup(layers[i].layer_name);
        }
        
        // 压缩测试
        double t_start = get_time_ms();
        
        size_t out_size;
        uint8_t* compressed = momentum_compressor_compress_model(
            compressor,
            (const NDArray**)arrays,
            (const char**)layer_names,
            num_layers,
            "client_0",
            &out_size
        );
        
        double t_end = get_time_ms();
        double elapsed = t_end - t_start;
        total_time += elapsed;
        
        if (compressed) {
            free(compressed);
        }
        
        // 清理
        for (size_t i = 0; i < num_layers; i++) {
            free(arrays[i]->shape);
            free(arrays[i]);
            free(layer_names[i]);
        }
        free(arrays);
        free(layer_names);
        free_test_data(layers, num_layers);
    }
    
    // 输出结果（格式化便于脚本解析）
    printf("BENCHMARK_RESULT: %.2f ms\n", total_time / num_files);
    
    momentum_compressor_destroy(compressor);
    return 0;
}
