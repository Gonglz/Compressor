#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_err));                            \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

__global__ void momentum_update_kernel(const float *cur, const float *prev, float *out,
                                       float lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lr * cur[i] + (1.0f - lr) * prev[i];
}

__global__ void residual_update_kernel(const float *cur, const float *pred, float *residual,
                                       float *next_pred, float lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] = cur[i] - pred[i];
        next_pred[i] = lr * cur[i] + (1.0f - lr) * pred[i];
    }
}

__global__ void threshold_scan_kernel(const float *x, unsigned char *mask, float threshold,
                                      size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mask[i] = fabsf(x[i]) > threshold ? 1 : 0;
}

__global__ void bitmap_pack_kernel(const unsigned char *mask, unsigned char *bitmap, size_t n) {
    size_t byte_i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = byte_i * 8;
    if (base >= n) return;
    unsigned char v = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        size_t j = base + static_cast<size_t>(b);
        if (j < n && mask[j]) v |= static_cast<unsigned char>(1u << b);
    }
    bitmap[byte_i] = v;
}

__global__ void sign_pack_kernel(const float *x, unsigned char *signs, size_t n) {
    size_t byte_i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = byte_i * 8;
    if (base >= n) return;
    unsigned char v = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        size_t j = base + static_cast<size_t>(b);
        if (j < n && x[j] >= 0.0f) v |= static_cast<unsigned char>(1u << b);
    }
    signs[byte_i] = v;
}

__global__ void reconstruct_kernel(const float *pred, const float *residual, float *out,
                                   size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = pred[i] + residual[i];
}

static double ms_since(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void fill_inputs(std::vector<float> &a, std::vector<float> &b) {
    for (size_t i = 0; i < a.size(); ++i) {
        int m = static_cast<int>(i % 1024);
        a[i] = (static_cast<float>(m) - 512.0f) * 0.001953125f;
        b[i] = (static_cast<float>((m * 17) % 1024) - 512.0f) * 0.001953125f;
    }
}

static double cpu_momentum(const std::vector<float> &a, const std::vector<float> &b,
                           std::vector<float> &out, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < a.size(); ++i) out[i] = 0.07f * a[i] + 0.93f * b[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

static double cpu_residual(const std::vector<float> &a, const std::vector<float> &b,
                           std::vector<float> &out, std::vector<float> &tmp, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] - b[i];
            tmp[i] = 0.07f * a[i] + 0.93f * b[i];
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

static double cpu_threshold(const std::vector<float> &a, std::vector<unsigned char> &mask,
                            int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < a.size(); ++i) mask[i] = std::fabs(a[i]) > 0.25f ? 1 : 0;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

static double cpu_bitmap(const std::vector<unsigned char> &mask, std::vector<unsigned char> &packed,
                         size_t n, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t byte_i = 0; byte_i < packed.size(); ++byte_i) {
            size_t base = byte_i * 8;
            unsigned char v = 0;
            for (int b = 0; b < 8; ++b) {
                size_t j = base + static_cast<size_t>(b);
                if (j < n && mask[j]) v |= static_cast<unsigned char>(1u << b);
            }
            packed[byte_i] = v;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

static double cpu_sign(const std::vector<float> &a, std::vector<unsigned char> &packed,
                       size_t n, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t byte_i = 0; byte_i < packed.size(); ++byte_i) {
            size_t base = byte_i * 8;
            unsigned char v = 0;
            for (int b = 0; b < 8; ++b) {
                size_t j = base + static_cast<size_t>(b);
                if (j < n && a[j] >= 0.0f) v |= static_cast<unsigned char>(1u << b);
            }
            packed[byte_i] = v;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

static double cpu_reconstruct(const std::vector<float> &a, const std::vector<float> &b,
                              std::vector<float> &out, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return ms_since(t0, t1) / iters;
}

enum KernelKind {
    MOMENTUM,
    RESIDUAL,
    THRESHOLD,
    BITMAP,
    SIGN,
    RECONSTRUCT,
};

static const char *kernel_name(KernelKind k) {
    switch (k) {
        case MOMENTUM: return "momentum_update";
        case RESIDUAL: return "residual_predmem_update";
        case THRESHOLD: return "threshold_scan";
        case BITMAP: return "bitmap_pack";
        case SIGN: return "dominant_sign_pack";
        case RECONSTRUCT: return "decompress_reconstruct";
    }
    return "unknown";
}

static double gpu_only_ms(KernelKind kind, const std::vector<float> &a,
                          const std::vector<float> &b,
                          const std::vector<unsigned char> &mask_host,
                          size_t n, size_t packed_n, int iters) {
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr, *d_tmp = nullptr;
    unsigned char *d_mask = nullptr, *d_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, n));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_n));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask_host.data(), n, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    int blocks_p = static_cast<int>((packed_n + threads - 1) / threads);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        switch (kind) {
            case MOMENTUM:
                momentum_update_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, 0.07f, n);
                break;
            case RESIDUAL:
                residual_update_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, d_tmp, 0.07f, n);
                break;
            case THRESHOLD:
                threshold_scan_kernel<<<blocks_n, threads>>>(d_a, d_mask, 0.25f, n);
                break;
            case BITMAP:
                bitmap_pack_kernel<<<blocks_p, threads>>>(d_mask, d_packed, n);
                break;
            case SIGN:
                sign_pack_kernel<<<blocks_p, threads>>>(d_a, d_packed, n);
                break;
            case RECONSTRUCT:
                reconstruct_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, n);
                break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = elapsed_ms(start, stop) / iters;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_tmp);
    cudaFree(d_mask);
    cudaFree(d_packed);
    return static_cast<double>(ms);
}

static double gpu_h2d_total_ms(KernelKind kind, const std::vector<float> &a,
                               const std::vector<float> &b,
                               const std::vector<unsigned char> &mask_host,
                               std::vector<float> &out,
                               std::vector<unsigned char> &packed,
                               size_t n, size_t packed_n, int iters) {
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr, *d_tmp = nullptr;
    unsigned char *d_mask = nullptr, *d_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, n));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_n));

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    int blocks_p = static_cast<int>((packed_n + threads - 1) / threads);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        if (kind == BITMAP) {
            CUDA_CHECK(cudaMemcpy(d_mask, mask_host.data(), n, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            if (kind == MOMENTUM || kind == RESIDUAL || kind == RECONSTRUCT) {
                CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
        switch (kind) {
            case MOMENTUM:
                momentum_update_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, 0.07f, n);
                CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
                break;
            case RESIDUAL:
                residual_update_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, d_tmp, 0.07f, n);
                CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
                break;
            case THRESHOLD:
                threshold_scan_kernel<<<blocks_n, threads>>>(d_a, d_mask, 0.25f, n);
                CUDA_CHECK(cudaMemcpy(packed.data(), d_mask, n, cudaMemcpyDeviceToHost));
                break;
            case BITMAP:
                bitmap_pack_kernel<<<blocks_p, threads>>>(d_mask, d_packed, n);
                CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed_n, cudaMemcpyDeviceToHost));
                break;
            case SIGN:
                sign_pack_kernel<<<blocks_p, threads>>>(d_a, d_packed, n);
                CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed_n, cudaMemcpyDeviceToHost));
                break;
            case RECONSTRUCT:
                reconstruct_kernel<<<blocks_n, threads>>>(d_a, d_b, d_out, n);
                CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_tmp);
    cudaFree(d_mask);
    cudaFree(d_packed);
    return ms_since(t0, t1) / iters;
}

static int launch_count(KernelKind kind) {
    return 1;
}

static double cpu_ms(KernelKind kind, const std::vector<float> &a, const std::vector<float> &b,
                     std::vector<float> &out, std::vector<float> &tmp,
                     std::vector<unsigned char> &mask,
                     std::vector<unsigned char> &packed, size_t n, int iters) {
    switch (kind) {
        case MOMENTUM: return cpu_momentum(a, b, out, iters);
        case RESIDUAL: return cpu_residual(a, b, out, tmp, iters);
        case THRESHOLD: return cpu_threshold(a, mask, iters);
        case BITMAP: return cpu_bitmap(mask, packed, n, iters);
        case SIGN: return cpu_sign(a, packed, n, iters);
        case RECONSTRUCT: return cpu_reconstruct(a, b, out, iters);
    }
    return 0.0;
}

int main(int argc, char **argv) {
    size_t n = 0;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--numel") == 0 && i + 1 < argc) {
            n = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }
    if (n == 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s --numel N [--iters I]\n", argv[0]);
        return 1;
    }

    size_t packed_n = (n + 7) / 8;
    std::vector<float> a(n), b(n), out(n), tmp(n);
    std::vector<unsigned char> mask(n), packed(std::max<size_t>(packed_n, n));
    fill_inputs(a, b);
    for (size_t i = 0; i < n; ++i) mask[i] = (i & 1) ? 1 : 0;
    CUDA_CHECK(cudaFree(nullptr));

    std::printf("kernel,numel,bytes,cpu_ms,gpu_only_ms,h2d_gpu_d2h_ms,launch_count,iters\n");
    KernelKind kinds[] = {MOMENTUM, RESIDUAL, THRESHOLD, BITMAP, SIGN, RECONSTRUCT};
    for (KernelKind kind : kinds) {
        double c_ms = cpu_ms(kind, a, b, out, tmp, mask, packed, n, iters);
        double g_ms = gpu_only_ms(kind, a, b, mask, n, packed_n, iters);
        double t_ms = gpu_h2d_total_ms(kind, a, b, mask, out, packed, n, packed_n, iters);
        size_t bytes = (kind == BITMAP) ? n : n * sizeof(float);
        std::printf("%s,%zu,%zu,%.6f,%.6f,%.6f,%d,%d\n",
                    kernel_name(kind), n, bytes, c_ms, g_ms, t_ms, launch_count(kind), iters);
    }
    return 0;
}
