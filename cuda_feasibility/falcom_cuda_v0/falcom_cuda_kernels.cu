#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <vector>

#define CUDA_CHECK_THROW(expr)                                      \
    do {                                                            \
        cudaError_t err__ = (expr);                                 \
        TORCH_CHECK(err__ == cudaSuccess, cudaGetErrorString(err__)); \
    } while (0)

__global__ void falcom_momentum_kernel(
    const float *current,
    const float *prev_grad,
    const float *prediction_memory,
    float *residual,
    float *next_prediction_memory,
    unsigned char *bitmap_flags,
    signed char *dominant_signs,
    unsigned long long *counters,
    size_t total_kernels,
    size_t kernel_size,
    float current_mean,
    float current_std,
    float prev_mean,
    float prev_std,
    float global_min,
    float global_max,
    float momentum_lr,
    float consistency_threshold,
    long long step) {
    size_t kernel_idx = static_cast<size_t>(blockIdx.x);
    if (kernel_idx >= total_kernels) return;

    size_t offset = kernel_idx * kernel_size;
    size_t positives = 0;
    size_t negatives = 0;
    size_t zeros = 0;

    for (size_t k = 0; k < kernel_size; ++k) {
        float v = current[offset + k];
        if (v > 0.0f) {
            positives++;
        } else if (v < 0.0f) {
            negatives++;
        } else {
            zeros++;
        }
    }

    size_t majority = (positives >= negatives) ? (positives + zeros) : (negatives + zeros);
    float consistency = ((static_cast<float>(majority) / static_cast<float>(kernel_size)) - 0.5f) * 2.0f;
    signed char dom_sign = (positives >= negatives) ? static_cast<signed char>(1) : static_cast<signed char>(-1);
    bool predicted = (step > 1) && (consistency >= consistency_threshold);

    bitmap_flags[kernel_idx] = predicted ? 1 : 0;
    dominant_signs[kernel_idx] = predicted ? dom_sign : 0;

    unsigned long long local_mismatch = 0;
    unsigned long long local_predicted_elements = 0;
    float inv_prev_std = (prev_std > 1.0e-8f) ? (1.0f / prev_std) : 1.0f;
    bool use_prev_std = (prev_std > 1.0e-8f);
    bool use_current_std = (current_std > 1.0e-8f);
    float one_minus_lr = 1.0f - momentum_lr;

    for (size_t k = 0; k < kernel_size; ++k) {
        size_t idx = offset + k;
        float cur = current[idx];

        if (!predicted) {
            residual[idx] = cur;
            continue;
        }

        float prev_abs = fabsf(prev_grad[idx]);
        float prev_norm = prev_abs - prev_mean;
        if (use_prev_std) prev_norm *= inv_prev_std;

        float new_mem = one_minus_lr * prediction_memory[idx] + momentum_lr * prev_norm;
        next_prediction_memory[idx] = new_mem;

        float abs_pred = use_current_std ? (new_mem * current_std + current_mean) : (new_mem + current_mean);
        abs_pred = fabsf(abs_pred);
        float predicted_val = static_cast<float>(dom_sign) * abs_pred;

        float pred_sign = (predicted_val > 0.0f) ? 1.0f : -1.0f;
        float actual_sign = (cur > 0.0f) ? 1.0f : -1.0f;
        if (pred_sign * actual_sign < 0.0f) {
            local_mismatch++;
        }
        local_predicted_elements++;

        float r = cur - predicted_val;
        r = fminf(fmaxf(r, global_min), global_max);
        residual[idx] = r;
    }

    if (predicted) {
        atomicAdd(&counters[0], 1ULL);
        atomicAdd(&counters[1], local_mismatch);
        atomicAdd(&counters[2], local_predicted_elements);
    }
}

std::vector<torch::Tensor> momentum_pack_cuda(
    torch::Tensor current,
    torch::Tensor prev_grad,
    torch::Tensor prediction_memory,
    double current_mean,
    double current_std,
    double prev_mean,
    double prev_std,
    double global_min,
    double global_max,
    double momentum_lr,
    double consistency_threshold,
    int64_t step) {
    TORCH_CHECK(current.is_cuda(), "current must be a CUDA tensor");
    TORCH_CHECK(prev_grad.is_cuda(), "prev_grad must be a CUDA tensor");
    TORCH_CHECK(prediction_memory.is_cuda(), "prediction_memory must be a CUDA tensor");
    TORCH_CHECK(current.dtype() == torch::kFloat32, "current must be float32");
    TORCH_CHECK(prev_grad.dtype() == torch::kFloat32, "prev_grad must be float32");
    TORCH_CHECK(prediction_memory.dtype() == torch::kFloat32, "prediction_memory must be float32");
    TORCH_CHECK(current.is_contiguous(), "current must be contiguous");
    TORCH_CHECK(prev_grad.is_contiguous(), "prev_grad must be contiguous");
    TORCH_CHECK(prediction_memory.is_contiguous(), "prediction_memory must be contiguous");
    TORCH_CHECK(current.dim() == 4, "v0 only supports 4D tensors");
    TORCH_CHECK(prev_grad.sizes() == current.sizes(), "prev_grad shape mismatch");
    TORCH_CHECK(prediction_memory.sizes() == current.sizes(), "prediction_memory shape mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(current));

    size_t total_kernels = static_cast<size_t>(current.size(0) * current.size(1));
    size_t kernel_size = static_cast<size_t>(current.size(2) * current.size(3));
    TORCH_CHECK(total_kernels > 0 && kernel_size > 0, "invalid tensor shape");

    auto residual = torch::empty_like(current);
    auto next_prediction_memory = prediction_memory.clone();
    auto flags = torch::empty({static_cast<long long>(total_kernels)},
                              torch::TensorOptions().dtype(torch::kUInt8).device(current.device()));
    auto signs = torch::empty({static_cast<long long>(total_kernels)},
                              torch::TensorOptions().dtype(torch::kInt8).device(current.device()));
    auto counters = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64).device(current.device()));
    auto kernel_ms = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK_THROW(cudaEventCreate(&start));
    CUDA_CHECK_THROW(cudaEventCreate(&stop));
    CUDA_CHECK_THROW(cudaEventRecord(start, stream));

    falcom_momentum_kernel<<<static_cast<unsigned int>(total_kernels), 1, 0, stream>>>(
        current.data_ptr<float>(),
        prev_grad.data_ptr<float>(),
        prediction_memory.data_ptr<float>(),
        residual.data_ptr<float>(),
        next_prediction_memory.data_ptr<float>(),
        flags.data_ptr<unsigned char>(),
        signs.data_ptr<signed char>(),
        reinterpret_cast<unsigned long long *>(counters.data_ptr<int64_t>()),
        total_kernels,
        kernel_size,
        static_cast<float>(current_mean),
        static_cast<float>(current_std),
        static_cast<float>(prev_mean),
        static_cast<float>(prev_std),
        static_cast<float>(global_min),
        static_cast<float>(global_max),
        static_cast<float>(momentum_lr),
        static_cast<float>(consistency_threshold),
        static_cast<long long>(step));

    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaEventRecord(stop, stream));
    CUDA_CHECK_THROW(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK_THROW(cudaEventDestroy(start));
    CUDA_CHECK_THROW(cudaEventDestroy(stop));
    kernel_ms.data_ptr<double>()[0] = static_cast<double>(elapsed);

    return {residual, next_prediction_memory, flags, signs, counters, kernel_ms};
}

