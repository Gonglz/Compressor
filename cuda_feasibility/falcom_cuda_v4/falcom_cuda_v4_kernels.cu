#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#define CUDA_CHECK_THROW(expr)                                          \
    do {                                                                \
        cudaError_t err__ = (expr);                                     \
        TORCH_CHECK(err__ == cudaSuccess, cudaGetErrorString(err__));   \
    } while (0)

namespace {

constexpr int THREADS = 256;

static double host_ms_since(std::chrono::high_resolution_clock::time_point start,
                            std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void reduce_abs_residual_kernel(const float *current,
                                           const float *base,
                                           const uint8_t has_base,
                                           float *block_max,
                                           int64_t n) {
    __shared__ float smem[THREADS];
    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    float v = 0.0f;
    if (idx < n) {
        float residual = has_base ? (current[idx] - base[idx]) : current[idx];
        v = fabsf(residual);
    }
    smem[tid] = v;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = smem[0];
}

__global__ void quantize_q8_kernel(const float *current,
                                   const float *base,
                                   const uint8_t has_base,
                                   int8_t *out,
                                   float scale,
                                   int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float residual = has_base ? (current[idx] - base[idx]) : current[idx];
    float qf = scale > 0.0f ? nearbyintf(residual / scale) : 0.0f;
    qf = fminf(127.0f, fmaxf(-127.0f, qf));
    out[idx] = static_cast<int8_t>(qf);
}

__global__ void decode_q8_kernel(const int8_t *q,
                                 const float *base,
                                 const uint8_t has_base,
                                 float *out,
                                 float scale,
                                 int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float residual = static_cast<float>(q[idx]) * scale;
    out[idx] = has_base ? (base[idx] + residual) : residual;
}

static int64_t numel_from_shape(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (int64_t dim : shape) {
        TORCH_CHECK(dim > 0, "invalid non-positive shape dimension");
        n *= dim;
    }
    return n;
}

static void validate_cpu_1d(torch::Tensor t, torch::ScalarType dtype, const char *name) {
    TORCH_CHECK(!t.is_cuda(), name, " must be CPU");
    TORCH_CHECK(t.dtype() == dtype, name, " dtype mismatch");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, torch::Tensor>
encode_q8_batch_cuda(std::vector<torch::Tensor> currents,
                     std::vector<torch::Tensor> bases,
                     torch::Tensor has_base_flags) {
    TORCH_CHECK(!currents.empty(), "currents must not be empty");
    TORCH_CHECK(currents.size() == bases.size(), "currents/bases size mismatch");
    validate_cpu_1d(has_base_flags, torch::kUInt8, "has_base_flags");
    TORCH_CHECK(has_base_flags.numel() == static_cast<int64_t>(currents.size()),
                "has_base_flags length mismatch");

    const auto device = currents[0].device();
    TORCH_CHECK(device.is_cuda(), "currents must be CUDA tensors");
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t total_bytes = 0;
    std::vector<int64_t> lengths_host;
    lengths_host.reserve(currents.size());
    const uint8_t *flags_host = has_base_flags.data_ptr<uint8_t>();
    for (size_t i = 0; i < currents.size(); ++i) {
        TORCH_CHECK(currents[i].is_cuda(), "current must be CUDA");
        TORCH_CHECK(currents[i].device() == device, "all currents must share a device");
        TORCH_CHECK(currents[i].dtype() == torch::kFloat32, "current must be float32");
        TORCH_CHECK(currents[i].is_contiguous(), "current must be contiguous");
        if (flags_host[i]) {
            TORCH_CHECK(bases[i].is_cuda(), "base must be CUDA when has_base is true");
            TORCH_CHECK(bases[i].device() == device, "base device mismatch");
            TORCH_CHECK(bases[i].dtype() == torch::kFloat32, "base must be float32");
            TORCH_CHECK(bases[i].is_contiguous(), "base must be contiguous");
            TORCH_CHECK(bases[i].sizes() == currents[i].sizes(), "base shape mismatch");
        }
        int64_t n = currents[i].numel();
        TORCH_CHECK(n > 0, "empty tensors are unsupported");
        lengths_host.push_back(n);
        total_bytes += n;
    }

    auto offsets = torch::empty({static_cast<int64_t>(currents.size())},
                                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto lengths = torch::empty_like(offsets);
    auto scales = torch::empty({static_cast<int64_t>(currents.size())},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto blob_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    torch::Tensor payload_blob;
    try {
        payload_blob = torch::empty({total_bytes}, blob_options);
    } catch (const c10::Error &) {
        payload_blob = torch::empty({total_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    }

    int64_t *offset_ptr = offsets.data_ptr<int64_t>();
    int64_t *length_ptr = lengths.data_ptr<int64_t>();
    float *scale_ptr = scales.data_ptr<float>();
    uint8_t *blob_ptr = payload_blob.data_ptr<uint8_t>();

    std::vector<torch::Tensor> candidates;
    candidates.reserve(currents.size());

    double encode_kernel_ms = 0.0;
    double payload_d2h_ms = 0.0;
    int64_t launches = 0;
    int64_t running_offset = 0;
    auto encode_start = std::chrono::high_resolution_clock::now();
    auto q_blob_device = torch::empty({total_bytes}, torch::TensorOptions().dtype(torch::kInt8).device(device));
    int8_t *q_blob_ptr = q_blob_device.data_ptr<int8_t>();
    auto k0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < currents.size(); ++i) {
        torch::Tensor current = currents[i];
        const int64_t n = lengths_host[i];
        const int64_t blocks = (n + THREADS - 1) / THREADS;
        const uint8_t has_base = flags_host[i] ? 1 : 0;
        const float *base_ptr = has_base ? bases[i].data_ptr<float>() : current.data_ptr<float>();
        auto block_max = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        reduce_abs_residual_kernel<<<static_cast<unsigned int>(blocks), THREADS, 0, stream>>>(
            current.data_ptr<float>(), base_ptr, has_base, block_max.data_ptr<float>(), n);
        CUDA_CHECK_THROW(cudaGetLastError());
        launches++;

        auto block_max_cpu = block_max.cpu();
        float max_abs = 0.0f;
        const float *max_ptr = block_max_cpu.data_ptr<float>();
        for (int64_t j = 0; j < blocks; ++j) max_abs = std::max(max_abs, max_ptr[j]);
        float scale = max_abs == 0.0f ? 1.0f : (max_abs / 127.0f);

        quantize_q8_kernel<<<static_cast<unsigned int>(blocks), THREADS, 0, stream>>>(
            current.data_ptr<float>(), base_ptr, has_base, q_blob_ptr + running_offset, scale, n);
        CUDA_CHECK_THROW(cudaGetLastError());
        launches++;

        offset_ptr[i] = running_offset;
        length_ptr[i] = n;
        scale_ptr[i] = scale;
        running_offset += n;
        candidates.push_back(current.detach().clone());
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto k1 = std::chrono::high_resolution_clock::now();
    encode_kernel_ms = host_ms_since(k0, k1);

    auto d0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_THROW(cudaMemcpyAsync(blob_ptr,
                                     q_blob_ptr,
                                     static_cast<size_t>(total_bytes),
                                     cudaMemcpyDeviceToHost,
                                     stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto d1 = std::chrono::high_resolution_clock::now();
    payload_d2h_ms = host_ms_since(d0, d1);
    auto encode_end = std::chrono::high_resolution_clock::now();

    auto stats = torch::empty({6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    double *stats_ptr = stats.data_ptr<double>();
    stats_ptr[0] = encode_kernel_ms;
    stats_ptr[1] = payload_d2h_ms;
    stats_ptr[2] = 1.0;  // one CUDA payload blob
    stats_ptr[3] = static_cast<double>(total_bytes);
    stats_ptr[4] = static_cast<double>(launches);
    stats_ptr[5] = host_ms_since(encode_start, encode_end);

    return {payload_blob, offsets, lengths, scales, candidates, stats};
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor>
decode_q8_batch_cuda(torch::Tensor payload_blob_cpu,
                     torch::Tensor offsets_cpu,
                     torch::Tensor lengths_cpu,
                     torch::Tensor scales_cpu,
                     std::vector<std::vector<int64_t>> shapes,
                     std::vector<torch::Tensor> bases,
                     torch::Tensor has_base_flags,
                     int64_t device_index) {
    validate_cpu_1d(payload_blob_cpu, torch::kUInt8, "payload_blob_cpu");
    validate_cpu_1d(offsets_cpu, torch::kInt64, "offsets_cpu");
    validate_cpu_1d(lengths_cpu, torch::kInt64, "lengths_cpu");
    validate_cpu_1d(scales_cpu, torch::kFloat32, "scales_cpu");
    validate_cpu_1d(has_base_flags, torch::kUInt8, "has_base_flags");
    const int64_t count = offsets_cpu.numel();
    TORCH_CHECK(lengths_cpu.numel() == count, "length length mismatch");
    TORCH_CHECK(scales_cpu.numel() == count, "scale length mismatch");
    TORCH_CHECK(has_base_flags.numel() == count, "flag length mismatch");
    TORCH_CHECK(static_cast<int64_t>(shapes.size()) == count, "shape count mismatch");
    TORCH_CHECK(static_cast<int64_t>(bases.size()) == count, "base count mismatch");

    c10::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index));
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint8_t *blob_ptr = payload_blob_cpu.data_ptr<uint8_t>();
    const int64_t *offset_ptr = offsets_cpu.data_ptr<int64_t>();
    const int64_t *length_ptr = lengths_cpu.data_ptr<int64_t>();
    const float *scale_ptr = scales_cpu.data_ptr<float>();
    const uint8_t *flag_ptr = has_base_flags.data_ptr<uint8_t>();

    std::vector<torch::Tensor> decoded;
    decoded.reserve(static_cast<size_t>(count));
    double payload_h2d_ms = 0.0;
    double decode_kernel_ms = 0.0;
    double materialize_ms = 0.0;
    int64_t launches = 0;

    auto q_blob_device = torch::empty({payload_blob_cpu.numel()},
                                      torch::TensorOptions().dtype(torch::kInt8).device(device));
    auto h0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_THROW(cudaMemcpyAsync(q_blob_device.data_ptr<int8_t>(),
                                     blob_ptr,
                                     static_cast<size_t>(payload_blob_cpu.numel()),
                                     cudaMemcpyHostToDevice,
                                     stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto h1 = std::chrono::high_resolution_clock::now();
    payload_h2d_ms = host_ms_since(h0, h1);
    int8_t *q_blob_ptr = q_blob_device.data_ptr<int8_t>();
    auto k0_all = std::chrono::high_resolution_clock::now();

    for (int64_t i = 0; i < count; ++i) {
        int64_t n = length_ptr[i];
        TORCH_CHECK(n == numel_from_shape(shapes[static_cast<size_t>(i)]), "shape/length mismatch");
        uint8_t has_base = flag_ptr[i] ? 1 : 0;
        if (has_base) {
            TORCH_CHECK(bases[static_cast<size_t>(i)].is_cuda(), "base must be CUDA when has_base is true");
            TORCH_CHECK(bases[static_cast<size_t>(i)].device() == device, "base device mismatch");
            TORCH_CHECK(bases[static_cast<size_t>(i)].dtype() == torch::kFloat32, "base must be float32");
            TORCH_CHECK(bases[static_cast<size_t>(i)].is_contiguous(), "base must be contiguous");
            TORCH_CHECK(bases[static_cast<size_t>(i)].numel() == n, "base length mismatch");
        }
        const int64_t blocks = (n + THREADS - 1) / THREADS;

        auto m0 = std::chrono::high_resolution_clock::now();
        auto out = torch::empty(shapes[static_cast<size_t>(i)],
                                torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto m1 = std::chrono::high_resolution_clock::now();
        materialize_ms += host_ms_since(m0, m1);

        const float *base_ptr = has_base ? bases[static_cast<size_t>(i)].data_ptr<float>() : out.data_ptr<float>();
        decode_q8_kernel<<<static_cast<unsigned int>(blocks), THREADS, 0, stream>>>(
            q_blob_ptr + offset_ptr[i], base_ptr, has_base, out.data_ptr<float>(), scale_ptr[i], n);
        CUDA_CHECK_THROW(cudaGetLastError());
        launches++;
        decoded.push_back(out);
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto k1_all = std::chrono::high_resolution_clock::now();
    decode_kernel_ms = host_ms_since(k0_all, k1_all);

    auto stats = torch::empty({5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    double *stats_ptr = stats.data_ptr<double>();
    stats_ptr[0] = payload_h2d_ms;
    stats_ptr[1] = decode_kernel_ms;
    stats_ptr[2] = materialize_ms;
    stats_ptr[3] = static_cast<double>(launches);
    stats_ptr[4] = static_cast<double>(payload_blob_cpu.numel());
    return {decoded, stats};
}
