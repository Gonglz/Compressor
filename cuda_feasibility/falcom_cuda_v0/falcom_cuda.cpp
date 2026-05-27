#include <torch/extension.h>

#include <vector>

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
    int64_t step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("momentum_pack_cuda", &momentum_pack_cuda,
          "FalCom v0 momentum residual/bitmap/sign CUDA kernel");
    m.def("is_cuda_build", []() { return true; });
}

