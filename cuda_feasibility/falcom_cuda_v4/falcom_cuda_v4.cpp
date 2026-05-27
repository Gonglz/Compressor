#include <torch/extension.h>

#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, torch::Tensor>
encode_q8_batch_cuda(std::vector<torch::Tensor> currents,
                     std::vector<torch::Tensor> bases,
                     torch::Tensor has_base_flags);

std::tuple<std::vector<torch::Tensor>, torch::Tensor>
decode_q8_batch_cuda(torch::Tensor payload_blob_cpu,
                     torch::Tensor offsets_cpu,
                     torch::Tensor lengths_cpu,
                     torch::Tensor scales_cpu,
                     std::vector<std::vector<int64_t>> shapes,
                     std::vector<torch::Tensor> bases,
                     torch::Tensor has_base_flags,
                     int64_t device_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_q8_batch", &encode_q8_batch_cuda,
          "FalCom CUDA v4 fused q8 batch encoder");
    m.def("decode_q8_batch", &decode_q8_batch_cuda,
          "FalCom CUDA v4 fused q8 batch decoder");
    m.def("is_cuda_build", []() { return true; });
}
