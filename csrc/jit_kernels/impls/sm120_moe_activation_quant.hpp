#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm::sm120_moe {

torch::Tensor sm120_silu_mul_quant_fp8_packed(const torch::Tensor& input,
                                              const torch::Tensor& output,
                                              int64_t group_size);

void register_apis(pybind11::module& m);

}  // namespace deep_gemm::sm120_moe

