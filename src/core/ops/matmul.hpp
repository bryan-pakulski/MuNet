#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                       bool transB);
Tensor batched_matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                               bool transB);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor batched_matmul(const Tensor &a, const Tensor &b);

} // namespace ops
} // namespace munet
