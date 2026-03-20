#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor relu(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor log(const Tensor &a);
Tensor sqrt(const Tensor &a);
Tensor rsqrt(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor softmax(const Tensor &a, int dim = -1);
Tensor log_softmax(const Tensor &a, int dim = -1);

} // namespace ops
} // namespace munet
