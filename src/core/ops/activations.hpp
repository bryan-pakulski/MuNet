#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor relu(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor softmax(const Tensor &a, int dim = -1);
Tensor log_softmax(const Tensor &a, int dim = -1);

} // namespace ops
} // namespace munet
