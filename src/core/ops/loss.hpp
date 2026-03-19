#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor mse_loss(const Tensor &pred, const Tensor &target);
Tensor cross_entropy(const Tensor &logits, const Tensor &targets);

} // namespace ops
} // namespace munet
