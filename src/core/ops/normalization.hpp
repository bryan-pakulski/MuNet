#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor batch_norm(const Tensor &in, Tensor &running_mean, Tensor &running_var,
                  const Tensor &weight, const Tensor &bias, bool training,
                  float momentum, float eps);
Tensor layer_norm(const Tensor &x, const Tensor &weight, const Tensor &bias,
                  float eps);

} // namespace ops
} // namespace munet
