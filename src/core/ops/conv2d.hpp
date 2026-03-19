#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor conv2d(const Tensor &in, const Tensor &weight, const Tensor &bias,
              int stride, int padding);

} // namespace ops
} // namespace munet
