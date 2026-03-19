#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor max_pool2d(const Tensor &in, int kernel_size, int stride,
                  int padding = 0);
Tensor upsample2d(const Tensor &in, int scale_factor);

} // namespace ops
} // namespace munet
