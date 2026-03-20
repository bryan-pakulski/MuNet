#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor cat(const std::vector<Tensor> &inputs, int dim = 1);
Tensor sum(const Tensor &a);
Tensor mean(const Tensor &a, int dim = -1, bool keepdim = false);
Tensor reshape(const Tensor &in, Shape new_shape);
Tensor transpose(const Tensor &in, int dim0, int dim1);
Tensor narrow(const Tensor &in, int dim, int start, int length);
Tensor zeros(Shape shape, Device device = Device{DeviceType::CPU, 0},
             bool requires_grad = false, DataType dtype = DataType::Float32);

} // namespace ops
} // namespace munet
