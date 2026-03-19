#pragma once

#include "common.hpp"

namespace munet {
namespace ops {

Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor masked_fill(const Tensor &a, const Tensor &mask,
                   const ScalarValue &value);

} // namespace ops
} // namespace munet
