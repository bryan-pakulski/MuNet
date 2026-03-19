#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct AddBackward : public Node {
  Shape shape_a, shape_b;
  AddBackward(Shape a, Shape b) : shape_a(std::move(a)), shape_b(std::move(b)) {}
  std::string name() const override { return "AddBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {ops::sum_to_shape(grads[0], shape_a),
            ops::sum_to_shape(grads[0], shape_b)};
  }
};

} // namespace autograd_nodes
} // namespace munet
