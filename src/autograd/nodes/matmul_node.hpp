#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MatmulBackward : public Node {
  MatmulBackward(Tensor a, Tensor b) {
    save_tensor(a, "matmul_lhs");
    save_tensor(b, "matmul_rhs");
  }
  std::string name() const override { return "MatmulBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor A = saved_tensor(0);
    Tensor B = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor grad_a;
    Tensor grad_b;

    if (next_edges.size() > 0 && next_edges[0].node) {
      grad_a = ops::matmul_internal(grad_out, B, false, true);
    }

    if (next_edges.size() > 1 && next_edges[1].node) {
      grad_b = ops::matmul_internal(A, grad_out, true, false);
    }

    return {grad_a, grad_b};
  }
};

} // namespace autograd_nodes
} // namespace munet
