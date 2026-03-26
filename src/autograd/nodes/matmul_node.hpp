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
    const int a_ndim = static_cast<int>(A.shape().size());
    const int b_ndim = static_cast<int>(B.shape().size());

    auto transpose_last_two = [](const Tensor &t) -> Tensor {
      const int ndim = static_cast<int>(t.shape().size());
      if (ndim < 2) {
        return t;
      }
      return ops::transpose(t, ndim - 2, ndim - 1);
    };

    if (next_edges.size() > 0 && next_edges[0].node) {
      if (a_ndim >= 3 || b_ndim >= 3 || grad_out.shape().size() >= 3) {
        grad_a = ops::batched_matmul(grad_out, transpose_last_two(B));
      } else {
        grad_a = ops::matmul(grad_out, ops::transpose(B, 0, 1));
      }
    }

    if (next_edges.size() > 1 && next_edges[1].node) {
      if (a_ndim >= 3 || b_ndim >= 3 || grad_out.shape().size() >= 3) {
        grad_b = ops::batched_matmul(transpose_last_two(A), grad_out);
      } else {
        grad_b = ops::matmul(ops::transpose(A, 0, 1), grad_out);
      }
    }

    return {grad_a, grad_b};
  }
};

} // namespace autograd_nodes
} // namespace munet
