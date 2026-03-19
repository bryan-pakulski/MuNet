#include "matmul.hpp"

#include "../../autograd/nodes/matmul_node.hpp"

namespace munet {
namespace ops {

Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                       bool transB) {
  detail::require_same_dtype(op_metadata(OpId::Matmul).name, a, b);
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    throw std::runtime_error("Matmul currently requires 2D tensors");
  }

  const int M = transA ? a.shape()[1] : a.shape()[0];
  const int K = transA ? a.shape()[0] : a.shape()[1];
  const int N = transB ? b.shape()[0] : b.shape()[1];

  const auto dispatch = resolve_dispatch(OpId::Matmul, a);
  if (!dispatch.use_backend) {
    return detail::matmul_cpu_fallback(a, b, transA, transB);
  }

  Tensor out({M, N}, a.device(), a.dtype());
  a.impl_->backend().matmul(*a.impl_->storage, *b.impl_->storage,
                            *out.impl_->storage, M, K, N, transA, transB);
  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device()) {
    throw std::runtime_error("Matmul: device mismatch");
  }

  Tensor out = matmul_internal(a, b, false, false);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<autograd_nodes::MatmulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Matmul, out, {a, b});
  return out;
}

} // namespace ops
} // namespace munet
