#include "add.hpp"

#include "../../autograd/nodes/add_node.hpp"

namespace munet {
namespace ops {

Tensor add(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device()) {
    throw std::runtime_error("Add: device mismatch");
  }
  detail::require_same_dtype(op_metadata(OpId::Add).name, a, b);

  const auto info =
      compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Add: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  const auto dispatch = resolve_dispatch(OpId::Add, a);
  Tensor out = dispatch.use_backend
                   ? Tensor(info.out_shape, a.device(), a.dtype())
                   : detail::binary_broadcast_cpu_fallback(
                         a, b, info,
                         [](double lhs, double rhs) { return lhs + rhs; });
  if (dispatch.use_backend) {
    a.impl_->backend().add(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn =
        std::make_shared<autograd_nodes::AddBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Add, out, {a, b});
  return out;
}

} // namespace ops
} // namespace munet
