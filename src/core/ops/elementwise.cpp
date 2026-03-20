#include "elementwise.hpp"

#include "../../autograd/nodes/elementwise_nodes.hpp"

namespace munet {
namespace ops {

Tensor masked_fill(const Tensor &a, const Tensor &mask,
                   const ScalarValue &value) {
  if (a.shape() != mask.shape())
    throw std::runtime_error("masked_fill: input/mask shape mismatch");
  if (a.device() != mask.device())
    throw std::runtime_error("masked_fill: input/mask device mismatch");

  Tensor a_cpu = a.to(Device{DeviceType::CPU, 0});
  Tensor m_cpu = mask.to(Device{DeviceType::CPU, 0});
  Tensor out_cpu(a.shape(), Device{DeviceType::CPU, 0}, a.dtype());

  const char *av = static_cast<const char *>(a_cpu.data());
  const char *mv = static_cast<const char *>(m_cpu.data());
  char *ov = static_cast<char *>(out_cpu.data());
  const size_t a_stride = dtype_size(a.dtype());
  const size_t mask_stride = dtype_size(mask.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());

  for (size_t i = 0; i < out_cpu.size(); ++i) {
    const ScalarValue mask_value =
        read_scalar_from_buffer(mv + i * mask_stride, mask.dtype());
    const ScalarValue input_value =
        read_scalar_from_buffer(av + i * a_stride, a.dtype());
    write_scalar_to_buffer(ov + i * out_stride, out_cpu.dtype(),
                           mask_value.is_nonzero() ? value.value
                                                   : input_value.value);
  }

  Tensor out =
      (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::MaskedFillBackward>(mask);
    link_backward_edges(fn.get(), {a, mask});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  if (value.dtype == DataType::Int32) {
    record_registered_trace(OpId::MaskedFill, out, {a, mask},
                            {{"value", {value.as_int32()}}});
  } else {
    record_registered_trace(OpId::MaskedFill, out, {a, mask}, {},
                            {{"value", value.as_float()}});
  }
  return out;
}

Tensor sub(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Sub: device mismatch");
  detail::require_same_dtype(op_metadata(OpId::Sub).name, a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Sub: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  const auto dispatch = resolve_dispatch(OpId::Sub, a);
  Tensor out =
      dispatch.use_backend
          ? Tensor(info.out_shape, a.device(), a.dtype())
          : detail::binary_broadcast_cpu_fallback(
                a, b, info, [](double lhs, double rhs) { return lhs - rhs; });
  if (dispatch.use_backend) {
    a.impl_->backend().sub(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn =
        std::make_shared<autograd_nodes::SubBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Sub, out, {a, b});
  return out;
}

Tensor mul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Mul: device mismatch");
  detail::require_same_dtype(op_metadata(OpId::Mul).name, a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast)
    throw std::runtime_error("Mul: shape mismatch");

  const auto dispatch = resolve_dispatch(OpId::Mul, a);
  Tensor out =
      dispatch.use_backend
          ? Tensor(info.out_shape, a.device(), a.dtype())
          : detail::binary_broadcast_cpu_fallback(
                a, b, info, [](double lhs, double rhs) { return lhs * rhs; });
  if (dispatch.use_backend) {
    a.impl_->backend().mul(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<autograd_nodes::MulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Mul, out, {a, b});
  return out;
}

Tensor div(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Div: device mismatch");
  detail::require_same_dtype(op_metadata(OpId::Div).name, a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast)
    throw std::runtime_error("Div: shape mismatch");

  const auto dispatch = resolve_dispatch(OpId::Div, a);
  Tensor out =
      dispatch.use_backend
          ? Tensor(info.out_shape, a.device(), a.dtype())
          : detail::binary_broadcast_cpu_fallback(
                a, b, info, [](double lhs, double rhs) { return lhs / rhs; });
  if (dispatch.use_backend) {
    a.impl_->backend().div(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<autograd_nodes::DivBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Div, out, {a, b});
  return out;
}

} // namespace ops
} // namespace munet
