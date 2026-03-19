#include "activations.hpp"

#include "../../autograd/nodes/activation_nodes.hpp"

namespace munet {
namespace ops {

Tensor relu(const Tensor &a) {
  const auto dispatch = resolve_dispatch(OpId::Relu, a);
  Tensor out = dispatch.use_backend
                   ? Tensor(a.shape(), a.device(), a.dtype())
                   : detail::unary_cpu_fallback(a, [](double v) {
                       return std::max(v, 0.0);
                     });
  if (dispatch.use_backend) {
    a.impl_->backend().relu(*a.impl_->storage, *out.impl_->storage, a.size());
  } else if (!is_floating(a.dtype()) && a.dtype() != DataType::Int32) {
    detail::require_backend_support(op_metadata(OpId::Relu).name, a,
                                    BackendFeature::UnaryActivation);
  }
  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::ReluBackward>(a);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Relu, out, {a});
  return out;
}

Tensor sigmoid(const Tensor &a) {
  const auto dispatch = resolve_dispatch(OpId::Sigmoid, a);
  Tensor out = dispatch.use_backend
                   ? Tensor(a.shape(), a.device(), a.dtype())
                   : detail::unary_cpu_fallback(a, [](double v) {
                       return 1.0 / (1.0 + std::exp(-v));
                     });
  if (dispatch.use_backend) {
    a.impl_->backend().sigmoid(*a.impl_->storage, *out.impl_->storage, a.size());
  }
  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::SigmoidBackward>(out);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Sigmoid, out, {a});
  return out;
}

Tensor softmax(const Tensor &a, int dim) {
  const auto dispatch = resolve_dispatch(OpId::Softmax, a);
  if (a.shape().empty())
    throw std::runtime_error("Softmax expects non-empty shape");

  const int rank = static_cast<int>(a.shape().size());
  const int resolved = (dim < 0) ? (rank + dim) : dim;
  if (resolved < 0 || resolved >= rank)
    throw std::runtime_error("Softmax: dim out of range");
  if (resolved != rank - 1)
    throw std::runtime_error("Softmax currently supports only the last dimension");

  const int num_classes = a.shape().back();
  const int batch_size = a.size() / num_classes;
  Tensor out(a.shape(), a.device(), a.dtype());
  if (dispatch.use_backend) {
    a.impl_->backend().softmax(*a.impl_->storage, *out.impl_->storage,
                               batch_size, num_classes);
  } else {
    Tensor a_cpu = a.to(Device{DeviceType::CPU, 0});
    Tensor out_cpu(a.shape(), Device{DeviceType::CPU, 0}, a.dtype());
    const char *ip = static_cast<const char *>(a_cpu.data());
    char *op = static_cast<char *>(out_cpu.data());
    const size_t stride = dtype_size(a.dtype());
    for (int b = 0; b < batch_size; ++b) {
      double max_val = read_scalar_from_buffer(ip + b * num_classes * stride,
                                               a.dtype())
                           .value;
      for (int i = 1; i < num_classes; ++i) {
        max_val = std::max(max_val,
                           read_scalar_from_buffer(
                               ip + (b * num_classes + i) * stride, a.dtype())
                               .value);
      }
      double sum_exp = 0.0;
      for (int i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(read_scalar_from_buffer(
                                ip + (b * num_classes + i) * stride, a.dtype())
                                .value -
                            max_val);
      }
      for (int i = 0; i < num_classes; ++i) {
        const double prob =
            std::exp(read_scalar_from_buffer(
                         ip + (b * num_classes + i) * stride, a.dtype())
                         .value -
                     max_val) /
            sum_exp;
        write_scalar_to_buffer(op + (b * num_classes + i) * stride,
                               out_cpu.dtype(), prob);
      }
    }
    out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::SoftmaxBackward>(out, resolved);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Softmax, out, {a}, {{"dim", {resolved}}});
  return out;
}

Tensor log_softmax(const Tensor &a, int dim) {
  resolve_dispatch(OpId::LogSoftmax, a);
  Tensor p = softmax(a, dim);

  Tensor p_cpu = p.to(Device{DeviceType::CPU, 0});
  Tensor out_cpu(a.shape(), Device{DeviceType::CPU, 0}, a.dtype());
  const char *pv = static_cast<const char *>(p_cpu.data());
  char *ov = static_cast<char *>(out_cpu.data());
  const size_t stride = dtype_size(a.dtype());
  for (size_t i = 0; i < p_cpu.size(); ++i) {
    const double prob =
        read_scalar_from_buffer(pv + i * stride, p_cpu.dtype()).value;
    write_scalar_to_buffer(ov + i * stride, out_cpu.dtype(),
                           std::log(std::max(prob, 1e-20)));
  }

  Tensor out =
      (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  const int rank = static_cast<int>(a.shape().size());
  const int resolved = (dim < 0) ? (rank + dim) : dim;
  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::LogSoftmaxBackward>(out, resolved);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::LogSoftmax, out, {a}, {{"dim", {resolved}}});
  return out;
}

} // namespace ops
} // namespace munet
