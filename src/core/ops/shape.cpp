#include "shape.hpp"

#include "../../autograd/nodes/shape_nodes.hpp"

namespace munet {
namespace ops {

Tensor cat(const std::vector<Tensor> &inputs, int dim) {
  if (inputs.empty())
    return Tensor();

  Shape out_shape = inputs[0].shape();
  std::vector<Shape> shapes;
  std::vector<Storage *> storages;
  int concat_size = 0;
  for (const auto &t : inputs) {
    const auto &s = t.shape();
    if (s.size() != out_shape.size())
      throw std::runtime_error(
          "All tensors must have the same number of dimensions for cat");
    for (size_t d = 0; d < s.size(); ++d) {
      if (static_cast<int>(d) != dim && s[d] != out_shape[d])
        throw std::runtime_error(
            "Tensor shapes must match except along concat dim");
    }
    concat_size += s[dim];
    shapes.push_back(s);
    storages.push_back(t.impl_->storage.get());
  }
  out_shape[dim] = concat_size;

  resolve_dispatch(OpId::Cat, inputs[0]);
  Tensor out(out_shape, inputs[0].device(), inputs[0].dtype());
  inputs[0].impl_->backend().concat(storages, *out.impl_->storage, dim, shapes);

  bool req = false;
  for (const auto &t : inputs)
    req = req || t.requires_grad();

  if (GradMode::is_enabled() && req) {
    auto fn = std::make_shared<autograd_nodes::ConcatBackward>(dim, shapes);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Cat, out, inputs, {{"axis", {dim}}});
  return out;
}

Tensor sum(const Tensor &a) {
  const auto dispatch = resolve_dispatch(OpId::Sum, a);
  Tensor out({1}, a.device(), a.dtype());
  if (dispatch.use_backend) {
    a.impl_->backend().sum(*a.impl_->storage, *out.impl_->storage, a.size());
  } else {
    Tensor a_cpu = a.to(Device{DeviceType::CPU, 0});
    Tensor out_cpu({1}, Device{DeviceType::CPU, 0}, a.dtype());
    double total = 0.0;
    const char *ip = static_cast<const char *>(a_cpu.data());
    const size_t stride = dtype_size(a.dtype());
    for (size_t i = 0; i < a_cpu.size(); ++i) {
      total += read_scalar_from_buffer(ip + i * stride, a.dtype()).value;
    }
    write_scalar_to_buffer(out_cpu.data(), out_cpu.dtype(), total);
    out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::SumBackward>(a.shape(), a.device());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Sum, out, {a}, {{"keepdims", {0}}});
  return out;
}

Tensor reshape(const Tensor &in, Shape new_shape) {
  if (numel(in.shape()) != numel(new_shape))
    throw std::runtime_error("Reshape: element count mismatch");

  Tensor out;
  out.impl_ = std::make_shared<TensorImpl>(new_shape, in.device(), in.dtype(),
                                           in.requires_grad());
  out.impl_->storage = in.impl_->storage;

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::ReshapeBackward>(in.shape());
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Reshape, out, {in}, {{"shape", new_shape}});
  return out;
}

Tensor transpose(const Tensor &in, int dim0, int dim1) {
  Tensor out = in.transpose(dim0, dim1);
  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::TransposeBackward>(dim0, dim1);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Transpose, out, {in}, {{"dims", {dim0, dim1}}});
  return out;
}

Tensor zeros(Shape shape, Device device, bool requires_grad, DataType dtype) {
  (void)op_metadata(OpId::Zeros);
  Tensor t(shape, device, dtype, requires_grad);
  t.impl_->storage->zero_();
  return t;
}

} // namespace ops
} // namespace munet
