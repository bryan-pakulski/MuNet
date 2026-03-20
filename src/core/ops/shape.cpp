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
    out =
        (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn =
        std::make_shared<autograd_nodes::SumBackward>(a.shape(), a.device());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Sum, out, {a}, {{"keepdims", {0}}});
  return out;
}

Tensor mean(const Tensor &a, int dim, bool keepdim) {
  if (a.shape().empty()) {
    throw std::runtime_error("Mean expects at least 1 dimension");
  }
  const int rank = static_cast<int>(a.shape().size());
  const int resolved = (dim < 0) ? (rank + dim) : dim;
  if (resolved < 0 || resolved >= rank) {
    throw std::runtime_error("Mean: dim out of range");
  }
  const auto dispatch = resolve_dispatch(OpId::Mean, a);

  Shape out_shape = a.shape();
  out_shape[resolved] = 1;
  if (!keepdim) {
    out_shape.erase(out_shape.begin() + resolved);
    if (out_shape.empty()) {
      out_shape = {1};
    }
  }

  Tensor out(out_shape, a.device(), a.dtype());
  const bool backend_supported =
      dispatch.use_backend && resolved == rank - 1 && a.is_contiguous();
  if (backend_supported) {
    Tensor tmp =
        keepdim ? out
                : Tensor(Shape{static_cast<int>(a.size() / a.shape().back())},
                         a.device(), a.dtype());
    a.impl_->backend().mean_last_dim(
        *a.impl_->storage, *tmp.impl_->storage,
        static_cast<int>(a.size() / a.shape().back()), a.shape().back());
    if (keepdim) {
      out = tmp;
    } else {
      out = tmp.reshape(out_shape);
    }
  } else {
    Tensor src = a.to(Device{DeviceType::CPU, 0}).contiguous();
    Tensor dst_cpu(out_shape, Device{DeviceType::CPU, 0}, a.dtype());
    const Shape src_shape = src.shape();
    Shape reduced_shape = src_shape;
    reduced_shape[resolved] = 1;
    Shape cpu_out_shape = keepdim ? reduced_shape : out_shape;
    Strides src_strides = default_strides(src_shape);
    Strides reduced_strides = default_strides(reduced_shape);
    const size_t elem_size = dtype_size(a.dtype());
    const char *sp = static_cast<const char *>(src.data());
    char *dp = static_cast<char *>(dst_cpu.data());
    std::vector<double> accum(numel(cpu_out_shape), 0.0);
    for (size_t linear = 0; linear < src.size(); ++linear) {
      size_t curr = linear;
      size_t out_off = 0;
      for (int d = 0; d < rank; ++d) {
        const int coord = static_cast<int>(curr / src_strides[d]);
        curr %= static_cast<size_t>(src_strides[d]);
        if (d == resolved)
          continue;
        int out_dim = d;
        if (!keepdim && d > resolved) {
          out_dim -= 1;
        }
        out_off += static_cast<size_t>(coord) *
                   static_cast<size_t>(
                       keepdim ? reduced_strides[d]
                               : default_strides(cpu_out_shape)[out_dim]);
      }
      accum[out_off] +=
          read_scalar_from_buffer(sp + linear * elem_size, src.dtype()).value;
    }
    const double inv = 1.0 / static_cast<double>(src_shape[resolved]);
    for (size_t i = 0; i < accum.size(); ++i) {
      write_scalar_to_buffer(dp + i * elem_size, dst_cpu.dtype(),
                             accum[i] * inv);
    }
    out =
        (a.device().type == DeviceType::CPU) ? dst_cpu : dst_cpu.to(a.device());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::MeanBackward>(
        a.shape(), resolved, keepdim, a.device(), a.dtype());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(
      OpId::Mean, out, {a},
      {{"dim", {resolved}}, {"keepdims", {keepdim ? 1 : 0}}});
  return out;
}

Tensor reshape(const Tensor &in, Shape new_shape) {
  if (numel(in.shape()) != numel(new_shape))
    throw std::runtime_error("Reshape: element count mismatch");

  Tensor out;
  out.impl_ = std::make_shared<TensorImpl>(new_shape, in.device(), in.dtype(),
                                           in.requires_grad());
  out.impl_->storage = in.impl_->storage;
  out.impl_->storage_offset = in.impl_->storage_offset;

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

Tensor narrow(const Tensor &in, int dim, int start, int length) {
  if (in.shape().empty()) {
    throw std::runtime_error("narrow expects at least 1 dimension");
  }
  const int rank = static_cast<int>(in.shape().size());
  const int resolved = (dim < 0) ? (rank + dim) : dim;
  if (resolved < 0 || resolved >= rank) {
    throw std::runtime_error("narrow: dim out of range");
  }
  if (start < 0 || length < 0 || start + length > in.shape()[resolved]) {
    throw std::runtime_error("narrow: slice out of bounds");
  }

  Tensor out = in;
  out.impl_ = std::make_shared<TensorImpl>(*in.impl_);
  out.impl_->shape[resolved] = length;
  out.impl_->storage_offset =
      in.impl_->storage_offset +
      static_cast<size_t>(start) * in.impl_->strides[resolved];

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::NarrowBackward>(
        in.shape(), resolved, start, length, in.device(), in.dtype());
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(
      OpId::Narrow, out, {in},
      {{"dim", {resolved}}, {"start", {start}}, {"length", {length}}});
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
