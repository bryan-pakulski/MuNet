#include "pooling.hpp"

#include "../../autograd/nodes/pooling_nodes.hpp"

namespace munet {
namespace ops {

Tensor max_pool2d(const Tensor &in, int kernel_size, int stride, int padding) {
  const auto dispatch = resolve_dispatch(OpId::MaxPool2D, in);
  const int B = in.shape()[0];
  const int C = in.shape()[1];
  const int iH = in.shape()[2];
  const int iW = in.shape()[3];
  const int oH = (iH + 2 * padding - kernel_size) / stride + 1;
  const int oW = (iW + 2 * padding - kernel_size) / stride + 1;
  Tensor out({B, C, oH, oW}, in.device(), in.dtype());
  if (dispatch.use_cpu_fallback) {
    Device cpu{DeviceType::CPU, 0};
    Tensor in_exec = in.to(cpu);
    if (in_exec.dtype() != DataType::Float32) {
      in_exec = in_exec.to(DataType::Float32);
    }
    Tensor out_exec({B, C, oH, oW}, cpu, in_exec.dtype());
    in_exec.impl_->backend().max_pool2d(*in_exec.impl_->storage,
                                        *out_exec.impl_->storage, B, C, iH, iW,
                                        kernel_size, stride, padding);
    if (out_exec.dtype() != in.dtype()) {
      out_exec = out_exec.to(in.dtype());
    }
    out = (in.device().type == DeviceType::CPU) ? out_exec
                                                 : out_exec.to(in.device());
  } else {
    in.impl_->backend().max_pool2d(*in.impl_->storage, *out.impl_->storage, B,
                                   C, iH, iW, kernel_size, stride, padding);
  }

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::MaxPool2DBackward>(
        in, kernel_size, stride, padding);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::MaxPool2D, out, {in},
                          {{"kernel_shape", {kernel_size, kernel_size}},
                           {"strides", {stride, stride}},
                           {"pads", {padding, padding, padding, padding}}});
  return out;
}

Tensor upsample2d(const Tensor &in, int scale_factor) {
  const auto dispatch = resolve_dispatch(OpId::Upsample2D, in);
  const int B = in.shape()[0];
  const int C = in.shape()[1];
  const int iH = in.shape()[2];
  const int iW = in.shape()[3];
  Tensor out({B, C, iH * scale_factor, iW * scale_factor}, in.device(),
             in.dtype());
  if (dispatch.use_cpu_fallback) {
    Device cpu{DeviceType::CPU, 0};
    Tensor in_exec = in.to(cpu);
    if (in_exec.dtype() != DataType::Float32) {
      in_exec = in_exec.to(DataType::Float32);
    }
    Tensor out_exec({B, C, iH * scale_factor, iW * scale_factor}, cpu,
                    in_exec.dtype());
    in_exec.impl_->backend().upsample2d(*in_exec.impl_->storage,
                                        *out_exec.impl_->storage, B, C, iH, iW,
                                        scale_factor);
    if (out_exec.dtype() != in.dtype()) {
      out_exec = out_exec.to(in.dtype());
    }
    out = (in.device().type == DeviceType::CPU) ? out_exec
                                                 : out_exec.to(in.device());
  } else {
    in.impl_->backend().upsample2d(*in.impl_->storage, *out.impl_->storage, B,
                                   C, iH, iW, scale_factor);
  }

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn =
        std::make_shared<autograd_nodes::Upsample2DBackward>(in, scale_factor);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::Upsample2D, out, {in},
                          {{"scale", {scale_factor}}});
  return out;
}

} // namespace ops
} // namespace munet
