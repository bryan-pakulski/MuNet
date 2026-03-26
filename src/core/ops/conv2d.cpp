#include "conv2d.hpp"

#include "../../autograd/nodes/conv2d_node.hpp"

namespace munet {
namespace ops {

Tensor conv2d(const Tensor &in, const Tensor &weight, const Tensor &bias,
              int stride, int padding) {
  detail::require_same_dtype(op_metadata(OpId::Conv2D).name, in, weight);
  if (bias.impl_) {
    detail::require_same_dtype(op_metadata(OpId::Conv2D).name, in, bias);
  }
  if (in.device() != weight.device() ||
      (bias.impl_ && bias.device() != in.device())) {
    MUNET_ERROR << "conv2d: inputs not on same device: "
                << in.device().to_string()
                << " != " << weight.device().to_string() << std::endl;
    throw std::runtime_error("Conv2d: inputs must be on same device");
  }

  const auto dispatch = resolve_dispatch(OpId::Conv2D, in);
  const bool use_cpu_fallback = dispatch.use_cpu_fallback;

  if (in.shape().size() != 4 || weight.shape().size() != 4) {
    MUNET_ERROR << "conv2d: inputs must be 4D, in.shape: "
                << to_string(in.shape())
                << " weight shape: " << to_string(weight.shape()) << std::endl;
    throw std::runtime_error("Conv2d: inputs must be 4D (NCHW)");
  }

  if (in.shape()[1] != weight.shape()[1]) {
    MUNET_ERROR << "conv2d: channel mismatch. input=" << in.shape()[1]
                << ", weight=" << weight.shape()[1] << std::endl;
    throw std::runtime_error("Conv2d: input channels mismatch. Input=" +
                             std::to_string(in.shape()[1]) +
                             ", Weight=" + std::to_string(weight.shape()[1]));
  }

  const int B = in.shape()[0];
  const int iC = in.shape()[1];
  const int iH = in.shape()[2];
  const int iW = in.shape()[3];
  const int oC = weight.shape()[0];
  const int kH = weight.shape()[2];
  const int kW = weight.shape()[3];
  const int oH = (iH + 2 * padding - kH) / stride + 1;
  const int oW = (iW + 2 * padding - kW) / stride + 1;
  Tensor out({B, oC, oH, oW}, in.device(), in.dtype());
  if (use_cpu_fallback) {
    Device cpu{DeviceType::CPU, 0};
    Tensor in_exec = in.to(cpu);
    Tensor weight_exec = weight.to(cpu);
    Tensor bias_exec = bias.impl_ ? bias.to(cpu) : Tensor();
    if (in_exec.dtype() != DataType::Float32) {
      in_exec = in_exec.to(DataType::Float32);
      weight_exec = weight_exec.to(DataType::Float32);
      if (bias_exec.impl_) {
        bias_exec = bias_exec.to(DataType::Float32);
      }
    }

    Tensor out_exec({B, oC, oH, oW}, cpu, in_exec.dtype());
    in_exec.impl_->backend().conv2d(
        *in_exec.impl_->storage, *weight_exec.impl_->storage,
        bias_exec.impl_ ? bias_exec.impl_->storage.get() : nullptr,
        *out_exec.impl_->storage, B, iC, iH, iW, oC, kH, kW, stride, padding);

    if (out_exec.dtype() != in.dtype()) {
      out_exec = out_exec.to(in.dtype());
    }
    out = (in.device().type == DeviceType::CPU) ? out_exec : out_exec.to(in.device());
  } else {
    in.impl_->backend().conv2d(*in.impl_->storage, *weight.impl_->storage,
                               bias.impl_ ? bias.impl_->storage.get() : nullptr,
                               *out.impl_->storage, B, iC, iH, iW, oC, kH, kW,
                               stride, padding);
  }

  if (GradMode::is_enabled() && (in.requires_grad() || weight.requires_grad() ||
                                 (bias.impl_ && bias.requires_grad()))) {
    auto fn = std::make_shared<autograd_nodes::Conv2DBackward>(in, weight, bias,
                                                               stride, padding);
    std::vector<Tensor> inputs = {in, weight};
    if (bias.impl_) {
      inputs.push_back(bias);
    }
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  std::vector<Tensor> inputs = {in, weight};
  if (bias.impl_) {
    inputs.push_back(bias);
  }
  record_registered_trace(OpId::Conv2D, out, inputs,
                          {{"strides", {stride, stride}},
                           {"pads", {padding, padding, padding, padding}}});
  return out;
}

} // namespace ops
} // namespace munet
