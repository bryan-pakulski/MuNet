#include "normalization.hpp"

#include "../../autograd/nodes/normalization_node.hpp"

namespace munet {
namespace ops {

Tensor batch_norm(const Tensor &in, Tensor &running_mean, Tensor &running_var,
                  const Tensor &weight, const Tensor &bias, bool training,
                  float momentum, float eps) {
  detail::require_same_dtype(op_metadata(OpId::BatchNorm).name, in, weight);
  detail::require_same_dtype(op_metadata(OpId::BatchNorm).name, in, bias);
  const auto dispatch = resolve_dispatch(OpId::BatchNorm, in);
  const bool use_cpu_fallback = dispatch.use_cpu_fallback;

  const int B = in.shape()[0];
  const int C = in.shape()[1];
  const int H = in.shape()[2];
  const int W = in.shape()[3];
  Tensor out(in.shape(), in.device(), in.dtype());

  const DataType stats_dtype =
      accumulation_type(AccumulationOp::Normalization, in.dtype());
  Tensor save_mean({C}, in.device(), stats_dtype);
  Tensor save_var({C}, in.device(), stats_dtype);
  Tensor backward_in = in;
  Tensor backward_weight = weight;
  Tensor backward_save_mean = save_mean;
  Tensor backward_save_var = save_var;

  if (use_cpu_fallback) {
    Device cpu{DeviceType::CPU, 0};
    Tensor in_exec = in.to(cpu);
    Tensor weight_exec = weight.to(cpu);
    Tensor bias_exec = bias.to(cpu);
    Tensor running_mean_exec = running_mean.to(cpu);
    Tensor running_var_exec = running_var.to(cpu);

    if (in_exec.dtype() != DataType::Float32) {
      in_exec = in_exec.to(DataType::Float32);
      weight_exec = weight_exec.to(DataType::Float32);
      bias_exec = bias_exec.to(DataType::Float32);
    }
    if (running_mean_exec.dtype() != DataType::Float32) {
      running_mean_exec = running_mean_exec.to(DataType::Float32);
      running_var_exec = running_var_exec.to(DataType::Float32);
    }

    Tensor out_exec(in.shape(), cpu, in_exec.dtype());
    Tensor save_mean_exec({C}, cpu, running_mean_exec.dtype());
    Tensor save_var_exec({C}, cpu, running_var_exec.dtype());

    in_exec.impl_->backend().batch_norm(
        *in_exec.impl_->storage, *weight_exec.impl_->storage,
        *bias_exec.impl_->storage, *running_mean_exec.impl_->storage,
        *running_var_exec.impl_->storage, *save_mean_exec.impl_->storage,
        *save_var_exec.impl_->storage, *out_exec.impl_->storage, B, C, H, W,
        momentum, eps, training);

    if (out_exec.dtype() != in.dtype()) {
      out_exec = out_exec.to(in.dtype());
    }
    out = (in.device().type == DeviceType::CPU) ? out_exec : out_exec.to(in.device());
    save_mean = (save_mean_exec.dtype() == save_mean.dtype())
                    ? ((in.device().type == DeviceType::CPU) ? save_mean_exec
                                                             : save_mean_exec.to(in.device()))
                    : ((in.device().type == DeviceType::CPU)
                           ? save_mean_exec.to(save_mean.dtype())
                           : save_mean_exec.to(save_mean.dtype()).to(in.device()));
    save_var = (save_var_exec.dtype() == save_var.dtype())
                   ? ((in.device().type == DeviceType::CPU) ? save_var_exec
                                                            : save_var_exec.to(in.device()))
                   : ((in.device().type == DeviceType::CPU)
                          ? save_var_exec.to(save_var.dtype())
                          : save_var_exec.to(save_var.dtype()).to(in.device()));

    Tensor running_mean_updated =
        (running_mean.device().type == DeviceType::CPU)
            ? running_mean_exec.to(running_mean.dtype())
            : running_mean_exec.to(running_mean.dtype()).to(running_mean.device());
    Tensor running_var_updated =
        (running_var.device().type == DeviceType::CPU)
            ? running_var_exec.to(running_var.dtype())
            : running_var_exec.to(running_var.dtype()).to(running_var.device());
    running_mean.impl_->backend().copy(
        running_mean_updated.data(), running_mean.data(), running_mean.bytes(),
        running_mean_updated.device(), running_mean.device());
    running_var.impl_->backend().copy(
        running_var_updated.data(), running_var.data(), running_var.bytes(),
        running_var_updated.device(), running_var.device());
    backward_in = in_exec;
    backward_weight = weight_exec;
    backward_save_mean = save_mean_exec;
    backward_save_var = save_var_exec;
  } else {
    in.impl_->backend().batch_norm(
        *in.impl_->storage, *weight.impl_->storage, *bias.impl_->storage,
        *running_mean.impl_->storage, *running_var.impl_->storage,
        *save_mean.impl_->storage, *save_var.impl_->storage,
        *out.impl_->storage, B, C, H, W, momentum, eps, training);
  }

  if (GradMode::is_enabled() && training &&
      (in.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    std::shared_ptr<autograd_nodes::BatchNormBackward> fn;
    if (use_cpu_fallback) {
      fn = std::make_shared<autograd_nodes::BatchNormBackward>(
          backward_in, backward_weight, backward_save_mean, backward_save_var,
          eps, in.shape(), in.device(), in.dtype(), weight.shape(),
          weight.device(), weight.dtype());
    } else {
      fn = std::make_shared<autograd_nodes::BatchNormBackward>(
          in, weight, save_mean, save_var, eps);
    }
    link_backward_edges(fn.get(), {in, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::BatchNorm, out,
                          {in, weight, bias, running_mean, running_var}, {},
                          {{"epsilon", eps}, {"momentum", momentum}});
  return out;
}

Tensor layer_norm(const Tensor &x, const Tensor &weight, const Tensor &bias,
                  float eps) {
  detail::require_same_dtype(op_metadata(OpId::LayerNorm).name, x, weight);
  detail::require_same_dtype(op_metadata(OpId::LayerNorm).name, x, bias);

  if (x.shape().empty()) {
    throw std::runtime_error("LayerNorm: input must have at least 1 dim");
  }

  const int cols = x.shape().back();
  if (static_cast<int>(weight.size()) != cols ||
      static_cast<int>(bias.size()) != cols) {
    throw std::runtime_error("LayerNorm: weight/bias size must equal last dim");
  }

  const int rows = static_cast<int>(x.size() / cols);
  const Shape flat_shape{rows, cols};
  Tensor x2 = x.reshape(flat_shape);
  Tensor mean = x2.mean(1, true);
  Tensor centered = x2 - mean;
  Tensor var = (centered * centered).mean(1, true);
  Tensor eps_tensor(var.shape(), var.device(), var.dtype(), false);
  eps_tensor.fill_(eps);
  Tensor inv_std = (var + eps_tensor).rsqrt();
  Tensor normalized = centered * inv_std;
  Tensor w2 = weight.reshape({1, cols});
  Tensor b2 = bias.reshape({1, cols});
  Tensor out = (normalized * w2 + b2).reshape(x.shape());
  record_registered_trace(OpId::LayerNorm, out, {x, weight, bias}, {},
                          {{"eps", eps}});
  return out;
}

} // namespace ops
} // namespace munet
