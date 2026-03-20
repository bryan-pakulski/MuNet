#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct ReluBackward : public Node {
  Shape input_shape;
  Device input_device;
  DataType input_dtype;
  explicit ReluBackward(Tensor a)
      : input_shape(a.shape()), input_device(a.device()), input_dtype(a.dtype()) {
    save_tensor(a, "relu_input");
  }
  std::string name() const override { return "ReluBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor saved_input = saved_tensor(0);
    Tensor grad_out = grads[0];
    Tensor grad_in(input_shape, input_device, input_dtype);
    saved_input.impl_->backend().relu_backward(
        *grad_out.impl_->storage, *saved_input.impl_->storage,
        *grad_in.impl_->storage, saved_input.size());
    return {grad_in};
  }
};

struct SigmoidBackward : public Node {
  Shape output_shape;
  Device output_device;
  DataType output_dtype;
  explicit SigmoidBackward(Tensor o)
      : output_shape(o.shape()), output_device(o.device()), output_dtype(o.dtype()) {
    save_tensor(o, "sigmoid_output");
  }
  std::string name() const override { return "SigmoidBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor saved_out = saved_tensor(0);
    Tensor grad_out = grads[0];
    Tensor grad_in(output_shape, output_device, output_dtype);
    saved_out.impl_->backend().sigmoid_backward(
        *grad_out.impl_->storage, *saved_out.impl_->storage,
        *grad_in.impl_->storage, saved_out.size());
    return {grad_in};
  }
};


struct ExpBackward : public Node {
  Shape output_shape;
  Device output_device;
  DataType output_dtype;
  explicit ExpBackward(Tensor o)
      : output_shape(o.shape()), output_device(o.device()), output_dtype(o.dtype()) {
    save_tensor(o, "exp_output");
  }
  std::string name() const override { return "ExpBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor saved_out = saved_tensor(0);
    Tensor grad_out = grads[0];
    Tensor grad_in = ops::mul(grad_out, saved_out);
    return {grad_in};
  }
};

struct LogBackward : public Node {
  Shape input_shape;
  Device input_device;
  DataType input_dtype;
  explicit LogBackward(Tensor input)
      : input_shape(input.shape()), input_device(input.device()),
        input_dtype(input.dtype()) {
    save_tensor(input, "log_input");
  }
  std::string name() const override { return "LogBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor input = saved_tensor(0);
    Tensor grad_in = ops::div(grads[0], input);
    return {grad_in};
  }
};

struct SqrtBackward : public Node {
  Shape output_shape;
  Device output_device;
  DataType output_dtype;
  explicit SqrtBackward(Tensor o)
      : output_shape(o.shape()), output_device(o.device()), output_dtype(o.dtype()) {
    save_tensor(o, "sqrt_output");
  }
  std::string name() const override { return "SqrtBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor saved_out = saved_tensor(0);
    Tensor denom({1}, saved_out.device(), saved_out.dtype());
    denom.fill_(2.0f);
    Tensor grad_in = ops::div(grads[0], ops::mul(saved_out, denom));
    return {grad_in};
  }
};

struct SoftmaxBackward : public Node {
  int dim;
  Shape shape;
  Device device;
  DataType dtype;
  SoftmaxBackward(Tensor o, int d)
      : dim(d), shape(o.shape()), device(o.device()), dtype(o.dtype()) {
    save_tensor(o, "softmax_output");
  }
  std::string name() const override { return "SoftmaxBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override;
};

struct LogSoftmaxBackward : public Node {
  int dim;
  Shape shape;
  Device device;
  DataType dtype;
  LogSoftmaxBackward(Tensor lp, int d)
      : dim(d), shape(lp.shape()), device(lp.device()), dtype(lp.dtype()) {
    save_tensor(lp, "log_softmax_output");
  }
  std::string name() const override { return "LogSoftmaxBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override;
};

inline std::vector<Tensor>
SoftmaxBackward::apply(const std::vector<Tensor> &grads) {
  Tensor saved_out = saved_tensor(0);
  Device cpu{DeviceType::CPU, 0};
  Tensor go_cpu = grads[0].to(cpu);
  Tensor out_cpu = saved_out.to(cpu);
  Tensor gi_cpu(shape, cpu, dtype);

  int rank = static_cast<int>(shape.size());
  int resolved = (dim < 0) ? (rank + dim) : dim;
  int outer = 1;
  int inner = 1;
  int dim_size = shape[resolved];
  for (int i = 0; i < resolved; ++i)
    outer *= shape[i];
  for (int i = resolved + 1; i < rank; ++i)
    inner *= shape[i];

  const char *go = static_cast<const char *>(go_cpu.data());
  const char *out = static_cast<const char *>(out_cpu.data());
  char *gi = static_cast<char *>(gi_cpu.data());
  const size_t go_stride = dtype_size(go_cpu.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());
  const size_t gi_stride = dtype_size(gi_cpu.dtype());

  for (int o = 0; o < outer; ++o) {
    for (int in = 0; in < inner; ++in) {
      double dot = 0.0;
      for (int d = 0; d < dim_size; ++d) {
        int idx = (o * dim_size + d) * inner + in;
        dot += read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value *
               read_scalar_from_buffer(out + idx * out_stride, out_cpu.dtype()).value;
      }
      for (int d = 0; d < dim_size; ++d) {
        int idx = (o * dim_size + d) * inner + in;
        const double out_value =
            read_scalar_from_buffer(out + idx * out_stride, out_cpu.dtype()).value;
        const double go_value =
            read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
        write_scalar_to_buffer(gi + idx * gi_stride, gi_cpu.dtype(),
                               out_value * (go_value - dot));
      }
    }
  }

  Tensor gi_dev = (device.type == DeviceType::CPU) ? gi_cpu : gi_cpu.to(device);
  return {gi_dev};
}

inline std::vector<Tensor>
LogSoftmaxBackward::apply(const std::vector<Tensor> &grads) {
  Tensor saved_log_probs = saved_tensor(0);
  Device cpu{DeviceType::CPU, 0};
  Tensor go_cpu = grads[0].to(cpu);
  Tensor lp_cpu = saved_log_probs.to(cpu);
  Tensor gi_cpu(shape, cpu, dtype);

  int rank = static_cast<int>(shape.size());
  int resolved = (dim < 0) ? (rank + dim) : dim;
  int outer = 1;
  int inner = 1;
  int dim_size = shape[resolved];
  for (int i = 0; i < resolved; ++i)
    outer *= shape[i];
  for (int i = resolved + 1; i < rank; ++i)
    inner *= shape[i];

  const char *go = static_cast<const char *>(go_cpu.data());
  const char *lp = static_cast<const char *>(lp_cpu.data());
  char *gi = static_cast<char *>(gi_cpu.data());
  const size_t go_stride = dtype_size(go_cpu.dtype());
  const size_t lp_stride = dtype_size(lp_cpu.dtype());
  const size_t gi_stride = dtype_size(gi_cpu.dtype());

  for (int o = 0; o < outer; ++o) {
    for (int in = 0; in < inner; ++in) {
      double sum_go = 0.0;
      for (int d = 0; d < dim_size; ++d) {
        int idx = (o * dim_size + d) * inner + in;
        sum_go += read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
      }

      for (int d = 0; d < dim_size; ++d) {
        int idx = (o * dim_size + d) * inner + in;
        const double p = std::exp(
            read_scalar_from_buffer(lp + idx * lp_stride, lp_cpu.dtype()).value);
        const double go_value =
            read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
        write_scalar_to_buffer(gi + idx * gi_stride, gi_cpu.dtype(),
                               go_value - p * sum_go);
      }
    }
  }

  Tensor gi_dev = (device.type == DeviceType::CPU) ? gi_cpu : gi_cpu.to(device);
  return {gi_dev};
}

} // namespace autograd_nodes
} // namespace munet
