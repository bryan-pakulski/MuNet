#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct Conv2DBackward : public Node {
  Shape input_shape;
  Device input_device;
  DataType input_dtype;
  Shape weight_shape;
  Device weight_device;
  DataType weight_dtype;
  Shape bias_shape;
  Device bias_device;
  DataType bias_dtype = DataType::Float32;
  bool has_bias = false;
  int stride, padding;
  Conv2DBackward(Tensor i, Tensor w, Tensor b, int s, int p)
      : input_shape(i.shape()), input_device(i.device()), input_dtype(i.dtype()),
        weight_shape(w.shape()), weight_device(w.device()),
        weight_dtype(w.dtype()), stride(s), padding(p) {
    save_tensor(i, "conv2d_input");
    save_tensor(w, "conv2d_weight");
    if (b.impl_) {
      has_bias = true;
      bias_shape = b.shape();
      bias_device = b.device();
      bias_dtype = b.dtype();
      save_tensor(b, "conv2d_bias");
    }
  }
  std::string name() const override { return "Conv2DBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor in = saved_tensor(0);
    Tensor weight = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor grad_in(input_shape, input_device, input_dtype);
    Tensor grad_w(weight_shape, weight_device, weight_dtype);
    Tensor grad_b;
    if (has_bias) {
      grad_b = Tensor(bias_shape, bias_device, bias_dtype);
    }
    grad_in.impl_->storage->zero_();
    grad_w.impl_->storage->zero_();
    if (has_bias) {
      grad_b.impl_->storage->zero_();
    }

    const int B = input_shape[0];
    const int iC = input_shape[1];
    const int iH = input_shape[2];
    const int iW = input_shape[3];
    const int oC = weight_shape[0];
    const int kH = weight_shape[2];
    const int kW = weight_shape[3];
    in.impl_->backend().conv2d_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *grad_in.impl_->storage, *grad_w.impl_->storage,
        has_bias ? grad_b.impl_->storage.get() : nullptr, B, iC, iH, iW, oC,
        kH, kW, stride, padding);
    return {grad_in, grad_w, grad_b};
  }
};

} // namespace autograd_nodes
} // namespace munet
