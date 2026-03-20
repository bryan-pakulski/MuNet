#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MaxPool2DBackward : public Node {
  Shape input_shape;
  Device input_device;
  DataType input_dtype;
  int k, s, p;
  MaxPool2DBackward(Tensor i, int k_, int s_, int p_)
      : input_shape(i.shape()), input_device(i.device()),
        input_dtype(i.dtype()), k(k_), s(s_), p(p_) {
    save_tensor(i, "max_pool_input");
  }
  std::string name() const override { return "MaxPool2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor in = saved_tensor(0);
    Tensor grad_in(input_shape, input_device, input_dtype);
    grad_in.impl_->storage->zero_();
    const int B = input_shape[0];
    const int C = input_shape[1];
    const int iH = input_shape[2];
    const int iW = input_shape[3];
    in.impl_->backend().max_pool2d_backward(
        *grads[0].impl_->storage, *in.impl_->storage, *grad_in.impl_->storage,
        B, C, iH, iW, k, s, p);
    return {grad_in};
  }
};

struct Upsample2DBackward : public Node {
  Shape input_shape;
  Device input_device;
  DataType input_dtype;
  int scale;
  Upsample2DBackward(Tensor i, int sc)
      : input_shape(i.shape()), input_device(i.device()),
        input_dtype(i.dtype()), scale(sc) {
    save_tensor(i, "upsample_input");
  }
  std::string name() const override { return "Upsample2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor in = saved_tensor(0);
    Tensor grad_in(input_shape, input_device, input_dtype);
    grad_in.impl_->storage->zero_();
    const int B = input_shape[0];
    const int C = input_shape[1];
    const int iH = input_shape[2];
    const int iW = input_shape[3];
    in.impl_->backend().upsample2d_backward(
        *grads[0].impl_->storage, *grad_in.impl_->storage, B, C, iH, iW, scale);
    return {grad_in};
  }
};

} // namespace autograd_nodes
} // namespace munet
