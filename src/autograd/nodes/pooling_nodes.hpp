#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MaxPool2DBackward : public Node {
  Tensor in;
  int k, s, p;
  MaxPool2DBackward(Tensor i, int k_, int s_, int p_)
      : in(std::move(i)), k(k_), s(s_), p(p_) {}
  std::string name() const override { return "MaxPool2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    const int B = in.shape()[0];
    const int C = in.shape()[1];
    const int iH = in.shape()[2];
    const int iW = in.shape()[3];
    in.impl_->backend().max_pool2d_backward(
        *grads[0].impl_->storage, *in.impl_->storage, *grad_in.impl_->storage,
        B, C, iH, iW, k, s, p);
    return {grad_in};
  }
};

struct Upsample2DBackward : public Node {
  Tensor in;
  int scale;
  Upsample2DBackward(Tensor i, int sc) : in(std::move(i)), scale(sc) {}
  std::string name() const override { return "Upsample2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    const int B = in.shape()[0];
    const int C = in.shape()[1];
    const int iH = in.shape()[2];
    const int iW = in.shape()[3];
    in.impl_->backend().upsample2d_backward(
        *grads[0].impl_->storage, *grad_in.impl_->storage, B, C, iH, iW, scale);
    return {grad_in};
  }
};

} // namespace autograd_nodes
} // namespace munet
