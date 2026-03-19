#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MSELossBackward : public Node {
  Shape pred_shape;
  Device pred_device;
  DataType pred_dtype;
  MSELossBackward(Tensor p, Tensor t)
      : pred_shape(p.shape()), pred_device(p.device()), pred_dtype(p.dtype()) {
    save_tensor(p, "mse_pred");
    save_tensor(t, "mse_target");
  }
  std::string name() const override { return "MSELossBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor pred = saved_tensor(0);
    Tensor target = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor grad_in(pred_shape, pred_device, pred_dtype);
    pred.impl_->backend().mse_loss_backward(
        *grad_out.impl_->storage, *pred.impl_->storage, *target.impl_->storage,
        *grad_in.impl_->storage, pred.size());
    return {grad_in, Tensor()};
  }
};

struct CrossEntropyBackward : public Node {
  int batch_size, num_classes, spatial;
  Shape logits_shape;
  Device logits_device;
  DataType logits_dtype;
  CrossEntropyBackward(Tensor l, Tensor t, int b, int c, int s)
      : batch_size(b), num_classes(c), spatial(s), logits_shape(l.shape()),
        logits_device(l.device()), logits_dtype(l.dtype()) {
    save_tensor(l, "cross_entropy_logits");
    save_tensor(t, "cross_entropy_targets");
  }
  std::string name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor logits = saved_tensor(0);
    Tensor targets = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor grad_in(logits_shape, logits_device, logits_dtype);
    logits.impl_->backend().cross_entropy_backward(
        *grad_out.impl_->storage, *logits.impl_->storage,
        *targets.impl_->storage, *grad_in.impl_->storage, batch_size,
        num_classes, spatial);
    return {grad_in, Tensor()};
  }
};

} // namespace autograd_nodes
} // namespace munet
