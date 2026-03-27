#include "loss.hpp"

#include "../../autograd/nodes/loss_nodes.hpp"

namespace munet {
namespace ops {

Tensor mse_loss(const Tensor &pred, const Tensor &target) {
  const auto dispatch = resolve_dispatch(OpId::MSELoss, pred);
  detail::require_same_dtype(op_metadata(OpId::MSELoss).name, pred, target);
  if (pred.shape() == target.shape()) {
    Tensor out({1}, pred.device(), pred.dtype());
    if (dispatch.use_cpu_fallback) {
      Device cpu{DeviceType::CPU, 0};
      Tensor pred_exec = pred.to(cpu);
      Tensor target_exec = target.to(cpu);
      if (pred_exec.dtype() != DataType::Float32) {
        pred_exec = pred_exec.to(DataType::Float32);
        target_exec = target_exec.to(DataType::Float32);
      }
      Tensor out_exec({1}, cpu, pred_exec.dtype());
      pred_exec.impl_->backend().mse_loss(*pred_exec.impl_->storage,
                                          *target_exec.impl_->storage,
                                          *out_exec.impl_->storage,
                                          pred_exec.size());
      if (out_exec.dtype() != pred.dtype()) {
        out_exec = out_exec.to(pred.dtype());
      }
      out = (pred.device().type == DeviceType::CPU) ? out_exec
                                                     : out_exec.to(pred.device());
    } else {
      pred.impl_->backend().mse_loss(*pred.impl_->storage,
                                     *target.impl_->storage,
                                     *out.impl_->storage, pred.size());
    }
    if (GradMode::is_enabled() && pred.requires_grad()) {
      auto fn = std::make_shared<autograd_nodes::MSELossBackward>(pred, target);
      link_backward_edges(fn.get(), {pred, target});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    }
    record_registered_trace(OpId::MSELoss, out, {pred, target});
    return out;
  }

  if (pred.shape().size() == 4 && target.shape().size() == 4 &&
      pred.shape()[0] == target.shape()[0] &&
      pred.shape()[2] == target.shape()[2] &&
      pred.shape()[3] == target.shape()[3]) {
    const int pC = pred.shape()[1];
    const int tC = target.shape()[1];
    if (pC == 1 && tC > 1) {
      std::vector<Tensor> expanded(tC, pred);
      return mse_loss(cat(expanded, 1), target);
    }
    if (tC == 1 && pC > 1) {
      std::vector<Tensor> expanded(pC, target);
      return mse_loss(pred, cat(expanded, 1));
    }
  }

  throw std::runtime_error(
      "MSELoss: shape mismatch. Pred=" + to_string(pred.shape()) +
      " Target=" + to_string(target.shape()));
}

Tensor cross_entropy(const Tensor &logits, const Tensor &targets) {
  const auto dispatch = resolve_dispatch(OpId::CrossEntropy, logits);
  detail::require_same_dtype(op_metadata(OpId::CrossEntropy).name, logits,
                             targets);
  if (logits.shape() != targets.shape()) {
    throw std::runtime_error(
        "CrossEntropy: shape mismatch. Logits=" + to_string(logits.shape()) +
        " Targets=" + to_string(targets.shape()));
  }

  const int batch_size = logits.shape().empty() ? 1 : logits.shape()[0];
  int num_classes;
  int spatial;
  if (logits.shape().size() == 4) {
    num_classes = logits.shape()[1];
    spatial = logits.shape()[2] * logits.shape()[3];
  } else {
    num_classes = logits.size() / batch_size;
    spatial = 1;
  }

  Tensor out({1}, logits.device(), logits.dtype());
  if (dispatch.use_cpu_fallback) {
    Device cpu{DeviceType::CPU, 0};
    Tensor logits_exec = logits.to(cpu);
    Tensor targets_exec = targets.to(cpu);
    if (logits_exec.dtype() != DataType::Float32) {
      logits_exec = logits_exec.to(DataType::Float32);
      targets_exec = targets_exec.to(DataType::Float32);
    }
    Tensor out_exec({1}, cpu, logits_exec.dtype());
    logits_exec.impl_->backend().cross_entropy(
        *logits_exec.impl_->storage, *targets_exec.impl_->storage,
        *out_exec.impl_->storage, batch_size, num_classes, spatial);
    if (out_exec.dtype() != logits.dtype()) {
      out_exec = out_exec.to(logits.dtype());
    }
    out = (logits.device().type == DeviceType::CPU) ? out_exec
                                                     : out_exec.to(logits.device());
  } else {
    logits.impl_->backend().cross_entropy(
        *logits.impl_->storage, *targets.impl_->storage, *out.impl_->storage,
        batch_size, num_classes, spatial);
  }

  if (GradMode::is_enabled() && logits.requires_grad()) {
    auto fn = std::make_shared<autograd_nodes::CrossEntropyBackward>(
        logits, targets, batch_size, num_classes, spatial);
    link_backward_edges(fn.get(), {logits, targets});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_registered_trace(OpId::CrossEntropy, out, {logits, targets});
  return out;
}

} // namespace ops
} // namespace munet
