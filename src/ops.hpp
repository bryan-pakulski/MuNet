#pragma once
#include "autograd/autograd.hpp"
#include "types.hpp"
#include "util.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace munet {
namespace ops {

// 1. Forward Declarations (so structs can see the functions)
inline Tensor add(const Tensor &a, const Tensor &b);
inline Tensor sub(const Tensor &a, const Tensor &b);
inline Tensor mul(const Tensor &a, const Tensor &b);
inline Tensor layer_norm(const Tensor &x, const Tensor &weight,
                         const Tensor &bias, float eps = 1e-5f);
inline Tensor masked_fill(const Tensor &a, const Tensor &mask, float value);

inline void link_backward_edges(Node *node, const std::vector<Tensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &t = inputs[i];
    if (t.impl_->grad_fn) {
      node->next_edges.push_back({t.impl_->grad_fn, (int)i});
    } else if (t.requires_grad()) {
      auto acc_node = std::make_shared<AccumulateGrad>(t.impl_);
      node->next_edges.push_back({acc_node, (int)i});
    } else {
      node->next_edges.push_back({nullptr, 0});
    }
  }
}

inline void record_trace(
    Tensor &out, const std::string &op_name, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs = {},
    const std::unordered_map<std::string, float> &float_attrs = {}) {
  auto fn = std::make_shared<ForwardNode>();
  fn->op_name = op_name;
  for (const auto &t : inputs) {
    if (t.name().empty())
      t.impl_->name =
          "tensor_" +
          std::to_string(reinterpret_cast<uintptr_t>(t.impl_.get()));
    fn->input_names.push_back(t.name());
    fn->inputs.push_back(t);
  }
  fn->int_attributes = int_attrs;
  fn->attributes = float_attrs;
  out.impl_->trace_node = fn;
}

inline Tensor sum_to_shape(const Tensor &t, const Shape &target_shape) {
  if (t.shape() == target_shape)
    return t;

  Tensor out(target_shape, t.device(), t.dtype());
  t.impl_->backend().sum_to_shape(*t.impl_->storage, *out.impl_->storage,
                                  t.shape(), target_shape);
  return out;
}

// --- ADD ---
struct AddBackward : public Node {
  Shape shape_a, shape_b;
  AddBackward(Shape a, Shape b) : shape_a(a), shape_b(b) {}
  std::string name() const override { return "AddBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {sum_to_shape(grads[0], shape_a), sum_to_shape(grads[0], shape_b)};
  }
};

inline Tensor broadcast_expand(const Tensor &src, int N) {
  Tensor out({N, (int)src.size()}, src.device(), src.dtype());
  src.impl_->backend().broadcast_row(*src.impl_->storage, *out.impl_->storage,
                                     N, src.size());
  return out;
}

inline Tensor expand_scalar(const Tensor &scalar, const Shape &target_shape) {
  Tensor out(target_shape, scalar.device(), scalar.dtype());
  // Use uniform_ to fill the tensor with the scalar value
  Tensor cpu_scalar = scalar.to(Device{DeviceType::CPU, 0});
  float val = ((float *)cpu_scalar.data())[0];
  out.uniform_(val, val);
  return out;
}

inline Tensor add(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Add: device mismatch");

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Add: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  Tensor out(info.out_shape, a.device(), a.dtype());
  a.impl_->backend().add(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, info);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<AddBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Add", {a, b});
  return out;
}

// --- CONCAT ---
struct ConcatBackward : public Node {
  int dim;
  std::vector<Shape> input_shapes;
  ConcatBackward(int d, const std::vector<Shape> &s)
      : dim(d), input_shapes(s) {}

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    std::vector<Tensor> grad_ins;
    std::vector<Storage *> grad_stors;

    for (const auto &s : input_shapes) {
      grad_ins.emplace_back(s, grad_out.device(), grad_out.dtype());
      grad_stors.push_back(grad_ins.back().impl_->storage.get());
    }

    grad_out.impl_->backend().concat_backward(*grad_out.impl_->storage,
                                              grad_stors, dim, input_shapes);
    return grad_ins;
  }
};

inline Tensor cat(const std::vector<Tensor> &inputs, int dim = 1) {
  if (inputs.empty())
    return Tensor();

  Shape out_shape = inputs[0].shape();
  std::vector<Shape> shapes;
  std::vector<Storage *> storages;
  int concat_size = 0;

  // Collect shapes and storages
  for (const auto &t : inputs) {
    const auto &s = t.shape();
    if (s.size() != out_shape.size())
      throw std::runtime_error(
          "All tensors must have the same number of dimensions for cat");

    // Check other dims match
    for (size_t d = 0; d < s.size(); ++d) {
      if (d != dim && s[d] != out_shape[d])
        throw std::runtime_error(
            "Tensor shapes must match except along concat dim");
    }

    concat_size += s[dim]; // Add size along the concat axis
    shapes.push_back(s);
    storages.push_back(t.impl_->storage.get());
  }

  // Set concatenated dimension in output shape
  out_shape[dim] = concat_size;

  Tensor out(out_shape, inputs[0].device(), inputs[0].dtype());

  // Call backend to perform actual concatenation
  inputs[0].impl_->backend().concat(storages, *out.impl_->storage, dim, shapes);

  // Setup backward graph if any input requires grad
  bool req = false;
  for (auto &t : inputs)
    if (t.requires_grad())
      req = true;

  if (GradMode::is_enabled() && req) {
    auto fn = std::make_shared<ConcatBackward>(dim, shapes);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  record_trace(out, "Concat", inputs, {{"axis", {dim}}});
  return out;
}

// --- MSE LOSS ---
struct MSELossBackward : public Node {
  Tensor pred, target;
  MSELossBackward(Tensor p, Tensor t) : pred(p), target(t) {}
  std::string name() const override { return "MSELossBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(pred.shape(), pred.device(), pred.dtype());
    pred.impl_->backend().mse_loss_backward(
        *grad_out.impl_->storage, *pred.impl_->storage, *target.impl_->storage,
        *grad_in.impl_->storage, pred.size());
    // Only return gradient for predictions (targets don't need gradients)
    return {grad_in, Tensor()};
  }
};

inline Tensor mse_loss(const Tensor &pred, const Tensor &target) {
  if (pred.shape() == target.shape()) {
    Tensor out({1}, pred.device(), pred.dtype());
    pred.impl_->backend().mse_loss(*pred.impl_->storage, *target.impl_->storage,
                                   *out.impl_->storage, pred.size());

    if (GradMode::is_enabled() && pred.requires_grad()) {
      auto fn = std::make_shared<MSELossBackward>(pred, target);
      link_backward_edges(fn.get(), {pred, target});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    }
    record_trace(out, "MSELoss", {pred, target});
    return out;
  }

  // Auto-broadcast for NCHW if channels differ (1 vs C)
  if (pred.shape().size() == 4 && target.shape().size() == 4 &&
      pred.shape()[0] == target.shape()[0] &&
      pred.shape()[2] == target.shape()[2] &&
      pred.shape()[3] == target.shape()[3]) {

    int pC = pred.shape()[1];
    int tC = target.shape()[1];

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

// --- CROSS ENTROPY LOSS ---
struct CrossEntropyBackward : public Node {
  Tensor logits, targets;
  int batch_size, num_classes, spatial;
  CrossEntropyBackward(Tensor l, Tensor t, int b, int c, int s)
      : logits(l), targets(t), batch_size(b), num_classes(c), spatial(s) {}
  std::string name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(logits.shape(), logits.device(), logits.dtype());
    logits.impl_->backend().cross_entropy_backward(
        *grad_out.impl_->storage, *logits.impl_->storage,
        *targets.impl_->storage, *grad_in.impl_->storage, batch_size,
        num_classes, spatial);
    return {grad_in, Tensor()};
  }
};

inline Tensor cross_entropy(const Tensor &logits, const Tensor &targets) {
  if (logits.shape() != targets.shape()) {
    throw std::runtime_error(
        "CrossEntropy: shape mismatch. Logits=" + to_string(logits.shape()) +
        " Targets=" + to_string(targets.shape()));
  }

  int batch_size = logits.shape().size() > 0 ? logits.shape()[0] : 1;
  int num_classes, spatial;

  if (logits.shape().size() == 4) {
    // NCHW
    num_classes = logits.shape()[1];
    spatial = logits.shape()[2] * logits.shape()[3];
  } else {
    // NC or flattened
    num_classes = logits.size() / batch_size;
    spatial = 1;
  }

  Tensor out({1}, logits.device(), logits.dtype());
  logits.impl_->backend().cross_entropy(
      *logits.impl_->storage, *targets.impl_->storage, *out.impl_->storage,
      batch_size, num_classes, spatial);

  if (GradMode::is_enabled() && logits.requires_grad()) {
    auto fn = std::make_shared<CrossEntropyBackward>(
        logits, targets, batch_size, num_classes, spatial);
    link_backward_edges(fn.get(), {logits, targets});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "CrossEntropy", {logits, targets});
  return out;
}



// --- LAYER NORM (CPU fallback with autograd) ---
struct LayerNormBackward : public Node {
  Tensor x, weight, bias;
  Tensor mean_cpu, inv_std_cpu;
  int rows, cols;
  LayerNormBackward(Tensor x_, Tensor w_, Tensor b_, Tensor m_, Tensor is_,
                    int r, int c)
      : x(x_), weight(w_), bias(b_), mean_cpu(m_), inv_std_cpu(is_), rows(r),
        cols(c) {}

  std::string name() const override { return "LayerNormBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor go_cpu = grads[0].to(cpu);
    Tensor x_cpu = x.to(cpu);
    Tensor w_cpu = weight.to(cpu);

    Tensor dx_cpu(x.shape(), cpu, x.dtype());
    Tensor dw_cpu(weight.shape(), cpu, weight.dtype());
    Tensor db_cpu(bias.shape(), cpu, bias.dtype());
    dw_cpu.uniform_(0.0f, 0.0f);
    db_cpu.uniform_(0.0f, 0.0f);

    const float *go = static_cast<const float *>(go_cpu.data());
    const float *xv = static_cast<const float *>(x_cpu.data());
    const float *wv = static_cast<const float *>(w_cpu.data());
    const float *mv = static_cast<const float *>(mean_cpu.data());
    const float *iv = static_cast<const float *>(inv_std_cpu.data());

    float *dx = static_cast<float *>(dx_cpu.data());
    float *dw = static_cast<float *>(dw_cpu.data());
    float *db = static_cast<float *>(db_cpu.data());

    for (int r = 0; r < rows; ++r) {
      const float mean = mv[r];
      const float inv = iv[r];
      float sum_gy = 0.0f;
      float sum_gy_xhat = 0.0f;

      for (int c = 0; c < cols; ++c) {
        int idx = r * cols + c;
        float xhat = (xv[idx] - mean) * inv;
        float gy = go[idx] * wv[c];
        sum_gy += gy;
        sum_gy_xhat += gy * xhat;
        dw[c] += go[idx] * xhat;
        db[c] += go[idx];
      }

      for (int c = 0; c < cols; ++c) {
        int idx = r * cols + c;
        float xhat = (xv[idx] - mean) * inv;
        float gy = go[idx] * wv[c];
        dx[idx] = (inv / cols) *
                  (cols * gy - sum_gy - xhat * sum_gy_xhat);
      }
    }

    Tensor dx_dev = (x.device().type == DeviceType::CPU) ? dx_cpu
                                                          : dx_cpu.to(x.device());
    Tensor dw_dev = (weight.device().type == DeviceType::CPU)
                        ? dw_cpu
                        : dw_cpu.to(weight.device());
    Tensor db_dev = (bias.device().type == DeviceType::CPU) ? db_cpu
                                                             : db_cpu.to(bias.device());
    return {dx_dev, dw_dev, db_dev};
  }
};

inline Tensor layer_norm(const Tensor &x, const Tensor &weight,
                         const Tensor &bias, float eps) {
  if (x.shape().empty())
    throw std::runtime_error("LayerNorm: input must have at least 1 dim");

  int cols = x.shape().back();
  if ((int)weight.size() != cols || (int)bias.size() != cols)
    throw std::runtime_error("LayerNorm: weight/bias size must equal last dim");

  int rows = (int)(x.size() / cols);
  Device cpu{DeviceType::CPU, 0};

  Tensor x_cpu = x.to(cpu);
  Tensor w_cpu = weight.to(cpu);
  Tensor b_cpu = bias.to(cpu);

  Tensor out_cpu(x.shape(), cpu, x.dtype());
  Tensor mean_cpu({rows}, cpu, x.dtype(), false);
  Tensor inv_std_cpu({rows}, cpu, x.dtype(), false);

  const float *xv = static_cast<const float *>(x_cpu.data());
  const float *wv = static_cast<const float *>(w_cpu.data());
  const float *bv = static_cast<const float *>(b_cpu.data());
  float *ov = static_cast<float *>(out_cpu.data());
  float *mv = static_cast<float *>(mean_cpu.data());
  float *iv = static_cast<float *>(inv_std_cpu.data());

  for (int r = 0; r < rows; ++r) {
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c)
      mean += xv[r * cols + c];
    mean /= cols;

    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float d = xv[r * cols + c] - mean;
      var += d * d;
    }
    var /= cols;

    float inv = 1.0f / std::sqrt(var + eps);
    mv[r] = mean;
    iv[r] = inv;

    for (int c = 0; c < cols; ++c) {
      float xhat = (xv[r * cols + c] - mean) * inv;
      ov[r * cols + c] = xhat * wv[c] + bv[c];
    }
  }

  Tensor out = (x.device().type == DeviceType::CPU) ? out_cpu
                                                     : out_cpu.to(x.device());

  if (GradMode::is_enabled() &&
      (x.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    auto fn = std::make_shared<LayerNormBackward>(x, weight, bias, mean_cpu,
                                                  inv_std_cpu, rows, cols);
    link_backward_edges(fn.get(), {x, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  record_trace(out, "LayerNorm", {x, weight, bias}, {}, {{"eps", eps}});
  return out;
}

// --- RELU ---
struct ReluBackward : public Node {
  Tensor saved_input;
  ReluBackward(Tensor a) : saved_input(a) {}
  std::string name() const override { return "ReluBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(saved_input.shape(), saved_input.device(),
                   saved_input.dtype());
    saved_input.impl_->backend().relu_backward(
        *grad_out.impl_->storage, *saved_input.impl_->storage,
        *grad_in.impl_->storage, saved_input.size());
    return {grad_in};
  }
};

inline Tensor relu(const Tensor &a) {
  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().relu(*a.impl_->storage, *out.impl_->storage, a.size());

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<ReluBackward>(a);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Relu", {a});
  return out;
}

// --- SIGMOID ---
struct SigmoidBackward : public Node {
  Tensor saved_out;
  SigmoidBackward(Tensor o) : saved_out(o) {}
  std::string name() const override { return "SigmoidBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(saved_out.shape(), saved_out.device(), saved_out.dtype());
    saved_out.impl_->backend().sigmoid_backward(
        *grad_out.impl_->storage, *saved_out.impl_->storage,
        *grad_in.impl_->storage, saved_out.size());
    return {grad_in};
  }
};

inline Tensor sigmoid(const Tensor &a) {
  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().sigmoid(*a.impl_->storage, *out.impl_->storage, a.size());

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SigmoidBackward>(out);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Sigmoid", {a});
  return out;
}

// --- SOFTMAX ---
struct SoftmaxBackward : public Node {
  Tensor saved_out;
  int batch_size, num_classes;
  SoftmaxBackward(Tensor o, int b, int c)
      : saved_out(o), batch_size(b), num_classes(c) {}
  std::string name() const override { return "SoftmaxBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(saved_out.shape(), saved_out.device(), saved_out.dtype());
    saved_out.impl_->backend().softmax_backward(
        *grad_out.impl_->storage, *saved_out.impl_->storage,
        *grad_in.impl_->storage, batch_size, num_classes);
    return {grad_in};
  }
};

inline Tensor softmax(const Tensor &a, int dim = -1) {
  if (a.shape().size() < 2)
    throw std::runtime_error("Softmax expects at least 2D tensor");

  int rank = static_cast<int>(a.shape().size());
  int resolved = (dim < 0) ? (rank + dim) : dim;
  if (resolved < 0 || resolved >= rank)
    throw std::runtime_error("Softmax: dim out of range");
  if (resolved != rank - 1)
    throw std::runtime_error(
        "Softmax currently supports only the last dimension");

  int num_classes = a.shape().back();
  int batch_size = a.size() / num_classes;

  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().softmax(*a.impl_->storage, *out.impl_->storage, batch_size,
                             num_classes);

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SoftmaxBackward>(out, batch_size, num_classes);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Softmax", {a}, {{"dim", {resolved}}});
  return out;
}

inline Tensor masked_fill(const Tensor &a, const Tensor &mask, float value) {
  if (a.device() != mask.device())
    throw std::runtime_error("masked_fill: device mismatch");
  if (a.shape() != mask.shape())
    throw std::runtime_error("masked_fill: shape mismatch");

  Tensor one({1}, a.device(), a.dtype(), false);
  one.uniform_(1.0f, 1.0f);
  Tensor fill({1}, a.device(), a.dtype(), false);
  fill.uniform_(value, value);

  Tensor kept = a * (one - mask);
  Tensor filled = fill * mask;
  Tensor out = kept + filled;
  record_trace(out, "MaskedFill", {a, mask}, {}, {{"value", value}});
  return out;
}

// --- MATMUL ---
inline Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                              bool transB) {
  if (a.shape().size() != 2 || b.shape().size() != 2)
    throw std::runtime_error("Matmul currently requires 2D tensors");

  int M = transA ? a.shape()[1] : a.shape()[0];
  int K = transA ? a.shape()[0] : a.shape()[1];
  int N = transB ? b.shape()[0] : b.shape()[1];

  Tensor out({M, N}, a.device(), a.dtype());
  a.impl_->backend().matmul(*a.impl_->storage, *b.impl_->storage,
                            *out.impl_->storage, M, K, N, transA, transB);
  return out;
}

struct MatmulBackward : public Node {
  Tensor A, B;
  MatmulBackward(Tensor a, Tensor b) : A(a), B(b) {}
  std::string name() const override { return "MatmulBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_a, grad_b;

    // dA = dC @ B^T
    if (next_edges.size() > 0 && next_edges[0].node) {
      grad_a = matmul_internal(grad_out, B, false, true);
    } else {
      grad_a = Tensor();
    }

    // dB = A^T @ dC
    if (next_edges.size() > 1 && next_edges[1].node) {
      grad_b = matmul_internal(A, grad_out, true, false);
    } else {
      grad_b = Tensor();
    }

    return {grad_a, grad_b};
  }
};

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Matmul: device mismatch");

  Tensor out = matmul_internal(a, b, false, false);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<MatmulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "MatMul", {a, b});
  return out;
}

// --- SUB ---
struct SubBackward : public Node {
  Shape shape_a, shape_b;
  SubBackward(Shape a, Shape b) : shape_a(a), shape_b(b) {}
  std::string name() const override { return "SubBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    // db = -grad_out.
    Tensor zeros(grad_out.shape(), grad_out.device(), grad_out.dtype());
    zeros.impl_->storage->zero_();
    Tensor neg_grad_out = sub(zeros, grad_out); // Recursive call is safe here

    return {sum_to_shape(grad_out, shape_a),
            sum_to_shape(neg_grad_out, shape_b)};
  }
};

inline Tensor sub(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Sub: device mismatch");

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Sub: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  Tensor out(info.out_shape, a.device(), a.dtype());
  a.impl_->backend().sub(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, info);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<SubBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Sub", {a, b});
  return out;
}

// --- MUL ---
struct MulBackward : public Node {
  Tensor A, B;
  MulBackward(Tensor a, Tensor b) : A(a), B(b) {}
  std::string name() const override { return "MulBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    // dC/dA = B * grad_out, then reduce to A's shape
    Tensor da = mul(B, grad_out);
    // dC/dB = A * grad_out, then reduce to B's shape
    Tensor db = mul(A, grad_out);

    return {sum_to_shape(da, A.shape()), sum_to_shape(db, B.shape())};
  }
};

inline Tensor mul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Mul: device mismatch");

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Mul: shape mismatch");
  }

  Tensor out(info.out_shape, a.device(), a.dtype());
  a.impl_->backend().mul(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, info);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<MulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Mul", {a, b});
  return out;
}

// --- SUM (Global) ---
struct SumBackward : public Node {
  Shape shape;
  Device dev;
  SumBackward(Shape s, Device d) : shape(s), dev(d) {}
  std::string name() const override { return "SumBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // grad_out is a scalar on the device.
    // We need to broadcast it.
    // For simplicity, we can just copy that 1 value to a full tensor.
    // Since we don't have a "Broadcast" kernel yet, we can fallback to CPU for
    // the BACKWARD pass only or implement a fill kernel. For now, let's keep
    // the backward pass simple, but ensure FORWARD is fast.

    Tensor grad_out_cpu = grads[0].to(Device{DeviceType::CPU, 0});
    float g = ((float *)grad_out_cpu.data())[0];

    Tensor cpu_grad_in(shape, Device{DeviceType::CPU, 0}, DataType::Float32);
    float *dest = (float *)cpu_grad_in.data();
    for (size_t i = 0; i < numel(shape); ++i)
      dest[i] = g;

    return {cpu_grad_in.to(dev)};
  }
};

inline Tensor sum(const Tensor &a) {
  Tensor out({1}, a.device(), a.dtype());

  // Use Backend!
  a.impl_->backend().sum(*a.impl_->storage, *out.impl_->storage, a.size());

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SumBackward>(a.shape(), a.device());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "ReduceSum", {a}, {{"keepdims", {0}}});
  return out;
}

struct ReshapeBackward : public Node {
  Shape input_shape;
  ReshapeBackward(Shape s) : input_shape(s) {}
  std::string name() const override { return "ReshapeBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // Gradient just needs to be reshaped back to original input layout
    return {grads[0].reshape(input_shape)};
  }
};

inline Tensor reshape(const Tensor &in, Shape new_shape) {
  size_t src_el = numel(in.shape());
  size_t dst_el = numel(new_shape);
  if (src_el != dst_el)
    throw std::runtime_error("Reshape: element count mismatch");

  Tensor out;
  // Create new TensorImpl pointing to the SAME storage (View)
  out.impl_ = std::make_shared<TensorImpl>(new_shape, in.device(), in.dtype(),
                                           in.requires_grad());
  out.impl_->storage = in.impl_->storage;

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<ReshapeBackward>(in.shape());
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Reshape", {in}, {{"shape", new_shape}});
  return out;
}

// --- CONV2D ---
struct Conv2DBackward : public Node {
  Tensor in, weight, bias;
  int stride, padding;
  Conv2DBackward(Tensor i, Tensor w, Tensor b, int s, int p)
      : in(i), weight(w), bias(b), stride(s), padding(p) {}
  std::string name() const override { return "Conv2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_w(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_b;
    if (bias.impl_)
      grad_b = Tensor(bias.shape(), bias.device(), bias.dtype());
    grad_in.impl_->storage->zero_();
    grad_w.impl_->storage->zero_();
    if (bias.impl_)
      grad_b.impl_->storage->zero_();
    int B = in.shape()[0], iC = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    int oC = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    in.impl_->backend().conv2d_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *grad_in.impl_->storage, *grad_w.impl_->storage,
        bias.impl_ ? grad_b.impl_->storage.get() : nullptr, B, iC, iH, iW, oC,
        kH, kW, stride, padding);
    return {grad_in, grad_w, grad_b};
  }
};
inline Tensor conv2d(const Tensor &in, const Tensor &weight, const Tensor &bias,
                     int stride, int padding) {
  if (in.device() != weight.device() ||
      (bias.impl_ && bias.device() != in.device())) {
    MUNET_ERROR << "conv2d: inputs not on same device: "
                << in.device().to_string()
                << " != " << weight.device().to_string() << std::endl;
    throw std::runtime_error("Conv2d: inputs must be on same device");
  }

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

  int B = in.shape()[0], iC = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  int oC = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
  int oH = (iH + 2 * padding - kH) / stride + 1;
  int oW = (iW + 2 * padding - kW) / stride + 1;
  Tensor out({B, oC, oH, oW}, in.device(), in.dtype());
  in.impl_->backend().conv2d(*in.impl_->storage, *weight.impl_->storage,
                             bias.impl_ ? bias.impl_->storage.get() : nullptr,
                             *out.impl_->storage, B, iC, iH, iW, oC, kH, kW,
                             stride, padding);
  if (GradMode::is_enabled() && (in.requires_grad() || weight.requires_grad() ||
                                 (bias.impl_ && bias.requires_grad()))) {
    auto fn =
        std::make_shared<Conv2DBackward>(in, weight, bias, stride, padding);
    std::vector<Tensor> inputs = {in, weight};
    if (bias.impl_)
      inputs.push_back(bias);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  std::vector<Tensor> inputs = {in, weight};
  if (bias.impl_)
    inputs.push_back(bias);
  record_trace(out, "Conv", inputs,
               {{"strides", {stride, stride}},
                {"pads", {padding, padding, padding, padding}}});
  return out;
}
// --- MAXPOOL2D ---
struct MaxPool2DBackward : public Node {
  Tensor in;
  int k, s, p;
  MaxPool2DBackward(Tensor i, int k_, int s_, int p_)
      : in(i), k(k_), s(s_), p(p_) {}
  std::string name() const override { return "MaxPool2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    in.impl_->backend().max_pool2d_backward(
        *grads[0].impl_->storage, *in.impl_->storage, *grad_in.impl_->storage,
        B, C, iH, iW, k, s, p);
    return {grad_in};
  }
};
inline Tensor max_pool2d(const Tensor &in, int kernel_size, int stride,
                         int padding) {
  int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  int oH = (iH + 2 * padding - kernel_size) / stride + 1;
  int oW = (iW + 2 * padding - kernel_size) / stride + 1;
  Tensor out({B, C, oH, oW}, in.device(), in.dtype());
  in.impl_->backend().max_pool2d(*in.impl_->storage, *out.impl_->storage, B, C,
                                 iH, iW, kernel_size, stride, padding);

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn =
        std::make_shared<MaxPool2DBackward>(in, kernel_size, stride, padding);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "MaxPool", {in},
               {{"kernel_shape", {kernel_size, kernel_size}},
                {"strides", {stride, stride}},
                {"pads", {padding, padding, padding, padding}}});
  return out;
}
// --- UPSAMPLE2D ---
struct Upsample2DBackward : public Node {
  Tensor in;
  int scale;
  Upsample2DBackward(Tensor i, int sc) : in(i), scale(sc) {}
  std::string name() const override { return "Upsample2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    in.impl_->backend().upsample2d_backward(
        *grads[0].impl_->storage, *grad_in.impl_->storage, B, C, iH, iW, scale);
    return {grad_in};
  }
};
inline Tensor upsample2d(const Tensor &in, int scale_factor) {
  int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  Tensor out({B, C, iH * scale_factor, iW * scale_factor}, in.device(),
             in.dtype());
  in.impl_->backend().upsample2d(*in.impl_->storage, *out.impl_->storage, B, C,
                                 iH, iW, scale_factor);

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<Upsample2DBackward>(in, scale_factor);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Upsample2d", {in}, {{"scale", {scale_factor}}});
  return out;
}

// --- BATCH NORM ---
struct BatchNormBackward : public Node {
  Tensor in, weight, save_mean, save_var;
  float eps;
  BatchNormBackward(Tensor i, Tensor w, Tensor sm, Tensor sv, float e)
      : in(i), weight(w), save_mean(sm), save_var(sv), eps(e) {}
  std::string name() const override { return "BatchNormBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_scale(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_bias(weight.shape(), weight.device(), weight.dtype());

    int B = in.shape()[0], C = in.shape()[1], H = in.shape()[2],
        W = in.shape()[3];
    in.impl_->backend().batch_norm_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *save_mean.impl_->storage, *save_var.impl_->storage,
        *grad_in.impl_->storage, *grad_scale.impl_->storage,
        *grad_bias.impl_->storage, B, C, H, W, eps);

    // Inputs: in (0), weight (1), bias (2)
    return {grad_in, grad_scale, grad_bias};
  }
};

inline Tensor batch_norm(const Tensor &in, Tensor &running_mean,
                         Tensor &running_var, const Tensor &weight,
                         const Tensor &bias, bool training, float momentum,
                         float eps) {
  int B = in.shape()[0], C = in.shape()[1], H = in.shape()[2],
      W = in.shape()[3];
  Tensor out(in.shape(), in.device(), in.dtype());

  // We need to save mean/var for backward if training
  Tensor save_mean({C}, in.device(), DataType::Float32);
  Tensor save_var({C}, in.device(), DataType::Float32);

  in.impl_->backend().batch_norm(
      *in.impl_->storage, *weight.impl_->storage, *bias.impl_->storage,
      *running_mean.impl_->storage, *running_var.impl_->storage,
      *save_mean.impl_->storage, *save_var.impl_->storage, *out.impl_->storage,
      B, C, H, W, momentum, eps, training);

  if (GradMode::is_enabled() && training &&
      (in.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    auto fn = std::make_shared<BatchNormBackward>(in, weight, save_mean,
                                                  save_var, eps);
    // Link inputs: input, weight, bias. Running stats do not participate in
    // gradient graph for backprop.
    link_backward_edges(fn.get(), {in, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "BatchNormalization",
               {in, weight, bias, running_mean, running_var}, {},
               {{"epsilon", eps}, {"momentum", momentum}});
  return out;
}

struct TransposeBackward : public Node {
  int d0, d1;
  TransposeBackward(int dim0, int dim1) : d0(dim0), d1(dim1) {}
  std::string name() const override { return "TransposeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // Transpose is its own inverse
    return {grads[0].transpose(d0, d1)};
  }
};

inline Tensor transpose(const Tensor &in, int dim0, int dim1) {
  Tensor out = in.transpose(dim0, dim1);
  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<TransposeBackward>(dim0, dim1);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Transpose", {in}, {{"dims", {dim0, dim1}}});
  return out;
}

inline Tensor zeros(Shape shape, Device device = Device{DeviceType::CPU, 0},
                    bool requires_grad = false) {
  Tensor t(shape, device, DataType::Float32, requires_grad);
  t.impl_->storage->zero_();
  return t;
}

} // namespace ops
} // namespace munet
