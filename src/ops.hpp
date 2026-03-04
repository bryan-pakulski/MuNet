#pragma once
#include "autograd/autograd.hpp"
#include <cstdint>
#include <stdexcept>

namespace munet {
namespace ops {

inline void link_backward_edges(Node *node, const std::vector<Tensor> &inputs) {
  for (const auto &t : inputs) {
    if (t.impl_->grad_fn) {
      node->next_edges.push_back({t.impl_->grad_fn, 0});
    } else if (t.requires_grad()) {
      auto acc_node = std::make_shared<AccumulateGrad>(t.impl_);
      node->next_edges.push_back({acc_node, 0});
    } else {
      node->next_edges.push_back({nullptr, 0});
    }
  }
}

inline void record_trace(Tensor &out, const std::string &op_name,
                         const std::vector<Tensor> &inputs) {
  auto fn = std::make_shared<ForwardNode>();
  fn->op_name = op_name;
  for (const auto &t : inputs) {
    if (t.name().empty())
      t.impl_->name =
          "tensor_" +
          std::to_string(reinterpret_cast<uintptr_t>(t.impl_.get()));
    fn->input_names.push_back(t.name());
  }
  out.impl_->trace_node = fn;
}

// --- ADD ---
struct AddBackward : public Node {
  std::string name() const override { return "AddBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0], grads[0]};
  }
};

inline Tensor add(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape())
    throw std::runtime_error("Add: shape mismatch");
  if (a.device() != b.device())
    throw std::runtime_error("Add: device mismatch");

  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().add(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, a.size());

  if (a.requires_grad() || b.requires_grad()) {
    auto fn = std::make_shared<AddBackward>();
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Add", {a, b});
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
  if (pred.shape() != target.shape())
    throw std::runtime_error("MSELoss: shape mismatch");

  Tensor out({1}, pred.device(), pred.dtype());
  pred.impl_->backend().mse_loss(*pred.impl_->storage, *target.impl_->storage,
                                 *out.impl_->storage, pred.size());

  if (pred.requires_grad()) {
    auto fn = std::make_shared<MSELossBackward>(pred, target);
    link_backward_edges(fn.get(), {pred, target});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "MSELoss", {pred, target});
  return out;
}

// --- CROSS ENTROPY LOSS ---
struct CrossEntropyBackward : public Node {
  Tensor logits, targets;
  int batch_size, num_classes;
  CrossEntropyBackward(Tensor l, Tensor t, int b, int c)
      : logits(l), targets(t), batch_size(b), num_classes(c) {}
  std::string name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(logits.shape(), logits.device(), logits.dtype());
    logits.impl_->backend().cross_entropy_backward(
        *grad_out.impl_->storage, *logits.impl_->storage,
        *targets.impl_->storage, *grad_in.impl_->storage, batch_size,
        num_classes);
    return {grad_in, Tensor()};
  }
};

inline Tensor cross_entropy(const Tensor &logits, const Tensor &targets) {
  if (logits.shape() != targets.shape())
    throw std::runtime_error("CrossEntropy: shape mismatch");

  int batch_size = logits.shape().size() > 1 ? logits.shape()[0] : 1;
  int num_classes = logits.size() / batch_size;

  Tensor out({1}, logits.device(), logits.dtype());
  logits.impl_->backend().cross_entropy(
      *logits.impl_->storage, *targets.impl_->storage, *out.impl_->storage,
      batch_size, num_classes);

  if (logits.requires_grad()) {
    auto fn = std::make_shared<CrossEntropyBackward>(logits, targets,
                                                     batch_size, num_classes);
    link_backward_edges(fn.get(), {logits, targets});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "CrossEntropy", {logits, targets});
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

  if (a.requires_grad()) {
    auto fn = std::make_shared<ReluBackward>(a);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "ReLU", {a});
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

  if (a.requires_grad()) {
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

inline Tensor softmax(const Tensor &a) {
  Tensor out(a.shape(), a.device(), a.dtype());
  int batch_size = a.shape().size() > 1 ? a.shape()[0] : 1;
  int num_classes = a.size() / batch_size;
  a.impl_->backend().softmax(*a.impl_->storage, *out.impl_->storage, batch_size,
                             num_classes);

  if (a.requires_grad()) {
    auto fn = std::make_shared<SoftmaxBackward>(out, batch_size, num_classes);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Softmax", {a});
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

  if (a.requires_grad() || b.requires_grad()) {
    auto fn = std::make_shared<MatmulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Matmul", {a, b});
  return out;
}

// --- SUB ---
struct SubBackward : public Node {
  std::string name() const override { return "SubBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];

    // db = -grad_out. To avoid writing a 'neg' kernel, we do `0 - grad_out`
    Tensor zeros(grad_out.shape(), grad_out.device(), grad_out.dtype());
    zeros.impl_->storage->zero_();
    Tensor neg_grad_out(grad_out.shape(), grad_out.device(), grad_out.dtype());
    grad_out.impl_->backend().sub(
        *zeros.impl_->storage, *grad_out.impl_->storage,
        *neg_grad_out.impl_->storage, grad_out.size());

    return {grad_out, neg_grad_out};
  }
};

inline Tensor sub(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape() || a.device() != b.device())
    throw std::runtime_error("Sub: shape or device mismatch");
  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().sub(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, a.size());

  if (a.requires_grad() || b.requires_grad()) {
    auto fn = std::make_shared<SubBackward>();
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
    // dC/dA = B * grad_out
    Tensor da(A.shape(), A.device(), A.dtype());
    A.impl_->backend().mul(*B.impl_->storage, *grad_out.impl_->storage,
                           *da.impl_->storage, A.size());
    // dC/dB = A * grad_out
    Tensor db(B.shape(), B.device(), B.dtype());
    A.impl_->backend().mul(*A.impl_->storage, *grad_out.impl_->storage,
                           *db.impl_->storage, B.size());
    return {da, db};
  }
};

inline Tensor mul(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape() || a.device() != b.device())
    throw std::runtime_error("Mul: shape or device mismatch");
  Tensor out(a.shape(), a.device(), a.dtype());
  a.impl_->backend().mul(*a.impl_->storage, *b.impl_->storage,
                         *out.impl_->storage, a.size());

  if (a.requires_grad() || b.requires_grad()) {
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
    // grad_out is a size 1 scalar. We broadcast it to the full shape.
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
  // Route sum through CPU to avoid writing tree reductions for CUDA/Vulkan for
  // now
  Tensor a_cpu = a.to(Device{DeviceType::CPU, 0});
  float s = 0;
  float *data = (float *)a_cpu.data();
  for (size_t i = 0; i < a.size(); ++i)
    s += data[i];

  Tensor out({1}, Device{DeviceType::CPU, 0}, a.dtype());
  ((float *)out.data())[0] = s;
  out = out.to(a.device()); // move scalar back to original device

  if (a.requires_grad()) {
    auto fn = std::make_shared<SumBackward>(a.shape(), a.device());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Sum", {a});
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

  if (in.requires_grad()) {
    auto fn = std::make_shared<ReshapeBackward>(in.shape());
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Reshape", {in});
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
  if (in.requires_grad() || weight.requires_grad() ||
      (bias.impl_ && bias.requires_grad())) {
    auto fn =
        std::make_shared<Conv2DBackward>(in, weight, bias, stride, padding);
    std::vector<Tensor> inputs = {in, weight};
    if (bias.impl_)
      inputs.push_back(bias);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
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

  if (in.requires_grad()) {
    auto fn =
        std::make_shared<MaxPool2DBackward>(in, kernel_size, stride, padding);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
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

  if (in.requires_grad()) {
    auto fn = std::make_shared<Upsample2DBackward>(in, scale_factor);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
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

  if (training &&
      (in.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    auto fn = std::make_shared<BatchNormBackward>(in, weight, save_mean,
                                                  save_var, eps);
    // Link inputs: input, weight, bias. Running stats do not participate in
    // gradient graph for backprop.
    link_backward_edges(fn.get(), {in, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  return out;
}

} // namespace ops
} // namespace munet
