#include "op_dispatch.hpp"
#include "core/grad_mode.hpp"
#include "core/util.hpp"
#include "util/logging.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace munet {
namespace ops {
namespace {

enum class DispatchFallbackReason {
  DType,
  Shape,
  Feature,
  Policy,
};

const std::unordered_map<OpId, OpMetadata> &registry() {
  static const std::unordered_map<OpId, OpMetadata> kRegistry = {
      {OpId::Add,
       {OpId::Add, "Add", "Add", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Sub,
       {OpId::Sub, "Sub", "Sub", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Mul,
       {OpId::Mul, "Mul", "Mul", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Div,
       {OpId::Div, "Div", "Div", BackendFeature::ElementwiseBinary, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::MaskedFill,
       {OpId::MaskedFill, "MaskedFill", "MaskedFill", std::nullopt, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Matmul,
       {OpId::Matmul, "Matmul", "MatMul", BackendFeature::Matmul, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Relu,
       {OpId::Relu, "Relu", "Relu", BackendFeature::UnaryActivation, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Sigmoid,
       {OpId::Sigmoid, "Sigmoid", "Sigmoid", BackendFeature::UnaryActivation,
        true, BackendFallbackPolicy::CPUFallback}},
      {OpId::Exp,
       {OpId::Exp, "Exp", "Exp", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Log,
       {OpId::Log, "Log", "Log", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Sqrt,
       {OpId::Sqrt, "Sqrt", "Sqrt", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Rsqrt,
       {OpId::Rsqrt, "Rsqrt", "Rsqrt", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Sin,
       {OpId::Sin, "Sin", "Sin", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Cos,
       {OpId::Cos, "Cos", "Cos", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Softmax,
       {OpId::Softmax, "Softmax", "Softmax", BackendFeature::Softmax, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::LogSoftmax,
       {OpId::LogSoftmax, "LogSoftmax", "LogSoftmax", std::nullopt, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Cat,
       {OpId::Cat, "Concat", "Concat", BackendFeature::Concat, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sum,
       {OpId::Sum, "Sum", "ReduceSum", BackendFeature::Reduction, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Mean,
       {OpId::Mean, "Mean", "ReduceMean", BackendFeature::Reduction, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Reshape,
       {OpId::Reshape, "Reshape", "Reshape", std::nullopt, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Conv2D,
       {OpId::Conv2D, "Conv2d", "Conv", BackendFeature::Convolution, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::MaxPool2D,
       {OpId::MaxPool2D, "MaxPool2d", "MaxPool", BackendFeature::Pooling, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Upsample2D,
       {OpId::Upsample2D, "Upsample2d", "Upsample2d", BackendFeature::Pooling,
        true, BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::BatchNorm,
       {OpId::BatchNorm, "BatchNorm", "BatchNormalization",
        BackendFeature::BatchNorm, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::LayerNorm,
       {OpId::LayerNorm, "LayerNorm", "LayerNorm", std::nullopt, true,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::MSELoss,
       {OpId::MSELoss, "MSELoss", "MSELoss", BackendFeature::Loss, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::CrossEntropy,
       {OpId::CrossEntropy, "CrossEntropy", "CrossEntropy",
        BackendFeature::Loss, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Transpose,
       {OpId::Transpose, "Transpose", "Transpose", std::nullopt, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Narrow,
       {OpId::Narrow, "Narrow", "Slice", std::nullopt, false,
        BackendFallbackPolicy::CPUFallback}},
      {OpId::Zeros,
       {OpId::Zeros, "Zeros", "Zeros", std::nullopt, false,
        BackendFallbackPolicy::CPUFallback}},
  };
  return kRegistry;
}

ForwardNode make_trace_node(
    const char *op_name, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs,
    const std::unordered_map<std::string, float> &float_attrs) {
  ForwardNode node;
  node.op_name = op_name;
  for (const auto &tensor : inputs) {
    if (tensor.name().empty()) {
      tensor.impl_->name =
          "tensor_" +
          std::to_string(reinterpret_cast<uintptr_t>(tensor.impl_.get()));
    }
    node.input_names.push_back(tensor.name());
    node.input_shapes.push_back(tensor.shape());
  }
  node.int_attributes = int_attrs;
  node.attributes = float_attrs;
  return node;
}

void record_dispatch_profile(const std::string &path, const OpMetadata &meta,
                             const Tensor &tensor, double cpu_us) {
  if (!is_profile_enabled()) {
    return;
  }

  Profiler::get().record("dispatch.resolve." + path + "." + meta.name, cpu_us,
                         0.0, 0, to_string(tensor.shape()));
}

const char *dispatch_fallback_reason_name(DispatchFallbackReason reason) {
  switch (reason) {
  case DispatchFallbackReason::DType:
    return "dtype";
  case DispatchFallbackReason::Shape:
    return "shape";
  case DispatchFallbackReason::Feature:
    return "feature";
  case DispatchFallbackReason::Policy:
    return "policy";
  default:
    return "unknown";
  }
}

DispatchFallbackReason
classify_dispatch_fallback(const OpMetadata &meta, const Tensor &tensor,
                           BackendFeature feature,
                           const BackendSupport &support) {
  if (!supports_backend_feature_dtype(feature, tensor.dtype())) {
    return DispatchFallbackReason::DType;
  }
  if (!supports_backend_feature_shape(feature, tensor.dtype(),
                                      &tensor.shape())) {
    return DispatchFallbackReason::Shape;
  }
  if (support.fallback_policy != meta.fallback_policy) {
    return DispatchFallbackReason::Policy;
  }
  return DispatchFallbackReason::Feature;
}

std::string dispatch_fallback_detail(const OpMetadata &meta,
                                     const Tensor &tensor,
                                     BackendFeature feature,
                                     const BackendSupport &support,
                                     DispatchFallbackReason reason) {
  std::string detail = "op=" + std::string(meta.name);
  detail += " backend=" + std::string(tensor.impl_->backend().name());
  detail += " feature=" + std::string(backend_feature_name(feature));
  detail += " dtype=" + dtype_name(tensor.dtype());
  detail += " shape=" + to_string(tensor.shape());
  detail += " reason=" + std::string(dispatch_fallback_reason_name(reason));
  detail += " policy=" +
            std::string(backend_fallback_policy_name(support.fallback_policy));
  return detail;
}

void record_dispatch_fallback_reason(const OpMetadata &meta,
                                     const Tensor &tensor,
                                     BackendFeature feature,
                                     const BackendSupport &support,
                                     DispatchFallbackReason reason) {
  if (!is_profile_enabled()) {
    return;
  }
  Profiler::get().record(
      "dispatch.fallback.reason." +
          std::string(dispatch_fallback_reason_name(reason)),
      0.0, 0.0, 0,
      dispatch_fallback_detail(meta, tensor, feature, support, reason));
}

void log_dispatch_fallback_reason(const OpMetadata &meta, const Tensor &tensor,
                                  BackendFeature feature,
                                  const BackendSupport &support,
                                  DispatchFallbackReason reason) {
  if (!is_debug_enabled()) {
    return;
  }
  MUNET_INFO << "dispatch_fallback "
             << dispatch_fallback_detail(meta, tensor, feature, support, reason)
             << std::endl;
}

} // namespace

const OpMetadata &op_metadata(OpId id) {
  const auto &map = registry();
  const auto it = map.find(id);
  if (it == map.end()) {
    throw std::runtime_error("Unknown op metadata request");
  }
  return it->second;
}

DispatchDecision resolve_dispatch(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);
  std::unique_ptr<Timer> timer;
  if (is_profile_enabled()) {
    timer = std::make_unique<Timer>();
  }

  if (meta.requires_floating && !is_floating(tensor.dtype())) {
    if (timer) {
      record_dispatch_profile("dtype_error", meta, tensor, timer->elapsed_us());
    }
    throw std::runtime_error(std::string(meta.name) +
                             " requires a floating-point tensor, got " +
                             dtype_name(tensor.dtype()));
  }

  if (!meta.feature.has_value()) {
    BackendSupport support;
    support.available = false;
    support.fallback_policy = meta.fallback_policy;
    support.preferred_accumulation_dtype = tensor.dtype();
    if (timer) {
      record_dispatch_profile("metadata_fallback", meta, tensor,
                              timer->elapsed_us());
    }
    return {meta, false,
            meta.fallback_policy == BackendFallbackPolicy::CPUFallback,
            support};
  }

  const auto feature = *meta.feature;
  const auto support = tensor.impl_->backend().query_support(
      feature, tensor.dtype(), &tensor.shape());

  if (support.available) {
    if (timer) {
      record_dispatch_profile("backend", meta, tensor, timer->elapsed_us());
    }
    return {meta, true, false, support};
  }

  if (meta.fallback_policy == BackendFallbackPolicy::CPUFallback &&
      support.fallback_policy == BackendFallbackPolicy::CPUFallback) {
    const auto reason =
        classify_dispatch_fallback(meta, tensor, feature, support);
    log_dispatch_fallback_reason(meta, tensor, feature, support, reason);
    record_dispatch_fallback_reason(meta, tensor, feature, support, reason);
    if (timer) {
      record_dispatch_profile("cpu_fallback", meta, tensor,
                              timer->elapsed_us());
    }
    return {meta, false, true, support};
  }

  const auto reason =
      classify_dispatch_fallback(meta, tensor, feature, support);
  log_dispatch_fallback_reason(meta, tensor, feature, support, reason);
  record_dispatch_fallback_reason(meta, tensor, feature, support, reason);
  if (timer) {
    record_dispatch_profile("unsupported", meta, tensor, timer->elapsed_us());
  }
  throw std::runtime_error(
      std::string(meta.name) + ": backend '" +
      std::string(tensor.impl_->backend().name()) +
      "' does not support feature '" + backend_feature_name(feature) +
      "' for dtype " + dtype_name(tensor.dtype()) + " (fallback policy: " +
      backend_fallback_policy_name(support.fallback_policy) + ")");
}

void record_registered_trace(
    OpId id, Tensor &out, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs,
    const std::unordered_map<std::string, float> &float_attrs) {

  if (!GradMode::is_enabled() && !is_profile_enabled())
    return;

  NoGradGuard guard;
  const auto &meta = op_metadata(id);
  out.impl_->trace_node = std::make_shared<ForwardNode>(
      make_trace_node(meta.trace_name, inputs, int_attrs, float_attrs));
}

} // namespace ops
} // namespace munet
