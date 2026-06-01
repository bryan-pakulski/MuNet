#include "op_dispatch.hpp"
#include "core/grad_mode.hpp"
#include "core/util.hpp"
#include "util/logging.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace munet {
namespace ops {
namespace {

const std::unordered_map<OpId, OpMetadata> &registry() {
  static const std::unordered_map<OpId, OpMetadata> kRegistry = {
      {OpId::Add, {OpId::Add, "Add", "Add", BackendFeature::ElementwiseBinary, false}},
      {OpId::Sub, {OpId::Sub, "Sub", "Sub", BackendFeature::ElementwiseBinary, false}},
      {OpId::Mul, {OpId::Mul, "Mul", "Mul", BackendFeature::ElementwiseBinary, false}},
      {OpId::Div, {OpId::Div, "Div", "Div", BackendFeature::ElementwiseBinary, true}},
      {OpId::MaskedFill, {OpId::MaskedFill, "MaskedFill", "MaskedFill", std::nullopt, false}},
      {OpId::Matmul, {OpId::Matmul, "Matmul", "MatMul", BackendFeature::Matmul, false}},
      {OpId::Relu, {OpId::Relu, "Relu", "Relu", BackendFeature::UnaryActivation, false}},
      {OpId::Sigmoid, {OpId::Sigmoid, "Sigmoid", "Sigmoid", BackendFeature::UnaryActivation, true}},
      {OpId::Exp, {OpId::Exp, "Exp", "Exp", BackendFeature::UnaryActivation, true}},
      {OpId::Log, {OpId::Log, "Log", "Log", BackendFeature::UnaryActivation, true}},
      {OpId::Sqrt, {OpId::Sqrt, "Sqrt", "Sqrt", BackendFeature::UnaryActivation, true}},
      {OpId::Rsqrt, {OpId::Rsqrt, "Rsqrt", "Rsqrt", BackendFeature::UnaryActivation, true}},
      {OpId::Sin, {OpId::Sin, "Sin", "Sin", BackendFeature::UnaryActivation, true}},
      {OpId::Cos, {OpId::Cos, "Cos", "Cos", BackendFeature::UnaryActivation, true}},
      {OpId::Softmax, {OpId::Softmax, "Softmax", "Softmax", BackendFeature::Softmax, true}},
      {OpId::LogSoftmax, {OpId::LogSoftmax, "LogSoftmax", "LogSoftmax", BackendFeature::Softmax, true}},
      {OpId::Cat, {OpId::Cat, "Cat", "Cat", BackendFeature::Concat, false}},
      {OpId::Sum, {OpId::Sum, "Sum", "Sum", BackendFeature::Reduction, false}},
      {OpId::SumToShape, {OpId::SumToShape, "SumToShape", "SumToShape", BackendFeature::Reduction, false}},
      {OpId::Mean, {OpId::Mean, "Mean", "Mean", BackendFeature::Reduction, true}},
      {OpId::Reshape, {OpId::Reshape, "Reshape", "Reshape", std::nullopt, false}},
      {OpId::Transpose, {OpId::Transpose, "Transpose", "Transpose", std::nullopt, false}},
      {OpId::Narrow, {OpId::Narrow, "Narrow", "Narrow", std::nullopt, false}},
      {OpId::Zeros, {OpId::Zeros, "Zeros", "Zeros", std::nullopt, false}},
      {OpId::Conv2D, {OpId::Conv2D, "Conv2D", "Conv2D", BackendFeature::Convolution, false}},
      {OpId::MaxPool2D, {OpId::MaxPool2D, "MaxPool2D", "MaxPool2D", BackendFeature::Pooling, false}},
      {OpId::Upsample2D, {OpId::Upsample2D, "Upsample2D", "Upsample2D", BackendFeature::Pooling, false}},
      {OpId::BatchNorm, {OpId::BatchNorm, "BatchNorm", "BatchNorm", BackendFeature::BatchNorm, true}},
      {OpId::LayerNorm, {OpId::LayerNorm, "LayerNorm", "LayerNorm", std::nullopt, true}},
      {OpId::MSELoss, {OpId::MSELoss, "MSELoss", "MSELoss", BackendFeature::Loss, true}},
      {OpId::CrossEntropy, {OpId::CrossEntropy, "CrossEntropy", "CrossEntropy", BackendFeature::Loss, false}},
  };
  return kRegistry;
}

ForwardNode make_trace_node(
    const char *trace_name, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs,
    const std::unordered_map<std::string, float> &float_attrs) {
  ForwardNode node;
  node.op_name = trace_name;
  node.int_attributes = int_attrs;
  node.attributes = float_attrs;
  node.input_shapes.reserve(inputs.size());
  node.input_names.reserve(inputs.size());
  for (const auto &input : inputs) {
    node.input_shapes.push_back(input.shape());
    node.input_names.push_back(input.name());
  }
  return node;
}

void record_dispatch_profile(const char *result, const OpMetadata &meta,
                             const Tensor &tensor, double host_us) {
  if (!is_profile_enabled()) {
    return;
  }
  Profiler::get().record("dispatch.resolve." + std::string(result) + "." + meta.name,
                         host_us, 0.0, 0, to_string(tensor.shape()));
}

void record_dispatch_stage(const char *stage, const OpMetadata &meta,
                           const Tensor &tensor, Timer *timer) {
  if (!is_profile_enabled()) {
    return;
  }
  const double host_us = timer ? timer->elapsed_us() : 0.0;
  Profiler::get().record("dispatch.stage." + std::string(stage) + "." + meta.name,
                         host_us, 0.0, 0, to_string(tensor.shape()));
}

std::string dispatch_decision_line(const OpMetadata &meta, const Tensor &tensor,
                                   const char *result,
                                   const BackendSupport *support = nullptr,
                                   const char *error = nullptr) {
  std::ostringstream oss;
  oss << "dispatch_decision result=" << result << " op=" << meta.name
      << " backend=" << tensor.impl_->backend().name()
      << " device=" << tensor.device().to_string()
      << " dtype=" << dtype_name(tensor.dtype())
      << " shape=" << to_string(tensor.shape())
      << " feature="
      << (meta.feature.has_value() ? backend_feature_name(*meta.feature)
                                   : "reference_metadata");
  if (support) {
    oss << " support_available=" << (support->available ? "1" : "0")
        << " accumulation_dtype="
        << dtype_name(support->preferred_accumulation_dtype);
  }
  if (error) {
    oss << " error=\"" << error << "\"";
  }
  return oss.str();
}

[[noreturn]] void throw_unsupported(const OpMetadata &meta, const Tensor &tensor,
                                    const BackendSupport &support) {
  const std::string error = std::string(meta.name) +
                            ": Vulkan runtime does not support feature '" +
                            backend_feature_name(*meta.feature) + "' for dtype " +
                            dtype_name(tensor.dtype()) + " and shape " +
                            to_string(tensor.shape()) +
                            " (preferred accumulation dtype " +
                            dtype_name(support.preferred_accumulation_dtype) + ").";
  MUNET_WARNING << dispatch_decision_line(meta, tensor, "unsupported", &support,
                                          error.c_str())
                << std::endl;
  throw std::runtime_error(error);
}

} // namespace

const OpMetadata &op_metadata(OpId id) {
  const auto &r = registry();
  auto it = r.find(id);
  if (it == r.end()) {
    throw std::runtime_error("Unknown operation metadata requested");
  }
  return it->second;
}

DispatchDecision resolve_dispatch(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);
  std::unique_ptr<Timer> timer;
  if (is_profile_enabled()) {
    timer = std::make_unique<Timer>();
  }

  record_dispatch_stage("metadata_validation", meta, tensor, timer.get());
  if (meta.requires_floating && !is_floating(tensor.dtype())) {
    const std::string error = std::string(meta.name) +
                              " requires a floating-point tensor, got " +
                              dtype_name(tensor.dtype());
    record_dispatch_profile("dtype_error", meta, tensor,
                            timer ? timer->elapsed_us() : 0.0);
    throw std::runtime_error(error);
  }

  BackendSupport support;
  support.preferred_accumulation_dtype = tensor.dtype();
  if (!meta.feature.has_value()) {
    record_dispatch_profile("reference", meta, tensor,
                            timer ? timer->elapsed_us() : 0.0);
    return DispatchDecision{meta, false, true, support};
  }

  record_dispatch_stage("support_query", meta, tensor, timer.get());
  support = tensor.impl_->backend().query_support(*meta.feature, tensor.dtype(),
                                                  &tensor.shape());
  if (support.available) {
    record_dispatch_profile("backend", meta, tensor,
                            timer ? timer->elapsed_us() : 0.0);
    return DispatchDecision{meta, true, false, support};
  }

  record_dispatch_profile("unsupported", meta, tensor,
                          timer ? timer->elapsed_us() : 0.0);
  throw_unsupported(meta, tensor, support);
}

std::string dispatch_policy_snapshot() {
  return "Vulkan-only dispatch: supported ops execute on the Vulkan runtime; "
         "metadata-only tensor view ops use reference metadata paths; unsupported "
         "dtype/shape/feature combinations throw explicit errors.\n";
}

std::string dispatch_decision_debug_dump(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);
  try {
    const auto decision = resolve_dispatch(id, tensor);
    const char *result = decision.use_backend
                             ? "vulkan_runtime"
                             : (decision.use_reference_path ? "reference_path"
                                                            : "unresolved");
    return dispatch_decision_line(meta, tensor, result,
                                  &decision.backend_support);
  } catch (const std::runtime_error &err) {
    return dispatch_decision_line(meta, tensor, "error", nullptr, err.what());
  }
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
