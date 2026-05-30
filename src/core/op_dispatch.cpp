#include "op_dispatch.hpp"
#include "core/grad_mode.hpp"
#include "core/util.hpp"
#include "util/logging.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
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

struct DispatchFallbackRule;

struct FallbackTelemetryState {
  std::mutex mutex;
  uint64_t accelerator_host_fallback_total = 0;
  std::unordered_map<std::string, uint64_t> accelerator_host_fallback_counters;
};

FallbackTelemetryState &fallback_telemetry_state() {
  static FallbackTelemetryState state;
  return state;
}

bool is_accelerator_device(DeviceType type) {
  return type == DeviceType::VULKAN;
}

bool is_dispatch_decision_dump_enabled() {
  const char *env = std::getenv("MUNET_DISPATCH_DECISION_DUMP");
  if (!env) {
    return false;
  }
  return std::string(env) == "1" || std::string(env) == "true" ||
         std::string(env) == "TRUE";
}

bool is_fallback_fail_fast_enabled() {
  const char *env = std::getenv("MUNET_FAIL_FAST_ACCELERATOR_Host_FALLBACK");
  if (!env) {
    return false;
  }
  return std::string(env) == "1" || std::string(env) == "true" ||
         std::string(env) == "TRUE";
}

const std::unordered_map<OpId, OpMetadata> &registry() {
  static const std::unordered_map<OpId, OpMetadata> kRegistry = {
      {OpId::Add,
       {OpId::Add, "Add", "Add", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sub,
       {OpId::Sub, "Sub", "Sub", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Mul,
       {OpId::Mul, "Mul", "Mul", BackendFeature::ElementwiseBinary, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Div,
       {OpId::Div, "Div", "Div", BackendFeature::ElementwiseBinary, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::MaskedFill,
       {OpId::MaskedFill, "MaskedFill", "MaskedFill", std::nullopt, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Matmul,
       {OpId::Matmul, "Matmul", "MatMul", BackendFeature::Matmul, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Relu,
       {OpId::Relu, "Relu", "Relu", BackendFeature::UnaryActivation, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sigmoid,
       {OpId::Sigmoid, "Sigmoid", "Sigmoid", BackendFeature::UnaryActivation,
        true, BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Exp,
       {OpId::Exp, "Exp", "Exp", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Log,
       {OpId::Log, "Log", "Log", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sqrt,
       {OpId::Sqrt, "Sqrt", "Sqrt", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Rsqrt,
       {OpId::Rsqrt, "Rsqrt", "Rsqrt", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sin,
       {OpId::Sin, "Sin", "Sin", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Cos,
       {OpId::Cos, "Cos", "Cos", BackendFeature::UnaryActivation, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Softmax,
       {OpId::Softmax, "Softmax", "Softmax", BackendFeature::Softmax, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::LogSoftmax,
       {OpId::LogSoftmax, "LogSoftmax", "LogSoftmax", std::nullopt, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Cat,
       {OpId::Cat, "Concat", "Concat", BackendFeature::Concat, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Sum,
       {OpId::Sum, "Sum", "ReduceSum", BackendFeature::Reduction, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::SumToShape,
       {OpId::SumToShape, "SumToShape", "ReduceSum", BackendFeature::Reduction,
        false, BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Mean,
       {OpId::Mean, "Mean", "ReduceMean", BackendFeature::Reduction, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Reshape,
       {OpId::Reshape, "Reshape", "Reshape", std::nullopt, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
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
        BackendFeature::BatchNorm, true, BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::LayerNorm,
       {OpId::LayerNorm, "LayerNorm", "LayerNorm", std::nullopt, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::MSELoss,
       {OpId::MSELoss, "MSELoss", "MSELoss", BackendFeature::Loss, true,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::CrossEntropy,
       {OpId::CrossEntropy, "CrossEntropy", "CrossEntropy",
        BackendFeature::Loss, true, BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Transpose,
       {OpId::Transpose, "Transpose", "Transpose", std::nullopt, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Narrow,
       {OpId::Narrow, "Narrow", "Slice", std::nullopt, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
      {OpId::Zeros,
       {OpId::Zeros, "Zeros", "Zeros", std::nullopt, false,
        BackendFallbackPolicy::ExplicitUnsupported}},
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
                             const Tensor &tensor, double host_us) {
  if (!is_profile_enabled()) {
    return;
  }

  Profiler::get().record("dispatch.resolve." + path + "." + meta.name, host_us,
                         0.0, 0, to_string(tensor.shape()));
}

void record_dispatch_stage_profile(const char *stage, const OpMetadata &meta,
                                   const Tensor &tensor, double host_us) {
  if (!is_profile_enabled()) {
    return;
  }
  Profiler::get().record("dispatch.stage." + std::string(stage) + "." + meta.name,
                         host_us, 0.0, 0, to_string(tensor.shape()));
}

void log_dispatch_stage(const char *stage, const OpMetadata &meta,
                        const Tensor &tensor) {
  if (!is_debug_enabled()) {
    return;
  }
  MUNET_INFO << "dispatch_stage stage=" << stage << " op=" << meta.name
             << " backend=" << tensor.impl_->backend().name()
             << " dtype=" << dtype_name(tensor.dtype())
             << " shape=" << to_string(tensor.shape()) << std::endl;
}

void record_dispatch_stage(const char *stage, const OpMetadata &meta,
                           const Tensor &tensor, Timer *timer) {
  const double host_us = timer ? timer->elapsed_us() : 0.0;
  record_dispatch_stage_profile(stage, meta, tensor, host_us);
  log_dispatch_stage(stage, meta, tensor);
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
    return "vulkan";
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

struct DispatchFallbackRule {
  enum class Action {
    DenyExplicitUnsupported,
  };

  DeviceType device_type;
  std::optional<BackendFeature> feature;
  std::optional<OpId> op;
  std::optional<DataType> dtype;
  Action action = Action::DenyExplicitUnsupported;
  const char *error_message;
};

std::string dispatch_decision_detail_line(
    const OpMetadata &meta, const Tensor &tensor, const char *stage,
    const char *result, std::optional<BackendFeature> feature,
    const BackendSupport *support, const DispatchFallbackRule *rule,
    std::optional<DispatchFallbackReason> fallback_reason,
    const char *error = nullptr) {
  std::ostringstream oss;
  oss << "dispatch_decision"
      << " stage=" << stage << " result=" << result << " op=" << meta.name
      << " backend=" << tensor.impl_->backend().name()
      << " device=" << tensor.device().to_string()
      << " dtype=" << dtype_name(tensor.dtype())
      << " shape=" << to_string(tensor.shape());
  if (feature.has_value()) {
    oss << " feature=" << backend_feature_name(*feature);
  } else {
    oss << " feature=<metadata_none>";
  }
  if (support) {
    oss << " support_available=" << (support->available ? "1" : "0")
        << " support_policy="
        << backend_fallback_policy_name(support->fallback_policy)
        << " support_acc_dtype="
        << dtype_name(support->preferred_accumulation_dtype);
  }
  if (rule) {
    oss << " rule_action="
        << "deny_host_fallback"
        << " rule_error=\"" << rule->error_message << "\"";
  } else {
    oss << " rule_action=<none>";
  }
  if (fallback_reason.has_value()) {
    oss << " fallback_reason="
        << dispatch_fallback_reason_name(*fallback_reason);
  }
  if (error) {
    oss << " error=\"" << error << "\"";
  }
  return oss.str();
}

void maybe_log_dispatch_decision(const std::string &line) {
  if (!is_dispatch_decision_dump_enabled()) {
    return;
  }
  MUNET_INFO << line << std::endl;
}

void record_accelerator_fallback_telemetry(const OpMetadata &meta,
                                           const Tensor &tensor,
                                           const char *reason) {
  if (!is_accelerator_device(tensor.device().type)) {
    return;
  }
  const std::string key =
      std::string("device=") + tensor.device().to_string() +
      ";backend=" + tensor.impl_->backend().name() + ";op=" + meta.name +
      ";dtype=" + dtype_name(tensor.dtype()) + ";reason=" + reason;

  auto &state = fallback_telemetry_state();
  uint64_t total = 0;
  uint64_t key_count = 0;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    total = ++state.accelerator_host_fallback_total;
    key_count = ++state.accelerator_host_fallback_counters[key];
  }
  MUNET_WARNING << "accelerator_host_fallback total=" << total
                << " key_count=" << key_count << " " << key << std::endl;
}

void maybe_fail_fast_on_accelerator_fallback(const OpMetadata &meta,
                                             const Tensor &tensor,
                                             const char *reason) {
  if (!is_accelerator_device(tensor.device().type) ||
      !is_fallback_fail_fast_enabled()) {
    return;
  }
  throw std::runtime_error(
      std::string("Fail-fast: accelerator tensor fell back to Host unexpectedly; op=") +
      meta.name + " device=" + tensor.device().to_string() +
      " backend=" + tensor.impl_->backend().name() + " reason=" + reason +
      ". Disable by unsetting MUNET_FAIL_FAST_ACCELERATOR_Host_FALLBACK.");
}

const std::vector<DispatchFallbackRule> &dispatch_fallback_rules() {
  // Vulkan-only policy matrix: unsupported accelerator dtype/feature
  // combinations are explicit errors instead of Host fallbacks.
  static const std::vector<DispatchFallbackRule> kRules = {
      {DeviceType::VULKAN, BackendFeature::ElementwiseBinary, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 elementwise-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Matmul, std::nullopt,
       DataType::Float16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support float16 matmul-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Matmul, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 matmul-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Convolution, std::nullopt,
       DataType::Float16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Host backend does not support float16 convolution"},
      {DeviceType::VULKAN, BackendFeature::Convolution, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Host backend does not support bfloat16 convolution"},
      {DeviceType::VULKAN, BackendFeature::Convolution, std::nullopt,
       DataType::Float16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support float16 convolution"},
      {DeviceType::VULKAN, BackendFeature::Pooling, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 pooling-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Loss, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 loss-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Reduction, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 reduction-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Softmax, std::nullopt,
       DataType::BFloat16, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "Vulkan backend does not support bfloat16 softmax-feature fallback"},
      {DeviceType::VULKAN, BackendFeature::Softmax, std::nullopt, std::nullopt,
       DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "VULKAN softmax feature fallback denied"},
      {DeviceType::VULKAN, BackendFeature::Softmax, std::nullopt,
       DataType::Float32, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "VULKAN softmax float32 fallback denied"},
      {DeviceType::VULKAN, BackendFeature::Softmax, OpId::Softmax,
       DataType::Float32, DispatchFallbackRule::Action::DenyExplicitUnsupported,
       "VULKAN softmax op-specific float32 fallback denied"},
  };
  return kRules;
}

int dispatch_rule_specificity(const DispatchFallbackRule &rule) {
  int score = 0;
  if (rule.feature.has_value()) {
    score += 4;
  }
  if (rule.op.has_value()) {
    score += 2;
  }
  if (rule.dtype.has_value()) {
    score += 1;
  }
  return score;
}

const DispatchFallbackRule *find_matching_dispatch_rule(
    OpId id, BackendFeature feature, const Tensor &tensor) {
  const DeviceType device_type = tensor.device().type;
  const DataType dtype = tensor.dtype();
  const DispatchFallbackRule *best = nullptr;
  int best_score = -1;
  for (const auto &rule : dispatch_fallback_rules()) {
    if (rule.device_type != device_type) {
      continue;
    }
    if (rule.feature.has_value() && *rule.feature != feature) {
      continue;
    }
    if (rule.op.has_value() && *rule.op != id) {
      continue;
    }
    if (rule.dtype.has_value() && *rule.dtype != dtype) {
      continue;
    }
    const int score = dispatch_rule_specificity(rule);
    if (score > best_score) {
      best = &rule;
      best_score = score;
    }
  }
  return best;
}

} // namespace

namespace {

std::optional<DispatchDecision>
stage_metadata_validation(const OpMetadata &meta, const Tensor &tensor,
                          Timer *timer) {
  if (meta.requires_floating && !is_floating(tensor.dtype())) {
    const std::string error = std::string(meta.name) +
                              " requires a floating-point tensor, got " +
                              dtype_name(tensor.dtype());
    maybe_log_dispatch_decision(dispatch_decision_detail_line(
        meta, tensor, "metadata_validation", "dtype_error", meta.feature,
        nullptr, nullptr, std::nullopt, error.c_str()));
    if (timer) {
      record_dispatch_profile("dtype_error", meta, tensor, timer->elapsed_us());
    }
    throw std::runtime_error(error);
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
    maybe_log_dispatch_decision(dispatch_decision_detail_line(
        meta, tensor, "metadata_validation", "metadata_fallback", meta.feature,
        &support, nullptr, std::nullopt));
    return DispatchDecision{meta, false,
                            meta.fallback_policy == BackendFallbackPolicy::ExplicitUnsupported,
                            support};
  }

  return std::nullopt;
}

std::pair<BackendFeature, BackendSupport>
stage_backend_support_query(const Tensor &tensor, BackendFeature feature) {
  auto support = tensor.impl_->backend().query_support(feature, tensor.dtype(),
                                                       &tensor.shape());
  return {feature, support};
}

struct PolicyStageResult {
  bool use_backend = false;
  bool use_host_fallback = false;
  const DispatchFallbackRule *matched_rule = nullptr;
  BackendSupport support;
};

PolicyStageResult
stage_policy_engine_evaluation(OpId id, const OpMetadata &,
                               const Tensor &tensor, BackendFeature feature,
                               BackendSupport support, Timer *timer) {
  const DispatchFallbackRule *matched_rule =
      find_matching_dispatch_rule(id, feature, tensor);

  if (support.available) {
    if (timer) {
      record_dispatch_profile("backend", op_metadata(id), tensor, timer->elapsed_us());
    }
    return {true, false, matched_rule, support};
  }

  return {false, false, matched_rule, support};
}

[[noreturn]] void stage_final_decision_error(const OpMetadata &meta,
                                             const Tensor &tensor,
                                             BackendFeature feature,
                                             const PolicyStageResult &policy,
                                             Timer *timer) {
  const auto reason =
      classify_dispatch_fallback(meta, tensor, feature, policy.support);
  log_dispatch_fallback_reason(meta, tensor, feature, policy.support, reason);
  record_dispatch_fallback_reason(meta, tensor, feature, policy.support, reason);
  if (timer) {
    record_dispatch_profile("unsupported", meta, tensor, timer->elapsed_us());
  }

  const bool disallow_host_fallback =
      policy.matched_rule != nullptr &&
      policy.matched_rule->action == DispatchFallbackRule::Action::DenyExplicitUnsupported;
  if (disallow_host_fallback) {
    const std::string error =
        std::string(meta.name) + ": " +
        std::string(policy.matched_rule->error_message) + " for backend '" +
        std::string(tensor.impl_->backend().name()) +
        "' on device '" + tensor.device().to_string() + "'";
    maybe_log_dispatch_decision(dispatch_decision_detail_line(
        meta, tensor, "final_decision_error", "deny_host_fallback", feature,
        &policy.support, policy.matched_rule, reason, error.c_str()));
    throw std::runtime_error(error);
  }

  const std::string error =
      std::string(meta.name) + ": backend '" +
      std::string(tensor.impl_->backend().name()) +
      "' does not support feature '" + backend_feature_name(feature) +
      "' for dtype " + dtype_name(tensor.dtype()) + " (fallback policy: " +
      backend_fallback_policy_name(policy.support.fallback_policy) + ")";
  maybe_log_dispatch_decision(dispatch_decision_detail_line(
      meta, tensor, "final_decision_error", "unsupported", feature,
      &policy.support, policy.matched_rule, reason, error.c_str()));
  throw std::runtime_error(error);
}

} // namespace

const OpMetadata &op_metadata(OpId id) {
  const auto &map = registry();
  const auto it = map.find(id);
  if (it == map.end()) {
    throw std::runtime_error("Vulkan op metadata request");
  }
  return it->second;
}

DispatchDecision resolve_dispatch(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);
  std::unique_ptr<Timer> timer;
  if (is_profile_enabled()) {
    timer = std::make_unique<Timer>();
  }

  // Stage 1: metadata validation.
  record_dispatch_stage("metadata_validation", meta, tensor, timer.get());
  if (auto stage1 = stage_metadata_validation(meta, tensor, timer.get())) {
    if (stage1->use_host_fallback) {
      record_accelerator_fallback_telemetry(meta, tensor, "metadata");
      maybe_fail_fast_on_accelerator_fallback(meta, tensor, "metadata");
    }
    return *stage1;
  }

  // Stage 2: backend support query.
  record_dispatch_stage("backend_support_query", meta, tensor, timer.get());
  const auto feature = *meta.feature;
  auto [queried_feature, support] = stage_backend_support_query(tensor, feature);

  // Stage 3: policy engine evaluation.
  record_dispatch_stage("policy_engine_evaluation", meta, tensor, timer.get());
  const auto policy = stage_policy_engine_evaluation(
      id, meta, tensor, queried_feature, support, timer.get());
  if (policy.use_backend) {
    maybe_log_dispatch_decision(dispatch_decision_detail_line(
        meta, tensor, "policy_engine_evaluation", "backend", queried_feature,
        &policy.support, policy.matched_rule, std::nullopt));
    return {meta, true, false, policy.support};
  }
  if (policy.use_host_fallback) {
    const auto reason =
        classify_dispatch_fallback(meta, tensor, queried_feature, policy.support);
    maybe_log_dispatch_decision(dispatch_decision_detail_line(
        meta, tensor, "policy_engine_evaluation", "host_fallback",
        queried_feature, &policy.support, policy.matched_rule, reason));
    record_accelerator_fallback_telemetry(
        meta, tensor, dispatch_fallback_reason_name(reason));
    maybe_fail_fast_on_accelerator_fallback(
        meta, tensor, dispatch_fallback_reason_name(reason));
    return {meta, false, true, policy.support};
  }

  // Stage 4: final decision/error.
  record_dispatch_stage("final_decision_error", meta, tensor, timer.get());
  stage_final_decision_error(meta, tensor, queried_feature, policy, timer.get());
}

FallbackTelemetrySnapshot fallback_telemetry_snapshot() {
  auto &state = fallback_telemetry_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  FallbackTelemetrySnapshot snapshot;
  snapshot.accelerator_host_fallback_total = state.accelerator_host_fallback_total;
  snapshot.accelerator_host_fallback_counters =
      state.accelerator_host_fallback_counters;
  return snapshot;
}

void reset_fallback_telemetry() {
  auto &state = fallback_telemetry_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.accelerator_host_fallback_total = 0;
  state.accelerator_host_fallback_counters.clear();
}

std::string dispatch_policy_snapshot() {
  const auto device_name = [](DeviceType) { return "vulkan"; };  std::ostringstream oss;
  for (const auto &rule : dispatch_fallback_rules()) {
    oss << "device=" << device_name(rule.device_type)
        << ";feature="
        << (rule.feature.has_value() ? backend_feature_name(*rule.feature)
                                     : "<any>")
        << ";op=";
    if (rule.op.has_value()) {
      oss << op_metadata(*rule.op).name;
    } else {
      oss << "<any>";
    }
    oss << ";dtype="
        << (rule.dtype.has_value() ? dtype_name(*rule.dtype) : "<any>")
        << ";action="
        << "deny_host_fallback"
        << ";error=" << rule.error_message << "\n";
  }
  return oss.str();
}

std::string dispatch_decision_debug_dump(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);
  try {
    const auto decision = resolve_dispatch(id, tensor);
    const char *result =
        decision.use_backend
            ? "backend"
            : (decision.use_host_fallback ? "host_fallback" : "unresolved");
    std::optional<DispatchFallbackReason> reason = std::nullopt;
    if (meta.feature.has_value() && decision.use_host_fallback) {
      reason =
          classify_dispatch_fallback(meta, tensor, *meta.feature, decision.backend_support);
    }
    return dispatch_decision_detail_line(meta, tensor, "resolved", result,
                                         meta.feature, &decision.backend_support,
                                         nullptr, reason);
  } catch (const std::runtime_error &err) {
    return dispatch_decision_detail_line(meta, tensor, "resolved", "error",
                                         meta.feature, nullptr, nullptr,
                                         std::nullopt, err.what());
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
