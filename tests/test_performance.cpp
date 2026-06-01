#include "core/op_dispatch.hpp"
#include "core/ops/common.hpp"
#include "core/util/profiler.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace munet;
using namespace munet::ops;

namespace {

class ScopedProfileOverride {
public:
  explicit ScopedProfileOverride(bool enabled) {
    set_profile_enabled_override(enabled);
  }

  ~ScopedProfileOverride() { set_profile_enabled_override(std::nullopt); }
};

struct OperatorBenchmarkCase {
  OpId id;
  std::function<Tensor()> run;
  std::function<Tensor()> dispatch_probe;
};

std::vector<OperatorBenchmarkCase> build_operator_cases(const Device &device) {
  Tensor binary_a({64, 64}, device, DataType::Float32);
  Tensor binary_b({64, 64}, device, DataType::Float32);
  binary_a.fill_(2.0f);
  binary_b.fill_(0.5f);

  Tensor mask_input({64, 64}, device, DataType::Float32);
  Tensor mask({64, 64}, device, DataType::Float32);
  mask_input.fill_(1.0f);
  mask.fill_(1.0f);

  Tensor matmul_a({64, 64}, device, DataType::Float32);
  Tensor matmul_b({64, 64}, device, DataType::Float32);
  matmul_a.fill_(0.5f);
  matmul_b.fill_(1.5f);

  Tensor activation_input({128, 128}, device, DataType::Float32);
  activation_input.fill_(0.25f);
  Tensor positive_input({128, 128}, device, DataType::Float32);
  positive_input.fill_(1.25f);

  Tensor softmax_input({32, 64}, device, DataType::Float32);
  softmax_input.fill_(0.125f);

  Tensor cat_a({32, 16}, device, DataType::Float32);
  Tensor cat_b({32, 16}, device, DataType::Float32);
  cat_a.fill_(1.0f);
  cat_b.fill_(2.0f);

  Tensor reduce_input({16, 32}, device, DataType::Float32);
  reduce_input.fill_(1.0f);
  Tensor sum_to_shape_input({4, 8}, device, DataType::Float32);
  sum_to_shape_input.fill_(1.0f);
  Tensor shape_input({16, 16}, device, DataType::Float32);
  shape_input.fill_(1.0f);

  Tensor conv_input({2, 2, 8, 8}, device, DataType::Float32);
  Tensor conv_weight({4, 2, 3, 3}, device, DataType::Float32);
  Tensor conv_bias({4}, device, DataType::Float32);
  conv_input.fill_(1.0f);
  conv_weight.fill_(0.125f);
  conv_bias.fill_(0.0f);

  Tensor spatial_input({2, 2, 8, 8}, device, DataType::Float32);
  spatial_input.fill_(1.0f);

  Tensor bn_input({2, 3, 4, 4}, device, DataType::Float32);
  Tensor bn_running_mean({3}, device, DataType::Float32);
  Tensor bn_running_var({3}, device, DataType::Float32);
  Tensor bn_weight({3}, device, DataType::Float32);
  Tensor bn_bias({3}, device, DataType::Float32);
  bn_input.fill_(1.0f);
  bn_running_mean.fill_(0.0f);
  bn_running_var.fill_(1.0f);
  bn_weight.fill_(1.0f);
  bn_bias.fill_(0.0f);

  Tensor ln_input({32, 16}, device, DataType::Float32);
  Tensor ln_weight({16}, device, DataType::Float32);
  Tensor ln_bias({16}, device, DataType::Float32);
  ln_input.fill_(0.5f);
  ln_weight.fill_(1.0f);
  ln_bias.fill_(0.0f);

  Tensor loss_pred({32, 8}, device, DataType::Float32);
  Tensor loss_target({32, 8}, device, DataType::Float32);
  loss_pred.fill_(0.25f);
  loss_target.fill_(0.0f);

  return {
      {OpId::Add, [=]() { return binary_a + binary_b; },
       [=]() { return binary_a; }},
      {OpId::Sub, [=]() { return binary_a - binary_b; },
       [=]() { return binary_a; }},
      {OpId::Mul, [=]() { return binary_a * binary_b; },
       [=]() { return binary_a; }},
      {OpId::Div, [=]() { return binary_a / binary_b; },
       [=]() { return binary_a; }},
      {OpId::MaskedFill,
       [=]() { return mask_input.masked_fill(mask, make_scalar(3.0f)); },
       [=]() { return mask_input; }},
      {OpId::Matmul, [=]() { return matmul_a.matmul(matmul_b); },
       [=]() { return matmul_a; }},
      {OpId::Relu, [=]() { return activation_input.relu(); },
       [=]() { return activation_input; }},
      {OpId::Sigmoid, [=]() { return activation_input.sigmoid(); },
       [=]() { return activation_input; }},
      {OpId::Exp, [=]() { return activation_input.exp(); },
       [=]() { return activation_input; }},
      {OpId::Log, [=]() { return positive_input.log(); },
       [=]() { return positive_input; }},
      {OpId::Sqrt, [=]() { return positive_input.sqrt(); },
       [=]() { return positive_input; }},
      {OpId::Rsqrt, [=]() { return positive_input.rsqrt(); },
       [=]() { return positive_input; }},
      {OpId::Sin, [=]() { return activation_input.sin(); },
       [=]() { return activation_input; }},
      {OpId::Cos, [=]() { return activation_input.cos(); },
       [=]() { return activation_input; }},
      {OpId::Softmax, [=]() { return softmax_input.softmax(-1); },
       [=]() { return softmax_input; }},
      {OpId::LogSoftmax, [=]() { return softmax_input.log_softmax(-1); },
       [=]() { return softmax_input; }},
      {OpId::Cat, [=]() { return cat({cat_a, cat_b}, 1); },
       [=]() { return cat_a; }},
      {OpId::Sum, [=]() { return reduce_input.sum(); },
       [=]() { return reduce_input; }},
      {OpId::SumToShape,
       [=]() { return sum_to_shape(sum_to_shape_input, Shape{1, 8}); },
       [=]() { return sum_to_shape_input; }},
      {OpId::Mean, [=]() { return reduce_input.mean(-1, false); },
       [=]() { return reduce_input; }},
      {OpId::Reshape, [=]() { return shape_input.reshape({8, 32}); },
       [=]() { return shape_input; }},
      {OpId::Conv2D,
       [=]() { return conv_input.conv2d(conv_weight, conv_bias, 1, 1); },
       [=]() { return conv_input; }},
      {OpId::MaxPool2D, [=]() { return spatial_input.max_pool2d(2, 2, 0); },
       [=]() { return spatial_input; }},
      {OpId::Upsample2D, [=]() { return spatial_input.upsample2d(2); },
       [=]() { return spatial_input; }},
      {OpId::BatchNorm,
       [=]() mutable {
         return bn_input.batch_norm(bn_running_mean, bn_running_var, bn_weight,
                                    bn_bias, false, 0.1f, 1e-5f);
       },
       [=]() { return bn_input; }},
      {OpId::LayerNorm,
       [=]() { return ln_input.layer_norm(ln_weight, ln_bias, 1e-5f); },
       [=]() { return ln_input; }},
      {OpId::MSELoss, [=]() { return loss_pred.mse_loss(loss_target); },
       [=]() { return loss_pred; }},
      {OpId::CrossEntropy,
       [=]() { return loss_pred.cross_entropy(loss_target); },
       [=]() { return loss_pred; }},
      {OpId::Transpose, [=]() { return shape_input.transpose(0, 1); },
       [=]() { return shape_input; }},
      {OpId::Narrow, [=]() { return shape_input.narrow(1, 4, 8); },
       [=]() { return shape_input; }},
      {OpId::Zeros,
       [=]() { return zeros({64, 64}, device, false, DataType::Float32); },
       [=]() { return shape_input; }},
  };
}

struct ProfileAggregate {
  double host_us = 0.0;
  double gpu_us = 0.0;
  double min_host_us = std::numeric_limits<double>::max();
  double min_gpu_us = std::numeric_limits<double>::max();
  double max_host_us = 0.0;
  double max_gpu_us = 0.0;
  size_t bytes_processed = 0;
  int count = 0;
};

struct OperatorBaseline {
  std::string name;
  size_t iterations = 0;
  double min_us = 0.0;
  double avg_us = 0.0;
  double max_us = 0.0;
  std::map<std::string, ProfileAggregate> profile;
};

bool perf_tests_enabled() {
  const char *value = std::getenv("MUNET_RUN_PERF_TESTS");
  if (value == nullptr) {
    return false;
  }
  const std::string flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" ||
         flag == "ON" || flag == "yes" || flag == "YES";
}

void reset_profiler_quietly() {
  set_profile_enabled_override(false);
  Profiler::get().reset();
  set_profile_enabled_override(true);
}

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (char c : value) {
    switch (c) {
    case '\\':
      escaped << "\\\\";
      break;
    case '"':
      escaped << "\\\"";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      escaped << c;
      break;
    }
  }
  return escaped.str();
}

void add_profile_snapshot(OperatorBaseline &baseline,
                          const ProfilerSnapshot &snapshot) {
  for (const auto &[label, stats] : snapshot.stats) {
    auto &aggregate = baseline.profile[label];
    aggregate.host_us += stats.host_us;
    aggregate.gpu_us += stats.gpu_us;
    aggregate.min_host_us = std::min(aggregate.min_host_us, stats.min_host_us);
    aggregate.min_gpu_us = std::min(aggregate.min_gpu_us, stats.min_gpu_us);
    aggregate.max_host_us = std::max(aggregate.max_host_us, stats.max_host_us);
    aggregate.max_gpu_us = std::max(aggregate.max_gpu_us, stats.max_gpu_us);
    aggregate.bytes_processed += stats.bytes_processed;
    aggregate.count += stats.count;
  }
}

OperatorBaseline benchmark_operator(const std::string &name,
                                    const std::function<Tensor()> &run,
                                    size_t warmup_iterations,
                                    size_t measured_iterations) {
  for (size_t i = 0; i < warmup_iterations; ++i) {
    Tensor out = run();
    out.impl_->backend().synchronize();
  }

  OperatorBaseline baseline;
  baseline.name = name;
  baseline.iterations = measured_iterations;
  std::vector<double> samples;
  samples.reserve(measured_iterations);

  for (size_t i = 0; i < measured_iterations; ++i) {
    reset_profiler_quietly();
    const auto start = std::chrono::steady_clock::now();
    Tensor out = run();
    out.impl_->backend().synchronize();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration<double, std::micro>(end - start).count();
    samples.push_back(elapsed);
    add_profile_snapshot(baseline, Profiler::get().snapshot());
  }

  baseline.min_us = *std::min_element(samples.begin(), samples.end());
  baseline.max_us = *std::max_element(samples.begin(), samples.end());
  baseline.avg_us =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  return baseline;
}

struct ProfileBreakdownRow {
  std::string label;
  const ProfileAggregate *stats = nullptr;
  double total_us = 0.0;
  double avg_us = 0.0;
  double percent = 0.0;
};

std::string truncate_label(const std::string &label, size_t width) {
  if (label.size() <= width) {
    return label;
  }
  if (width <= 3) {
    return label.substr(0, width);
  }
  return label.substr(0, width - 3) + "...";
}

std::string scaled_bar(double percent, size_t width = 28) {
  const size_t filled =
      static_cast<size_t>(std::round((percent / 100.0) * width));
  return std::string(std::min(filled, width), '#') +
         std::string(width - std::min(filled, width), '.');
}

std::vector<ProfileBreakdownRow>
profile_breakdown_rows(const OperatorBaseline &baseline) {
  std::vector<ProfileBreakdownRow> rows;
  rows.reserve(baseline.profile.size());

  double total_profile_us = 0.0;
  for (const auto &[label, stats] : baseline.profile) {
    total_profile_us += stats.host_us + stats.gpu_us;
  }

  for (const auto &[label, stats] : baseline.profile) {
    const double total_us = stats.host_us + stats.gpu_us;
    const double avg_us = stats.count > 0 ? total_us / stats.count : 0.0;
    const double percent =
        total_profile_us > 0.0 ? (total_us / total_profile_us) * 100.0 : 0.0;
    rows.push_back(
        ProfileBreakdownRow{label, &stats, total_us, avg_us, percent});
  }

  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    if (a.total_us == b.total_us) {
      return a.label < b.label;
    }
    return a.total_us > b.total_us;
  });
  return rows;
}

std::string profile_breakdown_visual(const OperatorBaseline &baseline) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  const auto rows = profile_breakdown_rows(baseline);

  out << "\nPERF_BREAKDOWN operator=" << baseline.name << "\n";
  out << "  " << std::left << std::setw(38) << "label" << std::right
      << std::setw(8) << "calls" << std::setw(12) << "total_us" << std::setw(12)
      << "avg_us" << std::setw(10) << "share"
      << "  visual\n";
  out << "  " << std::string(38, '-') << " " << std::string(7, '-') << " "
      << std::string(11, '-') << " " << std::string(11, '-') << " "
      << std::string(9, '-') << "  " << std::string(28, '-') << "\n";

  for (const auto &row : rows) {
    out << "  " << std::left << std::setw(38) << truncate_label(row.label, 38)
        << std::right << std::setw(8) << row.stats->count << std::setw(12)
        << row.total_us << std::setw(12) << row.avg_us << std::setw(9)
        << row.percent << "%  " << scaled_bar(row.percent) << "\n";
  }

  out << "  wall_us min/avg/max = " << baseline.min_us << " / "
      << baseline.avg_us << " / " << baseline.max_us << "\n";
  return out.str();
}

std::string profile_breakdown_json(const OperatorBaseline &baseline) {
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const auto &[label, stats] : baseline.profile) {
    if (!first) {
      out << ",";
    }
    first = false;
    const double avg_host = stats.count > 0 ? stats.host_us / stats.count : 0.0;
    const double avg_gpu = stats.count > 0 ? stats.gpu_us / stats.count : 0.0;
    const double min_host =
        stats.min_host_us == std::numeric_limits<double>::max()
            ? 0.0
            : stats.min_host_us;
    const double min_gpu =
        stats.min_gpu_us == std::numeric_limits<double>::max()
            ? 0.0
            : stats.min_gpu_us;
    out << "{\"label\":\"" << json_escape(label)
        << "\",\"count\":" << stats.count
        << ",\"host_us_total\":" << stats.host_us
        << ",\"host_us_avg\":" << avg_host << ",\"host_us_min\":" << min_host
        << ",\"host_us_max\":" << stats.max_host_us
        << ",\"vulkan_us_total\":" << stats.gpu_us
        << ",\"vulkan_us_avg\":" << avg_gpu << ",\"vulkan_us_min\":" << min_gpu
        << ",\"vulkan_us_max\":" << stats.max_gpu_us
        << ",\"bytes\":" << stats.bytes_processed << "}";
  }
  out << "]";
  return out.str();
}

void print_operator_baseline(const OperatorBaseline &baseline) {
  std::cout << std::fixed << std::setprecision(3)
            << "PERF operator=" << baseline.name
            << " iterations=" << baseline.iterations
            << " min_us=" << baseline.min_us << " avg_us=" << baseline.avg_us
            << " max_us=" << baseline.max_us
            << " profile_breakdown=" << profile_breakdown_json(baseline)
            << profile_breakdown_visual(baseline) << std::endl;
}

void expect_valid_baseline(const OperatorBaseline &baseline) {
  EXPECT_GT(baseline.iterations, 0u);
  EXPECT_GE(baseline.min_us, 0.0);
  EXPECT_GE(baseline.avg_us, baseline.min_us);
  EXPECT_GE(baseline.max_us, baseline.avg_us);
  EXPECT_FALSE(baseline.profile.empty())
      << baseline.name << " did not emit profiler breakdown rows";
}

} // namespace

TEST(PerformanceTest, OperatorBaselineMinAvgMaxAndProfilerBreakdown) {
  if (!perf_tests_enabled()) {
    GTEST_SKIP() << "Set MUNET_RUN_PERF_TESTS=1 to collect Vulkan operator "
                    "performance baselines.";
  }

  ScopedProfileOverride profile(true);
  const Device device{DeviceType::VULKAN, 0};
  constexpr size_t kWarmupIterations = 3;
  constexpr size_t kMeasuredIterations = 10;

  const auto cases = build_operator_cases(device);
  const auto registered_ops = registered_op_ids();
  const std::set<OpId> expected_ops(registered_ops.begin(),
                                    registered_ops.end());
  std::set<OpId> covered_ops;
  std::vector<OperatorBaseline> baselines;

  for (const auto &op_case : cases) {
    covered_ops.insert(op_case.id);
    const auto &metadata = op_metadata(op_case.id);
    Tensor probe = op_case.dispatch_probe();
    const auto decision = resolve_dispatch(op_case.id, probe);
    if (metadata.feature.has_value()) {
      EXPECT_TRUE(decision.use_backend)
          << metadata.name << " must resolve to a Vulkan runtime kernel";
      EXPECT_FALSE(decision.use_reference_path)
          << metadata.name << " unexpectedly used a reference path";
    }

    baselines.push_back(benchmark_operator(
        metadata.name, op_case.run, kWarmupIterations, kMeasuredIterations));
  }

  EXPECT_EQ(covered_ops, expected_ops)
      << "Every OpId must have a perf baseline";

  for (const auto &baseline : baselines) {
    print_operator_baseline(baseline);
    expect_valid_baseline(baseline);
    RecordProperty((baseline.name + "_min_us").c_str(), baseline.min_us);
    RecordProperty((baseline.name + "_avg_us").c_str(), baseline.avg_us);
    RecordProperty((baseline.name + "_max_us").c_str(), baseline.max_us);
    RecordProperty((baseline.name + "_profile_breakdown").c_str(),
                   profile_breakdown_json(baseline));
    RecordProperty((baseline.name + "_profile_visual").c_str(),
                   profile_breakdown_visual(baseline));
  }
}
