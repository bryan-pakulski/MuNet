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
#include <sstream>
#include <string>
#include <vector>

using namespace munet;

namespace {

class ScopedProfileOverride {
public:
  explicit ScopedProfileOverride(bool enabled) {
    set_profile_enabled_override(enabled);
  }

  ~ScopedProfileOverride() { set_profile_enabled_override(std::nullopt); }
};

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
    const double percent = total_profile_us > 0.0
                               ? (total_us / total_profile_us) * 100.0
                               : 0.0;
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
      << std::setw(8) << "calls" << std::setw(12) << "total_us"
      << std::setw(12) << "avg_us" << std::setw(10) << "share"
      << "  visual\n";
  out << "  " << std::string(38, '-') << " " << std::string(7, '-') << " "
      << std::string(11, '-') << " " << std::string(11, '-') << " "
      << std::string(9, '-') << "  " << std::string(28, '-') << "\n";

  for (const auto &row : rows) {
    out << "  " << std::left << std::setw(38)
        << truncate_label(row.label, 38) << std::right << std::setw(8)
        << row.stats->count << std::setw(12) << row.total_us << std::setw(12)
        << row.avg_us << std::setw(9) << row.percent << "%  "
        << scaled_bar(row.percent) << "\n";
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
    const double min_host = stats.min_host_us == std::numeric_limits<double>::max()
                                ? 0.0
                                : stats.min_host_us;
    const double min_gpu = stats.min_gpu_us == std::numeric_limits<double>::max()
                               ? 0.0
                               : stats.min_gpu_us;
    out << "{\"label\":\"" << json_escape(label) << "\",\"count\":" << stats.count
        << ",\"host_us_total\":" << stats.host_us
        << ",\"host_us_avg\":" << avg_host
        << ",\"host_us_min\":" << min_host
        << ",\"host_us_max\":" << stats.max_host_us
        << ",\"vulkan_us_total\":" << stats.gpu_us
        << ",\"vulkan_us_avg\":" << avg_gpu
        << ",\"vulkan_us_min\":" << min_gpu
        << ",\"vulkan_us_max\":" << stats.max_gpu_us
        << ",\"bytes\":" << stats.bytes_processed << "}";
  }
  out << "]";
  return out.str();
}

void print_operator_baseline(const OperatorBaseline &baseline) {
  std::cout << std::fixed << std::setprecision(3) << "PERF operator="
            << baseline.name << " iterations=" << baseline.iterations
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

  Tensor add_a({256, 256}, device, DataType::Float32);
  Tensor add_b({256, 256}, device, DataType::Float32);
  add_a.fill_(1.0f);
  add_b.fill_(2.0f);

  Tensor matmul_a({64, 64}, device, DataType::Float32);
  Tensor matmul_b({64, 64}, device, DataType::Float32);
  matmul_a.fill_(0.5f);
  matmul_b.fill_(1.5f);

  Tensor activation_input({256, 256}, device, DataType::Float32);
  activation_input.fill_(0.25f);

  Tensor softmax_input({64, 128}, device, DataType::Float32);
  softmax_input.fill_(0.125f);

  std::vector<OperatorBaseline> baselines;
  baselines.push_back(benchmark_operator(
      "add", [&]() { return add_a + add_b; }, kWarmupIterations,
      kMeasuredIterations));
  baselines.push_back(benchmark_operator(
      "matmul", [&]() { return matmul_a.matmul(matmul_b); },
      kWarmupIterations, kMeasuredIterations));
  baselines.push_back(benchmark_operator(
      "relu", [&]() { return activation_input.relu(); }, kWarmupIterations,
      kMeasuredIterations));
  baselines.push_back(benchmark_operator(
      "softmax", [&]() { return softmax_input.softmax(-1); },
      kWarmupIterations, kMeasuredIterations));

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
