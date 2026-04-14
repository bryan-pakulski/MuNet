/**
 * @file vulkan_profiling.cpp
 * @brief Fine-grained Vulkan backend profiling benchmark suite
 *
 * This test suite provides detailed timing breakdown for Vulkan operations:
 * - Kernel compilation and dispatch
 * - Descriptor set allocation
 * - Command buffer recording
 * - Buffer transfers (H2D and D2H)
 * - Pipeline creation
 * - Shader compilation
 * - Cold start vs warm performance
 */

#include "../test_utils.hpp"
#include "core/util/profiler.hpp"
#include "tensor.hpp"
#include "backend/vulkan_backend.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace munet {
namespace test {
namespace {

// ============================================================================
// Timing Infrastructure
// ============================================================================

struct TimingCategory {
    std::string name;
    std::string description;
};

const std::vector<TimingCategory> kTimingCategories = {
    {"kernel_compile", "Time to compile GLSL compute shaders"},
    {"kernel_dispatch", "Time to dispatch kernel to GPU"},
    {"descriptor_set_alloc", "Time to allocate Vulkan descriptor sets"},
    {"command_buffer_record", "Time to record Vulkan command buffers"},
    {"buffer_transfer_h2d", "Time to transfer data from host to device"},
    {"buffer_transfer_d2h", "Time to transfer data from device to host"},
    {"kernel_execute", "Time for kernel execution on GPU"},
    {"pipeline_creation", "Time to create Vulkan pipelines"},
    {"shader_compilation", "Time to compile GLSL to SPIR-V"},
    {"total_operation", "End-to-end operation time"},
};

class TimingResults {
  public:
    void add(const std::string &category, double time_ms) {
        results_[category].push_back(time_ms);
    }

    void print_summary(std::ostream &os = std::cout) const {
        os << "\n=== Vulkan Profiling Results ===\n";
        os << std::left << std::setw(30) << "Category" << std::setw(12)
           << "Min (ms)" << std::setw(12) << "Max (ms)" << std::setw(12)
           << "Avg (ms)" << std::setw(12) << "StdDev" << "\n";
        os << std::string(78, '-') << "\n";

        for (const auto &cat : kTimingCategories) {
            auto it = results_.find(cat.name);
            if (it == results_.end() || it->second.empty())
                continue;

            const auto &times = it->second;
            double min = *std::min_element(times.begin(), times.end());
            double max = *std::max_element(times.begin(), times.end());
            double avg =
                std::accumulate(times.begin(), times.end(), 0.0) / times.size();

            double variance = 0.0;
            for (double t : times) {
                variance += (t - avg) * (t - avg);
            }
            double stddev = std::sqrt(variance / times.size());

            os << std::left << std::setw(30) << cat.name << std::fixed
               << std::setprecision(3) << std::setw(12) << min << std::setw(12)
               << max << std::setw(12) << avg << std::setw(12) << stddev
               << "\n";
        }
        os << std::string(78, '=') << "\n";
    }

    void write_csv(const std::string &filename) const {
        std::ofstream file(filename);
        file << "category,min_ms,max_ms,avg_ms,stddev_ms,samples\n";
        for (const auto &cat : kTimingCategories) {
            auto it = results_.find(cat.name);
            if (it == results_.end() || it->second.empty())
                continue;

            const auto &times = it->second;
            double min = *std::min_element(times.begin(), times.end());
            double max = *std::max_element(times.begin(), times.end());
            double avg =
                std::accumulate(times.begin(), times.end(), 0.0) / times.size();

            double variance = 0.0;
            for (double t : times) {
                variance += (t - avg) * (t - avg);
            }
            double stddev = std::sqrt(variance / times.size());

            file << cat.name << "," << std::fixed << std::setprecision(6) << min
                 << "," << max << "," << avg << "," << stddev << ","
                 << times.size() << "\n";
        }
    }

    void merge(const TimingResults &other) {
        for (const auto &[cat, times] : other.results_) {
            for (double t : times) {
                results_[cat].push_back(t);
            }
        }
    }

    void clear() { results_.clear(); }

  private:
    std::map<std::string, std::vector<double>> results_;
};

// ============================================================================
// Benchmark Functions
// ============================================================================

// High-resolution timer
template <typename Func>
double measure_time_ms(Func &&func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Helper to extract profiler statistics and map to timing categories
TimingResults extract_profiler_stats() {
    TimingResults results;
    auto snapshot = Profiler::get().snapshot();
    
    // Helper to get average time in ms from OpStats
    auto avg_ms = [](const OpStats& s) -> double {
        if (s.count == 0) return 0.0;
        return (s.cpu_us / s.count) / 1000.0;  // Convert avg us to ms
    };
    
    // Map profiler events to timing categories
    // kernel_compile: shader compilation time
    if (snapshot.stats.count("vulkan.shader_compile") > 0) {
        results.add("kernel_compile", avg_ms(snapshot.stats.at("vulkan.shader_compile")));
    }
    
    // kernel_dispatch: time to dispatch kernel to GPU
    if (snapshot.stats.count("vulkan.dispatch_encode") > 0) {
        results.add("kernel_dispatch", avg_ms(snapshot.stats.at("vulkan.dispatch_encode")));
    }
    
    // descriptor_set_alloc: time to allocate Vulkan descriptor sets
    if (snapshot.stats.count("vulkan.update_descriptors") > 0) {
        results.add("descriptor_set_alloc", avg_ms(snapshot.stats.at("vulkan.update_descriptors")));
    }
    
    // command_buffer_record: time to record Vulkan command buffers
    if (snapshot.stats.count("vulkan.dispatch_encode") > 0) {
        results.add("command_buffer_record", avg_ms(snapshot.stats.at("vulkan.dispatch_encode")));
    }
    
    // buffer_transfer_h2d: time to transfer data from host to device
    if (snapshot.stats.count("transfer.h2d") > 0) {
        results.add("buffer_transfer_h2d", avg_ms(snapshot.stats.at("transfer.h2d")));
    }
    
    // buffer_transfer_d2h: time to transfer data from device to host
    if (snapshot.stats.count("transfer.d2h") > 0) {
        results.add("buffer_transfer_d2h", avg_ms(snapshot.stats.at("transfer.d2h")));
    }
    
    // kernel_execute: time for kernel execution on GPU (sync time)
    if (snapshot.stats.count("sync.explicit.vulkan") > 0) {
        results.add("kernel_execute", avg_ms(snapshot.stats.at("sync.explicit.vulkan")));
    }
    
    // pipeline_creation: time to create Vulkan pipelines
    if (snapshot.stats.count("vulkan.pipeline_create") > 0) {
        results.add("pipeline_creation", avg_ms(snapshot.stats.at("vulkan.pipeline_create")));
    }
    
    // shader_compilation: time to compile GLSL to SPIR-V
    if (snapshot.stats.count("vulkan.shader_compile") > 0) {
        results.add("shader_compilation", avg_ms(snapshot.stats.at("vulkan.shader_compile")));
    }
    
    // allocator operations
    if (snapshot.stats.count("allocator.pool_growth.vulkan") > 0) {
        results.add("allocator_pool_growth", avg_ms(snapshot.stats.at("allocator.pool_growth.vulkan")));
    }
    
    return results;
}

// Matmul benchmark with timing breakdown
double benchmark_matmul(Device device, int M, int K, int N, int warmup_iters = 3,
                        int test_iters = 10, TimingResults *timing_out = nullptr) {
    // Create matrices using the correct API
    Tensor a = Tensor({M, K}, device, DataType::Float32);
    Tensor b = Tensor({K, N}, device, DataType::Float32);

    // Fill with data
    a.fill_(1.0f);
    b.fill_(1.0f);

    // Warmup iterations
    for (int i = 0; i < warmup_iters; ++i) {
        Tensor c = a * b;
        c.impl_->backend().synchronize();
    }

    // Timed iterations with profiler capture
    std::vector<double> total_times;
    for (int i = 0; i < test_iters; ++i) {
        Profiler::get().reset();
        double t = measure_time_ms([&]() {
            Tensor c = a * b;
            c.impl_->backend().synchronize();
        });
        total_times.push_back(t);
        
        if (timing_out) {
            auto stats = extract_profiler_stats();
            timing_out->merge(stats);
            timing_out->add("total_operation", t);
        }
    }

    return std::accumulate(total_times.begin(), total_times.end(), 0.0) /
           total_times.size();
}

// Elementwise operation benchmark
double benchmark_elementwise(Device device, int rows, int cols,
                            int warmup_iters = 3, int test_iters = 10) {
    TimingResults timing;

    Tensor a = Tensor({rows, cols}, device, DataType::Float32);
    Tensor b = Tensor({rows, cols}, device, DataType::Float32);

    a.fill_(1.0f);
    b.fill_(2.0f);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        Tensor c = a + b;
        c.impl_->backend().synchronize();
    }

    // Timed iterations
    std::vector<double> total_times;
    for (int i = 0; i < test_iters; ++i) {
        double t = measure_time_ms([&]() {
            Tensor c = a + b;
            c.impl_->backend().synchronize();
        });
        total_times.push_back(t);
    }

    return std::accumulate(total_times.begin(), total_times.end(), 0.0) /
           total_times.size();
}

// Elementwise operation types for benchmarking
enum class ElementwiseOp { Add, Mul, Sigmoid, Relu };

const char* op_name(ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return "add";
        case ElementwiseOp::Mul: return "mul";
        case ElementwiseOp::Sigmoid: return "sigmoid";
        case ElementwiseOp::Relu: return "relu";
    }
    return "unknown";
}

// Bandwidth factor: how many bytes transferred per element (read inputs + write output)
int op_bandwidth_factor(ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return 3;   // read 2 + write 1
        case ElementwiseOp::Mul: return 3;   // read 2 + write 1
        case ElementwiseOp::Sigmoid: return 2; // read 1 + write 1
        case ElementwiseOp::Relu: return 2;   // read 1 + write 1
    }
    return 3;
}

// Benchmark a specific elementwise operation on a device
double benchmark_elementwise_op(Device device, int rows, int cols,
                                ElementwiseOp op,
                                int warmup_iters = 3, int test_iters = 10) {
    Tensor a = Tensor({rows, cols}, device, DataType::Float32);
    Tensor b = Tensor({rows, cols}, device, DataType::Float32);

    a.fill_(1.0f);
    b.fill_(2.0f);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        Tensor c = (op == ElementwiseOp::Add)  ? (Tensor)(a + b) :
                   (op == ElementwiseOp::Mul)  ? (Tensor)(a * b) :
                   (op == ElementwiseOp::Sigmoid) ? a.sigmoid() :
                   a.relu();
        c.impl_->backend().synchronize();
    }

    // Timed iterations
    std::vector<double> total_times;
    for (int i = 0; i < test_iters; ++i) {
        double t = measure_time_ms([&]() {
            Tensor c = (op == ElementwiseOp::Add)  ? (Tensor)(a + b) :
                       (op == ElementwiseOp::Mul)  ? (Tensor)(a * b) :
                       (op == ElementwiseOp::Sigmoid) ? a.sigmoid() :
                       a.relu();
            c.impl_->backend().synchronize();
        });
        total_times.push_back(t);
    }

    return std::accumulate(total_times.begin(), total_times.end(), 0.0) /
           total_times.size();
}

// Benchmark a sequential chain of elementwise operations on a device
double benchmark_elementwise_chain(Device device, int rows, int cols,
                                   const std::vector<ElementwiseOp>& ops,
                                   int warmup_iters = 3, int test_iters = 10) {
    Tensor a = Tensor({rows, cols}, device, DataType::Float32);
    Tensor b = Tensor({rows, cols}, device, DataType::Float32);

    a.fill_(1.0f);
    b.fill_(2.0f);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        Tensor cur = a;
        for (auto op : ops) {
            switch (op) {
                case ElementwiseOp::Add: cur = cur + b; break;
                case ElementwiseOp::Mul: cur = cur * b; break;
                case ElementwiseOp::Sigmoid: cur = cur.sigmoid(); break;
                case ElementwiseOp::Relu: cur = cur.relu(); break;
            }
        }
        cur.impl_->backend().synchronize();
    }

    // Timed iterations
    std::vector<double> total_times;
    for (int i = 0; i < test_iters; ++i) {
        double t = measure_time_ms([&]() {
            Tensor cur = a;
            for (auto op : ops) {
                switch (op) {
                    case ElementwiseOp::Add: cur = cur + b; break;
                    case ElementwiseOp::Mul: cur = cur * b; break;
                    case ElementwiseOp::Sigmoid: cur = cur.sigmoid(); break;
                    case ElementwiseOp::Relu: cur = cur.relu(); break;
                }
            }
            cur.impl_->backend().synchronize();
        });
        total_times.push_back(t);
    }

    return std::accumulate(total_times.begin(), total_times.end(), 0.0) /
           total_times.size();
}

// Host-to-device transfer benchmark
double benchmark_h2d_transfer(Device device, int size_bytes,
                              int test_iters = 10) {
    int size_elements = size_bytes / sizeof(float);

    // Create CPU tensor
    Device cpu{DeviceType::CPU, 0};
    Tensor cpu_tensor = Tensor({size_elements}, cpu, DataType::Float32);
    cpu_tensor.fill_(1.0f);

    std::vector<double> times;
    for (int i = 0; i < test_iters; ++i) {
        double t = measure_time_ms([&]() {
            Tensor gpu_tensor = cpu_tensor.to(device);
            gpu_tensor.impl_->backend().synchronize();
        });
        times.push_back(t);
    }

    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

// Device-to-host transfer benchmark
double benchmark_d2h_transfer(Device device, int size_bytes,
                              int test_iters = 10) {
    int size_elements = size_bytes / sizeof(float);

    // Create GPU tensor
    Tensor gpu_tensor = Tensor({size_elements}, device, DataType::Float32);
    gpu_tensor.fill_(1.0f);

    std::vector<double> times;
    for (int i = 0; i < test_iters; ++i) {
        double t = measure_time_ms([&]() {
            Tensor cpu_tensor = gpu_tensor.to({DeviceType::CPU, 0});
        });
        times.push_back(t);
    }

    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

// Cold start benchmark (first operation after device init)
double benchmark_cold_start(Device device, int rows, int cols) {
    // Create tensors but don't do any operation yet
    Tensor a = Tensor({rows, cols}, device, DataType::Float32);
    Tensor b = Tensor({rows, cols}, device, DataType::Float32);

    a.fill_(1.0f);
    b.fill_(2.0f);

    // Measure first operation (cold start)
    double t = measure_time_ms([&]() {
        Tensor c = a + b;
        c.impl_->backend().synchronize();
    });

    return t;
}

// ============================================================================
// Test Fixtures
// ============================================================================

class VulkanProfilingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Check if Vulkan is available
        auto devices = get_available_devices();
        vulkan_available_ = false;
        for (const auto &dev : devices) {
            if (dev.type == DeviceType::VULKAN) {
                vulkan_device_ = dev;
                vulkan_available_ = true;
                break;
            }
        }
    }

    Device vulkan_device_{DeviceType::VULKAN, 0};
    bool vulkan_available_ = false;
};

// ============================================================================
// Vulkan Profiling Tests
// ============================================================================

TEST_F(VulkanProfilingTest, MatmulPerfBreakdown) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}};

    std::cout << "\n=== Matmul Performance Breakdown ===\n";
    std::cout << std::left << std::setw(20) << "Size (MxKxN)" << std::setw(15)
              << "Avg Time (ms)" << std::setw(15) << "GFLOPS"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    TimingResults aggregate_timing;
    
    for (auto [M, K, N] : sizes) {
        double avg_time =
            benchmark_matmul(vulkan_device_, M, K, N, 3, 10, &aggregate_timing);

        // Compute GFLOPS
        double gflops = (2.0 * M * K * N) / (avg_time * 1e-3) / 1e9;

        std::cout << std::left << std::setw(20)
                  << (std::to_string(M) + "x" + std::to_string(K) + "x" +
                      std::to_string(N))
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << avg_time << std::setw(15) << std::setprecision(2) << gflops
                  << "\n";
    }

    // Print detailed timing breakdown from profiler
    std::cout << "\n--- Detailed Timing Breakdown (from Profiler) ---\n";
    aggregate_timing.print_summary();
}

TEST_F(VulkanProfilingTest, ElementwisePerfBreakdown) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    std::vector<std::pair<int, int>> sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}};

    std::cout << "\n=== Elementwise Performance Breakdown ===\n";
    std::cout << std::left << std::setw(20) << "Size (rows x cols)"
              << std::setw(15) << "Avg Time (ms)" << std::setw(15)
              << "Bandwidth (GB/s)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto [rows, cols] : sizes) {
        double avg_time =
            benchmark_elementwise(vulkan_device_, rows, cols, 3, 10);

        // Compute bandwidth (read 2 inputs + write 1 output = 3x size)
        double bytes = 3.0 * rows * cols * sizeof(float);
        double bandwidth_gbs = (bytes / (avg_time * 1e-3)) / 1e9;

        std::cout << std::left << std::setw(20)
                  << (std::to_string(rows) + "x" + std::to_string(cols))
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << avg_time << std::setw(15) << std::setprecision(2)
                  << bandwidth_gbs << "\n";
    }
}

TEST_F(VulkanProfilingTest, TransferPerfBreakdown) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    // Test various transfer sizes (1KB to 64MB)
    std::vector<int> sizes_kb = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536};

    std::cout << "\n=== Transfer Performance Breakdown ===\n";
    std::cout << std::left << std::setw(15) << "Size (KB)" << std::setw(20)
              << "H2D Time (ms)" << std::setw(20) << "D2H Time (ms)"
              << std::setw(20) << "H2D BW (GB/s)"
              << std::setw(20) << "D2H BW (GB/s)"
              << "\n";
    std::cout << std::string(95, '-') << "\n";

    for (int size_kb : sizes_kb) {
        int size_bytes = size_kb * 1024;

        double h2d_time = benchmark_h2d_transfer(vulkan_device_, size_bytes, 10);
        double d2h_time = benchmark_d2h_transfer(vulkan_device_, size_bytes, 10);

        double h2d_bw = (size_bytes / (h2d_time * 1e-3)) / 1e9;
        double d2h_bw = (size_bytes / (d2h_time * 1e-3)) / 1e9;

        std::cout << std::left << std::setw(15) << size_kb << std::fixed
                  << std::setprecision(3) << std::setw(20) << h2d_time
                  << std::setw(20) << d2h_time << std::setw(20)
                  << std::setprecision(2) << h2d_bw << std::setw(20) << d2h_bw
                  << "\n";
    }
}

TEST_F(VulkanProfilingTest, ColdStartVsWarm) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    std::cout << "\n=== Cold Start vs Warm Performance ===\n";

    // Create a fresh context by performing some operations first
    Tensor warmup_a = Tensor({256, 256}, vulkan_device_, DataType::Float32);
    Tensor warmup_b = Tensor({256, 256}, vulkan_device_, DataType::Float32);
    warmup_a.fill_(1.0f);
    warmup_b.fill_(1.0f);

    // Warm run
    double warm_time = measure_time_ms([&]() {
        Tensor c = warmup_a + warmup_b;
        c.impl_->backend().synchronize();
    });

    std::cout << "Warm operation time: " << std::fixed << std::setprecision(3)
              << warm_time << " ms\n";

    // Note: True cold start testing would require device reset
    // which is not currently exposed in the API
    std::cout << "(Note: True cold start measurement requires device reset)\n";
}

TEST_F(VulkanProfilingTest, CompareWithCUDA) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    // Check for CUDA availability
    Device cuda_device{DeviceType::CUDA, 0};
    bool cuda_available = false;
    for (const auto &dev : get_available_devices()) {
        if (dev.type == DeviceType::CUDA) {
            cuda_device = dev;
            cuda_available = true;
            break;
        }
    }

    if (!cuda_available) {
        std::cout << "\n=== CUDA not available, skipping comparison ===\n";
        GTEST_SKIP() << "CUDA device not available";
    }

    std::cout << "\n=== Vulkan vs CUDA Performance Comparison ===\n";

    // Matmul comparison
    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 128}, {256, 256, 256}, {512, 512, 512}};

    std::cout << "\nMatmul Comparison:\n";
    std::cout << std::left << std::setw(15) << "Size" << std::setw(15)
              << "Vulkan (ms)" << std::setw(15) << "CUDA (ms)" << std::setw(15)
              << "Ratio (V/C)"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto [M, K, N] : sizes) {
        double vulkan_time =
            benchmark_matmul(vulkan_device_, M, K, N, 3, 10);
        double cuda_time =
            benchmark_matmul(cuda_device, M, K, N, 3, 10);

        double ratio = vulkan_time / cuda_time;

        std::cout << std::left << std::setw(15)
                  << (std::to_string(M) + "x" + std::to_string(K) + "x" +
                      std::to_string(N))
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << vulkan_time << std::setw(15) << cuda_time << std::setw(15)
                  << std::setprecision(2) << ratio << "x\n";
    }

    // Transfer comparison
    std::cout << "\nTransfer Comparison (1MB):\n";
    int size_1mb = 1024 * 1024;

    double vk_h2d = benchmark_h2d_transfer(vulkan_device_, size_1mb, 10);
    double vk_d2h = benchmark_d2h_transfer(vulkan_device_, size_1mb, 10);
    double cuda_h2d = benchmark_h2d_transfer(cuda_device, size_1mb, 10);
    double cuda_d2h = benchmark_d2h_transfer(cuda_device, size_1mb, 10);

    std::cout << std::left << std::setw(15) << "H2D 1MB" << std::setw(15)
              << std::fixed << std::setprecision(3) << vk_h2d << std::setw(15)
              << cuda_h2d << std::setw(15) << std::setprecision(2)
              << (vk_h2d / cuda_h2d) << "x\n";
    std::cout << std::left << std::setw(15) << "D2H 1MB" << std::setw(15)
              << std::fixed << std::setprecision(3) << vk_d2h << std::setw(15)
              << cuda_d2h << std::setw(15) << std::setprecision(2)
              << (vk_d2h / cuda_d2h) << "x\n";
}

TEST_F(VulkanProfilingTest, CompareElementwiseWithCUDA) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    // Check for CUDA availability
    Device cuda_device{DeviceType::CUDA, 0};
    bool cuda_available = false;
    for (const auto &dev : get_available_devices()) {
        if (dev.type == DeviceType::CUDA) {
            cuda_device = dev;
            cuda_available = true;
            break;
        }
    }

    if (!cuda_available) {
        std::cout << "\n=== CUDA not available, skipping elementwise comparison ===\n";
        GTEST_SKIP() << "CUDA device not available";
    }

    std::cout << "\n=== Vulkan vs CUDA Elementwise Performance Comparison ===\n";

    // Sizes to benchmark
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    // Individual op benchmarks: add, mul, sigmoid, relu
    std::vector<ElementwiseOp> ops = {
        ElementwiseOp::Add, ElementwiseOp::Mul,
        ElementwiseOp::Sigmoid, ElementwiseOp::Relu
    };

    for (auto op : ops) {
        std::cout << "\n--- " << op_name(op) << " ---\n";
        std::cout << std::left << std::setw(15) << "Size"
                  << std::setw(15) << "Vulkan (ms)" << std::setw(15) << "CUDA (ms)"
                  << std::setw(15) << "Vulkan GB/s" << std::setw(15) << "CUDA GB/s"
                  << std::setw(12) << "Ratio (V/C)"
                  << "\n";
        std::cout << std::string(87, '-') << "\n";

        for (int size : sizes) {
            double vulkan_time = benchmark_elementwise_op(vulkan_device_, size, size, op, 3, 10);
            double cuda_time = benchmark_elementwise_op(cuda_device, size, size, op, 3, 10);

            int64_t num_elements = (int64_t)size * size;
            double bytes = (double)op_bandwidth_factor(op) * num_elements * sizeof(float);
            double vulkan_bw = (bytes / (vulkan_time * 1e-3)) / 1e9;
            double cuda_bw = (bytes / (cuda_time * 1e-3)) / 1e9;
            double ratio = vulkan_time / cuda_time;

            std::cout << std::left << std::setw(15)
                      << (std::to_string(size) + "x" + std::to_string(size))
                      << std::fixed << std::setprecision(3) << std::setw(15)
                      << vulkan_time << std::setw(15) << cuda_time
                      << std::setprecision(2) << std::setw(15)
                      << vulkan_bw << std::setw(15) << cuda_bw
                      << std::setprecision(2) << std::setw(12) << ratio << "x\n";
        }
    }

    // Chain benchmarks: Vulkan fused vs CUDA sequential
    std::cout << "\n=== Chain Comparison: Vulkan Fused vs CUDA Sequential ===\n";

    // Chain 1: sigmoid + relu (2 unary ops)
    {
        std::cout << "\n--- 2-op chain: sigmoid + relu ---\n";
        std::cout << std::left << std::setw(15) << "Size"
                  << std::setw(18) << "Vulkan fused (ms)" << std::setw(18) << "CUDA seq (ms)"
                  << std::setw(15) << "Vulkan GB/s" << std::setw(15) << "CUDA GB/s"
                  << std::setw(12) << "Ratio (V/C)"
                  << "\n";
        std::cout << std::string(93, '-') << "\n";

        std::vector<ElementwiseOp> chain = {ElementwiseOp::Sigmoid, ElementwiseOp::Relu};

        // Op codes for fused_elementwise_chain
        const uint32_t OP_SIGMOID = 5;
        const uint32_t OP_RELU = 4;

        for (int size : sizes) {
            int N = size * size;

            // Vulkan fused chain
            auto vk_backend = BackendManager::get(vulkan_device_);
            Tensor x_vk = Tensor({N}, vulkan_device_, DataType::Float32);
            x_vk.fill_(0.5f);
            Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);

            // Warmup fused
            for (int i = 0; i < 3; ++i) {
                vk_backend->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{x_vk.impl_->storage.get(), x_vk.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_SIGMOID, OP_RELU}, N);
                vk_backend->synchronize();
            }

            std::vector<double> fused_times;
            for (int i = 0; i < 10; ++i) {
                double t = measure_time_ms([&]() {
                    vk_backend->elementwise_capability()->fused_elementwise_chain(
                        std::vector<munet::Storage*>{x_vk.impl_->storage.get(), x_vk.impl_->storage.get()},
                        *fused_out.impl_->storage,
                        std::vector<uint32_t>{OP_SIGMOID, OP_RELU}, N);
                    vk_backend->synchronize();
                });
                fused_times.push_back(t);
            }
            double vulkan_fused_time = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();

            // CUDA sequential chain
            double cuda_seq_time = benchmark_elementwise_chain(cuda_device, size, size, chain, 3, 10);

            // Bandwidth: 2 ops, each reads 1 input + writes 1 output = 4 bytes/elem total
            double bytes = 4.0 * N * sizeof(float);
            double vulkan_bw = (bytes / (vulkan_fused_time * 1e-3)) / 1e9;
            double cuda_bw = (bytes / (cuda_seq_time * 1e-3)) / 1e9;
            double ratio = vulkan_fused_time / cuda_seq_time;

            std::cout << std::left << std::setw(15)
                      << (std::to_string(size) + "x" + std::to_string(size))
                      << std::fixed << std::setprecision(3) << std::setw(18)
                      << vulkan_fused_time << std::setw(18) << cuda_seq_time
                      << std::setprecision(2) << std::setw(15)
                      << vulkan_bw << std::setw(15) << cuda_bw
                      << std::setprecision(2) << std::setw(12) << ratio << "x\n";
        }
    }

    // Chain 2: add + mul (2 binary ops)
    {
        std::cout << "\n--- 2-op chain: add + mul (binary) ---\n";
        std::cout << std::left << std::setw(15) << "Size"
                  << std::setw(18) << "Vulkan fused (ms)" << std::setw(18) << "CUDA seq (ms)"
                  << std::setw(15) << "Vulkan GB/s" << std::setw(15) << "CUDA GB/s"
                  << std::setw(12) << "Ratio (V/C)"
                  << "\n";
        std::cout << std::string(93, '-') << "\n";

        std::vector<ElementwiseOp> chain = {ElementwiseOp::Add, ElementwiseOp::Mul};

        const uint32_t OP_ADD = 0;
        const uint32_t OP_MUL = 1;

        for (int size : sizes) {
            int N = size * size;

            // Vulkan fused chain
            auto vk_backend = BackendManager::get(vulkan_device_);
            Tensor a_vk = Tensor({N}, vulkan_device_, DataType::Float32);
            Tensor b_vk = Tensor({N}, vulkan_device_, DataType::Float32);
            a_vk.fill_(1.0f);
            b_vk.fill_(2.0f);
            Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);

            // Warmup fused
            for (int i = 0; i < 3; ++i) {
                vk_backend->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{a_vk.impl_->storage.get(), b_vk.impl_->storage.get(), a_vk.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_ADD, OP_MUL}, N);
                vk_backend->synchronize();
            }

            std::vector<double> fused_times;
            for (int i = 0; i < 10; ++i) {
                double t = measure_time_ms([&]() {
                    vk_backend->elementwise_capability()->fused_elementwise_chain(
                        std::vector<munet::Storage*>{a_vk.impl_->storage.get(), b_vk.impl_->storage.get(), a_vk.impl_->storage.get()},
                        *fused_out.impl_->storage,
                        std::vector<uint32_t>{OP_ADD, OP_MUL}, N);
                    vk_backend->synchronize();
                });
                fused_times.push_back(t);
            }
            double vulkan_fused_time = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();

            // CUDA sequential chain
            double cuda_seq_time = benchmark_elementwise_chain(cuda_device, size, size, chain, 3, 10);

            // Bandwidth: add reads 2 + writes 1, mul reads 2 + writes 1 = 6 bytes/elem
            double bytes = 6.0 * N * sizeof(float);
            double vulkan_bw = (bytes / (vulkan_fused_time * 1e-3)) / 1e9;
            double cuda_bw = (bytes / (cuda_seq_time * 1e-3)) / 1e9;
            double ratio = vulkan_fused_time / cuda_seq_time;

            std::cout << std::left << std::setw(15)
                      << (std::to_string(size) + "x" + std::to_string(size))
                      << std::fixed << std::setprecision(3) << std::setw(18)
                      << vulkan_fused_time << std::setw(18) << cuda_seq_time
                      << std::setprecision(2) << std::setw(15)
                      << vulkan_bw << std::setw(15) << cuda_bw
                      << std::setprecision(2) << std::setw(12) << ratio << "x\n";
        }
    }

    // Chain 3: 4-op chain: relu + sigmoid + relu + sigmoid
    {
        std::cout << "\n--- 4-op chain: relu + sigmoid + relu + sigmoid ---\n";
        std::cout << std::left << std::setw(15) << "Size"
                  << std::setw(18) << "Vulkan fused (ms)" << std::setw(18) << "CUDA seq (ms)"
                  << std::setw(15) << "Vulkan GB/s" << std::setw(15) << "CUDA GB/s"
                  << std::setw(12) << "Ratio (V/C)"
                  << "\n";
        std::cout << std::string(93, '-') << "\n";

        std::vector<ElementwiseOp> chain = {ElementwiseOp::Relu, ElementwiseOp::Sigmoid,
                                            ElementwiseOp::Relu, ElementwiseOp::Sigmoid};

        const uint32_t OP_RELU = 4;
        const uint32_t OP_SIGMOID = 5;

        for (int size : {64, 128, 256, 512, 1024, 2048}) {
            int N = size * size;

            // Vulkan fused chain
            auto vk_backend = BackendManager::get(vulkan_device_);
            Tensor x_vk = Tensor({N}, vulkan_device_, DataType::Float32);
            x_vk.fill_(0.5f);
            Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);

            // Warmup fused
            for (int i = 0; i < 3; ++i) {
                vk_backend->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{x_vk.impl_->storage.get(), x_vk.impl_->storage.get(),
                                                  x_vk.impl_->storage.get(), x_vk.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_RELU, OP_SIGMOID, OP_RELU, OP_SIGMOID}, N);
                vk_backend->synchronize();
            }

            std::vector<double> fused_times;
            for (int i = 0; i < 10; ++i) {
                double t = measure_time_ms([&]() {
                    vk_backend->elementwise_capability()->fused_elementwise_chain(
                        std::vector<munet::Storage*>{x_vk.impl_->storage.get(), x_vk.impl_->storage.get(),
                                                      x_vk.impl_->storage.get(), x_vk.impl_->storage.get()},
                        *fused_out.impl_->storage,
                        std::vector<uint32_t>{OP_RELU, OP_SIGMOID, OP_RELU, OP_SIGMOID}, N);
                    vk_backend->synchronize();
                });
                fused_times.push_back(t);
            }
            double vulkan_fused_time = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();

            // CUDA sequential chain
            double cuda_seq_time = benchmark_elementwise_chain(cuda_device, size, size, chain, 3, 10);

            // Bandwidth: 4 ops, each reads 1 + writes 1 = 8 bytes/elem total
            double bytes = 8.0 * N * sizeof(float);
            double vulkan_bw = (bytes / (vulkan_fused_time * 1e-3)) / 1e9;
            double cuda_bw = (bytes / (cuda_seq_time * 1e-3)) / 1e9;
            double ratio = vulkan_fused_time / cuda_seq_time;

            std::cout << std::left << std::setw(15)
                      << (std::to_string(size) + "x" + std::to_string(size))
                      << std::fixed << std::setprecision(3) << std::setw(18)
                      << vulkan_fused_time << std::setw(18) << cuda_seq_time
                      << std::setprecision(2) << std::setw(15)
                      << vulkan_bw << std::setw(15) << cuda_bw
                      << std::setprecision(2) << std::setw(12) << ratio << "x\n";
        }
    }
}

TEST_F(VulkanProfilingTest, GenerateBaselineReport) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    TimingResults results;

    // Collect comprehensive timing data
    std::cout << "\n=== Generating Baseline Performance Report ===\n";

    // Matmul benchmarks
    for (int size : {128, 256, 512}) {
        double t = benchmark_matmul(vulkan_device_, size, size, size, 3, 10);
        results.add("matmul_" + std::to_string(size), t);
    }

    // Elementwise benchmarks
    for (int size : {128, 256, 512}) {
        double t =
            benchmark_elementwise(vulkan_device_, size, size, 3, 10);
        results.add("elementwise_" + std::to_string(size), t);
    }

    // Transfer benchmarks
    for (int size_kb : {1, 16, 256, 1024}) {
        int size_bytes = size_kb * 1024;
        double h2d = benchmark_h2d_transfer(vulkan_device_, size_bytes, 10);
        double d2h = benchmark_d2h_transfer(vulkan_device_, size_bytes, 10);
        results.add("h2d_" + std::to_string(size_kb) + "kb", h2d);
        results.add("d2h_" + std::to_string(size_kb) + "kb", d2h);
    }

    // Print summary
    results.print_summary();

    // Note: CSV output can be enabled for CI baseline tracking
    // results.write_csv("vulkan_baseline.csv");
}

TEST_F(VulkanProfilingTest, CompareWithCPU) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    Device cpu_device{DeviceType::CPU, 0};

    std::cout << "\n=== Vulkan vs CPU Performance Comparison ===\n";

    // Matmul comparison - Vulkan vs CPU
    std::vector<std::tuple<int, int, int>> sizes = {
        {64, 64, 64}, {128, 128, 128}, {256, 256, 256}};

    std::cout << "\nMatmul Comparison (Vulkan vs CPU):\n";
    std::cout << std::left << std::setw(15) << "Size" << std::setw(15)
              << "Vulkan (ms)" << std::setw(15) << "CPU (ms)" << std::setw(15)
              << "Speedup (V/C)"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto [M, K, N] : sizes) {
        double vulkan_time =
            benchmark_matmul(vulkan_device_, M, K, N, 3, 10);
        double cpu_time =
            benchmark_matmul(cpu_device, M, K, N, 3, 10);

        double speedup = cpu_time / vulkan_time;

        std::cout << std::left << std::setw(15)
                  << (std::to_string(M) + "x" + std::to_string(K) + "x" +
                      std::to_string(N))
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << vulkan_time << std::setw(15) << cpu_time << std::setw(15)
                  << std::setprecision(2) << speedup << "x\n";
    }

    // Elementwise comparison - Vulkan vs CPU
    std::cout << "\nElementwise Comparison (Vulkan vs CPU):\n";
    std::cout << std::left << std::setw(15) << "Size" << std::setw(15)
              << "Vulkan (ms)" << std::setw(15) << "CPU (ms)" << std::setw(15)
              << "Speedup (V/C)"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int size : {64, 128, 256, 512}) {
        double vulkan_time =
            benchmark_elementwise(vulkan_device_, size, size, 3, 10);
        double cpu_time =
            benchmark_elementwise(cpu_device, size, size, 3, 10);

        double speedup = cpu_time / vulkan_time;

        std::cout << std::left << std::setw(15)
                  << (std::to_string(size) + "x" + std::to_string(size))
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << vulkan_time << std::setw(15) << cpu_time << std::setw(15)
                  << std::setprecision(2) << speedup << "x\n";
    }
}

// ============================================================================
// Kernel Fusion Benchmark
// ============================================================================

TEST_F(VulkanProfilingTest, KernelFusionOverheadReduction) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan device not available";
    }

    // Get the backend through BackendManager (same device context as tensors)
    auto bk = BackendManager::get(vulkan_device_);

    std::cout << "\n=== Kernel Fusion Overhead Reduction ===\n";
    std::cout << "Measuring real fused vs unfused elementwise chain performance\n\n";

    // Op codes: 4=relu, 5=sigmoid
    const uint32_t OP_ADD = 0;
    const uint32_t OP_MUL = 1;
    const uint32_t OP_RELU = 4;
    const uint32_t OP_SIGMOID = 5;

    int N = 1024 * 1024;  // 1M elements

    // -----------------------------------------------------------------------
    // Benchmark 1: Two unary ops (sigmoid + relu) — unfused vs fused
    // -----------------------------------------------------------------------
    {
        Tensor x = Tensor({N}, vulkan_device_, DataType::Float32);
        x.fill_(0.5f);

        // Warmup unfused
        for (int i = 0; i < 3; ++i) {
            Tensor tmp = x.sigmoid();
            Tensor result = tmp.relu();
            result.impl_->backend().synchronize();
        }

        // Measure unfused: two separate dispatches
        std::vector<double> unfused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                Tensor tmp = x.sigmoid();
                Tensor result = tmp.relu();
                result.impl_->backend().synchronize();
            });
            unfused_times.push_back(t);
        }

        // Warmup fused
        Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);
        for (int i = 0; i < 3; ++i) {
            bk->elementwise_capability()->fused_elementwise_chain(
                std::vector<munet::Storage*>{x.impl_->storage.get(), x.impl_->storage.get()},
                *fused_out.impl_->storage,
                std::vector<uint32_t>{OP_SIGMOID, OP_RELU}, N);
            bk->synchronize();
        }

        // Measure fused: single dispatch
        std::vector<double> fused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                bk->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{x.impl_->storage.get(), x.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_SIGMOID, OP_RELU}, N);
                bk->synchronize();
            });
            fused_times.push_back(t);
        }

        double avg_unfused = std::accumulate(unfused_times.begin(), unfused_times.end(), 0.0) / unfused_times.size();
        double avg_fused = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();
        double speedup = avg_unfused / avg_fused;
        double overhead_reduction = 1.0 - (avg_fused / avg_unfused);

        std::cout << "--- 2-op chain: sigmoid + relu ---\n";
        std::cout << std::left << std::setw(25) << "Unfused avg" << std::fixed << std::setprecision(3)
                  << avg_unfused << " ms\n";
        std::cout << std::left << std::setw(25) << "Fused avg" << std::fixed << std::setprecision(3)
                  << avg_fused << " ms\n";
        std::cout << std::left << std::setw(25) << "Speedup" << std::fixed << std::setprecision(2)
                  << speedup << "x\n";
        std::cout << std::left << std::setw(25) << "Overhead reduction" << std::fixed << std::setprecision(1)
                  << (overhead_reduction * 100) << "%\n\n";

        EXPECT_GT(avg_unfused, 0.0) << "Unfused time should be measurable";
        EXPECT_LT(avg_fused, avg_unfused) << "Fused should be faster than unfused";
    }

    // -----------------------------------------------------------------------
    // Benchmark 2: Longer chain (4 ops: relu + sigmoid + relu + sigmoid)
    // -----------------------------------------------------------------------
    {
        Tensor x = Tensor({N}, vulkan_device_, DataType::Float32);
        x.fill_(0.5f);

        // Measure unfused: four separate dispatches
        std::vector<double> unfused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                Tensor a = x.relu();
                Tensor b = a.sigmoid();
                Tensor c = b.relu();
                Tensor d = c.sigmoid();
                d.impl_->backend().synchronize();
            });
            unfused_times.push_back(t);
        }

        // Measure fused: single dispatch with 4 ops
        Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);
        std::vector<double> fused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                bk->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{x.impl_->storage.get(), x.impl_->storage.get(), x.impl_->storage.get(), x.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_RELU, OP_SIGMOID, OP_RELU, OP_SIGMOID}, N);
                bk->synchronize();
            });
            fused_times.push_back(t);
        }

        double avg_unfused = std::accumulate(unfused_times.begin(), unfused_times.end(), 0.0) / unfused_times.size();
        double avg_fused = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();
        double speedup = avg_unfused / avg_fused;
        double overhead_reduction = 1.0 - (avg_fused / avg_unfused);

        std::cout << "--- 4-op chain: relu + sigmoid + relu + sigmoid ---\n";
        std::cout << std::left << std::setw(25) << "Unfused avg" << std::fixed << std::setprecision(3)
                  << avg_unfused << " ms\n";
        std::cout << std::left << std::setw(25) << "Fused avg" << std::fixed << std::setprecision(3)
                  << avg_fused << " ms\n";
        std::cout << std::left << std::setw(25) << "Speedup" << std::fixed << std::setprecision(2)
                  << speedup << "x\n";
        std::cout << std::left << std::setw(25) << "Overhead reduction" << std::fixed << std::setprecision(1)
                  << (overhead_reduction * 100) << "%\n\n";

        EXPECT_GT(avg_unfused, 0.0) << "Unfused time should be measurable";
        EXPECT_LT(avg_fused, avg_unfused) << "Fused should be faster than unfused for 4-op chain";
    }

    // -----------------------------------------------------------------------
    // Benchmark 3: Binary op chain (add + mul)
    // -----------------------------------------------------------------------
    {
        Tensor a = Tensor({N}, vulkan_device_, DataType::Float32);
        Tensor b = Tensor({N}, vulkan_device_, DataType::Float32);
        a.fill_(1.0f);
        b.fill_(2.0f);

        // Measure unfused: two separate dispatches
        std::vector<double> unfused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                Tensor sum = a + b;       // add
                Tensor result = sum * a;  // mul
                result.impl_->backend().synchronize();
            });
            unfused_times.push_back(t);
        }

        // Measure fused: single dispatch with add + mul
        Tensor fused_out = Tensor({N}, vulkan_device_, DataType::Float32);
        std::vector<double> fused_times;
        for (int i = 0; i < 10; ++i) {
            double t = measure_time_ms([&]() {
                bk->elementwise_capability()->fused_elementwise_chain(
                    std::vector<munet::Storage*>{a.impl_->storage.get(), b.impl_->storage.get(), a.impl_->storage.get()},
                    *fused_out.impl_->storage,
                    std::vector<uint32_t>{OP_ADD, OP_MUL}, N);
                bk->synchronize();
            });
            fused_times.push_back(t);
        }

        double avg_unfused = std::accumulate(unfused_times.begin(), unfused_times.end(), 0.0) / unfused_times.size();
        double avg_fused = std::accumulate(fused_times.begin(), fused_times.end(), 0.0) / fused_times.size();
        double speedup = avg_unfused / avg_fused;
        double overhead_reduction = 1.0 - (avg_fused / avg_unfused);

        std::cout << "--- 2-op chain: add + mul (binary) ---\n";
        std::cout << std::left << std::setw(25) << "Unfused avg" << std::fixed << std::setprecision(3)
                  << avg_unfused << " ms\n";
        std::cout << std::left << std::setw(25) << "Fused avg" << std::fixed << std::setprecision(3)
                  << avg_fused << " ms\n";
        std::cout << std::left << std::setw(25) << "Speedup" << std::fixed << std::setprecision(2)
                  << speedup << "x\n";
        std::cout << std::left << std::setw(25) << "Overhead reduction" << std::fixed << std::setprecision(1)
                  << (overhead_reduction * 100) << "%\n\n";

        EXPECT_GT(avg_unfused, 0.0) << "Unfused time should be measurable";
    }
}

} // namespace
} // namespace test
} // namespace munet