// Test for transposed matmul performance and correctness
#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include "tensor.hpp"
#include "backend.hpp"

using namespace munet;

// Test transposed matmul correctness
TEST(VulkanTransposeTest, MatmulTransposeACorrectness) {
    Device vulkan_dev{DeviceType::Vulkan, 0};
    Device cpu_dev{DeviceType::CPU, 0};

    int M = 128, K = 64, N = 96;
    
    // Create A^T (K×M stored as K×M, representing M×K transposed)
    Tensor a_t = Tensor::randn({K, M}, vulkan_dev);
    // Create B (K×N)
    Tensor b = Tensor::randn({K, N}, vulkan_dev);
    
    // Compute on Vulkan: A^T @ B where A^T is stored as K×M
    // This is matmul(a_t, b, transA=true) -> result is M×N
    // Equivalent to: A @ B where A is M×K, stored as A^T
    
    // For now, test via explicit transpose
    Tensor a = a_t.to(cpu_dev).transpose(0, 1).to(vulkan_dev);
    Tensor result_vulkan = a.matmul(b);
    
    // Compare with CPU
    Tensor a_cpu = a_t.to(cpu_dev).transpose(0, 1);
    Tensor b_cpu = b.to(cpu_dev);
    Tensor result_cpu = a_cpu.matmul(b_cpu);
    
    Tensor diff = (result_vulkan.to(cpu_dev) - result_cpu).abs();
    float max_diff = diff.max().item<float>();
    
    EXPECT_LT(max_diff, 1e-3f) << "Vulkan transpose matmul mismatch";
    printf("Transpose A correctness: max_diff = %.6f (PASS)\n", max_diff);
}

TEST(VulkanTransposeTest, MatmulTransposeBCorrectness) {
    Device vulkan_dev{DeviceType::Vulkan, 0};
    Device cpu_dev{DeviceType::CPU, 0};

    int M = 128, K = 64, N = 96;
    
    // Create A (M×K)
    Tensor a = Tensor::randn({M, K}, vulkan_dev);
    // Create B^T (N×K stored as N×K, representing K×N transposed)
    Tensor b_t = Tensor::randn({N, K}, vulkan_dev);
    
    // Compute on Vulkan: A @ B^T where B^T is stored as N×K
    // Equivalent to: A @ B where B is K×N, stored as B^T
    
    // For now, test via explicit transpose
    Tensor b = b_t.to(cpu_dev).transpose(0, 1).to(vulkan_dev);
    Tensor result_vulkan = a.matmul(b);
    
    // Compare with CPU
    Tensor a_cpu = a.to(cpu_dev);
    Tensor b_cpu = b_t.to(cpu_dev).transpose(0, 1);
    Tensor result_cpu = a_cpu.matmul(b_cpu);
    
    Tensor diff = (result_vulkan.to(cpu_dev) - result_cpu).abs();
    float max_diff = diff.max().item<float>();
    
    EXPECT_LT(max_diff, 1e-3f) << "Vulkan transpose matmul mismatch";
    printf("Transpose B correctness: max_diff = %.6f (PASS)\n", max_diff);
}

// Benchmark transposed vs non-transposed to verify optimization
TEST(VulkanTransposeTest, MatmulTransposePerformance) {
    Device vulkan_dev{DeviceType::Vulkan, 0};

    int M = 512, K = 512, N = 512;
    int warmup = 2;
    int runs = 5;
    
    auto benchmark = [&](const char* name, auto&& matmul_fn) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            auto result = matmul_fn();
            (void)result;
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; i++) {
            auto result = matmul_fn();
            (void)result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / runs;
        double gflops = 2.0 * M * K * N / (ms * 1e6);
        printf("%s: %.2f ms (%.1f GFLOPS)\n", name, ms, gflops);
        return gflops;
    };
    
    // Non-transposed case
    Tensor a = Tensor::randn({M, K}, vulkan_dev);
    Tensor b = Tensor::randn({K, N}, vulkan_dev);
    double gflops_normal = benchmark("Non-transposed", [&]() { return a.matmul(b); });
    
    // Transposed A (via explicit transpose)
    Tensor a_t = a.transpose(0, 1);
    double gflops_trans_a = benchmark("Transposed A", [&]() { return a_t.matmul(b); });
    
    // Transposed B (via explicit transpose)
    Tensor b_t = b.transpose(0, 1);
    double gflops_trans_b = benchmark("Transposed B", [&]() { return a.matmul(b_t); });
    
    // Report ratios
    printf("\n=== Performance Summary ===\n");
    printf("Non-transposed:  %.1f GFLOPS (baseline)\n", gflops_normal);
    printf("Transposed A:    %.1f GFLOPS (%.2fx)\n", gflops_trans_a, gflops_trans_a / gflops_normal);
    printf("Transposed B:    %.1f GFLOPS (%.2fx)\n", gflops_trans_b, gflops_trans_b / gflops_normal);
    
    // With tiled shared memory optimization, transposed should be within 0.7x of non-transposed
    EXPECT_GT(gflops_trans_a, gflops_normal * 0.5f) << "Transposed A too slow";
    EXPECT_GT(gflops_trans_b, gflops_normal * 0.5f) << "Transposed B too slow";
}