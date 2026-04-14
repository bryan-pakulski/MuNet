#!/bin/bash
cd /home/bryanp/dev/projects/ai/MuNet
export MUNET_RUN_PERF_TESTS=1

# Run remaining tests that were truncated
for test in LogSoftmaxCudaVsVulkan MatmulSmallCudaVsVulkan MatmulLargeCudaVsVulkan \
  SoftmaxLargeClassCountCudaVsVulkan TinyAddDispatchOverheadCudaVsVulkan \
  ForwardGraphBuildChainCudaVsVulkan BackwardStepOverheadCudaVsVulkan \
  CopyOnlyCpuToGpuCudaVsVulkan CopyOnlyGpuToCpuCudaVsVulkan \
  Conv2DForwardCudaVsVulkan MaxPool2DCudaVsVulkan Upsample2DCudaVsVulkan \
  ConcatCudaVsVulkan OptimizerStepCudaVsVulkan; do
  ./build/release/munet_tests --gtest_filter="PerformanceTest.$test" 2>/dev/null | grep "^\[PERF\]"
done