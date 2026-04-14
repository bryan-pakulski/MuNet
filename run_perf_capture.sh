#!/bin/bash
cd /home/bryanp/dev/projects/ai/MuNet
MUNET_RUN_PERF_TESTS=1 ./build/release/munet_tests --gtest_filter=PerformanceTest.* 2>/dev/null | grep -E "\[PERF\]|\[  |PASSED|FAILED" > /tmp/perf_results.txt 2>&1
cat /tmp/perf_results.txt