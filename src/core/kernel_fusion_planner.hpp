#pragma once

#include "tensor.hpp"
#include <string>
#include <vector>

namespace munet {
namespace fusion {

struct FusionGroup {
  size_t begin = 0;
  size_t end = 0; // inclusive
  bool fusible = false;
  std::vector<std::string> ops;
};

// Backend-agnostic first-pass grouping over a forward op list.
//
// This planner intentionally only groups contiguous cheap elementwise ops,
// which is valid across CPU/CUDA/Vulkan and gives a safe starting point for
// backend-specific joint-kernel lowering.
std::vector<FusionGroup>
plan_elementwise_fusion_groups(const std::vector<ForwardNode> &nodes,
                               size_t max_group_size = 8);

bool is_elementwise_fusible_op(const std::string &op_name);

} // namespace fusion
} // namespace munet
