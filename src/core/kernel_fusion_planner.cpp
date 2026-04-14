#include "core/kernel_fusion_planner.hpp"

#include <unordered_set>

namespace munet {
namespace fusion {
namespace {

const std::unordered_set<std::string> &fusible_ops() {
  static const std::unordered_set<std::string> kOps = {
      "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Exp", "Log",
      "Sqrt", "Rsqrt", "Sin", "Cos", "MaskedFill"};
  return kOps;
}

} // namespace

bool is_elementwise_fusible_op(const std::string &op_name) {
  return fusible_ops().count(op_name) != 0;
}

std::vector<FusionGroup>
plan_elementwise_fusion_groups(const std::vector<ForwardNode> &nodes,
                               size_t max_group_size) {
  std::vector<FusionGroup> groups;
  if (nodes.empty()) {
    return groups;
  }

  if (max_group_size == 0) {
    max_group_size = 1;
  }

  size_t i = 0;
  while (i < nodes.size()) {
    if (!is_elementwise_fusible_op(nodes[i].op_name)) {
      groups.push_back(FusionGroup{i, i, false, {nodes[i].op_name}});
      ++i;
      continue;
    }

    const size_t begin = i;
    size_t end = i;
    std::vector<std::string> ops;
    ops.push_back(nodes[i].op_name);
    ++i;

    while (i < nodes.size() && is_elementwise_fusible_op(nodes[i].op_name) &&
           (i - begin) < max_group_size) {
      end = i;
      ops.push_back(nodes[i].op_name);
      ++i;
    }

    if (begin == end) {
      groups.push_back(FusionGroup{begin, end, false, std::move(ops)});
    } else {
      groups.push_back(FusionGroup{begin, end, true, std::move(ops)});
    }
  }

  return groups;
}

} // namespace fusion
} // namespace munet
