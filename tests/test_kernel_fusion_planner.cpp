#include "core/kernel_fusion_planner.hpp"

#include <gtest/gtest.h>

namespace munet {
namespace {

ForwardNode make_node(const char *name) {
  ForwardNode node;
  node.op_name = name;
  return node;
}

TEST(KernelFusionPlannerTest, GroupsContiguousElementwiseChains) {
  std::vector<ForwardNode> nodes = {make_node("Add"), make_node("Relu"),
                                    make_node("Mul"), make_node("MatMul"),
                                    make_node("Sigmoid")};

  const auto groups = fusion::plan_elementwise_fusion_groups(nodes);
  ASSERT_EQ(groups.size(), 3u);

  EXPECT_TRUE(groups[0].fusible);
  EXPECT_EQ(groups[0].begin, 0u);
  EXPECT_EQ(groups[0].end, 2u);
  EXPECT_EQ(groups[0].ops.size(), 3u);

  EXPECT_FALSE(groups[1].fusible);
  EXPECT_EQ(groups[1].ops.front(), "MatMul");

  EXPECT_FALSE(groups[2].fusible);
  EXPECT_EQ(groups[2].ops.front(), "Sigmoid");
}

TEST(KernelFusionPlannerTest, HonorsGroupSizeLimit) {
  std::vector<ForwardNode> nodes = {
      make_node("Add"), make_node("Relu"), make_node("Mul"),
      make_node("Div"), make_node("Exp"),  make_node("Log")};

  const auto groups = fusion::plan_elementwise_fusion_groups(nodes, 3);
  ASSERT_EQ(groups.size(), 2u);

  EXPECT_TRUE(groups[0].fusible);
  EXPECT_EQ(groups[0].begin, 0u);
  EXPECT_EQ(groups[0].end, 2u);

  EXPECT_TRUE(groups[1].fusible);
  EXPECT_EQ(groups[1].begin, 3u);
  EXPECT_EQ(groups[1].end, 5u);
}

TEST(KernelFusionPlannerTest, NonElementwiseOpsRemainStandalone) {
  std::vector<ForwardNode> nodes = {make_node("MatMul"), make_node("Conv2D"),
                                    make_node("Softmax")};

  const auto groups = fusion::plan_elementwise_fusion_groups(nodes);
  ASSERT_EQ(groups.size(), 3u);
  EXPECT_FALSE(groups[0].fusible);
  EXPECT_FALSE(groups[1].fusible);
  EXPECT_FALSE(groups[2].fusible);
}

} // namespace
} // namespace munet
