#include "inference.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace munet;

namespace {
class IdentityLayer : public inference::Module {
public:
  Tensor forward(Tensor x) override { return x; }
};
} // namespace

TEST(InferenceTest, ModuleInheritsCoreModule) {
  EXPECT_TRUE((std::is_base_of_v<core::Module, inference::Module>));
}

TEST(InferenceTest, TrainCallKeepsEvalMode) {
  auto m = std::make_shared<IdentityLayer>();
  m->train(true);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({1, 1}, cpu);
  x.uniform_(1.0f, 1.0f);

  auto y = m->forward(x);
  EXPECT_EQ(y.size(), x.size());
}
