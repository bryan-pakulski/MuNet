#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace munet;

class SpatialTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, SpatialTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(SpatialTest, TrainSpatialModel) {
  Device device = dev();
  Tensor x({2, 1, 6, 6}, device);
  x.uniform_(0.0f, 1.0f);
  Tensor y({2, 1, 6, 6}, device);
  y.uniform_(0.0f, 1.0f);

  Tensor w1({4, 1, 3, 3}, device, DataType::Float32, true);
  Tensor b1({4}, device, DataType::Float32, true);
  w1.uniform_(-0.1f, 0.1f);
  b1.uniform_(-0.01f, 0.01f);

  Tensor w2({1, 4, 3, 3}, device, DataType::Float32, true);
  Tensor b2({1}, device, DataType::Float32, true);
  w2.uniform_(-0.1f, 0.1f);
  b2.uniform_(-0.01f, 0.01f);

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  for (int i = 0; i < 5; ++i) {
    w1.zero_grad();
    b1.zero_grad();
    w2.zero_grad();
    b2.zero_grad();

    Tensor h1 = x.conv2d(w1, b1, 1, 1);
    Tensor a1 = h1.relu();
    Tensor p1 = a1.max_pool2d(2, 2, 0);
    Tensor u1 = p1.upsample2d(2);
    Tensor out = u1.conv2d(w2, b2, 1, 1);

    Tensor loss = out.mse_loss(y);
    Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
    float current_loss = static_cast<float *>(loss_cpu.data())[0];

    if (i == 0)
      initial_loss = current_loss;
    final_loss = current_loss;

    loss.backward();
    w1.step(0.1f);
    b1.step(0.1f);
    w2.step(0.1f);
    b2.step(0.1f);
  }

  EXPECT_FALSE(std::isnan(final_loss));
  EXPECT_LT(final_loss, initial_loss);
}

TEST_P(SpatialTest, Conv2DShapes) {
  Tensor in({1, 1, 4, 4}, dev());
  Tensor w({1, 1, 3, 3}, dev());

  Tensor out = in.conv2d(w, Tensor(), 1, 0);
  EXPECT_EQ(out.shape()[2], 2);
  EXPECT_EQ(out.shape()[3], 2);

  Tensor out2 = in.conv2d(w, Tensor(), 1, 1);
  EXPECT_EQ(out2.shape()[2], 4);
  EXPECT_EQ(out2.shape()[3], 4);
}
