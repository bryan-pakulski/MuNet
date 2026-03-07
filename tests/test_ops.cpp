#include "tensor.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <algorithm>

using namespace munet;

class OpsTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, OpsTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(OpsTest, ReshapeGradient) {
  Tensor x({2, 2}, dev(), DataType::Float32, true);
  x.uniform_(1.0f, 1.0f);

  Tensor y = x.reshape({4, 1});
  Tensor z = y.sum();
  z.backward();

  ASSERT_TRUE(x.has_grad());
  EXPECT_EQ(x.grad().shape()[0], 2);
  EXPECT_EQ(x.grad().shape()[1], 2);

  Tensor g = x.grad().to({DeviceType::CPU, 0});
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ((*(float*)g.data()), 1.0f);
}

TEST_P(OpsTest, CatDim0) {
  Tensor a({2, 2}, dev());
  Tensor b({1, 2}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  Tensor c = Tensor::cat({a, b}, 0);
  EXPECT_EQ(c.shape()[0], 3);
  EXPECT_EQ(c.shape()[1], 2);

  Tensor res = c.to({DeviceType::CPU, 0});
  float *data = (float *)res.data();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[4], 2.0f);
}

TEST_P(OpsTest, BroadCastSubMul) {
  Tensor a({2, 2}, dev());
  Tensor b({2}, dev()); // To be broadcasted as row

  Tensor val_a({2, 2}, {DeviceType::CPU, 0});
  ((float *)val_a.data())[0] = 10;
  ((float *)val_a.data())[1] = 20;
  ((float *)val_a.data())[2] = 30;
  ((float *)val_a.data())[3] = 40;
  a.impl_->backend().copy(val_a.data(), a.data(), a.bytes(), val_a.device(), dev());

  Tensor val_b({2}, {DeviceType::CPU, 0});
  ((float *)val_b.data())[0] = 1;
  ((float *)val_b.data())[1] = 2;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), val_b.device(), dev());

  Tensor c = a - b; // [10-1, 20-2, 30-1, 40-2]
  Tensor res = c.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)res.data())[0], 9.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[1], 18.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[2], 29.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[3], 38.0f);

  Tensor d = a * b; // [10*1, 20*2, 30*1, 40*2]
  res = d.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)res.data())[0], 10.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[1], 40.0f);
}
