#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace munet;

class TensorTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, TensorTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(TensorTest, CreationAndMetadata) {
  Tensor t({2, 3}, dev());
  EXPECT_EQ(t.size(), 6);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.device().type, dev().type);
}

TEST_P(TensorTest, Addition) {
  Tensor a({2}, dev());
  Tensor b({2}, dev());

  Tensor val({2}, {DeviceType::CPU, 0});
  float *data = (float *)val.data();
  data[0] = 1.0f;
  data[1] = 2.0f;
  a.impl_->backend().copy(val.data(), a.data(), a.bytes(), val.device(), dev());

  data[0] = 3.0f;
  data[1] = 4.0f;
  b.impl_->backend().copy(val.data(), b.data(), b.bytes(), val.device(), dev());

  Tensor c = a + b;
  Tensor res = c.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)res.data())[0], 4.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[1], 6.0f);
}

TEST_P(TensorTest, ItemMethod) {
  Tensor t({1}, dev());
  Tensor val({1}, {DeviceType::CPU, 0});
  ((float*)val.data())[0] = 3.14f;
  
  // Copy to device
  t.impl_->backend().copy(val.data(), t.data(), t.bytes(), val.device(), dev());
  
  EXPECT_FLOAT_EQ(t.item(), 3.14f);
}

TEST_P(TensorTest, ItemMethodFailure) {
  Tensor t({2}, dev());
  // item() should throw if size != 1
  EXPECT_THROW(t.item(), std::runtime_error);
}
