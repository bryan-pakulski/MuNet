#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace munet;

class AutogradTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, AutogradTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

// Helper to compute numerical gradient of a function f(x)
float numerical_gradient(std::function<Tensor(Tensor)> f, Tensor x,
                         float eps = 1e-3f) {
  Tensor x_cpu = x.to({DeviceType::CPU, 0});
  float original_val = ((float *)x_cpu.data())[0];

  ((float *)x_cpu.data())[0] = original_val + eps;
  Tensor out_pos = f(x_cpu.to(x.device())).to({DeviceType::CPU, 0});

  ((float *)x_cpu.data())[0] = original_val - eps;
  Tensor out_neg = f(x_cpu.to(x.device())).to({DeviceType::CPU, 0});

  float y_pos = ((float *)out_pos.data())[0];
  float y_neg = ((float *)out_neg.data())[0];

  return (y_pos - y_neg) / (2.0f * eps);
}

TEST_P(AutogradTest, AddBackward) {
  Tensor a({1}, dev(), DataType::Float32, true);
  Tensor b({1}, dev(), DataType::Float32, true);

  Tensor val({1}, {DeviceType::CPU, 0});
  ((float *)val.data())[0] = 2.0f;
  a.impl_->backend().copy(val.data(), a.data(), a.bytes(), val.device(), dev());

  ((float *)val.data())[0] = 3.0f;
  b.impl_->backend().copy(val.data(), b.data(), b.bytes(), val.device(), dev());

  Tensor c = a + b;
  c.backward();

  ASSERT_TRUE(a.has_grad());
  ASSERT_TRUE(b.has_grad());

  Tensor ga = a.grad().to({DeviceType::CPU, 0});
  Tensor gb = b.grad().to({DeviceType::CPU, 0});

  EXPECT_FLOAT_EQ(((float *)ga.data())[0], 1.0f);
  EXPECT_FLOAT_EQ(((float *)gb.data())[0], 1.0f);
}

TEST(AutogradTest, GradCheckLinear) {
  Device dev{DeviceType::CPU, 0};
  Tensor x({1, 1}, dev, DataType::Float32, true);
  ((float *)x.data())[0] = 1.5f;

  auto model = [](Tensor input) {
    return input.sigmoid(); // Test a non-linear op
  };

  // Analytical
  Tensor out = model(x);
  out.backward();
  float analytical = ((float *)x.grad().to({DeviceType::CPU, 0}).data())[0];

  // Numerical
  float numerical = numerical_gradient(model, x);

  EXPECT_NEAR(analytical, numerical, 1e-3);
}
