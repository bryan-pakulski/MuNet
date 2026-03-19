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

TEST_P(AutogradTest, SimpleAutograd) {
  Tensor w({2, 2}, dev(), DataType::Float32, true);
  Tensor x({2, 2}, dev(), DataType::Float32, false);

  w.uniform_(1.0f, 1.0f);
  x.uniform_(1.0f, 1.0f);

  Tensor z = w * x;
  Tensor loss = z.sum();
  loss.backward();

  ASSERT_TRUE(w.has_grad());
  Tensor grad = w.grad().to({DeviceType::CPU, 0});
  const float *g_ptr = static_cast<const float *>(grad.data());
  // d(sum(w*x))/dw = x. Since x is all 1s, grad is all 1s.
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(g_ptr[i], 1.0f);
}

TEST_P(AutogradTest, GradientAccumulatesAcrossMultipleGraphPaths) {
  Tensor x({1}, dev(), DataType::Float32, true);
  Tensor val({1}, {DeviceType::CPU, 0});
  static_cast<float *>(val.data())[0] = 3.0f;
  x.impl_->backend().copy(val.data(), x.data(), x.bytes(), val.device(), dev());

  Tensor y = x + x;
  Tensor loss = y.sum();
  loss.backward();

  ASSERT_TRUE(x.has_grad());
  Tensor grad = x.grad().to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(static_cast<float *>(grad.data())[0], 2.0f);
}

TEST(AutogradHardeningTest, BackwardRetainGraphControlsRepeatedExecution) {
  Device dev{DeviceType::CPU, 0};
  Tensor x({1}, dev, DataType::Float32, true);
  static_cast<float *>(x.data())[0] = 3.0f;

  Tensor y = x + x;
  Tensor loss = y.sum();

  EXPECT_NO_THROW(loss.backward(true));
  ASSERT_TRUE(x.has_grad());
  EXPECT_FLOAT_EQ(static_cast<float *>(x.grad().data())[0], 2.0f);

  EXPECT_NO_THROW(loss.backward());
  ASSERT_TRUE(x.has_grad());
  EXPECT_FLOAT_EQ(static_cast<float *>(x.grad().data())[0], 4.0f);

  try {
    loss.backward();
    FAIL() << "Expected released graph error";
  } catch (const std::runtime_error &err) {
    EXPECT_NE(std::string(err.what()).find("retain_graph=true"), std::string::npos);
  }
}

TEST(AutogradHardeningTest, InPlaceMutationOnSavedTensorThrowsClearly) {
  Device dev{DeviceType::CPU, 0};
  Tensor x({1}, dev, DataType::Float32, true);
  static_cast<float *>(x.data())[0] = 2.0f;

  Tensor y = x * x;
  x.fill_(3.0f);

  try {
    y.backward();
    FAIL() << "Expected in-place mutation detection";
  } catch (const std::runtime_error &err) {
    EXPECT_NE(std::string(err.what()).find("In-place mutation"), std::string::npos);
  }
}
