#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace munet;


TEST(TensorDTypeTest, DTypeUtilitiesProvidePromotionAndAccumulationRules) {
  EXPECT_TRUE(is_floating(DataType::Float32));
  EXPECT_TRUE(is_low_precision(DataType::Float16));
  EXPECT_TRUE(is_integral(DataType::Int32));
  EXPECT_EQ(dtype_name(DataType::Float16), "float16");
  EXPECT_EQ(promote_types(DataType::Int32, DataType::Float16),
            DataType::Float16);
  EXPECT_EQ(accumulation_type(AccumulationOp::Matmul, DataType::Float16),
            DataType::Float32);
}

TEST(TensorDTypeTest, TensorOptionsAndDTypeConversionRoundTrip) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;
  options.requires_grad = true;

  Tensor configured({2}, options);
  EXPECT_EQ(configured.dtype(), DataType::Float16);
  EXPECT_TRUE(configured.requires_grad());

  Tensor base({2}, Device{DeviceType::CPU, 0}, DataType::Float32, true);
  float *base_ptr = static_cast<float *>(base.data());
  base_ptr[0] = 1.5f;
  base_ptr[1] = -2.25f;

  Tensor half = base.to(DataType::Float16);
  EXPECT_EQ(half.dtype(), DataType::Float16);
  EXPECT_EQ(half.options().dtype, DataType::Float16);

  Tensor roundtrip = half.to(DataType::Float32);
  const float *roundtrip_ptr = static_cast<const float *>(roundtrip.data());
  EXPECT_NEAR(roundtrip_ptr[0], 1.5f, 1e-3f);
  EXPECT_NEAR(roundtrip_ptr[1], -2.25f, 1e-3f);
}

TEST(TensorDTypeTest, ItemValueSupportsInt32AndFloat16) {
  Tensor ints({1}, Device{DeviceType::CPU, 0}, DataType::Int32);
  static_cast<int32_t *>(ints.data())[0] = 7;
  ScalarValue int_value = ints.item_value();
  EXPECT_EQ(int_value.dtype, DataType::Int32);
  EXPECT_EQ(int_value.as_int32(), 7);

  Tensor base({1}, Device{DeviceType::CPU, 0}, DataType::Float32);
  static_cast<float *>(base.data())[0] = 1.25f;
  Tensor half = base.to(DataType::Float16);
  ScalarValue half_value = half.item_value();
  EXPECT_EQ(half_value.dtype, DataType::Float16);
  EXPECT_NEAR(half_value.as_float(), 1.25f, 1e-3f);
}

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

TEST_P(TensorTest, DeviceMovePreservesDTypeAndRequiresGrad) {
  Device cpu{DeviceType::CPU, 0};
  Tensor base({1}, cpu, DataType::Float32, true);
  static_cast<float *>(base.data())[0] = 2.5f;

  Tensor half = base.to(DataType::Float16);
  Tensor moved = half.to(dev());

  EXPECT_EQ(moved.dtype(), DataType::Float16);
  EXPECT_TRUE(moved.requires_grad());
  EXPECT_EQ(moved.device(), dev());

  ScalarValue roundtrip = moved.to(cpu).item_value();
  EXPECT_EQ(roundtrip.dtype, DataType::Float16);
  EXPECT_NEAR(roundtrip.as_float(), 2.5f, 1e-3f);
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
  ((float *)val.data())[0] = 3.14f;

  // Copy to device
  t.impl_->backend().copy(val.data(), t.data(), t.bytes(), val.device(), dev());

  EXPECT_FLOAT_EQ(t.item(), 3.14f);
}

TEST_P(TensorTest, ItemMethodFailure) {
  Tensor t({2}, dev());
  // item() should throw if size != 1
  EXPECT_THROW(t.item(), std::runtime_error);
}


TEST_P(TensorTest, MaskedFill) {
  Tensor a({2, 2}, dev());
  Tensor mask({2, 2}, dev());

  Tensor a_cpu({2, 2}, {DeviceType::CPU, 0});
  Tensor m_cpu({2, 2}, {DeviceType::CPU, 0});
  float *ad = (float *)a_cpu.data();
  float *md = (float *)m_cpu.data();
  ad[0] = 1.0f; ad[1] = 2.0f; ad[2] = 3.0f; ad[3] = 4.0f;
  md[0] = 0.0f; md[1] = 1.0f; md[2] = 0.0f; md[3] = 1.0f;

  a.impl_->backend().copy(a_cpu.data(), a.data(), a.bytes(), a_cpu.device(), dev());
  mask.impl_->backend().copy(m_cpu.data(), mask.data(), mask.bytes(), m_cpu.device(), dev());

  Tensor out = a.masked_fill(mask, -5.0f).to({DeviceType::CPU, 0});
  const float *o = (const float *)out.data();
  EXPECT_FLOAT_EQ(o[0], 1.0f);
  EXPECT_FLOAT_EQ(o[1], -5.0f);
  EXPECT_FLOAT_EQ(o[2], 3.0f);
  EXPECT_FLOAT_EQ(o[3], -5.0f);
}

TEST_P(TensorTest, SoftmaxDimValidation) {
  Tensor x({2, 3}, dev());
  EXPECT_NO_THROW(x.softmax(-1));
  EXPECT_THROW(x.softmax(0), std::runtime_error);
}


TEST_P(TensorTest, PermuteViewMetadata) {
  Tensor x({2, 3, 4}, dev());
  Tensor y = x.permute({1, 0, 2});
  EXPECT_EQ(y.shape()[0], 3);
  EXPECT_EQ(y.shape()[1], 2);
  EXPECT_EQ(y.shape()[2], 4);
}

TEST_P(TensorTest, LogSoftmaxDim) {
  Tensor x({1, 3}, dev());
  Tensor x_cpu({1, 3}, {DeviceType::CPU, 0});
  float *d = (float *)x_cpu.data();
  d[0] = 0.0f; d[1] = 1.0f; d[2] = 2.0f;
  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(), dev());

  Tensor ls = x.log_softmax(-1).to({DeviceType::CPU, 0});
  const float *o = (const float *)ls.data();
  float p0 = std::exp(o[0]);
  float p1 = std::exp(o[1]);
  float p2 = std::exp(o[2]);
  EXPECT_NEAR(p0 + p1 + p2, 1.0f, 1e-4f);
}
