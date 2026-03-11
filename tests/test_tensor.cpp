#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
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


TEST_P(TensorTest, TopKBasic) {
  Tensor x({2, 3}, dev());
  Tensor x_cpu({2, 3}, {DeviceType::CPU, 0});
  float *d = (float *)x_cpu.data();
  d[0] = 1.0f; d[1] = 3.0f; d[2] = 2.0f;
  d[3] = 4.0f; d[4] = 0.0f; d[5] = 5.0f;
  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(), dev());

  auto out = x.topk(2, 1, true, true);
  Tensor v = out.first.to({DeviceType::CPU, 0});
  Tensor i = out.second.to({DeviceType::CPU, 0});
  const float *vv = (const float *)v.data();
  const float *ii = (const float *)i.data();

  EXPECT_FLOAT_EQ(vv[0], 3.0f);
  EXPECT_FLOAT_EQ(vv[1], 2.0f);
  EXPECT_FLOAT_EQ(vv[2], 5.0f);
  EXPECT_FLOAT_EQ(vv[3], 4.0f);
  EXPECT_FLOAT_EQ(ii[0], 1.0f);
  EXPECT_FLOAT_EQ(ii[1], 2.0f);
  EXPECT_FLOAT_EQ(ii[2], 2.0f);
  EXPECT_FLOAT_EQ(ii[3], 0.0f);
}

TEST_P(TensorTest, GatherElementsBasic) {
  Tensor x({2, 3}, dev());
  Tensor idx({2, 3}, dev());

  Tensor x_cpu({2, 3}, {DeviceType::CPU, 0});
  Tensor i_cpu({2, 3}, {DeviceType::CPU, 0});
  float *xd = (float *)x_cpu.data();
  float *id = (float *)i_cpu.data();

  xd[0] = 10.0f; xd[1] = 20.0f; xd[2] = 30.0f;
  xd[3] = 40.0f; xd[4] = 50.0f; xd[5] = 60.0f;

  id[0] = 0.0f; id[1] = 2.0f; id[2] = 1.0f;
  id[3] = -1.0f; id[4] = 1.0f; id[5] = 0.0f;

  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(), dev());
  idx.impl_->backend().copy(i_cpu.data(), idx.data(), idx.bytes(), i_cpu.device(), dev());

  Tensor y = x.gather_elements(idx, 1).to({DeviceType::CPU, 0});
  const float *o = (const float *)y.data();
  EXPECT_FLOAT_EQ(o[0], 10.0f);
  EXPECT_FLOAT_EQ(o[1], 30.0f);
  EXPECT_FLOAT_EQ(o[2], 20.0f);
  EXPECT_FLOAT_EQ(o[3], 60.0f);
  EXPECT_FLOAT_EQ(o[4], 50.0f);
  EXPECT_FLOAT_EQ(o[5], 40.0f);
}



TEST_P(TensorTest, GridSampleNearestIdentity) {
  Tensor x({1, 1, 2, 2}, dev());
  Tensor g({1, 2, 2, 2}, dev());

  Tensor x_cpu({1, 1, 2, 2}, {DeviceType::CPU, 0});
  Tensor g_cpu({1, 2, 2, 2}, {DeviceType::CPU, 0});
  float *xd = (float *)x_cpu.data();
  float *gd = (float *)g_cpu.data();
  xd[0] = 1.0f; xd[1] = 2.0f; xd[2] = 3.0f; xd[3] = 4.0f;

  // normalized coords for corners with align_corners=true
  gd[0] = -1.0f; gd[1] = -1.0f;
  gd[2] =  1.0f; gd[3] = -1.0f;
  gd[4] = -1.0f; gd[5] =  1.0f;
  gd[6] =  1.0f; gd[7] =  1.0f;

  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(), dev());
  g.impl_->backend().copy(g_cpu.data(), g.data(), g.bytes(), g_cpu.device(), dev());

  Tensor y = x.grid_sample(g, "nearest", true).to({DeviceType::CPU, 0});
  const float *o = (const float *)y.data();
  EXPECT_FLOAT_EQ(o[0], 1.0f);
  EXPECT_FLOAT_EQ(o[1], 2.0f);
  EXPECT_FLOAT_EQ(o[2], 3.0f);
  EXPECT_FLOAT_EQ(o[3], 4.0f);
}

TEST_P(TensorTest, MatmulLeftBatched) {
  Tensor a({2, 3, 4}, dev());
  Tensor b({4, 2}, dev());

  Tensor a_cpu({2, 3, 4}, {DeviceType::CPU, 0});
  Tensor b_cpu({4, 2}, {DeviceType::CPU, 0});
  float *ad = (float *)a_cpu.data();
  float *bd = (float *)b_cpu.data();

  for (int i = 0; i < 24; ++i)
    ad[i] = (float)(i + 1); // 1..24

  // [ [1,2], [3,4], [5,6], [7,8] ]
  bd[0] = 1.0f;
  bd[1] = 2.0f;
  bd[2] = 3.0f;
  bd[3] = 4.0f;
  bd[4] = 5.0f;
  bd[5] = 6.0f;
  bd[6] = 7.0f;
  bd[7] = 8.0f;

  a.impl_->backend().copy(a_cpu.data(), a.data(), a.bytes(), a_cpu.device(), dev());
  b.impl_->backend().copy(b_cpu.data(), b.data(), b.bytes(), b_cpu.device(), dev());

  Tensor y = a.matmul(b).to({DeviceType::CPU, 0});
  ASSERT_EQ(y.shape().size(), 3);
  EXPECT_EQ(y.shape()[0], 2);
  EXPECT_EQ(y.shape()[1], 3);
  EXPECT_EQ(y.shape()[2], 2);

  const float *o = (const float *)y.data();
  // Reference from numpy-style reshape-batched matmul.
  const float expected[12] = {
      50.0f, 60.0f, 114.0f, 140.0f, 178.0f, 220.0f,
      242.0f, 300.0f, 306.0f, 380.0f, 370.0f, 460.0f};
  for (int i = 0; i < 12; ++i)
    EXPECT_FLOAT_EQ(o[i], expected[i]);
}

TEST_P(TensorTest, MatmulUnsupportedFullBatchedRhs) {
  Tensor a({2, 3, 4}, dev());
  Tensor b({2, 4, 5}, dev());
  EXPECT_THROW(a.matmul(b), std::runtime_error);
}

TEST_P(TensorTest, MatmulRhsSingletonLeadingDims) {
  Tensor a({1, 3, 4}, dev());
  Tensor b({1, 4, 2}, dev());

  Tensor a_cpu({1, 3, 4}, {DeviceType::CPU, 0});
  Tensor b_cpu({1, 4, 2}, {DeviceType::CPU, 0});
  float *ad = (float *)a_cpu.data();
  float *bd = (float *)b_cpu.data();

  for (int i = 0; i < 12; ++i)
    ad[i] = (float)(i + 1);

  bd[0] = 1.0f; bd[1] = 2.0f;
  bd[2] = 3.0f; bd[3] = 4.0f;
  bd[4] = 5.0f; bd[5] = 6.0f;
  bd[6] = 7.0f; bd[7] = 8.0f;

  a.impl_->backend().copy(a_cpu.data(), a.data(), a.bytes(), a_cpu.device(), dev());
  b.impl_->backend().copy(b_cpu.data(), b.data(), b.bytes(), b_cpu.device(), dev());

  Tensor y = a.matmul(b).to({DeviceType::CPU, 0});
  ASSERT_EQ(y.shape().size(), 3);
  EXPECT_EQ(y.shape()[0], 1);
  EXPECT_EQ(y.shape()[1], 3);
  EXPECT_EQ(y.shape()[2], 2);

  const float *o = (const float *)y.data();
  const float expected[6] = {50.0f, 60.0f, 114.0f, 140.0f, 178.0f, 220.0f};
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(o[i], expected[i]);
}
