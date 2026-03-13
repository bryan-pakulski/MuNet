#include "types.hpp"
#include "inference.hpp"
#include <gtest/gtest.h>

using namespace munet;

TEST(DataTypeTest, DtypeSizesIncludeNewTypes) {
  EXPECT_EQ(dtype_size(DataType::Float8E4M3FN), 1u);
  EXPECT_EQ(dtype_size(DataType::Float8E5M2), 1u);
  EXPECT_EQ(dtype_size(DataType::BFloat16), 2u);
  EXPECT_EQ(dtype_size(DataType::Int8), 1u);
  EXPECT_EQ(dtype_size(DataType::Int4), 1u);
  EXPECT_EQ(dtype_size(DataType::Float64), 8u);
}

TEST(DataTypeTest, ClassificationHelpers) {
  EXPECT_TRUE(is_float_dtype(DataType::Float32));
  EXPECT_TRUE(is_float_dtype(DataType::Float8E4M3FN));
  EXPECT_FALSE(is_float_dtype(DataType::Int8));
  EXPECT_FALSE(is_float_dtype(DataType::Int4));
  EXPECT_FALSE(is_float_dtype(DataType::Int32));

  EXPECT_TRUE(is_fp8(DataType::Float8E4M3FN));
  EXPECT_TRUE(is_fp8(DataType::Float8E5M2));
  EXPECT_FALSE(is_fp8(DataType::Float16));

  EXPECT_TRUE(is_low_precision(DataType::Float16));
  EXPECT_TRUE(is_low_precision(DataType::BFloat16));
  EXPECT_TRUE(is_low_precision(DataType::Float8E5M2));
  EXPECT_FALSE(is_low_precision(DataType::Float32));

  EXPECT_EQ(accumulation_dtype(DataType::Float16), DataType::Float32);
  EXPECT_EQ(accumulation_dtype(DataType::Float8E4M3FN), DataType::Float32);
  EXPECT_EQ(accumulation_dtype(DataType::Float32), DataType::Float32);
}

TEST(DataTypeTest, DtypeNamesAreStable) {
  EXPECT_STREQ(dtype_name(DataType::Float8E4M3FN), "float8_e4m3fn");
  EXPECT_STREQ(dtype_name(DataType::Float8E5M2), "float8_e5m2");
  EXPECT_STREQ(dtype_name(DataType::BFloat16), "bfloat16");
  EXPECT_STREQ(dtype_name(DataType::Int8), "int8");
  EXPECT_STREQ(dtype_name(DataType::Int4), "int4");
}

TEST(InferencePrecisionPolicyTest, EngineExposesPrecisionPolicy) {
  inference::Engine engine;

  inference::PrecisionPolicy policy;
  policy.param_dtype = DataType::Float16;
  policy.activation_dtype = DataType::BFloat16;
  policy.gradient_dtype = DataType::Float32;
  policy.optimizer_state_dtype = DataType::Float32;
  policy.accumulation_dtype = DataType::Float32;
  policy.loss_scale_mode = inference::LossScaleMode::Dynamic;
  policy.fallback_mode = inference::PrecisionFallbackMode::WarnAndUpcast;

  engine.set_precision_policy(policy);

  const auto &configured = engine.precision_policy();
  EXPECT_EQ(configured.param_dtype, DataType::Float16);
  EXPECT_EQ(configured.activation_dtype, DataType::BFloat16);
  EXPECT_EQ(configured.loss_scale_mode, inference::LossScaleMode::Dynamic);
  EXPECT_EQ(configured.fallback_mode,
            inference::PrecisionFallbackMode::WarnAndUpcast);
}


TEST(DataTypeTest, BackendDispatchUsesStorageDtypes) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2}, cpu, DataType::Int8, false);
  Tensor b({2}, cpu, DataType::Int8, false);
  auto *ap = static_cast<int8_t *>(a.data());
  auto *bp = static_cast<int8_t *>(b.data());
  ap[0] = 2; ap[1] = 3;
  bp[0] = 4; bp[1] = 5;

  Tensor c = a + b;
  auto *cp = static_cast<int8_t *>(c.data());
  EXPECT_EQ(cp[0], 6);
  EXPECT_EQ(cp[1], 8);

  Tensor m1({1, 2}, cpu, DataType::Int8, false);
  Tensor m2({2, 1}, cpu, DataType::Int8, false);
  auto *m1p = static_cast<int8_t *>(m1.data());
  auto *m2p = static_cast<int8_t *>(m2.data());
  m1p[0] = 2; m1p[1] = 3;
  m2p[0] = 4; m2p[1] = 5;

  Tensor out = m1.matmul(m2);
  auto *op = static_cast<int8_t *>(out.data());
  EXPECT_EQ(op[0], 23);

  Tensor s = a.sum();
  auto *sp = static_cast<int8_t *>(s.data());
  EXPECT_EQ(sp[0], 5);
}


TEST(DataTypeTest, DTypeDispatchWarnAndUpcastFallbackWorks) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2}, cpu, DataType::Int8, false);
  Tensor b({2}, cpu, DataType::Int8, false);
  auto *ap = static_cast<int8_t *>(a.data());
  auto *bp = static_cast<int8_t *>(b.data());
  ap[0] = 2; ap[1] = 3;
  bp[0] = 4; bp[1] = 5;

  DTypeDispatchConfig cfg;
  cfg.has_compute_dtype = true;
  cfg.compute_dtype = DataType::Int8; // unsupported compute dtype
  cfg.fallback_mode = KernelFallbackMode::WarnAndUpcast;
  DTypeDispatchGuard guard(cfg);

  Tensor c = a + b;
  auto *cp = static_cast<int8_t *>(c.data());
  EXPECT_EQ(cp[0], 6);
  EXPECT_EQ(cp[1], 8);
}

TEST(DataTypeTest, DTypeDispatchErrorModeThrowsOnUnsupportedComputeDType) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2}, cpu, DataType::Int8, false);
  Tensor b({2}, cpu, DataType::Int8, false);

  DTypeDispatchConfig cfg;
  cfg.has_compute_dtype = true;
  cfg.compute_dtype = DataType::Int8; // unsupported compute dtype
  cfg.fallback_mode = KernelFallbackMode::Error;
  DTypeDispatchGuard guard(cfg);

  EXPECT_THROW((void)(a + b), std::runtime_error);
}
