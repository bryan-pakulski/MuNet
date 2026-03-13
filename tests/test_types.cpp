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
