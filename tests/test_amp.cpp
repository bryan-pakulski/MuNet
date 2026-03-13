#include "amp.hpp"
#include "backend/debug_backend.hpp"
#include "nn.hpp"
#include "optim.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace munet;

TEST(AMPTest, AutocastGuardTogglesState) {
  EXPECT_FALSE(amp::AutocastMode::is_enabled());
  {
    amp::AutoCastGuard guard(DataType::BFloat16);
    EXPECT_TRUE(amp::AutocastMode::is_enabled());
    EXPECT_EQ(amp::AutocastMode::dtype(), DataType::BFloat16);
  }
  EXPECT_FALSE(amp::AutocastMode::is_enabled());
}





TEST(AMPTest, AutocastPolicyTableReflectsCurrentCoverage) {
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Add));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Matmul));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::MSELoss));

  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Conv2D));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::MaxPool2D));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Upsample2D));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::BatchNorm));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::LayerNorm));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Tanh));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::GELU));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::LeakyRelu));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Dropout));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::GlobalAvgPool2d));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::Embedding));
  EXPECT_TRUE(amp::should_autocast(amp::AutocastOp::MultiHeadAttention));
}





TEST(AMPTest, AutocastPolicyClearOverrideRestoresDefault) {
  EXPECT_TRUE(amp::AutocastPolicy::should_autocast(amp::AutocastOp::Matmul));
  amp::AutocastPolicy::set_override(amp::AutocastOp::Matmul, false);
  EXPECT_FALSE(amp::AutocastPolicy::should_autocast(amp::AutocastOp::Matmul));
  amp::AutocastPolicy::clear_override(amp::AutocastOp::Matmul);
  EXPECT_TRUE(amp::AutocastPolicy::should_autocast(amp::AutocastOp::Matmul));
}

TEST(AMPTest, AutocastPolicyGuardCanDisableSupportedOpTemporarily) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2}, cpu, DataType::Float32, false);
  Tensor b({2}, cpu, DataType::Float32, false);
  auto *ap = static_cast<float *>(a.data());
  auto *bp = static_cast<float *>(b.data());
  ap[0] = 1.0f; ap[1] = 2.0f;
  bp[0] = 3.0f; bp[1] = 4.0f;

  {
    amp::AutoCastGuard guard(DataType::Float16);

    Tensor base = a + b;
    EXPECT_EQ(base.dtype(), DataType::Float16);

    {
      amp::AutocastPolicyGuard disable_add(amp::AutocastOp::Add, false);
      Tensor no_cast = a + b;
      EXPECT_EQ(no_cast.dtype(), DataType::Float32);
    }

    Tensor restored = a + b;
    EXPECT_EQ(restored.dtype(), DataType::Float16);
  }
}

TEST(AMPTest, AutocastCastsCoreForwardOps) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2, 2}, cpu, DataType::Float32, false);
  Tensor b({2, 2}, cpu, DataType::Float32, false);
  auto *ap = static_cast<float *>(a.data());
  auto *bp = static_cast<float *>(b.data());
  ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f; ap[3] = 4.0f;
  bp[0] = 1.0f; bp[1] = 1.0f; bp[2] = 1.0f; bp[3] = 1.0f;

  Tensor add_fp32 = a + b;
  EXPECT_EQ(add_fp32.dtype(), DataType::Float32);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor add_out = a + b;
    Tensor mul_out = a * b;
    Tensor mm_out = a.matmul(b);
    Tensor relu_out = a.relu();
    Tensor loss = a.mse_loss(b);

    EXPECT_EQ(add_out.dtype(), DataType::Float16);
    EXPECT_EQ(mul_out.dtype(), DataType::Float16);
    EXPECT_EQ(mm_out.dtype(), DataType::Float16);
    EXPECT_EQ(relu_out.dtype(), DataType::Float16);
    EXPECT_EQ(loss.dtype(), DataType::Float32);
  }
}



TEST(AMPTest, AutocastCastsSpatialAndNormEntryPoints) {
  Device cpu{DeviceType::CPU, 0};

  Tensor x({1, 1, 4, 4}, cpu, DataType::Float32, false);
  Tensor w({1, 1, 3, 3}, cpu, DataType::Float32, false);
  Tensor b({1}, cpu, DataType::Float32, false);
  auto *xp = static_cast<float *>(x.data());
  auto *wp = static_cast<float *>(w.data());
  auto *bp = static_cast<float *>(b.data());
  for (int i = 0; i < 16; ++i)
    xp[i] = 1.0f;
  for (int i = 0; i < 9; ++i)
    wp[i] = 1.0f;
  bp[0] = 0.0f;

  Tensor running_mean({1}, cpu, DataType::Float32, false);
  Tensor running_var({1}, cpu, DataType::Float32, false);
  Tensor bn_w({1}, cpu, DataType::Float32, false);
  Tensor bn_b({1}, cpu, DataType::Float32, false);
  static_cast<float *>(running_mean.data())[0] = 0.0f;
  static_cast<float *>(running_var.data())[0] = 1.0f;
  static_cast<float *>(bn_w.data())[0] = 1.0f;
  static_cast<float *>(bn_b.data())[0] = 0.0f;

  Tensor ln_w({4}, cpu, DataType::Float32, false);
  Tensor ln_b({4}, cpu, DataType::Float32, false);
  for (int i = 0; i < 4; ++i) {
    static_cast<float *>(ln_w.data())[i] = 1.0f;
    static_cast<float *>(ln_b.data())[i] = 0.0f;
  }

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor conv = x.conv2d(w, b, 1, 0);
    Tensor pool = x.max_pool2d(2, 2, 0);
    Tensor up = pool.upsample2d(2);
    Tensor bn = x.batch_norm(running_mean, running_var, bn_w, bn_b, true, 0.1f, 1e-5f);
    Tensor ln = x.layer_norm(ln_w, ln_b, 1e-5f);

    EXPECT_EQ(conv.dtype(), DataType::Float16);
    EXPECT_EQ(pool.dtype(), DataType::Float16);
    EXPECT_EQ(up.dtype(), DataType::Float16);
    EXPECT_EQ(bn.dtype(), DataType::Float16);
    EXPECT_EQ(ln.dtype(), DataType::Float16);
  }
}


TEST(AMPTest, AutocastCoversNNModuleForwardPaths) {
  Device cpu{DeviceType::CPU, 0};

  auto linear = std::make_shared<nn::Linear>(4, 3);
  Tensor x2d({2, 4}, cpu, DataType::Float32, false);
  x2d.uniform_(0.0f, 1.0f);

  auto bn = std::make_shared<nn::BatchNorm2d>(3);
  Tensor x4d({2, 3, 4, 4}, cpu, DataType::Float32, false);
  x4d.uniform_(0.0f, 1.0f);

  auto ln = std::make_shared<nn::LayerNorm>(4);
  Tensor xln({2, 4}, cpu, DataType::Float32, false);
  xln.uniform_(0.0f, 1.0f);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_linear = linear->forward(x2d);
    Tensor y_bn = bn->forward(x4d);
    Tensor y_ln = ln->forward(xln);

    EXPECT_EQ(y_linear.dtype(), DataType::Float16);
    EXPECT_EQ(y_bn.dtype(), DataType::Float16);
    EXPECT_EQ(y_ln.dtype(), DataType::Float16);
  }
}

TEST(AMPTest, AutocastCoversActivationModulesWithPolicyOverride) {
  Device cpu{DeviceType::CPU, 0};

  Tensor x({2, 4}, cpu, DataType::Float32, false);
  x.uniform_(-1.0f, 1.0f);

  nn::Tanh tanh;
  nn::GELU gelu;
  nn::LeakyReLU lrelu(0.1f);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_tanh = tanh.forward(x);
    Tensor y_gelu = gelu.forward(x);
    Tensor y_lrelu = lrelu.forward(x);

    EXPECT_EQ(y_tanh.dtype(), DataType::Float16);
    EXPECT_EQ(y_gelu.dtype(), DataType::Float16);
    EXPECT_EQ(y_lrelu.dtype(), DataType::Float16);

    {
      amp::AutocastPolicyGuard disable_tanh(amp::AutocastOp::Tanh, false);
      Tensor y_tanh_fp32 = tanh.forward(x);
      EXPECT_EQ(y_tanh_fp32.dtype(), DataType::Float32);
      Tensor y_gelu_still_cast = gelu.forward(x);
      EXPECT_EQ(y_gelu_still_cast.dtype(), DataType::Float16);
    }
  }
}

TEST(AMPTest, AutocastCoversDropoutAndGlobalAvgPoolModules) {
  Device cpu{DeviceType::CPU, 0};

  nn::Dropout drop(0.2f);
  nn::GlobalAvgPool2d gap;

  Tensor x2d({2, 4}, cpu, DataType::Float32, false);
  x2d.uniform_(0.0f, 1.0f);

  Tensor x4d({1, 3, 4, 4}, cpu, DataType::Float32, false);
  x4d.uniform_(0.0f, 1.0f);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_drop = drop.forward(x2d);
    Tensor y_gap = gap.forward(x4d);
    EXPECT_EQ(y_drop.dtype(), DataType::Float16);
    EXPECT_EQ(y_gap.dtype(), DataType::Float16);

    {
      amp::AutocastPolicyGuard disable_drop(amp::AutocastOp::Dropout, false);
      Tensor y_drop_fp32 = drop.forward(x2d);
      EXPECT_EQ(y_drop_fp32.dtype(), DataType::Float32);
      Tensor y_gap_still_cast = gap.forward(x4d);
      EXPECT_EQ(y_gap_still_cast.dtype(), DataType::Float16);
    }

    {
      amp::AutocastPolicyGuard disable_gap(amp::AutocastOp::GlobalAvgPool2d,
                                           false);
      Tensor y_gap_fp32 = gap.forward(x4d);
      EXPECT_EQ(y_gap_fp32.dtype(), DataType::Float32);
      Tensor y_drop_still_cast = drop.forward(x2d);
      EXPECT_EQ(y_drop_still_cast.dtype(), DataType::Float16);
    }
  }
}

TEST(AMPTest, AutocastCoversMultiHeadAttentionModuleWithPolicyOverride) {
  Device cpu{DeviceType::CPU, 0};
  nn::MultiHeadAttention mha(4, 2, true);

  Tensor x({2, 3, 4}, cpu, DataType::Float32, false);
  x.uniform_(0.0f, 1.0f);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_cast = mha.forward(x);
    EXPECT_EQ(y_cast.dtype(), DataType::Float16);

    {
      amp::AutocastPolicyGuard disable_mha(amp::AutocastOp::MultiHeadAttention,
                                           false);
      Tensor y_fp32 = mha.forward(x);
      EXPECT_EQ(y_fp32.dtype(), DataType::Float32);
    }
  }
}

TEST(AMPTest, AutocastCoversEmbeddingModuleWithPolicyOverride) {
  Device cpu{DeviceType::CPU, 0};

  nn::Embedding embedding(8, 4);

  Tensor one_hot({2, 3, 8}, cpu, DataType::Float32, false);
  one_hot.uniform_(0.0f, 0.0f);
  auto *oh = static_cast<float *>(one_hot.data());
  auto set_tok = [&](int b, int t, int token) {
    oh[(b * 3 + t) * 8 + token] = 1.0f;
  };
  set_tok(0, 0, 0);
  set_tok(0, 1, 1);
  set_tok(0, 2, 2);
  set_tok(1, 0, 3);
  set_tok(1, 1, 4);
  set_tok(1, 2, 5);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_embed = embedding.forward(one_hot);
    EXPECT_EQ(y_embed.dtype(), DataType::Float16);

    {
      amp::AutocastPolicyGuard disable_embed(amp::AutocastOp::Embedding, false);
      Tensor y_embed_fp32 = embedding.forward(one_hot);
      EXPECT_EQ(y_embed_fp32.dtype(), DataType::Float32);
    }
  }
}

TEST(AMPTest, AutocastCoversSpatialNNModuleForwardPaths) {
  Device cpu{DeviceType::CPU, 0};

  auto conv = std::make_shared<nn::Conv2d>(3, 4, 3, 1, 1);
  auto bn = std::make_shared<nn::BatchNorm2d>(4);
  auto pool = std::make_shared<nn::MaxPool2d>(2, 2, 0);
  auto up = std::make_shared<nn::Upsample>(2);

  Tensor x({1, 3, 8, 8}, cpu, DataType::Float32, false);
  x.uniform_(0.0f, 1.0f);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor y_conv = conv->forward(x);
    Tensor y_bn = bn->forward(y_conv);
    Tensor y_pool = pool->forward(y_bn);
    Tensor y_up = up->forward(y_pool);

    EXPECT_EQ(y_conv.dtype(), DataType::Float16);
    EXPECT_EQ(y_bn.dtype(), DataType::Float16);
    EXPECT_EQ(y_pool.dtype(), DataType::Float16);
    EXPECT_EQ(y_up.dtype(), DataType::Float16);
  }
}

TEST(AMPTest, SensitiveOpsAccumulateToFP32FromLowPrecisionInputs) {
  Device cpu{DeviceType::CPU, 0};

  Tensor x({1, 4}, cpu, DataType::Float16, false);
  Tensor y({1, 4}, cpu, DataType::Float16, false);
  static_cast<int8_t *>(x.data())[0] = 1;
  static_cast<int8_t *>(x.data())[1] = 2;
  static_cast<int8_t *>(x.data())[2] = 3;
  static_cast<int8_t *>(x.data())[3] = 4;
  static_cast<int8_t *>(y.data())[0] = 1;
  static_cast<int8_t *>(y.data())[1] = 0;
  static_cast<int8_t *>(y.data())[2] = 0;
  static_cast<int8_t *>(y.data())[3] = 0;

  Tensor s = x.sum();
  Tensor sm = x.softmax(-1);
  Tensor lsm = x.log_softmax(-1);
  Tensor mse = x.mse_loss(y);
  Tensor ce = x.cross_entropy(y);

  EXPECT_EQ(s.dtype(), DataType::Float32);
  EXPECT_EQ(sm.dtype(), DataType::Float32);
  EXPECT_EQ(lsm.dtype(), DataType::Float32);
  EXPECT_EQ(mse.dtype(), DataType::Float32);
  EXPECT_EQ(ce.dtype(), DataType::Float32);
}

TEST(AMPTest, DebugBackendForwardsNonFiniteCheckHook) {
  Device cpu{DeviceType::CPU, 0};
  auto base = BackendManager::get(cpu);
  auto dbg = wrap_with_debug_backend(base);

  ASSERT_TRUE(dbg->supports_non_finite_check());

  Storage finite(2 * dtype_size(DataType::Float32), cpu, DataType::Float32);
  auto *fp = static_cast<float *>(finite.data());
  fp[0] = 3.0f;
  fp[1] = 4.0f;
  EXPECT_FALSE(dbg->has_non_finite(finite, 2));

  Storage bad(2 * dtype_size(DataType::Float32), cpu, DataType::Float32);
  auto *bp = static_cast<float *>(bad.data());
  bp[0] = 3.0f;
  bp[1] = -std::numeric_limits<float>::infinity();
  EXPECT_TRUE(dbg->has_non_finite(bad, 2));
}

TEST(AMPTest, CPUBackendProvidesNonFiniteCheckHook) {
  Device cpu{DeviceType::CPU, 0};
  auto backend = BackendManager::get(cpu);
  ASSERT_TRUE(backend->supports_non_finite_check());

  Storage finite(2 * dtype_size(DataType::Float32), cpu, DataType::Float32);
  auto *fp = static_cast<float *>(finite.data());
  fp[0] = 1.0f;
  fp[1] = 2.0f;
  EXPECT_FALSE(backend->has_non_finite(finite, 2));

  Storage bad(2 * dtype_size(DataType::Float32), cpu, DataType::Float32);
  auto *bp = static_cast<float *>(bad.data());
  bp[0] = 1.0f;
  bp[1] = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(backend->has_non_finite(bad, 2));
}

TEST(AMPTest, GradScalerScalesLossAndSteps) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  w.uniform_(2.0f, 2.0f);

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 1);

  Tensor x({1}, cpu, DataType::Float32, false);
  x.uniform_(1.0f, 1.0f);

  Tensor y = w * x;
  Tensor target({1}, cpu, DataType::Float32, false);
  target.uniform_(0.0f, 0.0f);

  Tensor loss = y.mse_loss(target);
  Tensor scaled = scaler.scale(loss);
  scaled.backward();

  bool stepped = scaler.step(opt, {w});
  EXPECT_TRUE(stepped);
  EXPECT_LT(w.item(), 2.0f);
  EXPECT_GE(scaler.current_scale(), 8.0f);
}


TEST(AMPTest, FP32MasterSGDKeepsModelLowPrecision) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float16, true);
  // Float16 path currently uses fallback scalar conversion; integer proxy values.
  auto *wp = static_cast<int8_t *>(w.data());
  wp[0] = 4;

  amp::FP32MasterSGD opt({w}, 1.0f);

  Tensor grad({1}, cpu, DataType::Float16, false);
  auto *gp = static_cast<int8_t *>(grad.data());
  gp[0] = 2;
  w.impl_->grad = grad.impl_;

  opt.step();

  EXPECT_EQ(w.dtype(), DataType::Float16);
  auto *w_after = static_cast<int8_t *>(w.data());
  EXPECT_NE(w_after[0], 4);
}


TEST(AMPTest, FP32MasterAdamKeepsModelLowPrecision) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float16, true);
  auto *wp = static_cast<int8_t *>(w.data());
  wp[0] = 8;

  amp::FP32MasterAdam opt({w}, 4.0f);

  Tensor grad({1}, cpu, DataType::Float16, false);
  auto *gp = static_cast<int8_t *>(grad.data());
  gp[0] = 4;
  w.impl_->grad = grad.impl_;

  opt.step();

  EXPECT_EQ(w.dtype(), DataType::Float16);
  auto *w_after = static_cast<int8_t *>(w.data());
  EXPECT_NE(w_after[0], 8);
}

TEST(AMPTest, FP32MasterSGDStepParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  auto *w32 = static_cast<float *>(w_fp32.data());
  w32[0] = 10.0f;
  Tensor g_fp32({1}, cpu, DataType::Float32, false);
  auto *g32 = static_cast<float *>(g_fp32.data());
  g32[0] = 4.0f;
  w_fp32.impl_->grad = g_fp32.impl_;
  optim::SGD sgd_fp32({w_fp32}, 0.5f);
  sgd_fp32.step();

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  auto *wlp = static_cast<int8_t *>(w_lp.data());
  wlp[0] = 10;
  Tensor g_lp({1}, cpu, DataType::Float16, false);
  auto *glp = static_cast<int8_t *>(g_lp.data());
  glp[0] = 4;
  w_lp.impl_->grad = g_lp.impl_;
  amp::FP32MasterSGD sgd_lp({w_lp}, 0.5f);
  sgd_lp.step();

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 1e-5f);
}



TEST(AMPTest, FP32MasterSGDMultiStepParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  static_cast<float *>(w_fp32.data())[0] = 10.0f;
  optim::SGD sgd_fp32({w_fp32}, 0.5f);

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  static_cast<int8_t *>(w_lp.data())[0] = 10;
  amp::FP32MasterSGD sgd_lp({w_lp}, 0.5f);

  for (int step = 0; step < 5; ++step) {
    Tensor g_fp32({1}, cpu, DataType::Float32, false);
    static_cast<float *>(g_fp32.data())[0] = 2.0f;
    w_fp32.impl_->grad = g_fp32.impl_;
    sgd_fp32.step();

    Tensor g_lp({1}, cpu, DataType::Float16, false);
    static_cast<int8_t *>(g_lp.data())[0] = 2;
    w_lp.impl_->grad = g_lp.impl_;
    sgd_lp.step();

    Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
    EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 1e-5f);
  }
}



TEST(AMPTest, FP32MasterSGDTrainingLoopParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  static_cast<float *>(w_fp32.data())[0] = 10.0f;
  optim::SGD sgd_fp32({w_fp32}, 0.1f);

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  static_cast<int8_t *>(w_lp.data())[0] = 10;
  amp::FP32MasterSGD sgd_lp({w_lp}, 0.1f);

  Tensor x_fp32({1}, cpu, DataType::Float32, false);
  static_cast<float *>(x_fp32.data())[0] = 1.0f;
  Tensor t_fp32({1}, cpu, DataType::Float32, false);
  static_cast<float *>(t_fp32.data())[0] = 0.0f;

  Tensor x_lp({1}, cpu, DataType::Float16, false);
  static_cast<int8_t *>(x_lp.data())[0] = 1;
  Tensor t_lp({1}, cpu, DataType::Float16, false);
  static_cast<int8_t *>(t_lp.data())[0] = 0;

  for (int step = 0; step < 5; ++step) {
    sgd_fp32.zero_grad();
    Tensor y_fp32 = w_fp32 * x_fp32;
    Tensor loss_fp32 = y_fp32.mse_loss(t_fp32);
    loss_fp32.backward();
    sgd_fp32.step();

    sgd_lp.zero_grad();
    Tensor y_lp = w_lp * x_lp;
    Tensor loss_lp = y_lp.mse_loss(t_lp);
    loss_lp.backward();
    sgd_lp.step();
  }

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_LT(w_fp32.item(), 10.0f);
  EXPECT_LT(w_lp_fp32.item(), 10.0f);
  // Low-precision path uses proxy storage conversion; keep tolerant parity bound.
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 0.5f);
}

TEST(AMPTest, FP32MasterAdamTrainingLoopParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  static_cast<float *>(w_fp32.data())[0] = 10.0f;
  optim::Adam adam_fp32({w_fp32}, 0.05f);

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  static_cast<int8_t *>(w_lp.data())[0] = 10;
  amp::FP32MasterAdam adam_lp({w_lp}, 0.05f);

  Tensor x_fp32({1}, cpu, DataType::Float32, false);
  static_cast<float *>(x_fp32.data())[0] = 1.0f;
  Tensor t_fp32({1}, cpu, DataType::Float32, false);
  static_cast<float *>(t_fp32.data())[0] = 0.0f;

  Tensor x_lp({1}, cpu, DataType::Float16, false);
  static_cast<int8_t *>(x_lp.data())[0] = 1;
  Tensor t_lp({1}, cpu, DataType::Float16, false);
  static_cast<int8_t *>(t_lp.data())[0] = 0;

  for (int step = 0; step < 10; ++step) {
    adam_fp32.zero_grad();
    Tensor y_fp32 = w_fp32 * x_fp32;
    Tensor loss_fp32 = y_fp32.mse_loss(t_fp32);
    loss_fp32.backward();
    adam_fp32.step();

    adam_lp.zero_grad();
    Tensor y_lp = w_lp * x_lp;
    Tensor loss_lp = y_lp.mse_loss(t_lp);
    loss_lp.backward();
    adam_lp.step();
  }

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_LT(w_fp32.item(), 10.0f);
  EXPECT_LT(w_lp_fp32.item(), 10.0f);
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 1.0f);
}

TEST(AMPTest, FP32MasterAdamStepParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  auto *w32 = static_cast<float *>(w_fp32.data());
  w32[0] = 10.0f;
  Tensor g_fp32({1}, cpu, DataType::Float32, false);
  auto *g32 = static_cast<float *>(g_fp32.data());
  g32[0] = 4.0f;
  w_fp32.impl_->grad = g_fp32.impl_;
  optim::Adam adam_fp32({w_fp32}, 2.0f);
  adam_fp32.step();

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  auto *wlp = static_cast<int8_t *>(w_lp.data());
  wlp[0] = 10;
  Tensor g_lp({1}, cpu, DataType::Float16, false);
  auto *glp = static_cast<int8_t *>(g_lp.data());
  glp[0] = 4;
  w_lp.impl_->grad = g_lp.impl_;
  amp::FP32MasterAdam adam_lp({w_lp}, 2.0f);
  adam_lp.step();

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_LT(w_fp32.item(), 10.0f);
  EXPECT_LT(w_lp_fp32.item(), 10.0f);
  // Low-precision path uses proxy storage conversion; keep tolerant parity bound.
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 0.5f);
}


TEST(AMPTest, GradScalerSkipsStepAndBacksOffOnInfGradients) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  auto *wp = static_cast<float *>(w.data());
  wp[0] = 2.0f;

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2);

  Tensor grad({1}, cpu, DataType::Float32, false);
  auto *gp = static_cast<float *>(grad.data());
  gp[0] = std::numeric_limits<float>::infinity();
  w.impl_->grad = grad.impl_;

  bool stepped = scaler.step(opt, {w});
  EXPECT_FALSE(stepped);
  EXPECT_FLOAT_EQ(w.item(), 2.0f);
  EXPECT_FLOAT_EQ(scaler.current_scale(), 4.0f);
}

TEST(AMPTest, GradScalerUnscaleDividesGradientsByScale) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  static_cast<float *>(w.data())[0] = 2.0f;

  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2);

  Tensor grad({1}, cpu, DataType::Float32, false);
  static_cast<float *>(grad.data())[0] = 16.0f;
  w.impl_->grad = grad.impl_;

  bool found_inf = scaler.unscale_({w});
  EXPECT_FALSE(found_inf);
  EXPECT_FLOAT_EQ(w.grad().item(), 2.0f);
}

TEST(AMPTest, GradScalerGrowsScaleAfterConfiguredInterval) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  auto *wp = static_cast<float *>(w.data());
  wp[0] = 2.0f;

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2);

  Tensor grad({1}, cpu, DataType::Float32, false);
  auto *gp = static_cast<float *>(grad.data());
  gp[0] = 1.0f;

  w.impl_->grad = grad.impl_;
  EXPECT_TRUE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 8.0f);

  w.impl_->grad = grad.impl_;
  EXPECT_TRUE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 16.0f);
}

TEST(AMPTest, GradScalerStaticModeKeepsScaleConstant) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  static_cast<float *>(w.data())[0] = 2.0f;

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2, amp::GradScaler::Mode::Static);

  Tensor good_grad({1}, cpu, DataType::Float32, false);
  static_cast<float *>(good_grad.data())[0] = 1.0f;

  w.impl_->grad = good_grad.impl_;
  EXPECT_TRUE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 8.0f);

  Tensor bad_grad({1}, cpu, DataType::Float32, false);
  static_cast<float *>(bad_grad.data())[0] = std::numeric_limits<float>::infinity();
  w.impl_->grad = bad_grad.impl_;
  EXPECT_FALSE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 8.0f);
}

TEST(AMPTest, GradScalerRejectsInvalidHyperparameters) {
  EXPECT_THROW((amp::GradScaler(0.0f, 2.0f, 0.5f, 2)), std::runtime_error);
  EXPECT_THROW((amp::GradScaler(8.0f, 1.0f, 0.5f, 2)), std::runtime_error);
  EXPECT_THROW((amp::GradScaler(8.0f, 2.0f, 1.0f, 2)), std::runtime_error);
  EXPECT_THROW((amp::GradScaler(8.0f, 2.0f, 0.5f, 0)), std::runtime_error);
}

TEST(AMPTest, FP32MasterOptimizersExposeFloat32StateForBFloat16Model) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::BFloat16, true);

  amp::FP32MasterSGD sgd({w}, 0.1f);
  EXPECT_EQ(sgd.master_dtype(), DataType::Float32);

  amp::FP32MasterAdam adam({w}, 1e-3f);
  EXPECT_EQ(adam.master_dtype(), DataType::Float32);
  EXPECT_EQ(adam.state_dtype(), DataType::Float32);
  EXPECT_EQ(adam.step_count(), 0);

  Tensor g({1}, cpu, DataType::BFloat16, false);
  static_cast<int8_t *>(g.data())[0] = 4;
  w.impl_->grad = g.impl_;
  adam.step();
  EXPECT_EQ(adam.step_count(), 1);
}
