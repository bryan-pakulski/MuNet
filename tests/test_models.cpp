#include "nn.hpp"
#include "ops.hpp"
#include "optim.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>

using namespace munet;

class ModelTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, ModelTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

class ReproUNet : public nn::Module {
public:
  ReproUNet() {
    enc1 =
        register_module("enc1", std::make_shared<nn::Conv2d>(1, 16, 3, 1, 1));
    pool = std::make_shared<nn::MaxPool2d>(2, 2);
    up = std::make_shared<nn::Upsample>(2);
    dec = register_module("dec",
                          std::make_shared<nn::Conv2d>(16 + 1, 1, 3, 1, 1));
  }

  Tensor forward_impl(Tensor x) override {
    auto e1 = std::dynamic_pointer_cast<nn::Conv2d>(enc1)->forward(x).relu();
    auto p1 = pool->forward(e1);
    auto up1 = up->forward(p1);
    auto merge = Tensor::cat({up1, x}, 1);
    return std::dynamic_pointer_cast<nn::Conv2d>(dec)->forward(merge);
  }

  std::shared_ptr<nn::Module> enc1, pool, up, dec;
};

TEST_P(ModelTest, UnetRegistration) {

  auto model = std::make_shared<ReproUNet>();
  model->to(dev());

  Tensor x({1, 1, 32, 32}, dev());
  x.uniform_();

  ASSERT_NO_THROW({
    auto out = model->forward(x);
    out.sum().backward();
  });
}

TEST_P(ModelTest, UnetTraining) {
  auto model = std::make_shared<ReproUNet>();
  model->to(dev());

  Tensor y({1, 1, 32, 32}, dev());
  y.uniform_();

  int batches = 100;
  auto optimizer = munet::optim::SGD(model->parameters(), 0.01f);

  for (int i = 0; i < batches; ++i) {
    optimizer.zero_grad();

    Tensor x({1, 1, 32, 32}, dev());
    x.uniform_();

    auto out = model->forward(x);
    auto loss = out.mse_loss(y);

    loss.backward();
    optimizer.step();
  }
}
