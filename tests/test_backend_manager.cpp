#include "backend.hpp"
#include "backend/cpu_backend.hpp"
#include <gtest/gtest.h>
#include <mutex>
#include <vector>

using namespace munet;

TEST(BackendManagerTest, CanOverrideBackendFactoryForDeviceType) {
  BackendManager::register_backend(
      DeviceType::CPU,
      [](Device) { return std::make_shared<CPUBackend>(); });

  auto backend = BackendManager::get(Device{DeviceType::CPU, 0});
  EXPECT_NE(backend, nullptr);
}

TEST(BackendManagerTest, ForwardsDeviceIndexToFactoryAndCachesPerIndex) {
  std::mutex mu;
  std::vector<int> requested_indices;

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device d) {
    std::lock_guard<std::mutex> lock(mu);
    requested_indices.push_back(d.index);
    return std::make_shared<CPUBackend>();
  });

  auto b0 = BackendManager::get(Device{DeviceType::UNKNOWN, 0});
  auto b1 = BackendManager::get(Device{DeviceType::UNKNOWN, 1});
  auto b1_again = BackendManager::get(Device{DeviceType::UNKNOWN, 1});

  EXPECT_NE(b0, nullptr);
  EXPECT_NE(b1, nullptr);
  EXPECT_EQ(b1, b1_again);

  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(requested_indices.size(), 2);
  EXPECT_EQ(requested_indices[0], 0);
  EXPECT_EQ(requested_indices[1], 1);
}

TEST(BackendManagerTest, ReRegisteringBackendTypeClearsCachedInstances) {
  int generation = 0;

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device) {
    ++generation;
    return std::make_shared<CPUBackend>();
  });

  auto first = BackendManager::get(Device{DeviceType::UNKNOWN, 7});
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(generation, 1);

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device) {
    ++generation;
    return std::make_shared<CPUBackend>();
  });

  auto second = BackendManager::get(Device{DeviceType::UNKNOWN, 7});
  EXPECT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(generation, 2);
}

TEST(BackendManagerTest, ExposesBackendCapabilitySurface) {
  auto backend = BackendManager::get(Device{DeviceType::CPU, 0});
  ASSERT_NE(backend, nullptr);
  EXPECT_STREQ(backend->name(), "cpu");
  EXPECT_TRUE(backend->supports(BackendFeature::Matmul, DataType::Float32));
  EXPECT_FALSE(backend->supports(BackendFeature::Matmul, DataType::Float16));
  EXPECT_TRUE(backend->supports(BackendFeature::RandomFill, DataType::Float16));
  EXPECT_FALSE(backend->supports(BackendFeature::RandomFill, DataType::Int32));
}

TEST(BackendManagerTest, CapabilityDTypePolicyLivesInOnePlace) {
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::ElementwiseBinary,
                                            DataType::Float32));
  EXPECT_FALSE(supports_backend_feature_dtype(BackendFeature::ElementwiseBinary,
                                             DataType::Float16));
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::RandomFill,
                                            DataType::Float16));
  EXPECT_FALSE(supports_backend_feature_dtype(BackendFeature::RandomFill,
                                             DataType::Int32));
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::Reduction,
                                            DataType::Float32));
}
