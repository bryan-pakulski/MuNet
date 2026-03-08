#include "backend.hpp"
#include "backend/cpu_backend.hpp"
#include <gtest/gtest.h>

using namespace munet;

TEST(BackendManagerTest, CanOverrideBackendFactoryForDeviceType) {
  BackendManager::register_backend(
      DeviceType::CPU,
      [](Device) { return std::make_shared<CPUBackend>(); });

  auto backend = BackendManager::get(Device{DeviceType::CPU, 0});
  EXPECT_NE(backend, nullptr);
}
