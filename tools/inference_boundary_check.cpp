#include "inference.hpp"

#ifdef MUNET_ENABLE_TRAINING
#error "munet_inference boundary check should not compile with training enabled"
#endif

int main() {
  munet::inference::EngineConfig config;
  config.strict_shape_check = true;
  config.allow_autograd_inputs = false;
  config.capture_profiler_memory = true;

  munet::inference::Engine engine(config);
  (void)engine;
  return 0;
}
