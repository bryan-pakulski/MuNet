#pragma once

#include "backend.hpp"
#include <memory>

namespace munet {

std::shared_ptr<Backend> wrap_with_debug_backend(std::shared_ptr<Backend> base);

} // namespace munet
