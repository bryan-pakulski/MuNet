#pragma once
#include "core.hpp"
namespace munet {
bool extended_shape(const Graph&, const std::string&, const std::vector<Id>&, Shape&, Shape&);
bool extended_gradient(Graph&, Id, Id, const std::function<void(Id,Id)>&);
}
