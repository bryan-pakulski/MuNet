#pragma once

#include "../core/module.hpp"

namespace munet {
namespace nn {

class Module : public core::Module {
public:
  virtual ~Module() = default;

  std::shared_ptr<Module> register_module(std::string name,
                                          std::shared_ptr<Module> m) {
    core::Module::register_module(name, m);
    return m;
  }

  std::map<std::string, std::shared_ptr<Module>>
  named_modules_typed(std::string prefix = "") {
    std::map<std::string, std::shared_ptr<Module>> mods;
    auto base_mods = core::Module::named_modules(prefix);
    for (auto &[name, m] : base_mods) {
      auto casted = std::dynamic_pointer_cast<Module>(m);
      if (casted) {
        mods[name] = casted;
      }
    }
    return mods;
  }
};

} // namespace nn
} // namespace munet
