#pragma once

#include <nlohmann/json.hpp>

namespace pioneerml::utils::timing {

class TimingConfigurator {
 public:
  bool Configure(const nlohmann::json& timing_config) const;
  static nlohmann::json DefaultConfig();
};

}  // namespace pioneerml::utils::timing
