#pragma once

#include <nlohmann/json.hpp>

namespace pioneerml::configurable {

class Configurable {
 public:
  virtual ~Configurable() = default;

  // Optional configuration hook. Default is no-op.
  virtual void LoadConfig(const nlohmann::json& cfg) { (void)cfg; }
};

}  // namespace pioneerml::configurable
