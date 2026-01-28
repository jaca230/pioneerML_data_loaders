#pragma once

#include <nlohmann/json.hpp>

namespace pioneerml::utils::logging {

class LoggerConfigurator {
 public:
  bool Configure(const nlohmann::json& logger_config) const;
  static nlohmann::json DefaultConfig();
};

}  // namespace pioneerml::utils::logging
