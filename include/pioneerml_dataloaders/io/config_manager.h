#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pioneerml::io {

class ConfigManager {
 public:
  static ConfigManager& Instance();

  bool LoadFiles(const std::vector<std::string>& filepaths);
  bool LoadFile(const std::string& filepath);
  bool AddJson(const nlohmann::json& j);
  void Reset();

  const nlohmann::json& LoggerConfig() const;
  bool ConfigureLogger() const;

 private:
  ConfigManager() = default;
  ConfigManager(const ConfigManager&) = delete;
  ConfigManager& operator=(const ConfigManager&) = delete;

  bool MergeJson(const nlohmann::json& new_json);

  nlohmann::json merged_json_{nlohmann::json::object()};
  nlohmann::json logger_config_{};
};

}  // namespace pioneerml::io
