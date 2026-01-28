#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pioneerml::utils::json {

class JsonManager {
 public:
  static JsonManager& Instance();

  bool LoadFiles(const std::vector<std::string>& filepaths);
  bool LoadFile(const std::string& filepath);
  bool AddJson(const nlohmann::json& j);
  void Reset();

  const nlohmann::json& MergedJson() const;

 private:
  JsonManager() = default;
  JsonManager(const JsonManager&) = delete;
  JsonManager& operator=(const JsonManager&) = delete;

  bool MergeJson(const nlohmann::json& new_json);

  nlohmann::json merged_json_{nlohmann::json::object()};
};

}  // namespace pioneerml::utils::json
