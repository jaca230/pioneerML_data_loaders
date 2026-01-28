#include "pioneerml_dataloaders/utils/json/json_manager.h"

#include "pioneerml_dataloaders/io/json_reader.h"

#include <spdlog/spdlog.h>

namespace pioneerml::utils::json {

JsonManager& JsonManager::Instance() {
  static JsonManager manager;
  return manager;
}

void JsonManager::Reset() {
  merged_json_ = nlohmann::json::object();
}

bool JsonManager::LoadFiles(const std::vector<std::string>& filepaths) {
  Reset();
  for (const auto& filepath : filepaths) {
    if (!LoadFile(filepath)) {
      return false;
    }
  }
  return true;
}

bool JsonManager::LoadFile(const std::string& filepath) {
  io::JsonReader reader;
  try {
    auto j = reader.ReadFile(filepath);
    return AddJson(j);
  } catch (const std::exception& e) {
    spdlog::warn("[JsonManager] Failed to load JSON file {}: {}", filepath, e.what());
    return false;
  }
}

bool JsonManager::AddJson(const nlohmann::json& j) {
  if (!MergeJson(j)) {
    spdlog::error("[JsonManager] Failed to merge JSON.");
    return false;
  }
  return true;
}

bool JsonManager::MergeJson(const nlohmann::json& new_json) {
  for (auto it = new_json.begin(); it != new_json.end(); ++it) {
    merged_json_[it.key()] = it.value();
  }
  return true;
}

const nlohmann::json& JsonManager::MergedJson() const {
  return merged_json_;
}

}  // namespace pioneerml::utils::json
