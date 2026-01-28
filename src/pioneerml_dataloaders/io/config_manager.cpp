#include "pioneerml_dataloaders/io/config_manager.h"

#include <fstream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace pioneerml::io {

ConfigManager& ConfigManager::Instance() {
  static ConfigManager manager;
  return manager;
}

void ConfigManager::Reset() {
  merged_json_ = nlohmann::json::object();
  logger_config_.clear();
}

bool ConfigManager::LoadFiles(const std::vector<std::string>& filepaths) {
  Reset();
  for (const auto& filepath : filepaths) {
    if (!LoadFile(filepath)) {
      return false;
    }
  }
  return true;
}

bool ConfigManager::LoadFile(const std::string& filepath) {
  std::ifstream in(filepath);
  if (!in.is_open()) {
    spdlog::warn("[ConfigManager] Could not open config file: {}", filepath);
    return false;
  }
  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception& e) {
    spdlog::error("[ConfigManager] Failed to parse JSON: {}", e.what());
    return false;
  }
  return AddJson(j);
}

bool ConfigManager::AddJson(const nlohmann::json& j) {
  if (!MergeJson(j)) {
    spdlog::error("[ConfigManager] Failed to merge JSON.");
    return false;
  }
  if (merged_json_.contains("logger")) {
    logger_config_ = merged_json_["logger"];
  }
  return true;
}

bool ConfigManager::MergeJson(const nlohmann::json& new_json) {
  for (auto it = new_json.begin(); it != new_json.end(); ++it) {
    merged_json_[it.key()] = it.value();
  }
  return true;
}

const nlohmann::json& ConfigManager::LoggerConfig() const {
  return logger_config_;
}

bool ConfigManager::ConfigureLogger() const {
  if (logger_config_.empty()) {
    return false;
  }

  try {
    spdlog::level::level_enum level = spdlog::level::info;
    if (logger_config_.contains("level")) {
      level = spdlog::level::from_str(logger_config_.at("level").get<std::string>());
    }

    std::vector<spdlog::sink_ptr> sinks;
    auto sinks_json = logger_config_.value("sinks", nlohmann::json::object());
    bool console_enabled = true;
    bool console_color = true;
    if (sinks_json.contains("console")) {
      console_enabled = sinks_json["console"].value("enabled", true);
      console_color = sinks_json["console"].value("color", true);
    }
    if (console_enabled) {
      if (console_color) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
      } else {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
      }
    }

    if (sinks_json.contains("file")) {
      const auto& file_cfg = sinks_json["file"];
      if (file_cfg.value("enabled", false)) {
        auto filename = file_cfg.value("filename", std::string("pioneerml_dataloaders.log"));
        if (file_cfg.contains("max_size") && file_cfg.contains("max_files")) {
          auto max_size = file_cfg.value("max_size", 10485760);
          auto max_files = file_cfg.value("max_files", 3);
          sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
              filename, max_size, max_files));
        } else {
          sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true));
        }
      }
    }

    std::string logger_name = logger_config_.value("name", "pioneerml_dataloaders");
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);

    logger->set_pattern(logger_config_.value("pattern", "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"));
    logger->set_level(level);
    return true;
  } catch (const std::exception& e) {
    spdlog::error("[ConfigManager] Logger config error: {}", e.what());
    return false;
  }
}

}  // namespace pioneerml::io
