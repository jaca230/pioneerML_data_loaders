#include "pioneerml_dataloaders/utils/logging/logger_configurator.h"

#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace pioneerml::utils::logging {

bool LoggerConfigurator::Configure(const nlohmann::json& logger_config) const {
  nlohmann::json config = logger_config;
  if (config.empty()) {
    config = DefaultConfig();
  }

  try {
    spdlog::level::level_enum level = spdlog::level::info;
    if (config.contains("level")) {
      level = spdlog::level::from_str(config.at("level").get<std::string>());
    }

    std::vector<spdlog::sink_ptr> sinks;
    auto sinks_json = config.value("sinks", nlohmann::json::object());
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

    std::string logger_name = config.value("name", "PioneerML_DataLoaders");
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);

    logger->set_pattern(config.value("pattern", "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v"));
    logger->set_level(level);
    return true;
  } catch (const std::exception& e) {
    spdlog::error("[LoggerConfigurator] Logger config error: {}", e.what());
    return false;
  }
}

nlohmann::json LoggerConfigurator::DefaultConfig() {
  return nlohmann::json{
      {"name", "PioneerML_DataLoaders"},
      {"level", "info"},
      {"pattern", "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v"},
      {"sinks",
       {{"console", {{"enabled", true}, {"color", true}}}}}};
}

}  // namespace pioneerml::utils::logging
