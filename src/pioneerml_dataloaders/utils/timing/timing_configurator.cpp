#include "pioneerml_dataloaders/utils/timing/timing_configurator.h"

#include "pioneerml_dataloaders/utils/timing/timing_settings.h"

namespace pioneerml::utils::timing {

bool TimingConfigurator::Configure(const nlohmann::json& timing_config) const {
  nlohmann::json config = timing_config;
  if (config.empty()) {
    config = DefaultConfig();
  }

  if (config.is_boolean()) {
    TimingSettings::Instance().SetEnabled(config.get<bool>());
    return true;
  }

  if (config.contains("enabled")) {
    TimingSettings::Instance().SetEnabled(config.at("enabled").get<bool>());
    return true;
  }

  return false;
}

nlohmann::json TimingConfigurator::DefaultConfig() {
  return nlohmann::json{{"enabled", false}};
}

}  // namespace pioneerml::utils::timing
