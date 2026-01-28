#include "pioneerml_dataloaders/utils/timing/timing_settings.h"

namespace pioneerml::utils::timing {

TimingSettings& TimingSettings::Instance() {
  static TimingSettings settings;
  return settings;
}

void TimingSettings::SetEnabled(bool enabled) {
  enabled_.store(enabled, std::memory_order_relaxed);
}

bool TimingSettings::Enabled() const {
  return enabled_.load(std::memory_order_relaxed);
}

}  // namespace pioneerml::utils::timing
