#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

#include "pioneerml_dataloaders/utils/timing/timing_registry.h"
#include "pioneerml_dataloaders/utils/timing/timing_settings.h"

namespace pioneerml::utils::timing {

ScopedTimer::ScopedTimer(std::string name)
    : name_(std::move(name)),
      timer_(),
      enabled_(TimingSettings::Instance().Enabled()) {}

ScopedTimer::~ScopedTimer() {
  if (!enabled_) {
    return;
  }
  TimingRegistry::Instance().Record(name_, timer_.ElapsedMs());
}

}  // namespace pioneerml::utils::timing
