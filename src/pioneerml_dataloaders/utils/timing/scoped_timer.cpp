#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

#include <cstdlib>

#include "pioneerml_dataloaders/utils/timing/timing_registry.h"

#include <spdlog/spdlog.h>

namespace pioneerml::utils::timing {

ScopedTimer::ScopedTimer(std::string name)
    : name_(std::move(name)),
      timer_(),
      enabled_(TimingEnabled()) {}

ScopedTimer::~ScopedTimer() {
  if (!enabled_) {
    return;
  }
  TimingRegistry::Instance().Record(name_, timer_.ElapsedMs());
}

bool TimingEnabled() {
  const char* env = std::getenv("PIONEERML_DATALOADERS_TIMING");
  if (!env) {
    return false;
  }
  return std::string(env) != "0";
}

void LogTimings() {
  auto stats = TimingRegistry::Instance().Snapshot();
  if (stats.empty()) {
    spdlog::info("timing: no stats collected");
    return;
  }
  spdlog::info("timing: {} entries", stats.size());
  for (const auto& [name, stat] : stats) {
    double avg = stat.count ? (stat.total_ms / static_cast<double>(stat.count)) : 0.0;
    spdlog::info(
        "timing: {} count={} total_ms={:.3f} avg_ms={:.3f} min_ms={:.3f} max_ms={:.3f}",
        name,
        stat.count,
        stat.total_ms,
        avg,
        stat.min_ms,
        stat.max_ms);
  }
}

}  // namespace pioneerml::utils::timing
