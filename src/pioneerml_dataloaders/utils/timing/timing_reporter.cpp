#include "pioneerml_dataloaders/utils/timing/timing_reporter.h"

#include "pioneerml_dataloaders/utils/timing/timing_registry.h"

#include <spdlog/spdlog.h>

namespace pioneerml::utils::timing {

void TimingReporter::LogTimings() {
  auto stats = TimingRegistry::Instance().Snapshot();
  if (stats.empty()) {
    spdlog::debug("timing: no stats collected");
    return;
  }
  spdlog::debug("timing: {} entries", stats.size());
  for (const auto& [name, stat] : stats) {
    double avg = stat.count ? (stat.total_ms / static_cast<double>(stat.count)) : 0.0;
    spdlog::debug(
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
