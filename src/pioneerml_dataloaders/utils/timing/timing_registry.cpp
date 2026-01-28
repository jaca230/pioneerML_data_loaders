#include "pioneerml_dataloaders/utils/timing/timing_registry.h"

namespace pioneerml::utils::timing {

TimingRegistry& TimingRegistry::Instance() {
  static TimingRegistry registry;
  return registry;
}

void TimingRegistry::Record(const std::string& name, double elapsed_ms) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto& stat = stats_[name];
  stat.count += 1;
  stat.total_ms += elapsed_ms;
  if (stat.count == 1) {
    stat.min_ms = elapsed_ms;
    stat.max_ms = elapsed_ms;
  } else {
    if (elapsed_ms < stat.min_ms) stat.min_ms = elapsed_ms;
    if (elapsed_ms > stat.max_ms) stat.max_ms = elapsed_ms;
  }
}

std::unordered_map<std::string, TimingStats> TimingRegistry::Snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return stats_;
}

void TimingRegistry::Reset() {
  std::lock_guard<std::mutex> guard(mutex_);
  stats_.clear();
}

}  // namespace pioneerml::utils::timing
