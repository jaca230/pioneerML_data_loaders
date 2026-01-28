#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace pioneerml::utils::timing {

struct TimingStats {
  uint64_t count{0};
  double total_ms{0.0};
  double min_ms{0.0};
  double max_ms{0.0};
};

class TimingRegistry {
 public:
  static TimingRegistry& Instance();

  void Record(const std::string& name, double elapsed_ms);
  std::unordered_map<std::string, TimingStats> Snapshot() const;
  void Reset();

 private:
  TimingRegistry() = default;
  TimingRegistry(const TimingRegistry&) = delete;
  TimingRegistry& operator=(const TimingRegistry&) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, TimingStats> stats_;
};

}  // namespace pioneerml::utils::timing
