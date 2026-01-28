#pragma once

#include <atomic>

namespace pioneerml::utils::timing {

class TimingSettings {
 public:
  static TimingSettings& Instance();

  void SetEnabled(bool enabled);
  bool Enabled() const;

 private:
  TimingSettings() = default;
  TimingSettings(const TimingSettings&) = delete;
  TimingSettings& operator=(const TimingSettings&) = delete;

  std::atomic<bool> enabled_{false};
};

}  // namespace pioneerml::utils::timing
