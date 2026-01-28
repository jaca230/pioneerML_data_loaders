#pragma once

#include <string>

#include "pioneerml_dataloaders/utils/timing/timer.h"

namespace pioneerml::utils::timing {

class ScopedTimer {
 public:
  explicit ScopedTimer(std::string name);
  ~ScopedTimer();

 private:
  std::string name_;
  Timer timer_;
  bool enabled_{false};
};

bool TimingEnabled();
void LogTimings();

}  // namespace pioneerml::utils::timing
