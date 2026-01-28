#pragma once

#include <chrono>

namespace pioneerml::utils::timing {

class Timer {
 public:
  Timer();
  void Reset();
  double ElapsedMs() const;

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace pioneerml::utils::timing
