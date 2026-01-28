#include "pioneerml_dataloaders/utils/timing/timer.h"

namespace pioneerml::utils::timing {

Timer::Timer() : start_(std::chrono::steady_clock::now()) {}

void Timer::Reset() {
  start_ = std::chrono::steady_clock::now();
}

double Timer::ElapsedMs() const {
  auto now = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed = now - start_;
  return elapsed.count();
}

}  // namespace pioneerml::utils::timing
