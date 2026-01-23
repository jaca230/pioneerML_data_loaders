#pragma once

#include <cstdint>
#include <vector>

namespace pioneerml::derived {

class TimeGrouper {
 public:
  explicit TimeGrouper(double window_ns) : window_ns_(window_ns) {}
  std::vector<int64_t> Compute(const std::vector<double>& times) const;

 private:
  double window_ns_;
};

}  // namespace pioneerml::derived
