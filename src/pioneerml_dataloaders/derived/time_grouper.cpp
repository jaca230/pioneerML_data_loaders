#include "pioneerml_dataloaders/derived/time_grouper.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace pioneerml::derived {

std::vector<int64_t> TimeGrouper::Compute(const std::vector<double>& times) const {
  if (times.empty()) return {};
  std::vector<int64_t> order(times.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return times[a] < times[b]; });
  std::vector<int64_t> group_ids(times.size(), 0);
  int64_t current_group = 0;
  for (size_t i = 1; i < order.size(); ++i) {
    double prev_t = times[order[i - 1]];
    double curr_t = times[order[i]];
    if (std::abs(curr_t - prev_t) > window_ns_) {
      current_group++;
    }
    group_ids[order[i]] = current_group;
  }
  return group_ids;
}

}  // namespace pioneerml::derived
