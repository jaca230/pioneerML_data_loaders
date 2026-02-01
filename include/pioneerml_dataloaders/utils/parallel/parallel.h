#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace pioneerml::utils::parallel {

class Parallel {
 public:
  using Index = int64_t;
  using Fn = std::function<void(Index)>;

  static void For(Index begin, Index end, const Fn& fn);
  static std::vector<Index> PrefixSum(const std::vector<Index>& counts);
};

}  // namespace pioneerml::utils::parallel
