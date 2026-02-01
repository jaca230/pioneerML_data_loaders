#include "pioneerml_dataloaders/utils/parallel/parallel.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

namespace pioneerml::utils::parallel {

void Parallel::For(Index begin, Index end, const Fn& fn) {
  if (end <= begin) {
    return;
  }
  tbb::parallel_for(tbb::blocked_range<Index>(begin, end),
                    [&](const tbb::blocked_range<Index>& range) {
                      for (Index i = range.begin(); i != range.end(); ++i) {
                        fn(i);
                      }
                    });
}

std::vector<Parallel::Index> Parallel::PrefixSum(const std::vector<Index>& counts) {
  std::vector<Index> out(counts.size() + 1, 0);
  if (counts.empty()) {
    return out;
  }

  tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, counts.size()),
      Index{0},
      [&](const tbb::blocked_range<size_t>& range, Index sum, bool is_final) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          sum += counts[i];
          if (is_final) {
            out[i + 1] = sum;
          }
        }
        return sum;
      },
      std::plus<Index>());

  return out;
}

}  // namespace pioneerml::utils::parallel
