#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pioneerml_dataloaders/data_derivers/base_deriver.h"

namespace pioneerml::data_derivers {

class TimeGrouper : public BaseDeriver {
 public:
  explicit TimeGrouper(double window_ns, std::string time_column = "hits_time")
      : window_ns_(window_ns), time_column_(std::move(time_column)) {}

  std::shared_ptr<arrow::Array> DeriveColumn(const arrow::Table& table) const override;

 private:
  std::vector<int64_t> Compute(const std::vector<double>& times) const;

  double window_ns_;
  std::string time_column_;
};

}  // namespace pioneerml::data_derivers
