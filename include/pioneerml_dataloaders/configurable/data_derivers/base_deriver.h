#pragma once

#include <memory>
#include <vector>

#include "pioneerml_dataloaders/configurable/configurable.h"

namespace arrow {
class Table;
class Array;
}  // namespace arrow

namespace pioneerml::data_derivers {

// Contract for components that emit derived columns (Arrow Arrays) for a table.
class BaseDeriver : public pioneerml::configurable::Configurable {
 public:
  virtual ~BaseDeriver() = default;

  // Produce derived columns aligned with the input table rows.
  // Each returned array must have length == table.num_rows().
  virtual std::vector<std::shared_ptr<arrow::Array>> DeriveColumns(
      const arrow::Table& table) const = 0;
};

}  // namespace pioneerml::data_derivers
