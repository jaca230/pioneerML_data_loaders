#pragma once

#include <memory>
#include <vector>

namespace arrow {
class Table;
class Array;
}  // namespace arrow

namespace pioneerml::data_derivers {

// Contract for components that emit a derived column (Arrow Array) for a table.
class BaseDeriver {
 public:
  virtual ~BaseDeriver() = default;

  // Produce a derived column aligned with the input table rows.
  // The returned array must have length == table.num_rows().
  virtual std::shared_ptr<arrow::Array> DeriveColumn(const arrow::Table& table) const = 0;
};

// Contract for components that emit multiple derived columns in a single pass.
class MultiDeriver {
 public:
  virtual ~MultiDeriver() = default;

  // Produce derived columns aligned with the input table rows.
  // Each returned array must have length == table.num_rows().
  virtual std::vector<std::shared_ptr<arrow::Array>> DeriveColumns(
      const arrow::Table& table) const = 0;
};

}  // namespace pioneerml::data_derivers
