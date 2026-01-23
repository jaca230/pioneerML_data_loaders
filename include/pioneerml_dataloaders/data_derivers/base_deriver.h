#pragma once

#include <memory>

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

}  // namespace pioneerml::data_derivers
