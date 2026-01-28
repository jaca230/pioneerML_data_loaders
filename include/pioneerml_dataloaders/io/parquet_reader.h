#pragma once

#include <arrow/api.h>

#include <memory>
#include <string>

namespace pioneerml::io {

class ParquetReader {
 public:
  std::shared_ptr<arrow::Table> ReadTable(const std::string& path) const;
};

}  // namespace pioneerml::io
