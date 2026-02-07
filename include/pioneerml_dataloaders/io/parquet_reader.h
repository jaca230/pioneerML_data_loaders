#pragma once

#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

namespace pioneerml::io {

class ParquetReader {
 public:
  std::shared_ptr<arrow::Table> ReadTable(const std::string& path,
                                          const std::vector<std::string>& columns = {}) const;
};

}  // namespace pioneerml::io
