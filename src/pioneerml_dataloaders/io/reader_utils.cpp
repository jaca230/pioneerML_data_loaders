#include "pioneerml_dataloaders/io/reader_utils.h"

#include <stdexcept>

namespace pioneerml::io {

std::shared_ptr<arrow::Table> ReadParquet(const std::string& path) {
  auto infile_res = arrow::io::ReadableFile::Open(path, arrow::default_memory_pool());
  if (!infile_res.ok()) {
    throw std::runtime_error(infile_res.status().ToString());
  }
  auto infile = std::move(infile_res).MoveValueUnsafe();

  PARQUET_ASSIGN_OR_THROW(auto reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
  return table;
}

}  // namespace pioneerml::io
