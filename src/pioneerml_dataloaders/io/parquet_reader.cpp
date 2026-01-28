#include "pioneerml_dataloaders/io/parquet_reader.h"

#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <stdexcept>

namespace pioneerml::io {

std::shared_ptr<arrow::Table> ParquetReader::ReadTable(const std::string& path) const {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  auto open_result = arrow::io::ReadableFile::Open(path);
  if (!open_result.ok()) {
    throw std::runtime_error(open_result.status().ToString());
  }
  infile = open_result.ValueOrDie();

  auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
  if (!reader_result.ok()) {
    throw std::runtime_error(reader_result.status().ToString());
  }
  auto reader = std::move(reader_result).ValueOrDie();

  std::shared_ptr<arrow::Table> table;
  auto status = reader->ReadTable(&table);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
  return table;
}

}  // namespace pioneerml::io
