#include "pioneerml_dataloaders/io/parquet_manager.h"

#include <arrow/io/file.h>
#include <arrow/result.h>
#include <parquet/arrow/reader.h>

#include <stdexcept>

namespace pioneerml::io {

ParquetManager& ParquetManager::Instance() {
  static ParquetManager manager;
  return manager;
}

std::shared_ptr<arrow::Table> ParquetManager::ReadParquet(const std::string& path) const {
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

std::pair<int64_t, int64_t> ParquetManager::ListRange(const arrow::Array& list_arr,
                                                      int64_t idx) const {
  switch (list_arr.type_id()) {
    case arrow::Type::LIST: {
      const auto& list = static_cast<const arrow::ListArray&>(list_arr);
      auto start = list.value_offset(idx);
      auto end = start + list.value_length(idx);
      return {start, end};
    }
    case arrow::Type::LARGE_LIST: {
      const auto& list = static_cast<const arrow::LargeListArray&>(list_arr);
      auto start = list.value_offset(idx);
      auto end = start + list.value_length(idx);
      return {start, end};
    }
    default:
      throw std::runtime_error("ListRange: unsupported list array type");
  }
}

int64_t ParquetManager::ListLength(const arrow::Array& list_arr, int64_t idx) const {
  auto range = ListRange(list_arr, idx);
  return range.second - range.first;
}

std::shared_ptr<arrow::Array> ParquetManager::ListValues(const arrow::Array& list_arr) const {
  switch (list_arr.type_id()) {
    case arrow::Type::LIST: {
      const auto& list = static_cast<const arrow::ListArray&>(list_arr);
      return list.values();
    }
    case arrow::Type::LARGE_LIST: {
      const auto& list = static_cast<const arrow::LargeListArray&>(list_arr);
      return list.values();
    }
    default:
      throw std::runtime_error("ListValues: unsupported list array type");
  }
}

}  // namespace pioneerml::io
