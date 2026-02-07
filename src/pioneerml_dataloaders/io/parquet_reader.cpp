#include "pioneerml_dataloaders/io/parquet_reader.h"

#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <stdexcept>
#include <unordered_map>

#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

namespace pioneerml::io {

std::shared_ptr<arrow::Table> ParquetReader::ReadTable(
    const std::string& path,
    const std::vector<std::string>& columns) const {
  utils::timing::ScopedTimer total_timer("parquet.read_table");
  std::shared_ptr<arrow::io::ReadableFile> infile;
  {
    utils::timing::ScopedTimer open_file_timer("parquet.read_table.open_file");
    auto open_result = arrow::io::ReadableFile::Open(path);
    if (!open_result.ok()) {
      throw std::runtime_error(open_result.status().ToString());
    }
    infile = open_result.ValueOrDie();
  }

  std::unique_ptr<parquet::arrow::FileReader> reader;
  {
    utils::timing::ScopedTimer open_reader_timer("parquet.read_table.open_reader");
    auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
      throw std::runtime_error(reader_result.status().ToString());
    }
    reader = std::move(reader_result).ValueOrDie();
  }
  std::shared_ptr<arrow::Table> table;
  {
    utils::timing::ScopedTimer read_table_timer("parquet.read_table.read");
    parquet::arrow::FileReader* file_reader = reader.get();
    arrow::Status status;
    if (columns.empty()) {
      status = file_reader->ReadTable(&table);
    } else {
      std::shared_ptr<arrow::Schema> schema;
      status = file_reader->GetSchema(&schema);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
      std::unordered_map<std::string, int> index_by_name;
      index_by_name.reserve(static_cast<size_t>(schema->num_fields()));
      for (int i = 0; i < schema->num_fields(); ++i) {
        index_by_name.emplace(schema->field(i)->name(), i);
      }
      std::vector<int> column_indices;
      column_indices.reserve(columns.size());
      for (const auto& name : columns) {
        auto it = index_by_name.find(name);
        if (it == index_by_name.end()) {
          throw std::runtime_error("Parquet column not found in " + path + ": " + name);
        }
        column_indices.push_back(it->second);
      }
      status = file_reader->ReadTable(column_indices, &table);
    }
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  return table;
}

}  // namespace pioneerml::io
