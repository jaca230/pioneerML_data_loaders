#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"

#include <sstream>
#include <stdexcept>

#include "pioneerml_dataloaders/io/parquet_reader.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

namespace pioneerml::utils::parquet {

NumericAccessor NumericAccessor::FromArray(const std::shared_ptr<arrow::Array>& arr,
                                           const std::string& context) {
  NumericAccessor out;
  out.arr_ = arr;
  if (arr->type_id() == arrow::Type::DOUBLE) {
    auto values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(arr);
    out.d_ = values->raw_values();
    out.is_double_ = true;
    return out;
  }
  if (arr->type_id() == arrow::Type::FLOAT) {
    auto values = std::static_pointer_cast<arrow::NumericArray<arrow::FloatType>>(arr);
    out.f_ = values->raw_values();
    out.is_double_ = false;
    return out;
  }
  throw std::runtime_error("Unsupported numeric type in " + context + ".");
}

std::string JoinNames(const std::vector<std::string>& names, const std::string& sep) {
  std::ostringstream out;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << sep;
    out << names[i];
  }
  return out.str();
}

std::vector<std::string> MissingColumns(const arrow::Table& table,
                                        const std::vector<std::string>& required) {
  std::vector<std::string> missing;
  missing.reserve(required.size());
  for (const auto& name : required) {
    if (!table.GetColumnByName(name)) {
      missing.push_back(name);
    }
  }
  return missing;
}

void ValidateColumns(const arrow::Table& table,
                     const std::vector<std::string>& required,
                     const std::vector<std::string>& optional,
                     bool require_single_chunk,
                     const std::string& context) {
  auto missing = MissingColumns(table, required);
  if (!missing.empty()) {
    throw std::runtime_error(context + " missing columns: " + JoinNames(missing));
  }

  if (!require_single_chunk) {
    return;
  }

  auto check_chunks = [&](const std::string& name) {
    auto col = table.GetColumnByName(name);
    if (col && col->num_chunks() != 1) {
      throw std::runtime_error(context + " column has multiple chunks: " + name);
    }
  };

  for (const auto& name : required) check_chunks(name);
  for (const auto& name : optional) check_chunks(name);
}

std::vector<std::string> MergeColumns(const std::vector<std::string>& left,
                                      const std::vector<std::string>& right) {
  std::vector<std::string> out;
  out.reserve(left.size() + right.size());
  out.insert(out.end(), left.begin(), left.end());
  out.insert(out.end(), right.begin(), right.end());
  return out;
}

ColumnMap BindColumns(const arrow::Table& table,
                      const std::vector<std::string>& names,
                      bool require_all,
                      bool require_single_chunk,
                      const std::string& context) {
  ColumnMap out;
  out.reserve(names.size());
  std::vector<std::string> missing;
  for (const auto& name : names) {
    auto col = table.GetColumnByName(name);
    if (!col) {
      missing.push_back(name);
      continue;
    }
    if (require_single_chunk && col->num_chunks() != 1) {
      throw std::runtime_error(context + " column has multiple chunks: " + name);
    }
    out.emplace(name, std::move(col));
  }
  if (require_all && !missing.empty()) {
    throw std::runtime_error(context + " missing columns: " + JoinNames(missing));
  }
  return out;
}

std::pair<int64_t, int64_t> ParquetUtils::ListRange(const arrow::Array& list_arr,
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
      throw std::runtime_error("ParquetUtils::ListRange: unsupported list array type");
  }
}

int64_t ParquetUtils::ListLength(const arrow::Array& list_arr, int64_t idx) const {
  auto range = ListRange(list_arr, idx);
  return range.second - range.first;
}

std::shared_ptr<arrow::Array> ParquetUtils::ListValues(const arrow::Array& list_arr) const {
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
      throw std::runtime_error("ParquetUtils::ListValues: unsupported list array type");
  }
}

std::shared_ptr<arrow::Table> MergeTablesByColumns(
    const std::vector<std::shared_ptr<arrow::Table>>& tables) {
  utils::timing::ScopedTimer total_timer("parquet.merge_tables_by_columns");
  if (tables.empty()) {
    throw std::runtime_error("No tables provided to merge.");
  }

  auto merged = tables.front();
  if (!merged) {
    throw std::runtime_error("Encountered null base table while merging.");
  }

  for (size_t i = 1; i < tables.size(); ++i) {
    const auto& extra = tables[i];
    if (!extra) {
      throw std::runtime_error("Encountered null auxiliary table while merging.");
    }
    if (extra->num_rows() != merged->num_rows()) {
      throw std::runtime_error("Parquet table row count mismatch while merging shard inputs.");
    }

    for (int c = 0; c < extra->num_columns(); ++c) {
      const auto& field = extra->schema()->field(c);
      const auto& name = field->name();
      if (merged->schema()->GetFieldIndex(name) >= 0) {
        throw std::runtime_error("Duplicate column while merging shard inputs: " + name);
      }
      auto result = merged->AddColumn(merged->num_columns(), field, extra->column(c));
      if (!result.ok()) {
        throw std::runtime_error(result.status().ToString());
      }
      merged = result.MoveValueUnsafe();
    }
  }
  return merged;
}

std::shared_ptr<arrow::Table> ConcatenateTablesByRows(
    const std::vector<std::shared_ptr<arrow::Table>>& tables) {
  utils::timing::ScopedTimer total_timer("parquet.concatenate_tables_by_rows");
  if (tables.empty()) {
    throw std::runtime_error("No tables provided to concatenate.");
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result;
  {
    utils::timing::ScopedTimer concat_timer("parquet.concatenate_tables_by_rows.concat");
    result = arrow::ConcatenateTables(tables);
  }
  if (!result.ok()) {
    throw std::runtime_error(result.status().ToString());
  }
  auto combined = result.MoveValueUnsafe();
  arrow::Result<std::shared_ptr<arrow::Table>> combine_result;
  {
    utils::timing::ScopedTimer combine_chunks_timer(
        "parquet.concatenate_tables_by_rows.combine_chunks");
    combine_result = combined->CombineChunks(arrow::default_memory_pool());
  }
  if (!combine_result.ok()) {
    throw std::runtime_error(combine_result.status().ToString());
  }
  return combine_result.MoveValueUnsafe();
}

std::shared_ptr<arrow::Table> LoadAndMergeTablesByColumns(
    const std::vector<std::string>& parquet_paths) {
  std::vector<std::vector<std::string>> no_projection(parquet_paths.size());
  return LoadAndMergeTablesByColumns(parquet_paths, no_projection);
}

std::shared_ptr<arrow::Table> LoadAndMergeTablesByColumns(
    const std::vector<std::string>& parquet_paths,
    const std::vector<std::vector<std::string>>& columns_by_path) {
  utils::timing::ScopedTimer total_timer("parquet.load_and_merge_tables_by_columns");
  if (parquet_paths.empty()) {
    throw std::runtime_error("No parquet paths provided.");
  }
  if (!columns_by_path.empty() && columns_by_path.size() != parquet_paths.size()) {
    throw std::runtime_error("columns_by_path size must match parquet_paths size.");
  }
  pioneerml::io::ParquetReader reader;
  std::vector<std::shared_ptr<arrow::Table>> tables;
  tables.reserve(parquet_paths.size());
  for (size_t i = 0; i < parquet_paths.size(); ++i) {
    const auto& path = parquet_paths[i];
    utils::timing::ScopedTimer read_one_timer(
        "parquet.load_and_merge_tables_by_columns.read_file");
    if (columns_by_path.empty()) {
      tables.push_back(reader.ReadTable(path));
    } else {
      tables.push_back(reader.ReadTable(path, columns_by_path[i]));
    }
  }
  utils::timing::ScopedTimer merge_timer(
      "parquet.load_and_merge_tables_by_columns.merge_columns");
  return MergeTablesByColumns(tables);
}

}  // namespace pioneerml::utils::parquet
