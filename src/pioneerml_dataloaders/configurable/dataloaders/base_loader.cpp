#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

#include <sstream>
#include <stdexcept>

#include <arrow/table.h>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"
namespace pioneerml::dataloaders {

void DataLoader::AddDeriver(std::string name,
                            std::shared_ptr<data_derivers::BaseDeriver> deriver) {
  derivers_.push_back({{std::move(name)}, std::move(deriver)});
}

void DataLoader::AddDeriver(std::vector<std::string> names,
                            std::shared_ptr<data_derivers::BaseDeriver> deriver) {
  derivers_.push_back({std::move(names), std::move(deriver)});
}

TrainingBundle DataLoader::LoadTraining(
    const std::vector<std::string>& parquet_paths) const {
  throw std::runtime_error("LoadTraining(paths) not implemented for this loader.");
}

InferenceBundle DataLoader::LoadInference(
    const std::vector<std::string>& parquet_paths) const {
  throw std::runtime_error("LoadInference(paths) not implemented for this loader.");
}

std::shared_ptr<arrow::Table> DataLoader::AddDerivedColumns(
    const std::shared_ptr<arrow::Table>& table) const {
  auto current = table;
  for (size_t i = 0; i < derivers_.size(); ++i) {
    const auto& spec = derivers_[i];
    if (!spec.deriver) {
      continue;
    }
    auto arrays = spec.deriver->DeriveColumns(*current);
    if (arrays.empty()) {
      continue;
    }
    if (arrays.size() != spec.names.size()) {
      throw std::runtime_error("Derived column count does not match output names.");
    }
    for (size_t j = 0; j < spec.names.size(); ++j) {
      const auto& name = spec.names[j];
      const auto& array = arrays[j];
      if (!array) {
        continue;
      }
      if (array->length() != current->num_rows()) {
        throw std::runtime_error("Derived column has unexpected length: " + name);
      }
      auto field = arrow::field(name, array->type());
      auto chunked = std::make_shared<arrow::ChunkedArray>(array);
      int index = current->schema()->GetFieldIndex(name);
      if (index >= 0) {
        auto result = current->SetColumn(index, field, chunked);
        if (!result.ok()) {
          throw std::runtime_error(result.status().ToString());
        }
        current = result.MoveValueUnsafe();
      } else {
        auto result = current->AddColumn(current->num_columns(), field, chunked);
        if (!result.ok()) {
          throw std::runtime_error(result.status().ToString());
        }
        current = result.MoveValueUnsafe();
      }
    }
  }

  return current;
}

std::shared_ptr<arrow::Table> DataLoader::LoadAndConcatenateTables(
    const std::vector<std::string>& parquet_paths,
    bool add_derived) const {
  utils::timing::ScopedTimer total_timer("loader.load_and_concat");
  if (parquet_paths.empty()) {
    throw std::runtime_error("No parquet paths provided.");
  }
  std::vector<std::shared_ptr<arrow::Table>> tables(parquet_paths.size());
  std::vector<std::exception_ptr> errors(parquet_paths.size());

  utils::parallel::Parallel::For(0, static_cast<int64_t>(parquet_paths.size()),
                                 [&](int64_t idx) {
    const auto& path = parquet_paths[static_cast<size_t>(idx)];
    try {
      utils::timing::ScopedTimer read_timer("loader.read_table");
      auto table = LoadTable(path);
      if (add_derived) {
        utils::timing::ScopedTimer derive_timer("loader.add_derived");
        table = AddDerivedColumns(table);
      }
      tables[static_cast<size_t>(idx)] = std::move(table);
    } catch (...) {
      errors[static_cast<size_t>(idx)] = std::current_exception();
    }
  });

  for (const auto& err : errors) {
    if (err) {
      std::rethrow_exception(err);
    }
  }

  utils::timing::ScopedTimer concat_timer("loader.concat_tables");
  auto result = arrow::ConcatenateTables(tables);
  if (!result.ok()) {
    throw std::runtime_error(result.status().ToString());
  }
  auto combined = result.MoveValueUnsafe();
  utils::timing::ScopedTimer combine_timer("loader.combine_chunks");
  auto combine_result = combined->CombineChunks(arrow::default_memory_pool());
  if (!combine_result.ok()) {
    throw std::runtime_error(combine_result.status().ToString());
  }
  return combine_result.MoveValueUnsafe();
}

std::string DataLoader::JoinNames(const std::vector<std::string>& names,
                                  const std::string& sep) const {
  std::ostringstream out;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << sep;
    out << names[i];
  }
  return out.str();
}

std::vector<std::string> DataLoader::MissingColumns(
    const arrow::Table& table,
    const std::vector<std::string>& required) const {
  std::vector<std::string> missing;
  missing.reserve(required.size());
  for (const auto& name : required) {
    if (!table.GetColumnByName(name)) {
      missing.push_back(name);
    }
  }
  return missing;
}

void DataLoader::ValidateColumns(const arrow::Table& table,
                                 const std::vector<std::string>& required,
                                 const std::vector<std::string>& optional,
                                 bool require_single_chunk,
                                 const std::string& context) const {
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

std::vector<std::string> DataLoader::MergeColumns(
    const std::vector<std::string>& left,
    const std::vector<std::string>& right) const {
  std::vector<std::string> out;
  out.reserve(left.size() + right.size());
  out.insert(out.end(), left.begin(), left.end());
  out.insert(out.end(), right.begin(), right.end());
  return out;
}

DataLoader::ColumnMap DataLoader::BindColumns(const arrow::Table& table,
                                              const std::vector<std::string>& names,
                                              bool require_all,
                                              bool require_single_chunk,
                                              const std::string& context) const {
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

}  // namespace pioneerml::dataloaders
