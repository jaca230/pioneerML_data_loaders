#include "pioneerml_dataloaders/dataloaders/base_loader.h"

#include <sstream>
#include <stdexcept>

namespace pioneerml::dataloaders {

void DataLoader::AddDeriver(std::string name,
                            std::shared_ptr<data_derivers::BaseDeriver> deriver) {
  derivers_.push_back({{std::move(name)}, std::move(deriver)});
}

void DataLoader::AddDeriver(std::vector<std::string> names,
                            std::shared_ptr<data_derivers::BaseDeriver> deriver) {
  derivers_.push_back({std::move(names), std::move(deriver)});
}

std::shared_ptr<arrow::Table> DataLoader::AddDerivedColumns(
    const std::shared_ptr<arrow::Table>& table) const {
  auto current = table;
  for (const auto& spec : derivers_) {
    if (!spec.deriver) {
      continue;
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    if (spec.names.size() == 1U) {
      auto array = spec.deriver->DeriveColumn(*current);
      if (!array) {
        continue;
      }
      arrays.push_back(std::move(array));
    } else {
      auto* multi = dynamic_cast<data_derivers::MultiDeriver*>(spec.deriver.get());
      if (!multi) {
        throw std::runtime_error("Deriver does not support multiple outputs.");
      }
      arrays = multi->DeriveColumns(*current);
    }

    if (arrays.size() != spec.names.size()) {
      throw std::runtime_error("Derived column count does not match output names.");
    }

    for (size_t i = 0; i < spec.names.size(); ++i) {
      const auto& name = spec.names[i];
      const auto& array = arrays[i];
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
