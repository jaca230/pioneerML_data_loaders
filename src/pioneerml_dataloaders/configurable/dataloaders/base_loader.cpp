#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

#include <stdexcept>

#include <arrow/table.h>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"
#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"
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

std::shared_ptr<arrow::Table> DataLoader::PrepareTable(
    const std::shared_ptr<arrow::Table>& table,
    bool add_derived) const {
  utils::timing::ScopedTimer total_timer("loader.prepare_table");
  if (!table) {
    throw std::runtime_error("Null table provided.");
  }
  auto prepared = table;
  if (add_derived) {
    utils::timing::ScopedTimer derive_timer("loader.add_derived");
    prepared = AddDerivedColumns(prepared);
  }
  utils::timing::ScopedTimer combine_timer("loader.combine_chunks");
  auto combine_result = prepared->CombineChunks(arrow::default_memory_pool());
  if (!combine_result.ok()) {
    throw std::runtime_error(combine_result.status().ToString());
  }
  return combine_result.MoveValueUnsafe();
}

DataLoader::NumericAccessor DataLoader::MakeNumericAccessor(
    const std::shared_ptr<arrow::Array>& arr,
    const std::string& context) const {
  return utils::parquet::NumericAccessor::FromArray(arr, context);
}

std::shared_ptr<arrow::Buffer> DataLoader::AllocBuffer(int64_t bytes) const {
  auto result = arrow::AllocateBuffer(bytes);
  if (!result.ok()) {
    throw std::runtime_error(result.status().ToString());
  }
  return std::shared_ptr<arrow::Buffer>(std::move(result).ValueOrDie());
}

std::shared_ptr<arrow::Array> DataLoader::MakeArray(
    std::shared_ptr<arrow::Buffer> buffer,
    const std::shared_ptr<arrow::DataType>& type,
    int64_t length) const {
  auto data = arrow::ArrayData::Make(type, length, {nullptr, std::move(buffer)});
  return arrow::MakeArray(std::move(data));
}

double DataLoader::ResolveCoordinateForView(const NumericAccessor& x_values,
                                            const NumericAccessor& y_values,
                                            int32_t view,
                                            int64_t idx) const {
  if (view == 0) {
    if (x_values.IsValid(idx)) {
      return x_values.Value(idx);
    }
    if (y_values.IsValid(idx)) {
      return y_values.Value(idx);
    }
    return 0.0;
  }
  if (y_values.IsValid(idx)) {
    return y_values.Value(idx);
  }
  if (x_values.IsValid(idx)) {
    return x_values.Value(idx);
  }
  return 0.0;
}

std::vector<int64_t> DataLoader::BuildOffsets(const std::vector<int64_t>& counts) const {
  return utils::parallel::Parallel::PrefixSum(counts);
}

void DataLoader::FillPointerArrayFromOffsets(const std::vector<int64_t>& offsets,
                                             int64_t* out_ptr) const {
  for (size_t i = 0; i < offsets.size(); ++i) {
    out_ptr[i] = offsets[i];
  }
}

}  // namespace pioneerml::dataloaders
