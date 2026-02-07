#pragma once

#include <arrow/api.h>

#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pioneerml::utils::parquet {

class ParquetUtils {
 public:
  std::pair<int64_t, int64_t> ListRange(const arrow::Array& list_arr, int64_t idx) const;
  int64_t ListLength(const arrow::Array& list_arr, int64_t idx) const;
  std::shared_ptr<arrow::Array> ListValues(const arrow::Array& list_arr) const;

  template <typename ArrowType, typename OutType>
  std::vector<OutType> ListToVector(const arrow::Array& list_arr, int64_t idx) const {
    auto range = ListRange(list_arr, idx);
    auto values = std::static_pointer_cast<arrow::NumericArray<ArrowType>>(ListValues(list_arr));
    const auto* raw = values->raw_values();
    std::vector<OutType> out;
    out.reserve(static_cast<size_t>(range.second - range.first));
    for (int64_t i = range.first; i < range.second; ++i) {
      out.push_back(static_cast<OutType>(raw[i]));
    }
    return out;
  }

  template <typename T>
  const T* GetScalarPtr(const arrow::ChunkedArray& arr, int64_t row) const {
    auto chunk = arr.chunk(0);
    const auto& typed =
        static_cast<const arrow::NumericArray<typename arrow::CTypeTraits<T>::ArrowType>&>(
            *chunk);
    return typed.raw_values() + row;
  }
};

using ColumnMap = std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>>;

class NumericAccessor {
 public:
  NumericAccessor() = default;

  static NumericAccessor FromArray(const std::shared_ptr<arrow::Array>& arr,
                                   const std::string& context);

  bool IsValid(int64_t idx) const { return arr_ && arr_->IsValid(idx); }
  double Value(int64_t idx) const { return is_double_ ? d_[idx] : static_cast<double>(f_[idx]); }

 private:
  std::shared_ptr<arrow::Array> arr_;
  const float* f_{nullptr};
  const double* d_{nullptr};
  bool is_double_{false};
};

std::string JoinNames(const std::vector<std::string>& names,
                      const std::string& sep = ", ");

std::vector<std::string> MissingColumns(const arrow::Table& table,
                                        const std::vector<std::string>& required);

void ValidateColumns(const arrow::Table& table,
                     const std::vector<std::string>& required,
                     const std::vector<std::string>& optional,
                     bool require_single_chunk,
                     const std::string& context);

std::vector<std::string> MergeColumns(const std::vector<std::string>& left,
                                      const std::vector<std::string>& right);

ColumnMap BindColumns(const arrow::Table& table,
                      const std::vector<std::string>& names,
                      bool require_all,
                      bool require_single_chunk,
                      const std::string& context);

std::shared_ptr<arrow::Table> MergeTablesByColumns(
    const std::vector<std::shared_ptr<arrow::Table>>& tables);

std::shared_ptr<arrow::Table> ConcatenateTablesByRows(
    const std::vector<std::shared_ptr<arrow::Table>>& tables);

std::shared_ptr<arrow::Table> LoadAndMergeTablesByColumns(
    const std::vector<std::string>& parquet_paths);
std::shared_ptr<arrow::Table> LoadAndMergeTablesByColumns(
    const std::vector<std::string>& parquet_paths,
    const std::vector<std::vector<std::string>>& columns_by_path);

}  // namespace pioneerml::utils::parquet
