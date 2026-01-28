#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <memory>
#include <string>
#include <vector>

namespace pioneerml::io {

class ParquetManager {
 public:
  static ParquetManager& Instance();

  std::shared_ptr<arrow::Table> ReadParquet(const std::string& path) const;

  std::pair<int64_t, int64_t> ListRange(const arrow::Array& list_arr,
                                        int64_t idx) const;
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

 private:
  ParquetManager() = default;
  ParquetManager(const ParquetManager&) = delete;
  ParquetManager& operator=(const ParquetManager&) = delete;
};

}  // namespace pioneerml::io
