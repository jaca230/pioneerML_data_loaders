#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <memory>
#include <string>
#include <vector>

namespace pioneerml::io {

std::shared_ptr<arrow::Table> ReadParquet(const std::string& path);

// Convert a ListArray element to a std::vector<OutType> without copying array buffers.
template <typename ArrowType, typename OutType>
std::vector<OutType> ListToVector(const arrow::ListArray& list_arr, int64_t idx) {
  std::vector<OutType> out;
  auto offsets = list_arr.raw_value_offsets();
  int32_t start = offsets[idx];
  int32_t end = offsets[idx + 1];
  auto values = std::static_pointer_cast<arrow::NumericArray<ArrowType>>(list_arr.values());
  const auto* raw = values->raw_values();
  out.reserve(end - start);
  for (int32_t i = start; i < end; ++i) {
    out.push_back(static_cast<OutType>(raw[i]));
  }
  return out;
}

template <typename T>
inline const T* GetScalarPtr(const arrow::ChunkedArray& arr, int64_t row) {
  auto chunk = arr.chunk(0);
  const auto& typed = static_cast<const arrow::NumericArray<typename arrow::CTypeTraits<T>::ArrowType>&>(*chunk);
  return typed.raw_values() + row;
}

}  // namespace pioneerml::io
