#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <memory>
#include <string>
#include <vector>

namespace pioneerml::io {

std::shared_ptr<arrow::Table> ReadParquet(const std::string& path);

inline std::pair<int64_t, int64_t> ListRange(const arrow::Array& list_arr, int64_t idx) {
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

inline int64_t ListLength(const arrow::Array& list_arr, int64_t idx) {
  auto range = ListRange(list_arr, idx);
  return range.second - range.first;
}

inline std::shared_ptr<arrow::Array> ListValues(const arrow::Array& list_arr) {
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

// Convert a (List/LargeList) element to a std::vector<OutType> without copying array buffers.
template <typename ArrowType, typename OutType>
std::vector<OutType> ListToVector(const arrow::Array& list_arr, int64_t idx) {
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
inline const T* GetScalarPtr(const arrow::ChunkedArray& arr, int64_t row) {
  auto chunk = arr.chunk(0);
  const auto& typed = static_cast<const arrow::NumericArray<typename arrow::CTypeTraits<T>::ArrowType>&>(*chunk);
  return typed.raw_values() + row;
}

}  // namespace pioneerml::io
