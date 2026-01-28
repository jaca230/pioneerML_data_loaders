#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"

namespace pioneerml::utils::parquet {

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

}  // namespace pioneerml::utils::parquet
