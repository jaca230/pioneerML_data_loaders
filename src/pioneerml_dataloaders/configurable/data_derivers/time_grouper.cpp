#include "pioneerml_dataloaders/configurable/data_derivers/time_grouper.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"

namespace pioneerml::data_derivers {

void TimeGrouper::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("window_ns")) {
    window_ns_ = cfg.at("window_ns").get<double>();
  }
  if (cfg.contains("time_column")) {
    time_column_ = cfg.at("time_column").get<std::string>();
  }
}

std::vector<std::shared_ptr<arrow::Array>> TimeGrouper::DeriveColumns(
    const arrow::Table& table) const {
  auto col = table.GetColumnByName(time_column_);
  if (!col) {
    return {std::make_shared<arrow::NullArray>(table.num_rows())};
  }
  const auto& list = static_cast<const arrow::ListArray&>(*col->chunk(0));

  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>(arrow::default_memory_pool()));
  auto* int_builder = static_cast<arrow::Int64Builder*>(list_builder.value_builder());

  int64_t rows = table.num_rows();
  auto offsets = list.raw_value_offsets();
  const auto& values_arr = list.values();
  const double* raw = nullptr;
  std::vector<double> converted;

  switch (values_arr->type_id()) {
    case arrow::Type::DOUBLE: {
      auto values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(values_arr);
      raw = values->raw_values();
      break;
    }
    case arrow::Type::FLOAT: {
      auto values = std::static_pointer_cast<arrow::NumericArray<arrow::FloatType>>(values_arr);
      converted.assign(values->length(), 0.0);
      const float* raw_f = values->raw_values();
      for (int64_t i = 0; i < values->length(); ++i) {
        converted[static_cast<size_t>(i)] = static_cast<double>(raw_f[i]);
      }
      raw = converted.data();
      break;
    }
    default:
      throw std::runtime_error("TimeGrouper expects hits_time to be float or double.");
  }

  std::vector<std::vector<int64_t>> derived(rows);
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    auto start = offsets[row];
    auto end = offsets[row + 1];
    std::vector<double> times;
    times.reserve(end - start);
    for (int32_t i = start; i < end; ++i) {
      times.push_back(raw[i]);
    }
    derived[row] = Compute(times);
  });

  for (int64_t row = 0; row < rows; ++row) {
    auto st = list_builder.Append();
    if (!st.ok()) {
      throw std::runtime_error(st.ToString());
    }
    for (auto g : derived[row]) {
      st = int_builder->Append(g);
      if (!st.ok()) {
        throw std::runtime_error(st.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  auto st_finish = list_builder.Finish(&out);
  if (!st_finish.ok()) {
    throw std::runtime_error(st_finish.ToString());
  }
  return {std::move(out)};
}

std::vector<int64_t> TimeGrouper::Compute(const std::vector<double>& times) const {
  if (times.empty()) return {};
  std::vector<int64_t> order(times.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return times[a] < times[b]; });
  std::vector<int64_t> group_ids(times.size(), 0);
  int64_t current_group = 0;
  for (size_t i = 1; i < order.size(); ++i) {
    double prev_t = times[order[i - 1]];
    double curr_t = times[order[i]];
    if (std::abs(curr_t - prev_t) > window_ns_) {
      current_group++;
    }
    group_ids[order[i]] = current_group;
  }
  return group_ids;
}

}  // namespace pioneerml::data_derivers
