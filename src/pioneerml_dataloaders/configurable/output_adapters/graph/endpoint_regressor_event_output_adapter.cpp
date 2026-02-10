#include "pioneerml_dataloaders/configurable/output_adapters/graph/endpoint_regressor_event_output_adapter.h"

#include <stdexcept>

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace pioneerml::output_adapters::graph {

std::shared_ptr<arrow::Array> EndpointRegressorEventOutputAdapter::BuildGroupListColumn(
    const float* pred_raw,
    int64_t total_groups,
    const int64_t* group_ptr_raw,
    int64_t num_graphs,
    int feature_index) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* value_builder = static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = group_ptr_raw[graph_idx];
    const int64_t end = group_ptr_raw[graph_idx + 1];
    if (start < 0 || end < start || end > total_groups) {
      throw std::runtime_error("Invalid group_ptr while building endpoint output.");
    }
    for (int64_t group = start; group < end; ++group) {
      const int64_t idx = group * 6 + feature_index;
      status = value_builder->Append(pred_raw[idx]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize endpoint prediction list column.");
  }
  return out;
}

std::shared_ptr<arrow::Table> EndpointRegressorEventOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& group_ptr,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  if (!group_pred || !group_ptr || !graph_event_ids) {
    throw std::runtime_error("Missing required arrays for EndpointRegressor output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(*group_pred);
  const auto& group_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*group_ptr);
  const int64_t total_groups = pred_arr.length() / 6;
  const int64_t num_graphs = group_ptr_arr.length() - 1;

  const float* pred_raw = pred_arr.raw_values();
  const int64_t* group_ptr_raw = group_ptr_arr.raw_values();

  auto start_x = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 0);
  auto start_y = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 1);
  auto start_z = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 2);
  auto end_x = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 3);
  auto end_y = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 4);
  auto end_z = BuildGroupListColumn(pred_raw, total_groups, group_ptr_raw, num_graphs, 5);

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("event_id", arrow::int64()),
      arrow::field("pred_group_start_x", start_x->type()),
      arrow::field("pred_group_start_y", start_y->type()),
      arrow::field("pred_group_start_z", start_z->type()),
      arrow::field("pred_group_end_x", end_x->type()),
      arrow::field("pred_group_end_y", end_y->type()),
      arrow::field("pred_group_end_z", end_z->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {
      graph_event_ids,
      start_x,
      start_y,
      start_z,
      end_x,
      end_y,
      end_z,
  };

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void EndpointRegressorEventOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& group_ptr,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  auto table = BuildEventTable(group_pred, group_ptr, graph_event_ids);
  auto out_result = arrow::io::FileOutputStream::Open(output_path);
  if (!out_result.ok()) {
    throw std::runtime_error(out_result.status().ToString());
  }
  auto out = out_result.MoveValueUnsafe();
  auto result = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out);
  if (!result.ok()) {
    throw std::runtime_error(result.ToString());
  }
}

}  // namespace pioneerml::output_adapters::graph
