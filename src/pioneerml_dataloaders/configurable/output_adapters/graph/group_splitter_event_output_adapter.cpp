#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_splitter_event_output_adapter.h"

#include <stdexcept>

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace pioneerml::output_adapters::graph {

std::shared_ptr<arrow::Array> GroupSplitterEventOutputAdapter::BuildNodeListColumn(
    const float* pred_raw,
    int64_t total_nodes,
    const int64_t* node_ptr,
    int64_t num_graphs,
    int class_index) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* value_builder = static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = node_ptr[graph_idx];
    const int64_t end = node_ptr[graph_idx + 1];
    if (start < 0 || end < start || end > total_nodes) {
      throw std::runtime_error("Invalid node_ptr while building GroupSplitterEvent output.");
    }
    for (int64_t node = start; node < end; ++node) {
      const int64_t idx = node * 3 + class_index;
      status = value_builder->Append(pred_raw[idx]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize splitter-event node prediction list column.");
  }
  return out;
}

std::shared_ptr<arrow::Array> GroupSplitterEventOutputAdapter::BuildTimeGroupListColumn(
    const int64_t* tg_raw,
    int64_t total_nodes,
    const int64_t* node_ptr,
    int64_t num_graphs) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>());
  auto* value_builder = static_cast<arrow::Int64Builder*>(list_builder.value_builder());

  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = node_ptr[graph_idx];
    const int64_t end = node_ptr[graph_idx + 1];
    if (start < 0 || end < start || end > total_nodes) {
      throw std::runtime_error("Invalid node_ptr while building splitter-event time groups.");
    }
    for (int64_t node = start; node < end; ++node) {
      status = value_builder->Append(tg_raw[node]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize splitter-event time-group list column.");
  }
  return out;
}

std::shared_ptr<arrow::Table> GroupSplitterEventOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& node_pred,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  if (!node_pred || !node_ptr || !time_group_ids || !graph_event_ids) {
    throw std::runtime_error("Missing required arrays for GroupSplitterEvent output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(*node_pred);
  const auto& node_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*node_ptr);
  const auto& tg_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*time_group_ids);
  const int64_t total_nodes = pred_arr.length() / 3;
  const int64_t num_graphs = node_ptr_arr.length() - 1;

  const float* pred_raw = pred_arr.raw_values();
  const int64_t* node_ptr_raw = node_ptr_arr.raw_values();
  const int64_t* tg_raw = tg_arr.raw_values();

  auto pred_pion = BuildNodeListColumn(pred_raw, total_nodes, node_ptr_raw, num_graphs, 0);
  auto pred_muon = BuildNodeListColumn(pred_raw, total_nodes, node_ptr_raw, num_graphs, 1);
  auto pred_mip = BuildNodeListColumn(pred_raw, total_nodes, node_ptr_raw, num_graphs, 2);
  auto tg_list = BuildTimeGroupListColumn(tg_raw, total_nodes, node_ptr_raw, num_graphs);

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("event_id", arrow::int64()),
      arrow::field("time_group_ids", tg_list->type()),
      arrow::field("pred_hit_pion", pred_pion->type()),
      arrow::field("pred_hit_muon", pred_muon->type()),
      arrow::field("pred_hit_mip", pred_mip->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {
      graph_event_ids,
      tg_list,
      pred_pion,
      pred_muon,
      pred_mip,
  };

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void GroupSplitterEventOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& node_pred,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  auto table = BuildEventTable(node_pred, node_ptr, time_group_ids, graph_event_ids);
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
