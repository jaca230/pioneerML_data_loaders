#include "pioneerml_dataloaders/configurable/output_adapters/graph/event_splitter_event_output_adapter.h"

#include <stdexcept>

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace pioneerml::output_adapters::graph {

std::shared_ptr<arrow::Array> EventSplitterEventOutputAdapter::BuildEdgeFloatListColumn(
    const float* pred_raw,
    int64_t total_edges,
    const int64_t* edge_ptr,
    int64_t num_graphs) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* value_builder = static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = edge_ptr[graph_idx];
    const int64_t end = edge_ptr[graph_idx + 1];
    if (start < 0 || end < start || end > total_edges) {
      throw std::runtime_error("Invalid edge_ptr while building EventSplitterEvent output.");
    }
    for (int64_t edge = start; edge < end; ++edge) {
      status = value_builder->Append(pred_raw[edge]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize event-splitter edge prediction list column.");
  }
  return out;
}

std::shared_ptr<arrow::Array> EventSplitterEventOutputAdapter::BuildEdgeIndexListColumn(
    const int64_t* edge_index_raw,
    int64_t total_edges,
    const int64_t* edge_ptr,
    const int64_t* node_ptr,
    int64_t num_graphs,
    int endpoint) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>());
  auto* value_builder = static_cast<arrow::Int64Builder*>(list_builder.value_builder());

  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = edge_ptr[graph_idx];
    const int64_t end = edge_ptr[graph_idx + 1];
    const int64_t node_base = node_ptr[graph_idx];
    if (start < 0 || end < start || end > total_edges) {
      throw std::runtime_error("Invalid edge_ptr while building EventSplitterEvent output indices.");
    }
    for (int64_t edge = start; edge < end; ++edge) {
      const int64_t global_idx = edge_index_raw[edge * 2 + endpoint];
      const int64_t local_idx = global_idx - node_base;
      status = value_builder->Append(local_idx);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize event-splitter edge index list column.");
  }
  return out;
}

std::shared_ptr<arrow::Array> EventSplitterEventOutputAdapter::BuildNodeIntListColumn(
    const int64_t* values_raw,
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
      throw std::runtime_error("Invalid node_ptr while building EventSplitterEvent node list column.");
    }
    for (int64_t node = start; node < end; ++node) {
      status = value_builder->Append(values_raw[node]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize event-splitter node list column.");
  }
  return out;
}

std::shared_ptr<arrow::Table> EventSplitterEventOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& edge_pred,
    const std::shared_ptr<arrow::Array>& edge_ptr,
    const std::shared_ptr<arrow::Array>& edge_index,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  if (!edge_pred || !edge_ptr || !edge_index || !node_ptr || !time_group_ids || !graph_event_ids) {
    throw std::runtime_error("Missing required arrays for EventSplitterEvent output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(*edge_pred);
  const auto& edge_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*edge_ptr);
  const auto& edge_index_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*edge_index);
  const auto& node_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*node_ptr);
  const auto& tg_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*time_group_ids);

  const int64_t total_edges = pred_arr.length();
  const int64_t total_nodes = tg_arr.length();
  const int64_t num_graphs = edge_ptr_arr.length() - 1;

  const float* pred_raw = pred_arr.raw_values();
  const int64_t* edge_ptr_raw = edge_ptr_arr.raw_values();
  const int64_t* edge_index_raw = edge_index_arr.raw_values();
  const int64_t* node_ptr_raw = node_ptr_arr.raw_values();
  const int64_t* tg_raw = tg_arr.raw_values();

  auto pred_list = BuildEdgeFloatListColumn(pred_raw, total_edges, edge_ptr_raw, num_graphs);
  auto edge_src = BuildEdgeIndexListColumn(edge_index_raw, total_edges, edge_ptr_raw, node_ptr_raw, num_graphs, 0);
  auto edge_dst = BuildEdgeIndexListColumn(edge_index_raw, total_edges, edge_ptr_raw, node_ptr_raw, num_graphs, 1);
  auto tg_list = BuildNodeIntListColumn(tg_raw, total_nodes, node_ptr_raw, num_graphs);

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("event_id", arrow::int64()),
      arrow::field("time_group_ids", tg_list->type()),
      arrow::field("edge_src_index", edge_src->type()),
      arrow::field("edge_dst_index", edge_dst->type()),
      arrow::field("pred_edge_affinity", pred_list->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {
      graph_event_ids,
      tg_list,
      edge_src,
      edge_dst,
      pred_list,
  };

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void EventSplitterEventOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& edge_pred,
    const std::shared_ptr<arrow::Array>& edge_ptr,
    const std::shared_ptr<arrow::Array>& edge_index,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids,
    const std::shared_ptr<arrow::Array>& graph_event_ids) const {
  auto table = BuildEventTable(edge_pred,
                               edge_ptr,
                               edge_index,
                               node_ptr,
                               time_group_ids,
                               graph_event_ids);
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
