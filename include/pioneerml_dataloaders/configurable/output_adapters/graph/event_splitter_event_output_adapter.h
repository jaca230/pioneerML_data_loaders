#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "pioneerml_dataloaders/configurable/output_adapters/graph/graph_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class EventSplitterEventOutputAdapter : public GraphOutputAdapter {
 public:
  EventSplitterEventOutputAdapter() = default;

  std::shared_ptr<arrow::Table> BuildEventTable(
      const std::shared_ptr<arrow::Array>& edge_pred,
      const std::shared_ptr<arrow::Array>& edge_ptr,
      const std::shared_ptr<arrow::Array>& edge_index,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& time_group_ids,
      const std::shared_ptr<arrow::Array>& graph_event_ids) const;

  void WriteParquet(
      const std::string& output_path,
      const std::shared_ptr<arrow::Array>& edge_pred,
      const std::shared_ptr<arrow::Array>& edge_ptr,
      const std::shared_ptr<arrow::Array>& edge_index,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& time_group_ids,
      const std::shared_ptr<arrow::Array>& graph_event_ids) const;

 private:
  std::shared_ptr<arrow::Array> BuildEdgeFloatListColumn(
      const float* pred_raw,
      int64_t total_edges,
      const int64_t* edge_ptr,
      int64_t num_graphs) const;

  std::shared_ptr<arrow::Array> BuildEdgeIndexListColumn(
      const int64_t* edge_index_raw,
      int64_t total_edges,
      const int64_t* edge_ptr,
      const int64_t* node_ptr,
      int64_t num_graphs,
      int endpoint) const;

  std::shared_ptr<arrow::Array> BuildNodeIntListColumn(
      const int64_t* values_raw,
      int64_t total_nodes,
      const int64_t* node_ptr,
      int64_t num_graphs) const;
};

}  // namespace pioneerml::output_adapters::graph
