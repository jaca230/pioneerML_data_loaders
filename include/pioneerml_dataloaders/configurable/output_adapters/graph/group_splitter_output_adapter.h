#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "pioneerml_dataloaders/configurable/output_adapters/graph/graph_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class GroupSplitterOutputAdapter : public GraphOutputAdapter {
 public:
  GroupSplitterOutputAdapter() = default;

  std::shared_ptr<arrow::Table> BuildEventTable(
      const std::shared_ptr<arrow::Array>& node_pred,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& graph_event_ids,
      const std::shared_ptr<arrow::Array>& graph_group_ids) const;

  void WriteParquet(
      const std::string& output_path,
      const std::shared_ptr<arrow::Array>& node_pred,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& graph_event_ids,
      const std::shared_ptr<arrow::Array>& graph_group_ids) const;
};

}  // namespace pioneerml::output_adapters::graph
