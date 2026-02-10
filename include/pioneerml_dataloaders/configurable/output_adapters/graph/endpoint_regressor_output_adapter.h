#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/configurable/output_adapters/graph/graph_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class EndpointRegressorOutputAdapter : public GraphOutputAdapter {
 public:
  EndpointRegressorOutputAdapter() = default;

  std::shared_ptr<arrow::Table> BuildEventTable(
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& graph_event_ids,
      const std::shared_ptr<arrow::Array>& graph_group_ids) const;

  void WriteParquet(
      const std::string& output_path,
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& graph_event_ids,
      const std::shared_ptr<arrow::Array>& graph_group_ids) const;

 private:
  struct GroupIndexLists {
    std::vector<std::vector<int64_t>> graphs_by_event;
    std::vector<std::vector<int64_t>> group_ids_by_event;
  };

  GroupIndexLists BuildGraphIndexLists(
      const arrow::Array& graph_event_ids,
      const arrow::Array& graph_group_ids) const;

  std::shared_ptr<arrow::Array> BuildGroupListColumn(
      const float* pred_raw,
      int64_t num_groups,
      const std::vector<std::vector<int64_t>>& graphs_by_event,
      int feature_index) const;

  std::shared_ptr<arrow::Array> BuildGroupIdListColumn(
      const std::vector<std::vector<int64_t>>& group_ids_by_event) const;
};

}  // namespace pioneerml::output_adapters::graph
