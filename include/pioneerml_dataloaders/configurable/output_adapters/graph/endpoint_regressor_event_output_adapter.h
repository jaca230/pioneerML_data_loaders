#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "pioneerml_dataloaders/configurable/output_adapters/graph/graph_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class EndpointRegressorEventOutputAdapter : public GraphOutputAdapter {
 public:
  EndpointRegressorEventOutputAdapter() = default;

  std::shared_ptr<arrow::Table> BuildEventTable(
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& group_ptr,
      const std::shared_ptr<arrow::Array>& graph_event_ids) const;

  void WriteParquet(
      const std::string& output_path,
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& group_ptr,
      const std::shared_ptr<arrow::Array>& graph_event_ids) const;

 private:
  std::shared_ptr<arrow::Array> BuildGroupListColumn(
      const float* pred_raw,
      int64_t total_groups,
      const int64_t* group_ptr_raw,
      int64_t num_graphs,
      int feature_index) const;
};

}  // namespace pioneerml::output_adapters::graph
