#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/configurable/output_adapters/graph/graph_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class GroupClassifierEventOutputAdapter : public GraphOutputAdapter {
 public:
  GroupClassifierEventOutputAdapter() = default;

  std::shared_ptr<arrow::Table> BuildEventTable(
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& group_pred_energy,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& time_group_ids) const;

  void WriteParquet(
      const std::string& output_path,
      const std::shared_ptr<arrow::Array>& group_pred,
      const std::shared_ptr<arrow::Array>& group_pred_energy,
      const std::shared_ptr<arrow::Array>& node_ptr,
      const std::shared_ptr<arrow::Array>& time_group_ids) const;

 private:
  struct GroupOffsets {
    std::vector<int64_t> offsets;
    std::vector<int64_t> counts;
  };

  GroupOffsets ComputeGroupOffsets(
      const arrow::Array& node_ptr,
      const arrow::Array& time_group_ids) const;

  std::shared_ptr<arrow::Array> BuildGroupListColumn(
      const float* pred_raw,
      int64_t num_groups,
      const std::vector<int64_t>& offsets,
      const std::vector<int64_t>& counts,
      int class_index) const;
};

}  // namespace pioneerml::output_adapters::graph
