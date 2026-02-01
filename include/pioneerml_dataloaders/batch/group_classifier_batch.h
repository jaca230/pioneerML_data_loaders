#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/base_batch.h"

namespace pioneerml {

struct GroupClassifierInputs : public BaseBatch {
  // Arrow arrays/views over contiguous buffers for zero-copy adapters.
  std::shared_ptr<arrow::Array> node_features;   // float32, length = total_nodes * 4
  std::shared_ptr<arrow::Array> edge_index;      // int64,   length = total_edges * 2
  std::shared_ptr<arrow::Array> edge_attr;       // float32, length = total_edges * 4
  std::shared_ptr<arrow::Array> u;               // float32, length = num_graphs
  std::shared_ptr<arrow::Array> time_group_ids;  // int64,   length = total_nodes
  std::shared_ptr<arrow::Array> node_ptr;        // int64,   length = num_graphs + 1
  std::shared_ptr<arrow::Array> edge_ptr;        // int64,   length = num_graphs + 1
  std::shared_ptr<arrow::Array> group_ptr;       // int64,   length = num_graphs + 1
  // Labels kept here initially; stripped for inference.
  std::shared_ptr<arrow::Array> y;         // float32, length = num_groups * 3
  std::shared_ptr<arrow::Array> y_energy;  // float32, length = num_groups * 3
  size_t num_graphs{0};
  size_t num_groups{0};
};

struct GroupClassifierTargets : public BaseBatch {
  std::shared_ptr<arrow::Array> y;         // float32, length = num_groups * 3
  std::shared_ptr<arrow::Array> y_energy;  // float32, length = num_groups * 3
  size_t num_groups{0};
};

}  // namespace pioneerml
