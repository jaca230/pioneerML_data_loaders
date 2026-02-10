#pragma once

#include <cstddef>
#include <memory>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/base_batch.h"

namespace pioneerml {

struct EndpointRegressorInputs : public BaseBatch {
  std::shared_ptr<arrow::Array> node_features;    // float32, length = total_nodes * 4
  std::shared_ptr<arrow::Array> edge_index;       // int64,   length = total_edges * 2
  std::shared_ptr<arrow::Array> edge_attr;        // float32, length = total_edges * 4
  std::shared_ptr<arrow::Array> time_group_ids;   // int64,   length = total_nodes
  std::shared_ptr<arrow::Array> u;                // float32, length = num_graphs
  std::shared_ptr<arrow::Array> group_probs;      // float32, length = total_groups * 3
  std::shared_ptr<arrow::Array> splitter_probs;   // float32, length = total_nodes * 3
  std::shared_ptr<arrow::Array> node_ptr;         // int64,   length = num_graphs + 1
  std::shared_ptr<arrow::Array> edge_ptr;         // int64,   length = num_graphs + 1
  std::shared_ptr<arrow::Array> group_ptr;        // int64,   length = num_graphs + 1
  std::shared_ptr<arrow::Array> graph_event_ids;  // int64,   length = num_graphs
  std::shared_ptr<arrow::Array> graph_group_ids;  // int64,   length = num_graphs

  // Labels kept here initially; stripped for inference.
  std::shared_ptr<arrow::Array> y;  // float32, length = total_groups * 6

  size_t num_graphs{0};
  size_t num_groups{0};
};

struct EndpointRegressorTargets : public BaseBatch {
  std::shared_ptr<arrow::Array> y;  // float32, length = total_groups * 6
  size_t num_groups{0};
};

}  // namespace pioneerml
