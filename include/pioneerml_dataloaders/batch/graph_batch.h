#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pioneerml_dataloaders/batch/base_batch.h"

namespace pioneerml {

struct GraphBatch : public BaseBatch {
  // Flattened node features [total_nodes, 4]: coord, z, energy, view
  std::vector<float> node_features;
  // Edge indices [2, total_edges]
  std::vector<int64_t> edge_index;
  // Edge attrs [total_edges, 4]: dx, dz, dE, same_view
  std::vector<float> edge_attr;
  // Graph-level features [num_graphs, 1]
  std::vector<float> u;
  // Per-node metadata
  std::vector<uint8_t> hit_mask;        // [total_nodes]
  std::vector<int64_t> time_group_ids;  // [total_nodes]
  std::vector<int64_t> y_node;          // [total_nodes]
  // Per-graph labels
  std::vector<float> y;         // [num_graphs, 3]
  std::vector<float> y_energy;  // [num_graphs, 3]
  // Prefix sums to delimit graphs in flattened buffers
  std::vector<int64_t> node_ptr;  // length num_graphs+1
  std::vector<int64_t> edge_ptr;  // length num_graphs+1

  size_t num_graphs{0};
};

}  // namespace pioneerml
