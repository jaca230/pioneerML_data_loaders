#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace pioneerml {

struct GraphBatch {
  // Flattened node features [num_nodes, 4]: coord, z, energy, view
  std::vector<float> node_features;
  // Edge indices [2, num_edges]
  std::vector<int64_t> edge_index;
  // Edge attrs [num_edges, 4]: dx, dz, dE, same_view
  std::vector<float> edge_attr;
  // Graph-level features [num_graphs, 1]
  std::vector<float> u;
  // Masks
  std::vector<uint8_t> hit_mask;
  std::vector<int64_t> time_group_ids;
  // Labels
  std::vector<float> y;         // [num_graphs, 3]
  std::vector<float> y_energy;  // [num_graphs, 3]
  std::vector<int64_t> y_node;  // [num_nodes]

  size_t num_graphs{0};
  size_t max_hits{0};
};

struct GroupClassifierConfig {
  size_t max_hits{256};
  float pad_value{0.0f};
  float time_window_ns{1.0f};
  bool compute_time_groups{true};
};

// Load a parquet shard and emit a single GraphBatch.
// Implementation should use Arrow/Parquet for column projection and derive
// time-group labels + padding without Python involvement.
std::unique_ptr<GraphBatch> load_group_classifier_batch(const std::string& parquet_path,
                                                        const GroupClassifierConfig& cfg);

}  // namespace pioneerml
