#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/configurable/dataloaders/graph/graph_loader.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"

namespace pioneerml::dataloaders::graph {

class GroupClassifierEventLoader : public GraphLoader {
 public:
  GroupClassifierEventLoader();

  void LoadConfig(const nlohmann::json& cfg) override;

  TrainingBundle LoadTraining(
      const std::shared_ptr<arrow::Table>& table) const override;
  InferenceBundle LoadInference(
      const std::shared_ptr<arrow::Table>& table) const override;

 protected:
  std::unique_ptr<BaseBatch> BuildGraph(const arrow::Table& table) const override;
  TrainingBundle SplitInputsTargets(std::unique_ptr<BaseBatch> batch) const override;

 private:
  struct BuildContext {
    ColumnMap input_cols;
    ColumnMap target_cols;
    bool has_targets{false};

    const arrow::ListArray* hits_x{nullptr};
    const arrow::ListArray* hits_y{nullptr};
    const arrow::ListArray* hits_z{nullptr};
    const arrow::ListArray* hits_edep{nullptr};
    const arrow::ListArray* hits_view{nullptr};
    const arrow::ListArray* hits_time_group{nullptr};

    NumericAccessor x_values;
    NumericAccessor y_values;
    NumericAccessor z_values;
    NumericAccessor edep_values;

    const int32_t* view_raw{nullptr};
    const int64_t* tg_raw{nullptr};
    const int32_t* z_offsets{nullptr};
    const int32_t* tg_offsets{nullptr};

    int64_t rows{0};
    std::vector<int64_t> node_counts;
    std::vector<int64_t> edge_counts;
    std::vector<int64_t> group_counts;
    std::vector<int64_t> node_offsets;
    std::vector<int64_t> edge_offsets;
    std::vector<int64_t> row_group_offsets;
    int64_t total_nodes{0};
    int64_t total_edges{0};
    int64_t total_groups{0};
  };

  struct BuildBuffers {
    std::shared_ptr<arrow::Buffer> node_feat_buf;
    std::shared_ptr<arrow::Buffer> edge_index_buf;
    std::shared_ptr<arrow::Buffer> edge_attr_buf;
    std::shared_ptr<arrow::Buffer> time_group_buf;
    std::shared_ptr<arrow::Buffer> node_ptr_buf;
    std::shared_ptr<arrow::Buffer> edge_ptr_buf;
    std::shared_ptr<arrow::Buffer> group_ptr_buf;
    std::shared_ptr<arrow::Buffer> u_buf;
    std::shared_ptr<arrow::Buffer> y_buf;

    float* node_feat{nullptr};
    int64_t* edge_index{nullptr};
    float* edge_attr{nullptr};
    int64_t* time_group_ids{nullptr};
    int64_t* node_ptr{nullptr};
    int64_t* edge_ptr{nullptr};
    int64_t* group_ptr{nullptr};
    float* u{nullptr};
    float* y{nullptr};
  };

  void BuildGraphPhase0Initialize(const arrow::Table& table, BuildContext* ctx) const;
  void BuildGraphPhase1Count(BuildContext* ctx) const;
  void BuildGraphPhase2Offsets(BuildContext* ctx) const;
  void BuildGraphPhase3Allocate(const BuildContext& ctx, BuildBuffers* bufs) const;
  void BuildGraphPhase4Populate(const BuildContext& ctx, BuildBuffers* bufs) const;
  std::unique_ptr<BaseBatch> BuildGraphPhase5Finalize(const BuildContext& ctx,
                                                      BuildBuffers* bufs) const;

  void ConfigureDerivers(const nlohmann::json* derivers_cfg);
  void CountNodeEdgePerRow(const int32_t* z_offsets,
                           int64_t rows,
                           std::vector<int64_t>* node_counts,
                           std::vector<int64_t>* edge_counts) const;
  void CountGroupsForRows(const int32_t* z_offsets,
                          const int32_t* tg_offsets,
                          const int64_t* tg_raw,
                          int64_t rows,
                          std::vector<int64_t>* group_counts) const;
  void EncodeTargets(const ColumnMap& target_cols,
                     const std::vector<int64_t>& group_counts,
                     const std::vector<int64_t>& row_group_offsets,
                     int64_t rows,
                     float* y) const;

  double time_window_ns_{1.0};
};

}  // namespace pioneerml::dataloaders::graph
