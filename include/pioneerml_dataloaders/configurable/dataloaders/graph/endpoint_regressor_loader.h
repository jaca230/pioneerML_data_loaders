#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/batch/endpoint_regressor_batch.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/graph_loader.h"

namespace pioneerml::dataloaders::graph {

class EndpointRegressorLoader : public GraphLoader {
 public:
  EndpointRegressorLoader();

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
    ColumnMap prob_cols;
    ColumnMap splitter_cols;
    bool has_targets{false};
    bool has_prob_columns{false};
    bool has_splitter_columns{false};

    const arrow::ListArray* hits_x{nullptr};
    const arrow::ListArray* hits_y{nullptr};
    const arrow::ListArray* hits_z{nullptr};
    const arrow::ListArray* hits_edep{nullptr};
    const arrow::ListArray* hits_view{nullptr};
    const arrow::ListArray* hits_time_group{nullptr};
    const arrow::ListArray* hits_pdg_id{nullptr};

    NumericAccessor x_values;
    NumericAccessor y_values;
    NumericAccessor z_values;
    NumericAccessor edep_values;

    const int32_t* view_raw{nullptr};
    const int64_t* tg_raw{nullptr};
    const int32_t* pdg_raw{nullptr};
    const int32_t* offsets{nullptr};
    const int32_t* tg_offsets{nullptr};
    const int32_t* pdg_offsets{nullptr};

    int64_t rows{0};
    std::vector<int64_t> group_counts;
    std::vector<std::vector<int64_t>> group_node_counts;
    std::vector<int64_t> graph_offsets;
    std::vector<int64_t> node_counts;
    std::vector<int64_t> edge_counts;
    std::vector<int64_t> node_offsets;
    std::vector<int64_t> edge_offsets;
    int64_t total_graphs{0};
    int64_t total_nodes{0};
    int64_t total_edges{0};
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
    std::shared_ptr<arrow::Buffer> group_probs_buf;
    std::shared_ptr<arrow::Buffer> splitter_probs_buf;
    std::shared_ptr<arrow::Buffer> graph_event_ids_buf;
    std::shared_ptr<arrow::Buffer> graph_group_ids_buf;
    std::shared_ptr<arrow::Buffer> y_buf;

    float* node_feat{nullptr};
    int64_t* edge_index{nullptr};
    float* edge_attr{nullptr};
    int64_t* time_group_ids{nullptr};
    int64_t* node_ptr{nullptr};
    int64_t* edge_ptr{nullptr};
    int64_t* group_ptr{nullptr};
    float* u{nullptr};
    float* group_probs{nullptr};
    float* splitter_probs{nullptr};
    int64_t* graph_event_ids{nullptr};
    int64_t* graph_group_ids{nullptr};
    float* y{nullptr};

    std::vector<uint8_t> group_truth;
    std::vector<float> node_truth;
  };

  struct TargetReaders {
    NumericAccessor start_x;
    NumericAccessor start_y;
    NumericAccessor start_z;
    NumericAccessor end_x;
    NumericAccessor end_y;
    NumericAccessor end_z;
    const int32_t* sx_offsets{nullptr};
    const int32_t* sy_offsets{nullptr};
    const int32_t* sz_offsets{nullptr};
    const int32_t* ex_offsets{nullptr};
    const int32_t* ey_offsets{nullptr};
    const int32_t* ez_offsets{nullptr};
  };

  void BuildGraphPhase0Initialize(const arrow::Table& table, BuildContext* ctx) const;
  void BuildGraphPhase1Count(BuildContext* ctx) const;
  void BuildGraphPhase2Offsets(BuildContext* ctx) const;
  void BuildGraphPhase3Allocate(const BuildContext& ctx, BuildBuffers* bufs) const;
  void BuildGraphPhase4Populate(const BuildContext& ctx, BuildBuffers* bufs) const;
  std::unique_ptr<BaseBatch> BuildGraphPhase5Finalize(const BuildContext& ctx,
                                                      BuildBuffers* bufs) const;

  void ConfigureDerivers(const nlohmann::json* derivers_cfg);
  void CountGroupsForRows(const int32_t* offsets,
                          const int32_t* tg_offsets,
                          const int64_t* tg_raw,
                          const int32_t* pdg_offsets,
                          int64_t rows,
                          std::vector<int64_t>* group_counts,
                          std::vector<std::vector<int64_t>>* group_node_counts) const;
  TargetReaders BuildTargetReaders(const ColumnMap& target_cols) const;
  void ValidateTargetListLengths(const TargetReaders& readers,
                                 const std::vector<int64_t>& group_counts,
                                 int64_t rows) const;

  double time_window_ns_{1.0};
  bool use_group_probs_{true};
  bool use_splitter_probs_{true};
};

}  // namespace pioneerml::dataloaders::graph
