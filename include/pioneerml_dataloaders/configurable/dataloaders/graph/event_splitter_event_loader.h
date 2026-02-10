#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/batch/event_splitter_event_batch.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/graph_loader.h"

namespace pioneerml::dataloaders::graph {

class EventSplitterEventLoader : public GraphLoader {
 public:
  EventSplitterEventLoader();

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
    ColumnMap endpoint_cols;

    bool has_targets{false};
    bool has_prob_columns{false};
    bool has_splitter_columns{false};
    bool has_endpoint_columns{false};

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
    const int32_t* event_id_raw{nullptr};

    const int32_t* offsets{nullptr};
    const int32_t* tg_offsets{nullptr};

    const arrow::ListArray* contrib_mc_outer{nullptr};
    const arrow::ListArray* contrib_step_outer{nullptr};
    const arrow::ListArray* contrib_mc_inner{nullptr};
    const arrow::ListArray* contrib_step_inner{nullptr};
    const int32_t* contrib_mc_outer_offsets{nullptr};
    const int32_t* contrib_step_outer_offsets{nullptr};
    const int32_t* contrib_mc_inner_offsets{nullptr};
    const int32_t* contrib_step_inner_offsets{nullptr};
    const int32_t* contrib_mc_values_raw{nullptr};
    const int32_t* contrib_step_values_raw{nullptr};

    const arrow::ListArray* steps_mc{nullptr};
    const arrow::ListArray* steps_step{nullptr};
    const arrow::ListArray* steps_pdg{nullptr};
    const arrow::ListArray* steps_edep{nullptr};
    const int32_t* steps_offsets{nullptr};
    const int32_t* steps_step_offsets{nullptr};
    const int32_t* steps_pdg_offsets{nullptr};
    const int32_t* steps_edep_offsets{nullptr};
    const int32_t* steps_mc_values_raw{nullptr};
    const int32_t* steps_step_values_raw{nullptr};
    const int32_t* steps_pdg_values_raw{nullptr};
    NumericAccessor steps_edep_values;

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
    std::shared_ptr<arrow::Buffer> group_probs_buf;
    std::shared_ptr<arrow::Buffer> splitter_probs_buf;
    std::shared_ptr<arrow::Buffer> endpoint_preds_buf;
    std::shared_ptr<arrow::Buffer> graph_event_ids_buf;
    std::shared_ptr<arrow::Buffer> y_edge_buf;

    float* node_feat{nullptr};
    int64_t* edge_index{nullptr};
    float* edge_attr{nullptr};
    int64_t* time_group_ids{nullptr};
    int64_t* node_ptr{nullptr};
    int64_t* edge_ptr{nullptr};
    int64_t* group_ptr{nullptr};
    float* group_probs{nullptr};
    float* splitter_probs{nullptr};
    float* endpoint_preds{nullptr};
    int64_t* graph_event_ids{nullptr};
    float* y_edge{nullptr};

    std::vector<uint8_t> group_truth;
    std::vector<int8_t> node_class;
  };

  void BuildGraphPhase0Initialize(const arrow::Table& table, BuildContext* ctx) const;
  void BuildGraphPhase1Count(BuildContext* ctx) const;
  void BuildGraphPhase2Offsets(BuildContext* ctx) const;
  void BuildGraphPhase3Allocate(const BuildContext& ctx, BuildBuffers* bufs) const;
  void BuildGraphPhase4Populate(const BuildContext& ctx, BuildBuffers* bufs) const;
  std::unique_ptr<BaseBatch> BuildGraphPhase5Finalize(const BuildContext& ctx,
                                                      BuildBuffers* bufs) const;

  void ConfigureDerivers(const nlohmann::json* derivers_cfg);
  void CountRows(const int32_t* offsets,
                 const int32_t* tg_offsets,
                 const int64_t* tg_raw,
                 int64_t rows,
                 std::vector<int64_t>* node_counts,
                 std::vector<int64_t>* edge_counts,
                 std::vector<int64_t>* group_counts) const;

  double time_window_ns_{1.0};
  bool use_group_probs_{true};
  bool use_splitter_probs_{true};
  bool use_endpoint_preds_{true};
  bool derive_time_groups_{true};
};

}  // namespace pioneerml::dataloaders::graph
