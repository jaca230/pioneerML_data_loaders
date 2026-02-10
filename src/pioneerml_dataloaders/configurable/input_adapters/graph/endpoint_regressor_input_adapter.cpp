#include "pioneerml_dataloaders/configurable/input_adapters/graph/endpoint_regressor_input_adapter.h"

#include <vector>

namespace pioneerml::input_adapters::graph {
namespace {

const std::vector<std::string> kMainColumns = {
    "hits_x",
    "hits_y",
    "hits_z",
    "hits_edep",
    "hits_strip_type",
    "hits_time",
    "hits_contrib_mc_event_id",
    "hits_contrib_step_id",
    "steps_mc_event_id",
    "steps_step_id",
    "steps_pdg_id",
    "steps_x",
    "steps_y",
    "steps_z",
    "steps_edep",
    "steps_time",
};

const std::vector<std::string> kGroupProbColumns = {
    "pred_pion",
    "pred_muon",
    "pred_mip",
};

const std::vector<std::string> kSplitterProbColumns = {
    "pred_hit_pion",
    "pred_hit_muon",
    "pred_hit_mip",
    "time_group_ids",
};

}  // namespace

EndpointRegressorInputAdapter::EndpointRegressorInputAdapter() = default;

void EndpointRegressorInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void EndpointRegressorInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  if (cfg.contains("loader")) {
    loader_.LoadConfig(cfg.at("loader"));
  } else {
    loader_.LoadConfig(cfg);
  }
}

pioneerml::dataloaders::TrainingBundle EndpointRegressorInputAdapter::LoadTraining(
    const nlohmann::json& input_spec) const {
  return loader_.LoadTraining(BuildUnifiedTable(input_spec));
}

pioneerml::dataloaders::InferenceBundle EndpointRegressorInputAdapter::LoadInference(
    const nlohmann::json& input_spec) const {
  return loader_.LoadInference(BuildUnifiedTable(input_spec));
}

std::shared_ptr<arrow::Table> EndpointRegressorInputAdapter::BuildUnifiedTable(
    const nlohmann::json& input_spec) const {
  return BuildUnifiedTableFromFilesSpec(
      input_spec,
      {
          JsonFieldSpec{"main_file", {"main_file", "mainFile"}, kMainColumns, true},
          JsonFieldSpec{"group_probs",
                        {"group_probs", "group_probs_file"},
                        kGroupProbColumns,
                        false},
          JsonFieldSpec{"group_splitter_probs",
                        {"group_splitter_probs", "group_splitter_probs_file"},
                        kSplitterProbColumns,
                        false},
      },
      "EndpointRegressorInputAdapter");
}

}  // namespace pioneerml::input_adapters::graph
