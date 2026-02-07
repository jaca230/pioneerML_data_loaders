#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_splitter_input_adapter.h"

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
    "hits_pdg_id",
};

const std::vector<std::string> kGroupProbColumns = {
    "pred_pion",
    "pred_muon",
    "pred_mip",
};

}  // namespace

GroupSplitterInputAdapter::GroupSplitterInputAdapter() = default;

void GroupSplitterInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void GroupSplitterInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  if (cfg.contains("loader")) {
    loader_.LoadConfig(cfg.at("loader"));
  } else {
    loader_.LoadConfig(cfg);
  }
}

pioneerml::dataloaders::TrainingBundle GroupSplitterInputAdapter::LoadTraining(
    const nlohmann::json& input_spec) const {
  return loader_.LoadTraining(BuildUnifiedTable(input_spec));
}

pioneerml::dataloaders::InferenceBundle GroupSplitterInputAdapter::LoadInference(
    const nlohmann::json& input_spec) const {
  return loader_.LoadInference(BuildUnifiedTable(input_spec));
}

std::shared_ptr<arrow::Table> GroupSplitterInputAdapter::BuildUnifiedTable(
    const nlohmann::json& input_spec) const {
  return BuildUnifiedTableFromFilesSpec(
      input_spec,
      {
          JsonFieldSpec{"main_file", {"main_file", "mainFile"}, kMainColumns, true},
          JsonFieldSpec{"group_probs",
                        {"group_probs", "group_probs_file"},
                        kGroupProbColumns,
                        false},
      },
      "GroupSplitterInputAdapter");
}

}  // namespace pioneerml::input_adapters::graph
