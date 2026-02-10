#include "pioneerml_dataloaders/configurable/input_adapters/graph/event_splitter_event_input_adapter.h"

#include <vector>

namespace pioneerml::input_adapters::graph {
namespace {

const std::vector<std::string> kMainColumns = {
    "event_id",
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
    "steps_edep",
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

const std::vector<std::string> kEndpointColumns = {
    "pred_group_start_x",
    "pred_group_start_y",
    "pred_group_start_z",
    "pred_group_end_x",
    "pred_group_end_y",
    "pred_group_end_z",
};

}  // namespace

EventSplitterEventInputAdapter::EventSplitterEventInputAdapter() = default;

void EventSplitterEventInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void EventSplitterEventInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  if (cfg.contains("loader")) {
    loader_.LoadConfig(cfg.at("loader"));
  } else {
    loader_.LoadConfig(cfg);
  }
}

pioneerml::dataloaders::TrainingBundle EventSplitterEventInputAdapter::LoadTraining(
    const nlohmann::json& input_spec) const {
  return loader_.LoadTraining(BuildUnifiedTable(input_spec));
}

pioneerml::dataloaders::InferenceBundle EventSplitterEventInputAdapter::LoadInference(
    const nlohmann::json& input_spec) const {
  return loader_.LoadInference(BuildUnifiedTable(input_spec));
}

std::shared_ptr<arrow::Table> EventSplitterEventInputAdapter::BuildUnifiedTable(
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
          JsonFieldSpec{"endpoint_preds",
                        {"endpoint_preds", "endpoint_preds_file"},
                        kEndpointColumns,
                        false},
      },
      "EventSplitterEventInputAdapter");
}

}  // namespace pioneerml::input_adapters::graph
