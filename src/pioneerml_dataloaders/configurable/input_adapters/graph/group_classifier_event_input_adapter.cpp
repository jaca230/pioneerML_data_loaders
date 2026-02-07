#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_event_input_adapter.h"

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

}  // namespace

GroupClassifierEventInputAdapter::GroupClassifierEventInputAdapter() = default;

void GroupClassifierEventInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void GroupClassifierEventInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  loader_.LoadConfig(cfg);
}

pioneerml::dataloaders::TrainingBundle GroupClassifierEventInputAdapter::LoadTraining(
    const nlohmann::json& input_spec) const {
  return loader_.LoadTraining(BuildUnifiedTable(input_spec));
}

pioneerml::dataloaders::InferenceBundle GroupClassifierEventInputAdapter::LoadInference(
    const nlohmann::json& input_spec) const {
  return loader_.LoadInference(BuildUnifiedTable(input_spec));
}

std::shared_ptr<arrow::Table> GroupClassifierEventInputAdapter::BuildUnifiedTable(
    const nlohmann::json& input_spec) const {
  return BuildUnifiedTableFromFilesSpec(
      input_spec,
      {
          JsonFieldSpec{"main_file", {"main_file", "mainFile"}, kMainColumns, true},
      },
      "GroupClassifierEventInputAdapter");
}

}  // namespace pioneerml::input_adapters::graph
