#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_event_input_adapter.h"

namespace pioneerml::input_adapters::graph {

GroupClassifierEventInputAdapter::GroupClassifierEventInputAdapter() = default;

void GroupClassifierEventInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void GroupClassifierEventInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  loader_.LoadConfig(cfg);
}

pioneerml::dataloaders::TrainingBundle GroupClassifierEventInputAdapter::LoadTraining(
    const std::string& parquet_path) const {
  return loader_.LoadTraining(parquet_path);
}

pioneerml::dataloaders::TrainingBundle GroupClassifierEventInputAdapter::LoadTraining(
    const std::vector<std::string>& parquet_paths) const {
  return loader_.LoadTraining(parquet_paths);
}

pioneerml::dataloaders::InferenceBundle GroupClassifierEventInputAdapter::LoadInference(
    const std::string& parquet_path) const {
  return loader_.LoadInference(parquet_path);
}

pioneerml::dataloaders::InferenceBundle GroupClassifierEventInputAdapter::LoadInference(
    const std::vector<std::string>& parquet_paths) const {
  return loader_.LoadInference(parquet_paths);
}

}  // namespace pioneerml::input_adapters::graph
