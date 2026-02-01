#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_input_adapter.h"

namespace pioneerml::input_adapters::graph {

GroupClassifierInputAdapter::GroupClassifierInputAdapter() = default;

void GroupClassifierInputAdapter::LoadConfig(const nlohmann::json& cfg) {
  ApplyLoaderConfig(cfg);
}

void GroupClassifierInputAdapter::ApplyLoaderConfig(const nlohmann::json& cfg) {
  if (cfg.contains("loader")) {
    loader_.LoadConfig(cfg.at("loader"));
  } else {
    loader_.LoadConfig(cfg);
  }
}

pioneerml::dataloaders::TrainingBundle GroupClassifierInputAdapter::LoadTraining(
    const std::string& parquet_path) const {
  return loader_.LoadTraining(parquet_path);
}

pioneerml::dataloaders::TrainingBundle GroupClassifierInputAdapter::LoadTraining(
    const std::vector<std::string>& parquet_paths) const {
  return loader_.LoadTraining(parquet_paths);
}

pioneerml::dataloaders::InferenceBundle GroupClassifierInputAdapter::LoadInference(
    const std::string& parquet_path) const {
  return loader_.LoadInference(parquet_path);
}

pioneerml::dataloaders::InferenceBundle GroupClassifierInputAdapter::LoadInference(
    const std::vector<std::string>& parquet_paths) const {
  return loader_.LoadInference(parquet_paths);
}

}  // namespace pioneerml::input_adapters::graph
