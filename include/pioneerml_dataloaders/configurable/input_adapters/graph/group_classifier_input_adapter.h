#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/configurable/input_adapters/graph/graph_input_adapter.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.h"

namespace pioneerml::input_adapters::graph {

class GroupClassifierInputAdapter : public GraphInputAdapter {
 public:
  GroupClassifierInputAdapter();

  void LoadConfig(const nlohmann::json& cfg) override;

  pioneerml::dataloaders::TrainingBundle LoadTraining(
      const std::string& parquet_path) const override;
  pioneerml::dataloaders::TrainingBundle LoadTraining(
      const std::vector<std::string>& parquet_paths) const override;

  pioneerml::dataloaders::InferenceBundle LoadInference(
      const std::string& parquet_path) const override;
  pioneerml::dataloaders::InferenceBundle LoadInference(
      const std::vector<std::string>& parquet_paths) const override;

 private:
  void ApplyLoaderConfig(const nlohmann::json& cfg);

  pioneerml::dataloaders::graph::GroupClassifierLoader loader_;
};

}  // namespace pioneerml::input_adapters::graph
