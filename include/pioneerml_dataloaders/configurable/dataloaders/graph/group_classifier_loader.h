#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/configurable/dataloaders/graph/graph_loader.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"

namespace pioneerml::dataloaders::graph {

class GroupClassifierLoader : public GraphLoader {
 public:
  GroupClassifierLoader();

  void LoadConfig(const nlohmann::json& cfg) override;

  TrainingBundle LoadTraining(const std::string& parquet_path) const override;
  TrainingBundle LoadTraining(
      const std::vector<std::string>& parquet_paths) const override;
  InferenceBundle LoadInference(const std::string& parquet_path) const override;
  InferenceBundle LoadInference(
      const std::vector<std::string>& parquet_paths) const override;
  std::shared_ptr<arrow::Table> LoadTable(const std::string& parquet_path) const override;

 protected:
  std::unique_ptr<BaseBatch> BuildGraph(const arrow::Table& table) const override;
  TrainingBundle SplitInputsTargets(std::unique_ptr<BaseBatch> batch) const override;

 private:
  void ConfigureDerivers(const nlohmann::json* derivers_cfg);

  double time_window_ns_{1.0};
  bool compute_time_groups_{true};
};

}  // namespace pioneerml::dataloaders::graph
