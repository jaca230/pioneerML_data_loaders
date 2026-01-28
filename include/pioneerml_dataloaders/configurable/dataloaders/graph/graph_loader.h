#pragma once

#include <memory>
#include <string>

#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"

namespace pioneerml::dataloaders::graph {

// Specialization for loaders that emit GraphBatch.
class GraphLoader : public DataLoader {
 public:
  TrainingBundle LoadTraining(const std::string& parquet_path) const override {
    auto table = LoadTable(parquet_path);
    auto batch = BuildGraph(*table);
    return SplitInputsTargets(std::move(batch));
  }

  InferenceBundle LoadInference(const std::string& parquet_path) const override {
    auto table = LoadTable(parquet_path);
    InferenceBundle out;
    out.inputs = BuildGraph(*table);
    return out;
  }

 protected:
  // Implement in derived loaders to build a GraphBatch from a loaded table.
  virtual std::unique_ptr<BaseBatch> BuildGraph(const arrow::Table& table) const = 0;

  // Derived loaders can override if they need custom splitting.
  virtual TrainingBundle SplitInputsTargets(std::unique_ptr<BaseBatch> batch) const = 0;
};

}  // namespace pioneerml::dataloaders::graph
