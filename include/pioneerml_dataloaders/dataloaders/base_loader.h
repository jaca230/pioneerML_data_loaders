#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/base_batch.h"

namespace pioneerml::dataloaders {

struct TrainingBundle {
  std::unique_ptr<BaseBatch> inputs;
  std::unique_ptr<BaseBatch> targets;
};

struct InferenceBundle {
  std::unique_ptr<BaseBatch> inputs;
};

// Contract for loading parquet shards into a model-ready batch abstraction.
class DataLoader {
 public:
  virtual ~DataLoader() = default;

  // Load inputs + targets for training.
  virtual TrainingBundle LoadTraining(const std::string& parquet_path) const = 0;

  // Load inputs only for inference.
  virtual InferenceBundle LoadInference(const std::string& parquet_path) const = 0;

 protected:
  // Optional helper for derived loaders that need the raw table.
  virtual std::shared_ptr<arrow::Table> LoadTable(const std::string& parquet_path) const = 0;
};

}  // namespace pioneerml::dataloaders
