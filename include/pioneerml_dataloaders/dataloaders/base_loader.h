#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/base_batch.h"

namespace pioneerml::dataloaders {

// Contract for loading parquet shards into a model-ready batch abstraction.
class DataLoader {
 public:
  virtual ~DataLoader() = default;

  // Top-level API: load a batch from one parquet shard.
  virtual std::unique_ptr<BaseBatch> Load(const std::string& parquet_path) const = 0;

 protected:
  // Optional helper for derived loaders that need the raw table.
  virtual std::shared_ptr<arrow::Table> LoadTable(const std::string& parquet_path) const = 0;
};

}  // namespace pioneerml::dataloaders
