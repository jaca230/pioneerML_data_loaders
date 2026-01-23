#pragma once

#include <memory>
#include <string>

#include "pioneerml_dataloaders/dataloaders/base_loader.h"
#include "pioneerml_dataloaders/batch/graph_batch.h"

namespace pioneerml::dataloaders::graph {

// Specialization for loaders that emit GraphBatch.
class GraphLoader : public DataLoader {
 public:
  std::unique_ptr<BaseBatch> Load(const std::string& parquet_path) const override {
    auto table = LoadTable(parquet_path);
    return BuildGraph(*table);
  }

 protected:
  // Implement in derived loaders to build a GraphBatch from a loaded table.
  virtual std::unique_ptr<GraphBatch> BuildGraph(const arrow::Table& table) const = 0;
};

}  // namespace pioneerml::dataloaders::graph
