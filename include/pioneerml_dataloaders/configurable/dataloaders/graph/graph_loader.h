#pragma once

#include <memory>

#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

namespace pioneerml::dataloaders::graph {

// Specialization for loaders that emit GraphBatch.
class GraphLoader : public DataLoader {
 public:
  ~GraphLoader() override = default;
  using DataLoader::LoadInference;
  using DataLoader::LoadTraining;

 protected:
  virtual std::unique_ptr<BaseBatch> BuildGraph(const arrow::Table& table) const = 0;
};

}  // namespace pioneerml::dataloaders::graph
