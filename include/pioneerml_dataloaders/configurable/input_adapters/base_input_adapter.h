#pragma once

#include <string>
#include <vector>

#include "pioneerml_dataloaders/configurable/configurable.h"
#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

namespace pioneerml::input_adapters {

class InputAdapter : public pioneerml::configurable::Configurable {
 public:
  virtual ~InputAdapter() = default;

  virtual pioneerml::dataloaders::TrainingBundle LoadTraining(
      const std::string& parquet_path) const = 0;
  virtual pioneerml::dataloaders::TrainingBundle LoadTraining(
      const std::vector<std::string>& parquet_paths) const = 0;

  virtual pioneerml::dataloaders::InferenceBundle LoadInference(
      const std::string& parquet_path) const = 0;
  virtual pioneerml::dataloaders::InferenceBundle LoadInference(
      const std::vector<std::string>& parquet_paths) const = 0;
};

}  // namespace pioneerml::input_adapters
