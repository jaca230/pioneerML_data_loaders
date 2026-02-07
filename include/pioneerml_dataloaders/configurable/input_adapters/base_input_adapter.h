#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/configurable/configurable.h"
#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

namespace pioneerml::input_adapters {

class InputAdapter : public pioneerml::configurable::Configurable {
 public:
  virtual ~InputAdapter() = default;

  virtual pioneerml::dataloaders::TrainingBundle LoadTraining(
      const nlohmann::json& input_spec) const = 0;

  virtual pioneerml::dataloaders::InferenceBundle LoadInference(
      const nlohmann::json& input_spec) const = 0;

 protected:
  struct JsonFieldSpec {
    std::string canonical_name;
    std::vector<std::string> aliases;
    std::vector<std::string> projected_columns;
    bool required{true};
  };

  std::shared_ptr<arrow::Table> BuildUnifiedTableFromFilesSpec(
      const nlohmann::json& input_spec,
      const std::vector<JsonFieldSpec>& field_specs,
      const std::string& context) const;
};

}  // namespace pioneerml::input_adapters
