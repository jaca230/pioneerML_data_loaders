#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include "pioneerml_dataloaders/configurable/dataloaders/graph/endpoint_regressor_event_loader.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/graph_input_adapter.h"

namespace pioneerml::input_adapters::graph {

class EndpointRegressorEventInputAdapter : public GraphInputAdapter {
 public:
  EndpointRegressorEventInputAdapter();

  void LoadConfig(const nlohmann::json& cfg) override;

  pioneerml::dataloaders::TrainingBundle LoadTraining(
      const nlohmann::json& input_spec) const override;

  pioneerml::dataloaders::InferenceBundle LoadInference(
      const nlohmann::json& input_spec) const override;

 private:
  void ApplyLoaderConfig(const nlohmann::json& cfg);
  std::shared_ptr<arrow::Table> BuildUnifiedTable(const nlohmann::json& input_spec) const;

  pioneerml::dataloaders::graph::EndpointRegressorEventLoader loader_;
};

}  // namespace pioneerml::input_adapters::graph
