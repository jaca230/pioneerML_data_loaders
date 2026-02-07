#pragma once

#include "pioneerml_dataloaders/configurable/input_adapters/base_input_adapter.h"

namespace pioneerml::input_adapters::graph {

class GraphInputAdapter : public InputAdapter {
 public:
  ~GraphInputAdapter() override = default;
  using InputAdapter::LoadInference;
  using InputAdapter::LoadTraining;
};

}  // namespace pioneerml::input_adapters::graph
