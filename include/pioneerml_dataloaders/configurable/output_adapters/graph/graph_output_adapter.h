#pragma once

#include "pioneerml_dataloaders/configurable/output_adapters/base_output_adapter.h"

namespace pioneerml::output_adapters::graph {

class GraphOutputAdapter : public OutputAdapter {
 public:
  ~GraphOutputAdapter() override = default;
};

}  // namespace pioneerml::output_adapters::graph
