#pragma once

#include "pioneerml_dataloaders/configurable/configurable.h"

namespace pioneerml::output_adapters {

class OutputAdapter : public pioneerml::configurable::Configurable {
 public:
  virtual ~OutputAdapter() = default;
};

}  // namespace pioneerml::output_adapters
