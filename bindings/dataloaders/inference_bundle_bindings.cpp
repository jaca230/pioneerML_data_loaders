#include <pybind11/pybind11.h>

#include "bindings.h"
#include "pioneerml_dataloaders/batch/base_batch.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"
#include "pioneerml_dataloaders/dataloaders/base_loader.h"

namespace py = pybind11;

namespace {

pioneerml::GroupClassifierInputs* AsGroupInputs(pioneerml::BaseBatch* batch) {
  return dynamic_cast<pioneerml::GroupClassifierInputs*>(batch);
}

}  // namespace

namespace pioneerml::bindings {

void BindInferenceBundle(py::module_& m) {
  py::class_<pioneerml::dataloaders::InferenceBundle>(m, "InferenceBundle")
      .def_property_readonly("inputs",
                             [](pioneerml::dataloaders::InferenceBundle& self) {
                               return AsGroupInputs(self.inputs.get());
                             },
                             py::return_value_policy::reference);
}

}  // namespace pioneerml::bindings
