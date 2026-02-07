#include <pybind11/pybind11.h>

#include "bindings.h"
#include "pioneerml_dataloaders/batch/base_batch.h"
#include "pioneerml_dataloaders/configurable/dataloaders/base_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindBaseLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::TrainingBundle>(m, "TrainingBundle")
      .def_property_readonly("inputs",
                             [](pioneerml::dataloaders::TrainingBundle& self) {
                               return self.inputs.get();
                             },
                             py::return_value_policy::reference)
      .def_property_readonly("targets",
                             [](pioneerml::dataloaders::TrainingBundle& self) {
                               return self.targets.get();
                             },
                             py::return_value_policy::reference);

  py::class_<pioneerml::dataloaders::InferenceBundle>(m, "InferenceBundle")
      .def_property_readonly("inputs",
                             [](pioneerml::dataloaders::InferenceBundle& self) {
                               return self.inputs.get();
                             },
                             py::return_value_policy::reference);
}

}  // namespace pioneerml::bindings
