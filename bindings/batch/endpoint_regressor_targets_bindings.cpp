#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/endpoint_regressor_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEndpointRegressorTargets(py::module_& m) {
  py::class_<pioneerml::EndpointRegressorTargets, pioneerml::BaseBatch>(m, "EndpointRegressorTargets")
      .def_property_readonly("y", [](const pioneerml::EndpointRegressorTargets& self) {
        return WrapArray(self.y);
      })
      .def_property_readonly("num_groups",
                             [](const pioneerml::EndpointRegressorTargets& self) {
                               return self.num_groups;
                             });
}

}  // namespace pioneerml::bindings
