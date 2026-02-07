#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_event_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierEventLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::GroupClassifierEventLoader>(
      m, "GroupClassifierEventLoader")
      .def(py::init<>())
      .def("load_config", &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadConfig);
}

}  // namespace pioneerml::bindings
