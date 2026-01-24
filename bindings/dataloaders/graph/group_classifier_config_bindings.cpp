#include <pybind11/pybind11.h>

#include "bindings.h"
#include "pioneerml_dataloaders/dataloaders/graph/group_classifier_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierConfig(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::GroupClassifierConfig>(
      m, "GroupClassifierConfig")
      .def(py::init<>())
      .def_readwrite(
          "time_window_ns",
          &pioneerml::dataloaders::graph::GroupClassifierConfig::time_window_ns)
      .def_readwrite(
          "compute_time_groups",
          &pioneerml::dataloaders::graph::GroupClassifierConfig::compute_time_groups);
}

}  // namespace pioneerml::bindings
