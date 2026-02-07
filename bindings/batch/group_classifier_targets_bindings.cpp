#include <pybind11/pybind11.h>

#include "bindings.h"
#include "arrow_utils.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierTargets(py::module_& m) {
  py::class_<pioneerml::GroupClassifierTargets, pioneerml::BaseBatch>(
      m, "GroupClassifierTargets")
      .def_property_readonly("y",
                             [](const pioneerml::GroupClassifierTargets& self) {
                               return WrapArray(self.y);
                             })
      .def_property_readonly("num_groups",
                             [](const pioneerml::GroupClassifierTargets& self) {
                               return self.num_groups;
                             });
}

}  // namespace pioneerml::bindings
