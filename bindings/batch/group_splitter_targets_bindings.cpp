#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/group_splitter_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterTargets(py::module_& m) {
  py::class_<pioneerml::GroupSplitterTargets, pioneerml::BaseBatch>(m, "GroupSplitterTargets")
      .def_property_readonly("y_node",
                             [](const pioneerml::GroupSplitterTargets& self) {
                               return WrapArray(self.y_node);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::GroupSplitterTargets& self) {
                               return self.num_graphs;
                             });
}

}  // namespace pioneerml::bindings
