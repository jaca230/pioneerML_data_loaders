#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/group_splitter_event_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterEventTargets(py::module_& m) {
  py::class_<pioneerml::GroupSplitterEventTargets, pioneerml::BaseBatch>(m, "GroupSplitterEventTargets")
      .def_property_readonly("y_node",
                             [](const pioneerml::GroupSplitterEventTargets& self) {
                               return WrapArray(self.y_node);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::GroupSplitterEventTargets& self) {
                               return self.num_graphs;
                             });
}

}  // namespace pioneerml::bindings
