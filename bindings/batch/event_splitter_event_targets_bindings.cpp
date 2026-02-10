#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/event_splitter_event_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEventSplitterEventTargets(py::module_& m) {
  py::class_<pioneerml::EventSplitterEventTargets, pioneerml::BaseBatch>(m, "EventSplitterEventTargets")
      .def_property_readonly("y_edge", [](const pioneerml::EventSplitterEventTargets& self) {
        return WrapArray(self.y_edge);
      })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::EventSplitterEventTargets& self) {
                               return self.num_graphs;
                             });
}

}  // namespace pioneerml::bindings
