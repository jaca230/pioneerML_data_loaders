#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/event_splitter_event_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEventSplitterEventInputs(py::module_& m) {
  py::class_<pioneerml::EventSplitterEventInputs, pioneerml::BaseBatch>(m, "EventSplitterEventInputs")
      .def_property_readonly("node_features",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.node_features);
                             })
      .def_property_readonly("edge_index",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.edge_index);
                             })
      .def_property_readonly("edge_attr",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.edge_attr);
                             })
      .def_property_readonly("time_group_ids",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.time_group_ids);
                             })
      .def_property_readonly("group_probs",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.group_probs);
                             })
      .def_property_readonly("splitter_probs",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.splitter_probs);
                             })
      .def_property_readonly("endpoint_preds",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.endpoint_preds);
                             })
      .def_property_readonly("node_ptr",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.node_ptr);
                             })
      .def_property_readonly("edge_ptr",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.edge_ptr);
                             })
      .def_property_readonly("group_ptr",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.group_ptr);
                             })
      .def_property_readonly("graph_event_ids",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.graph_event_ids);
                             })
      .def_property_readonly("y_edge",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return WrapArray(self.y_edge);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return self.num_graphs;
                             })
      .def_property_readonly("num_groups",
                             [](const pioneerml::EventSplitterEventInputs& self) {
                               return self.num_groups;
                             });
}

}  // namespace pioneerml::bindings
