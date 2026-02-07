#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/group_splitter_event_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterEventInputs(py::module_& m) {
  py::class_<pioneerml::GroupSplitterEventInputs, pioneerml::BaseBatch>(m, "GroupSplitterEventInputs")
      .def_property_readonly("node_features",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.node_features);
                             })
      .def_property_readonly("edge_index",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.edge_index);
                             })
      .def_property_readonly("edge_attr",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.edge_attr);
                             })
      .def_property_readonly("time_group_ids",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.time_group_ids);
                             })
      .def_property_readonly("u", [](const pioneerml::GroupSplitterEventInputs& self) {
        return WrapArray(self.u);
      })
      .def_property_readonly("group_probs",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.group_probs);
                             })
      .def_property_readonly("node_ptr",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.node_ptr);
                             })
      .def_property_readonly("edge_ptr",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.edge_ptr);
                             })
      .def_property_readonly("group_ptr",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.group_ptr);
                             })
      .def_property_readonly("graph_event_ids",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.graph_event_ids);
                             })
      .def_property_readonly("y_node",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return WrapArray(self.y_node);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return self.num_graphs;
                             })
      .def_property_readonly("num_groups",
                             [](const pioneerml::GroupSplitterEventInputs& self) {
                               return self.num_groups;
                             });
}

}  // namespace pioneerml::bindings
