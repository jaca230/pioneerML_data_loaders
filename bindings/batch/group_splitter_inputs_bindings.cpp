#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/batch/group_splitter_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterInputs(py::module_& m) {
  py::class_<pioneerml::GroupSplitterInputs, pioneerml::BaseBatch>(m, "GroupSplitterInputs")
      .def_property_readonly("node_features",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.node_features);
                             })
      .def_property_readonly("edge_index",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.edge_index);
                             })
      .def_property_readonly("edge_attr",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.edge_attr);
                             })
      .def_property_readonly("u", [](const pioneerml::GroupSplitterInputs& self) {
        return WrapArray(self.u);
      })
      .def_property_readonly("group_probs",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.group_probs);
                             })
      .def_property_readonly("node_ptr",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.node_ptr);
                             })
      .def_property_readonly("edge_ptr",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.edge_ptr);
                             })
      .def_property_readonly("graph_event_ids",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.graph_event_ids);
                             })
      .def_property_readonly("graph_group_ids",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.graph_group_ids);
                             })
      .def_property_readonly("y_node",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return WrapArray(self.y_node);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::GroupSplitterInputs& self) {
                               return self.num_graphs;
                             });
}

}  // namespace pioneerml::bindings
