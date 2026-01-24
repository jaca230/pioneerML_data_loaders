#include <pybind11/pybind11.h>

#include "bindings.h"
#include "arrow_utils.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierInputs(py::module_& m) {
  py::class_<pioneerml::GroupClassifierInputs, pioneerml::BaseBatch>(
      m, "GroupClassifierInputs")
      .def_property_readonly("node_features",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.node_features);
                             })
      .def_property_readonly("edge_index",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.edge_index);
                             })
      .def_property_readonly("edge_attr",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.edge_attr);
                             })
      .def_property_readonly("u",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.u);
                             })
      .def_property_readonly("time_group_ids",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.time_group_ids);
                             })
      .def_property_readonly("node_ptr",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.node_ptr);
                             })
      .def_property_readonly("edge_ptr",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.edge_ptr);
                             })
      .def_property_readonly("y",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.y);
                             })
      .def_property_readonly("y_energy",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return WrapArray(self.y_energy);
                             })
      .def_property_readonly("num_graphs",
                             [](const pioneerml::GroupClassifierInputs& self) {
                               return self.num_graphs;
                             });
}

}  // namespace pioneerml::bindings
