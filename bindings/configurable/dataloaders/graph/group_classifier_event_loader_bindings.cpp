#include <pybind11/pybind11.h>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_event_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierEventLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::GroupClassifierEventLoader>(
      m, "GroupClassifierEventLoader")
      .def(py::init<>())
      .def("load_config", &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadConfig)
      .def("load_training",
           static_cast<pioneerml::dataloaders::TrainingBundle (
               pioneerml::dataloaders::graph::GroupClassifierEventLoader::*)(
               const std::string&) const>(
               &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadTraining))
      .def("load_training",
           static_cast<pioneerml::dataloaders::TrainingBundle (
               pioneerml::dataloaders::graph::GroupClassifierEventLoader::*)(
               const std::vector<std::string>&) const>(
               &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadTraining))
      .def("load_inference",
           static_cast<pioneerml::dataloaders::InferenceBundle (
               pioneerml::dataloaders::graph::GroupClassifierEventLoader::*)(
               const std::string&) const>(
               &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadInference))
      .def("load_inference",
           static_cast<pioneerml::dataloaders::InferenceBundle (
               pioneerml::dataloaders::graph::GroupClassifierEventLoader::*)(
               const std::vector<std::string>&) const>(
               &pioneerml::dataloaders::graph::GroupClassifierEventLoader::LoadInference));
}

}  // namespace pioneerml::bindings
