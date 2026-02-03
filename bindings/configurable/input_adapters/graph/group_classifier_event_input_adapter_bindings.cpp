#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_event_input_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierEventInputAdapter(py::module_& m) {
  py::class_<pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter>(
      m, "GroupClassifierEventInputAdapter")
      .def(py::init<>())
      .def("load_config", &pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::LoadConfig)
      .def("load_training",
           static_cast<pioneerml::dataloaders::TrainingBundle (
               pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::*)(
               const std::string&) const>(
               &pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::LoadTraining))
      .def("load_training",
           static_cast<pioneerml::dataloaders::TrainingBundle (
               pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::*)(
               const std::vector<std::string>&) const>(
               &pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::LoadTraining))
      .def("load_inference",
           static_cast<pioneerml::dataloaders::InferenceBundle (
               pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::*)(
               const std::string&) const>(
               &pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::LoadInference))
      .def("load_inference",
           static_cast<pioneerml::dataloaders::InferenceBundle (
               pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::*)(
               const std::vector<std::string>&) const>(
               &pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter::LoadInference))
      .def(
          "load_config_json",
          [](pioneerml::input_adapters::graph::GroupClassifierEventInputAdapter& adapter,
             const std::string& json_str) {
            adapter.LoadConfig(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"));
}

}  // namespace pioneerml::bindings
