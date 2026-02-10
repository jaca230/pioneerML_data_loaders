#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/endpoint_regressor_input_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEndpointRegressorInputAdapter(py::module_& m) {
  py::class_<pioneerml::input_adapters::graph::EndpointRegressorInputAdapter>(m, "EndpointRegressorInputAdapter")
      .def(py::init<>())
      .def("load_training",
           py::overload_cast<const nlohmann::json&>(
               &pioneerml::input_adapters::graph::EndpointRegressorInputAdapter::LoadTraining,
               py::const_),
           py::arg("input_spec"))
      .def("load_inference",
           py::overload_cast<const nlohmann::json&>(
               &pioneerml::input_adapters::graph::EndpointRegressorInputAdapter::LoadInference,
               py::const_),
           py::arg("input_spec"))
      .def("load_training_json",
           [](const pioneerml::input_adapters::graph::EndpointRegressorInputAdapter& adapter,
              const std::string& json_str) {
             return adapter.LoadTraining(nlohmann::json::parse(json_str));
           },
           py::arg("json_str"))
      .def("load_inference_json",
           [](const pioneerml::input_adapters::graph::EndpointRegressorInputAdapter& adapter,
              const std::string& json_str) {
             return adapter.LoadInference(nlohmann::json::parse(json_str));
           },
           py::arg("json_str"))
      .def("load_config_json",
           [](pioneerml::input_adapters::graph::EndpointRegressorInputAdapter& adapter,
              const std::string& json_str) { adapter.LoadConfig(nlohmann::json::parse(json_str)); },
           py::arg("json_str"));
}

}  // namespace pioneerml::bindings
