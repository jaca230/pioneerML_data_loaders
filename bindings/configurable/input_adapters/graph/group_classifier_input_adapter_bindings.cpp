#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_input_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierInputAdapter(py::module_& m) {
  py::class_<pioneerml::input_adapters::graph::GroupClassifierInputAdapter>(
      m, "GroupClassifierInputAdapter")
      .def(py::init<>())
      .def("load_training",
           py::overload_cast<const std::string&>(
               &pioneerml::input_adapters::graph::GroupClassifierInputAdapter::LoadTraining,
               py::const_),
           py::arg("parquet_path"))
      .def("load_training",
           py::overload_cast<const std::vector<std::string>&>(
               &pioneerml::input_adapters::graph::GroupClassifierInputAdapter::LoadTraining,
               py::const_),
           py::arg("parquet_paths"))
      .def("load_inference",
           py::overload_cast<const std::string&>(
               &pioneerml::input_adapters::graph::GroupClassifierInputAdapter::LoadInference,
               py::const_),
           py::arg("parquet_path"))
      .def("load_inference",
           py::overload_cast<const std::vector<std::string>&>(
               &pioneerml::input_adapters::graph::GroupClassifierInputAdapter::LoadInference,
               py::const_),
           py::arg("parquet_paths"))
      .def(
          "load_config_json",
          [](pioneerml::input_adapters::graph::GroupClassifierInputAdapter& adapter,
             const std::string& json_str) {
            adapter.LoadConfig(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"));
}

}  // namespace pioneerml::bindings
