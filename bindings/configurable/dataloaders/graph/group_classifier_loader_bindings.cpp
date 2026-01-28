#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::GroupClassifierLoader>(
      m, "GroupClassifierLoader")
      .def(py::init<>())
      .def("load_training",
           py::overload_cast<const std::string&>(
               &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadTraining,
               py::const_),
           py::arg("parquet_path"))
      .def("load_training",
           py::overload_cast<const std::vector<std::string>&>(
               &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadTraining,
               py::const_),
           py::arg("parquet_paths"))
      .def("load_inference",
           py::overload_cast<const std::string&>(
               &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadInference,
               py::const_),
           py::arg("parquet_path"))
      .def("load_inference",
           py::overload_cast<const std::vector<std::string>&>(
               &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadInference,
               py::const_),
           py::arg("parquet_paths"))
      .def(
          "load_config_json",
          [](pioneerml::dataloaders::graph::GroupClassifierLoader& loader,
             const std::string& json_str) {
            loader.LoadConfig(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"));
}

}  // namespace pioneerml::bindings
