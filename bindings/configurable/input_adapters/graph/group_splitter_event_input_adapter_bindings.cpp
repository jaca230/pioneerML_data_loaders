#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_splitter_event_input_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterEventInputAdapter(py::module_& m) {
  py::class_<pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter>(m, "GroupSplitterEventInputAdapter")
      .def(py::init<>())
      .def("load_training",
           py::overload_cast<const nlohmann::json&>(
               &pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter::LoadTraining, py::const_),
           py::arg("input_spec"))
      .def("load_inference",
           py::overload_cast<const nlohmann::json&>(
               &pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter::LoadInference, py::const_),
           py::arg("input_spec"))
      .def(
          "load_training_json",
          [](const pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter& adapter,
             const std::string& json_str) {
            return adapter.LoadTraining(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"))
      .def(
          "load_inference_json",
          [](const pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter& adapter,
             const std::string& json_str) {
            return adapter.LoadInference(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"))
      .def("load_config_json",
           [](pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter& adapter,
              const std::string& json_str) { adapter.LoadConfig(nlohmann::json::parse(json_str)); },
           py::arg("json_str"));
}

}  // namespace pioneerml::bindings
