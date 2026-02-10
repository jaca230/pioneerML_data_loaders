#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/configurable/dataloaders/graph/event_splitter_event_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEventSplitterEventLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::EventSplitterEventLoader>(m, "EventSplitterEventLoader")
      .def(py::init<>())
      .def("load_config_json",
           [](pioneerml::dataloaders::graph::EventSplitterEventLoader& loader,
              const std::string& json_str) { loader.LoadConfig(nlohmann::json::parse(json_str)); },
           py::arg("json_str"));
}

}  // namespace pioneerml::bindings
