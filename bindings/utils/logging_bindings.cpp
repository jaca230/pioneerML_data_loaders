#include <pybind11/pybind11.h>

#include <string>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/io/json_reader.h"
#include "pioneerml_dataloaders/utils/logging/logger_configurator.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindLogging(py::module_& m) {
  m.def(
      "configure_logger_defaults",
      []() {
        utils::logging::LoggerConfigurator configurator;
        return configurator.Configure(nlohmann::json::object());
      },
      "Configure the logger with library defaults.");

  m.def(
      "configure_logger_json",
      [](const std::string& json_str) {
        utils::logging::LoggerConfigurator configurator;
        return configurator.Configure(nlohmann::json::parse(json_str));
      },
      py::arg("json_str"),
      "Configure the logger from a JSON string.");

  m.def(
      "configure_logger_from_file",
      [](const std::string& filepath) {
        io::JsonReader reader;
        auto j = reader.ReadFile(filepath);
        if (j.contains("logger")) {
          j = j.at("logger");
        }
        utils::logging::LoggerConfigurator configurator;
        return configurator.Configure(j);
      },
      py::arg("filepath"),
      "Configure the logger from a JSON file (uses the 'logger' key if present)."
  );
}

}  // namespace pioneerml::bindings
