#include <pybind11/pybind11.h>

#include <string>

#include <nlohmann/json.hpp>

#include "bindings.h"
#include "pioneerml_dataloaders/io/json_reader.h"
#include "pioneerml_dataloaders/utils/timing/timing_configurator.h"
#include "pioneerml_dataloaders/utils/timing/timing_reporter.h"
#include "pioneerml_dataloaders/utils/timing/timing_settings.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindTiming(py::module_& m) {
  m.def(
      "configure_timing_defaults",
      []() {
        utils::timing::TimingConfigurator configurator;
        return configurator.Configure(nlohmann::json::object());
      },
      "Configure timing with library defaults.");

  m.def(
      "configure_timing_json",
      [](const std::string& json_str) {
        utils::timing::TimingConfigurator configurator;
        return configurator.Configure(nlohmann::json::parse(json_str));
      },
      py::arg("json_str"),
      "Configure timing from a JSON string.");

  m.def(
      "configure_timing_from_file",
      [](const std::string& filepath) {
        io::JsonReader reader;
        auto j = reader.ReadFile(filepath);
        if (j.contains("timing")) {
          j = j.at("timing");
        }
        utils::timing::TimingConfigurator configurator;
        return configurator.Configure(j);
      },
      py::arg("filepath"),
      "Configure timing from a JSON file (uses the 'timing' key if present)."
  );

  m.def(
      "configure_timing_enabled",
      [](bool enabled) {
        utils::timing::TimingSettings::Instance().SetEnabled(enabled);
      },
      py::arg("enabled"),
      "Enable or disable timing collection.");

  m.def(
      "log_timings",
      []() { utils::timing::TimingReporter::LogTimings(); },
      "Log collected timing stats at debug level.");
}

}  // namespace pioneerml::bindings
