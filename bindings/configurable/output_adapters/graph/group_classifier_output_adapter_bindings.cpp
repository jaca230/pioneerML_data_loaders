#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_classifier_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::GroupClassifierOutputAdapter>(
      m, "GroupClassifierOutputAdapter")
      .def(py::init<>())
      .def(
          "write_parquet",
          [](pioneerml::output_adapters::graph::GroupClassifierOutputAdapter& adapter,
             const std::string& output_path,
             py::handle pred,
             py::handle pred_energy,
             py::handle graph_event_ids,
             py::handle graph_group_ids) {
            auto pred_arr = ImportArray(pred);
            auto energy_arr = pred_energy.is_none() ? nullptr : ImportArray(pred_energy);
            auto event_arr = ImportArray(graph_event_ids);
            auto group_arr = ImportArray(graph_group_ids);
            adapter.WriteParquet(output_path, pred_arr, energy_arr, event_arr, group_arr);
          },
          py::arg("output_path"),
          py::arg("predictions"),
          py::arg("predictions_energy") = py::none(),
          py::arg("graph_event_ids"),
          py::arg("graph_group_ids"));
}

}  // namespace pioneerml::bindings
