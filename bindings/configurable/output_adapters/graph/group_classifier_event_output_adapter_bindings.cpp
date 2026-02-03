#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_classifier_event_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierEventOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::GroupClassifierEventOutputAdapter>(
      m, "GroupClassifierEventOutputAdapter")
      .def(py::init<>())
      .def(
          "write_parquet",
          [](pioneerml::output_adapters::graph::GroupClassifierEventOutputAdapter& adapter,
             const std::string& output_path,
             py::handle pred,
             py::handle pred_energy,
             py::handle node_ptr,
             py::handle time_group_ids) {
            auto pred_arr = ImportArray(pred);
            auto energy_arr = pred_energy.is_none() ? nullptr : ImportArray(pred_energy);
            auto node_ptr_arr = ImportArray(node_ptr);
            auto tg_arr = ImportArray(time_group_ids);
            adapter.WriteParquet(output_path, pred_arr, energy_arr, node_ptr_arr, tg_arr);
          },
          py::arg("output_path"),
          py::arg("predictions"),
          py::arg("predictions_energy") = py::none(),
          py::arg("node_ptr"),
          py::arg("time_group_ids"));
}

}  // namespace pioneerml::bindings
