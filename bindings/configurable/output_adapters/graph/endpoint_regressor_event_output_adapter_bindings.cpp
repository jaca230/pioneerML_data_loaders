#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/endpoint_regressor_event_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEndpointRegressorEventOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::EndpointRegressorEventOutputAdapter>(m, "EndpointRegressorEventOutputAdapter")
      .def(py::init<>())
      .def("write_parquet",
           [](pioneerml::output_adapters::graph::EndpointRegressorEventOutputAdapter& adapter,
              const std::string& output_path,
              py::handle group_pred,
              py::handle group_ptr,
              py::handle graph_event_ids) {
             auto pred_arr = ImportArray(group_pred);
             auto group_ptr_arr = ImportArray(group_ptr);
             auto event_arr = ImportArray(graph_event_ids);
             adapter.WriteParquet(output_path, pred_arr, group_ptr_arr, event_arr);
           },
           py::arg("output_path"),
           py::arg("group_predictions"),
           py::arg("group_ptr"),
           py::arg("graph_event_ids"));
}

}  // namespace pioneerml::bindings
