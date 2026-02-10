#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/endpoint_regressor_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEndpointRegressorOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::EndpointRegressorOutputAdapter>(m, "EndpointRegressorOutputAdapter")
      .def(py::init<>())
      .def("write_parquet",
           [](pioneerml::output_adapters::graph::EndpointRegressorOutputAdapter& adapter,
              const std::string& output_path,
              py::handle group_pred,
              py::handle graph_event_ids,
              py::handle graph_group_ids) {
             auto pred_arr = ImportArray(group_pred);
             auto event_arr = ImportArray(graph_event_ids);
             auto group_arr = ImportArray(graph_group_ids);
             adapter.WriteParquet(output_path, pred_arr, event_arr, group_arr);
           },
           py::arg("output_path"),
           py::arg("group_predictions"),
           py::arg("graph_event_ids"),
           py::arg("graph_group_ids"));
}

}  // namespace pioneerml::bindings
