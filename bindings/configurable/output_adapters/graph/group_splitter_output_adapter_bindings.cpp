#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_splitter_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::GroupSplitterOutputAdapter>(m, "GroupSplitterOutputAdapter")
      .def(py::init<>())
      .def("write_parquet",
           [](pioneerml::output_adapters::graph::GroupSplitterOutputAdapter& adapter,
              const std::string& output_path,
              py::handle node_pred,
              py::handle node_ptr,
              py::handle graph_event_ids,
              py::handle graph_group_ids) {
             auto pred_arr = ImportArray(node_pred);
             auto node_ptr_arr = ImportArray(node_ptr);
             auto event_arr = ImportArray(graph_event_ids);
             auto group_arr = ImportArray(graph_group_ids);
             adapter.WriteParquet(output_path, pred_arr, node_ptr_arr, event_arr, group_arr);
           },
           py::arg("output_path"),
           py::arg("node_predictions"),
           py::arg("node_ptr"),
           py::arg("graph_event_ids"),
           py::arg("graph_group_ids"));
}

}  // namespace pioneerml::bindings
