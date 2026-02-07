#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_splitter_event_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupSplitterEventOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::GroupSplitterEventOutputAdapter>(m, "GroupSplitterEventOutputAdapter")
      .def(py::init<>())
      .def("write_parquet",
           [](pioneerml::output_adapters::graph::GroupSplitterEventOutputAdapter& adapter,
              const std::string& output_path,
              py::handle node_pred,
              py::handle node_ptr,
              py::handle time_group_ids,
              py::handle graph_event_ids) {
             auto pred_arr = ImportArray(node_pred);
             auto node_ptr_arr = ImportArray(node_ptr);
             auto tg_arr = ImportArray(time_group_ids);
             auto event_arr = ImportArray(graph_event_ids);
             adapter.WriteParquet(output_path, pred_arr, node_ptr_arr, tg_arr, event_arr);
           },
           py::arg("output_path"),
           py::arg("node_predictions"),
           py::arg("node_ptr"),
           py::arg("time_group_ids"),
           py::arg("graph_event_ids"));
}

}  // namespace pioneerml::bindings
