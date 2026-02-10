#include <pybind11/pybind11.h>

#include "arrow_utils.h"
#include "bindings.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/event_splitter_event_output_adapter.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindEventSplitterEventOutputAdapter(py::module_& m) {
  py::class_<pioneerml::output_adapters::graph::EventSplitterEventOutputAdapter>(m, "EventSplitterEventOutputAdapter")
      .def(py::init<>())
      .def("write_parquet",
           [](pioneerml::output_adapters::graph::EventSplitterEventOutputAdapter& adapter,
              const std::string& output_path,
              py::handle edge_pred,
              py::handle edge_ptr,
              py::handle edge_index,
              py::handle node_ptr,
              py::handle time_group_ids,
              py::handle graph_event_ids) {
             auto pred_arr = ImportArray(edge_pred);
             auto edge_ptr_arr = ImportArray(edge_ptr);
             auto edge_index_arr = ImportArray(edge_index);
             auto node_ptr_arr = ImportArray(node_ptr);
             auto time_group_arr = ImportArray(time_group_ids);
             auto event_arr = ImportArray(graph_event_ids);
             adapter.WriteParquet(output_path,
                                 pred_arr,
                                 edge_ptr_arr,
                                 edge_index_arr,
                                 node_ptr_arr,
                                 time_group_arr,
                                 event_arr);
           },
           py::arg("output_path"),
           py::arg("edge_predictions"),
           py::arg("edge_ptr"),
           py::arg("edge_index"),
           py::arg("node_ptr"),
           py::arg("time_group_ids"),
           py::arg("graph_event_ids"));
}

}  // namespace pioneerml::bindings
