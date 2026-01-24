#include <pybind11/pybind11.h>

#include "bindings.h"
#include "pioneerml_dataloaders/dataloaders/graph/group_classifier_loader.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindGroupClassifierLoader(py::module_& m) {
  py::class_<pioneerml::dataloaders::graph::GroupClassifierLoader>(
      m, "GroupClassifierLoader")
      .def(py::init<pioneerml::dataloaders::graph::GroupClassifierConfig>(),
           py::arg("config") =
               pioneerml::dataloaders::graph::GroupClassifierConfig{})
      .def("load_training",
           &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadTraining,
           py::arg("parquet_path"))
      .def("load_inference",
           &pioneerml::dataloaders::graph::GroupClassifierLoader::LoadInference,
           py::arg("parquet_path"));
}

}  // namespace pioneerml::bindings
