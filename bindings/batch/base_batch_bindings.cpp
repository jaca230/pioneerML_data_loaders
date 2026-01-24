#include <pybind11/pybind11.h>

#include "pioneerml_dataloaders/batch/base_batch.h"
#include "bindings.h"

namespace py = pybind11;

namespace pioneerml::bindings {

void BindBaseBatch(py::module_& m) {
  py::class_<pioneerml::BaseBatch>(m, "BaseBatch");
}

}  // namespace pioneerml::bindings
