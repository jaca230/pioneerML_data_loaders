#include <stdexcept>

#include <pybind11/pybind11.h>

#include "bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(pioneerml_dataloaders_python, m) {
  m.doc() = "Pybind11 bindings for pioneerml_dataloaders";

  auto m_batch = m.def_submodule("batch");
  pioneerml::bindings::BindBaseBatch(m_batch);
  pioneerml::bindings::BindGroupClassifierInputs(m_batch);
  pioneerml::bindings::BindGroupClassifierTargets(m_batch);

  auto m_dataloaders = m.def_submodule("dataloaders");
  pioneerml::bindings::BindBaseLoader(m_dataloaders);

  auto m_graph = m_dataloaders.def_submodule("graph");
  pioneerml::bindings::BindGroupClassifierLoader(m_graph);
}
