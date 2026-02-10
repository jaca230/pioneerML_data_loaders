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
  pioneerml::bindings::BindGroupSplitterInputs(m_batch);
  pioneerml::bindings::BindGroupSplitterTargets(m_batch);
  pioneerml::bindings::BindGroupSplitterEventInputs(m_batch);
  pioneerml::bindings::BindGroupSplitterEventTargets(m_batch);
  pioneerml::bindings::BindEventSplitterEventInputs(m_batch);
  pioneerml::bindings::BindEventSplitterEventTargets(m_batch);
  pioneerml::bindings::BindEndpointRegressorInputs(m_batch);
  pioneerml::bindings::BindEndpointRegressorTargets(m_batch);

  auto m_dataloaders = m.def_submodule("dataloaders");
  pioneerml::bindings::BindBaseLoader(m_dataloaders);

  auto m_graph = m_dataloaders.def_submodule("graph");
  pioneerml::bindings::BindGroupClassifierLoader(m_graph);
  pioneerml::bindings::BindGroupClassifierEventLoader(m_graph);
  pioneerml::bindings::BindGroupSplitterLoader(m_graph);
  pioneerml::bindings::BindGroupSplitterEventLoader(m_graph);
  pioneerml::bindings::BindEventSplitterEventLoader(m_graph);
  pioneerml::bindings::BindEndpointRegressorLoader(m_graph);
  pioneerml::bindings::BindEndpointRegressorEventLoader(m_graph);

  auto m_adapters = m.def_submodule("adapters");
  auto m_input = m_adapters.def_submodule("input");
  auto m_output = m_adapters.def_submodule("output");
  auto m_input_graph = m_input.def_submodule("graph");
  pioneerml::bindings::BindGroupClassifierInputAdapter(m_input_graph);
  pioneerml::bindings::BindGroupClassifierEventInputAdapter(m_input_graph);
  pioneerml::bindings::BindGroupSplitterInputAdapter(m_input_graph);
  pioneerml::bindings::BindGroupSplitterEventInputAdapter(m_input_graph);
  pioneerml::bindings::BindEventSplitterEventInputAdapter(m_input_graph);
  pioneerml::bindings::BindEndpointRegressorInputAdapter(m_input_graph);
  pioneerml::bindings::BindEndpointRegressorEventInputAdapter(m_input_graph);
  auto m_output_graph = m_output.def_submodule("graph");
  pioneerml::bindings::BindGroupClassifierOutputAdapter(m_output_graph);
  pioneerml::bindings::BindGroupClassifierEventOutputAdapter(m_output_graph);
  pioneerml::bindings::BindGroupSplitterOutputAdapter(m_output_graph);
  pioneerml::bindings::BindGroupSplitterEventOutputAdapter(m_output_graph);
  pioneerml::bindings::BindEventSplitterEventOutputAdapter(m_output_graph);
  pioneerml::bindings::BindEndpointRegressorOutputAdapter(m_output_graph);
  pioneerml::bindings::BindEndpointRegressorEventOutputAdapter(m_output_graph);

  auto m_utils = m.def_submodule("utils");
  auto m_logging = m_utils.def_submodule("logging");
  pioneerml::bindings::BindLogging(m_logging);
  auto m_timing = m_utils.def_submodule("timing");
  pioneerml::bindings::BindTiming(m_timing);
}
