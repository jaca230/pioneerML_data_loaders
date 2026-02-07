#pragma once

#include <pybind11/pybind11.h>

namespace pioneerml::bindings {

void BindBaseBatch(pybind11::module_& m);
void BindGroupClassifierInputs(pybind11::module_& m);
void BindGroupClassifierTargets(pybind11::module_& m);
void BindGroupSplitterInputs(pybind11::module_& m);
void BindGroupSplitterTargets(pybind11::module_& m);
void BindGroupSplitterEventInputs(pybind11::module_& m);
void BindGroupSplitterEventTargets(pybind11::module_& m);

void BindBaseLoader(pybind11::module_& m);
void BindGroupClassifierLoader(pybind11::module_& m);
void BindGroupClassifierEventLoader(pybind11::module_& m);
void BindGroupSplitterLoader(pybind11::module_& m);
void BindGroupSplitterEventLoader(pybind11::module_& m);

void BindGroupClassifierInputAdapter(pybind11::module_& m);
void BindGroupClassifierEventInputAdapter(pybind11::module_& m);
void BindGroupSplitterInputAdapter(pybind11::module_& m);
void BindGroupSplitterEventInputAdapter(pybind11::module_& m);
void BindGroupClassifierOutputAdapter(pybind11::module_& m);
void BindGroupClassifierEventOutputAdapter(pybind11::module_& m);
void BindGroupSplitterOutputAdapter(pybind11::module_& m);
void BindGroupSplitterEventOutputAdapter(pybind11::module_& m);

void BindLogging(pybind11::module_& m);
void BindTiming(pybind11::module_& m);

}  // namespace pioneerml::bindings
