#pragma once

#include <pybind11/pybind11.h>

namespace pioneerml::bindings {

void BindBaseBatch(pybind11::module_& m);
void BindGroupClassifierInputs(pybind11::module_& m);
void BindGroupClassifierTargets(pybind11::module_& m);

void BindBaseLoader(pybind11::module_& m);
void BindGroupClassifierLoader(pybind11::module_& m);

}  // namespace pioneerml::bindings
