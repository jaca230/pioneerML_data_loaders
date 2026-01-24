#pragma once

#include <memory>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <pybind11/pybind11.h>

namespace pioneerml::bindings {

inline pybind11::object WrapArray(const std::shared_ptr<arrow::Array>& array) {
  if (!array) {
    return pybind11::none();
  }

  auto* out_array = new ArrowArray();
  auto* out_schema = new ArrowSchema();

  auto status = arrow::ExportArray(*array, out_array, out_schema);
  if (!status.ok()) {
    delete out_array;
    delete out_schema;
    throw std::runtime_error(status.ToString());
  }

  auto capsule_schema = pybind11::capsule(out_schema, "arrow_schema", [](PyObject* capsule) {
    auto* schema =
        reinterpret_cast<ArrowSchema*>(PyCapsule_GetPointer(capsule, "arrow_schema"));
    if (schema && schema->release) {
      schema->release(schema);
    }
    delete schema;
  });

  auto capsule_array = pybind11::capsule(out_array, "arrow_array", [](PyObject* capsule) {
    auto* arr =
        reinterpret_cast<ArrowArray*>(PyCapsule_GetPointer(capsule, "arrow_array"));
    if (arr && arr->release) {
      arr->release(arr);
    }
    delete arr;
  });

  return pybind11::make_tuple(capsule_schema, capsule_array);
}

}  // namespace pioneerml::bindings
