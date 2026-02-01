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

inline std::shared_ptr<arrow::Array> ImportArray(pybind11::handle obj) {
  pybind11::object capsules;
  if (pybind11::isinstance<pybind11::tuple>(obj)) {
    capsules = pybind11::reinterpret_borrow<pybind11::object>(obj);
  } else if (pybind11::hasattr(obj, "__arrow_c_array__")) {
    capsules = obj.attr("__arrow_c_array__")();
  } else {
    throw std::runtime_error("Expected Arrow C data capsule tuple.");
  }

  auto tup = pybind11::cast<pybind11::tuple>(capsules);
  if (tup.size() != 2) {
    throw std::runtime_error("Arrow C data tuple must have 2 items.");
  }

  auto* schema = reinterpret_cast<ArrowSchema*>(
      PyCapsule_GetPointer(tup[0].ptr(), "arrow_schema"));
  auto* array = reinterpret_cast<ArrowArray*>(
      PyCapsule_GetPointer(tup[1].ptr(), "arrow_array"));
  if (!schema || !array) {
    throw std::runtime_error("Invalid Arrow C data capsules.");
  }

  auto result = arrow::ImportArray(array, schema);
  if (!result.ok()) {
    throw std::runtime_error(result.status().ToString());
  }

  return result.MoveValueUnsafe();
}

}  // namespace pioneerml::bindings
