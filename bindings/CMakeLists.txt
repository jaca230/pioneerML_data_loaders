find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    pioneerml_dataloaders_python
    bindings.cpp
    batch/base_batch_bindings.cpp
    batch/group_classifier_inputs_bindings.cpp
    batch/group_classifier_targets_bindings.cpp
    configurable/dataloaders/base_loader_bindings.cpp
    configurable/dataloaders/graph/group_classifier_loader_bindings.cpp
    configurable/input_adapters/graph/group_classifier_input_adapter_bindings.cpp
    configurable/output_adapters/graph/group_classifier_output_adapter_bindings.cpp
    utils/logging_bindings.cpp
    utils/timing_bindings.cpp
)
target_link_libraries(pioneerml_dataloaders_python PRIVATE pioneerml_dataloaders)
target_include_directories(
    pioneerml_dataloaders_python
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
