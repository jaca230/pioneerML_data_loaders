find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    pioneerml_dataloaders_python
    bindings.cpp
    batch/base_batch_bindings.cpp
    batch/group_classifier_inputs_bindings.cpp
    batch/group_classifier_targets_bindings.cpp
    dataloaders/training_bundle_bindings.cpp
    dataloaders/inference_bundle_bindings.cpp
    dataloaders/graph/group_classifier_config_bindings.cpp
    dataloaders/graph/group_classifier_loader_bindings.cpp
)
target_link_libraries(pioneerml_dataloaders_python PRIVATE pioneerml_dataloaders)
target_include_directories(
    pioneerml_dataloaders_python
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
