cmake_minimum_required(VERSION 3.16)

project(pioneerml_dataloaders VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Arrow/Parquet
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE PIONEERML_DATALOADERS_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pioneerml_dataloaders/**/*.cpp)

add_library(${PROJECT_NAME} SHARED ${PIONEERML_DATALOADERS_SRC})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Arrow::arrow_shared
        Parquet::parquet_shared)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "pioneerml_dataloaders")

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(bindings)
