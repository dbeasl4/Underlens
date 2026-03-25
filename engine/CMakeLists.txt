cmake_minimum_required(VERSION 3.16)
project(underlens_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get pybind11
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Build the Python extension module
pybind11_add_module(underlens_engine
    src/models.cpp
    src/bindings.cpp
)

target_include_directories(underlens_engine PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    include
)

target_link_libraries(underlens_engine PRIVATE
    ${CERES_LIBRARIES}
)
