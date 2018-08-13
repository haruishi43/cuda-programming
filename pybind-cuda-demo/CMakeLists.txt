cmake_minimum_required(VERSION 3.4)
set(PROJECT_NAME cuda_pybind)
project(${PROJECT_NAME} LANGUAGES CXX)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_subdirectory(pybind11)
cuda_add_library(hello SHARED gpu_library.cu)

set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


pybind11_add_module(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE hello)



set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
