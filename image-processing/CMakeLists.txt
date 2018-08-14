cmake_minimum_required(VERSION 3.4)
set(PROJECT_NAME extension)
project(${PROJECT_NAME} LANGUAGES CXX)

find_package(CUDA QUIET REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(DEPENDENCIES ${OpenCV_LIBS} ${EIGEN3_LIBS})

add_subdirectory(pybind11)
cuda_add_library(hello SHARED gpu_library.cu)
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
pybind11_add_module(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../")
target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES} hello)
