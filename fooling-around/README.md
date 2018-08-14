# Using Pybind11 and CUDA

## Motivation

- In my project, I need to build a custom fast image processing application using Python, but Python is extremely slow.
- Pybind11 is the way to go since you can create a C++ wrapper for Python.
- CUDA is another tool that I need to use since there are many thread execution that can be done in parallel.
    - OpenMP and `std::thread` also worked well, but CUDA can do even better.

## Setup

### Building

Created a `setup.py` based on the [cmake_examples]() tutorial on the official Pybind Github page.

__Note__: the Python paths should be changed since I hard coded it.


```
python setup.py build
```

This outputs the library in the build folder.

### Cleaning

```
python setup.py clean
```

## CMake

The worst part about setting up is creating the `CMakeLists.txt` file since I don't have much experience in creating build templates.
The problem that took a while to solve was building and linking the CUDA library with Pybind.
For this, instead of using `cuda_add_executable()`, you have to use `cuda_add_library()` since you cannot link executables (I did not know this until now...).
Also, when making the library, you also have make it `SHARED`.

For this project,

```
cuda_add_library(hello SHARED gpu_library.cu)
```

In the `CMakeLists.txt`, there are some parts that I don't understand:

- What is this property, and do we always need this?
    - `set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)`
- This part I'm sure that it only changes the library name, but I'm not too confident:
    - `set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")`


## Code...

