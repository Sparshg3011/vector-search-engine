# Builds the _vecstore_gpu extension into gpu/vecstore_gpu/, where
# vecstore_gpu/__init__.py imports it from. Nothing here is pip
# installed - gpu/ goes on sys.path instead.
#
#   cmake -S gpu -B gpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build gpu/build
#
# gpu/setup_pod.sh runs exactly that.

cmake_minimum_required(VERSION 3.24)  # 3.24 for CUDA_ARCHITECTURES "native"
project(vecstore_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# "native" targets whatever card the pod has. Override when building for
# a different one, e.g. -DCMAKE_CUDA_ARCHITECTURES=89 for Ada (4090,
# L40S), 86 for Ampere (A10G, 3090), 80 for A100.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

# An installed pybind11 wins, so the build needs no network - cluster
# compute nodes are often offline. Point at a pip install with
# -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir); otherwise fetch it.
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  message(STATUS "pybind11 ${pybind11_VERSION} from ${pybind11_DIR}")
else()
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_vecstore_gpu
  src/bindings.cpp
  src/launchers.cu
)

target_include_directories(_vecstore_gpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(_vecstore_gpu PRIVATE CUDA::cudart CUDA::cublas)

# -lineinfo keeps source correlation in ncu without the cost of -G, and
# the gap analysis is read off ncu. Kernels are one translation unit
# (launchers.cu includes them), so no separable compilation.
target_compile_options(_vecstore_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(_vecstore_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vecstore_gpu
)

# multi-config generators ignore the plain property above
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${cfg} CFG)
  set_target_properties(_vecstore_gpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_CURRENT_SOURCE_DIR}/vecstore_gpu
  )
endforeach()
