# Cross-platform build for SOAT Miner (Linux + Windows).
#
# Linux users can also just run `make`. This exists mainly so Windows builds
# work without hand-rolling MSVC command lines.
#
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build --config Release
#
# CUDA architectures:
#   89 = Ada    (RTX 4090/4080/4070)
#   86 = Ampere (RTX 3090/3080/3060)
#   75 = Turing (RTX 2080/1660)

cmake_minimum_required(VERSION 3.18)
project(soat-miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(soat-miner
  src/core/miner.cu
  src/core/registry.cu
  src/algos/autolykos2/algo.cu
)

target_include_directories(soat-miner PRIVATE src)

set_target_properties(soat-miner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(soat-miner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# NVML is loaded at runtime (dlopen/LoadLibrary), so it is not linked here.
# Only the dynamic loader itself needs linking, and only on Unix.
if(UNIX)
  target_link_libraries(soat-miner PRIVATE dl)
endif()

if(WIN32)
  target_link_libraries(soat-miner PRIVATE ws2_32)
endif()

install(TARGETS soat-miner RUNTIME DESTINATION bin)
