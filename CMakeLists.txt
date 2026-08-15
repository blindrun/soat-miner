# Cross-platform build for SOAT Miner (Linux + Windows).
#
# Linux users can also just run `make`. This exists mainly so Windows builds
# work without hand-rolling MSVC command lines - it is the only supported way
# to build the CUDA binary on Windows, because nvcc needs MSVC as its host
# compiler and so cannot be cross-compiled from Linux.
#
#   cmake -B build
#   cmake --build build --config Release
#
# By default this builds SASS for the common generations plus PTX for the
# newest one the installed toolkit knows, so the binary runs on cards newer
# than the toolkit via driver JIT. Pass -DCMAKE_CUDA_ARCHITECTURES=89 to build
# a single architecture, which compiles about 4x faster.
#
#   120 = Blackwell (RTX 5090/5080/5070) - needs CUDA Toolkit 12.8 or newer
#    89 = Ada       (RTX 4090/4080/4070)
#    86 = Ampere    (RTX 3090/3080/3060)
#    75 = Turing    (RTX 2080/1660)

cmake_minimum_required(VERSION 3.18)

# Has to be tested BEFORE project(): enabling the CUDA language defines
# CMAKE_CUDA_ARCHITECTURES itself (to 52), so afterwards there is no way left
# to tell "the user asked for an architecture" from "CMake filled in a default".
# Checking after project() silently built everything for sm_52.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(SOAT_ARCH_FROM_USER TRUE)
else()
  set(SOAT_ARCH_FROM_USER FALSE)
endif()

project(soat-miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Architecture selection. This has to come after project(), which is what sets
# CMAKE_CUDA_COMPILER_VERSION.
#
# "-real" emits SASS for that architecture; "-virtual" emits PTX, which the
# driver JITs onto anything newer. Shipping the PTX is what lets a build from
# CUDA 12.4 (whose newest target is sm_90) run on a Blackwell card at all.
if(NOT SOAT_ARCH_FROM_USER)
  set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120-real 120-virtual)
  else()
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90-virtual)
    message(STATUS
      "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot emit Blackwell (sm_120) SASS; "
      "RTX 50-series will run via PTX JIT. Install CUDA 12.8+ for native code.")
  endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# run.cpp and stratum.cpp are shared with the Vulkan backend, so they are .cpp
# rather than .cu - but miner.cu calls straight into them, so the CUDA binary
# has to compile them too. Leaving them out is not a missing optimisation, it
# is a link error (undefined reference to om::runMiner), which is what this
# file shipped with.
set(SOAT_CUDA_SOURCES
  src/core/miner.cu
  src/core/registry.cu
  src/algos/autolykos2/algo.cu
  src/core/run.cpp
  src/core/stratum.cpp
)

# Push the two .cpp files through nvcc, matching the Makefile's `-x cu`.
set_source_files_properties(
  src/core/run.cpp
  src/core/stratum.cpp
  PROPERTIES LANGUAGE CUDA
)

add_executable(soat-miner ${SOAT_CUDA_SOURCES})

target_include_directories(soat-miner PRIVATE src)

# Separable compilation is deliberately OFF. It runs the device linker, and
# nvlink resolves to real SASS and discards PTX - so a separably-compiled
# binary carries no JIT fallback and refuses to start on any architecture it
# was not explicitly built for. Whole-program keeps the PTX at identical
# measured hashrate (217.6 vs 217.5 MH/s on a 4090).
set_target_properties(soat-miner PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
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
