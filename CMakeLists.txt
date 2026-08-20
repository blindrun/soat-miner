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

# "Did the user pick an architecture?" - and it takes two tricks to answer.
#
# It has to be tested BEFORE project(), because enabling the CUDA language
# defines CMAKE_CUDA_ARCHITECTURES itself (to 52); after that there is no way
# left to tell a user's choice from CMake's default.
#
# And the answer has to be cached on the FIRST configure only. `cmake --build`
# re-runs configure, and by then CMake's default IS in the cache, so
# recomputing makes the second pass believe the user asked for 52.
#
# Both were found the same way: the build printed the right architectures and
# produced an sm_52 binary anyway.
if(NOT DEFINED SOAT_ARCH_FROM_USER)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(SOAT_ARCH_FROM_USER TRUE CACHE INTERNAL "CUDA arch came from the user")
  else()
    set(SOAT_ARCH_FROM_USER FALSE CACHE INTERNAL "CUDA arch came from the user")
  endif()
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
  # Ask the toolkit what it can actually target rather than assuming. Every
  # CUDA major release drops old architectures - 12.x dropped Kepler, 13.x
  # dropped Maxwell/Pascal/Volta - so a hardcoded list eventually turns into a
  # hard build failure on a newer toolkit, for a wanted architecture nobody
  # noticed had gone.
  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
    OUTPUT_VARIABLE SOAT_NVCC_ARCHES
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(CMAKE_CUDA_ARCHITECTURES "")
  foreach(cc 75 86 89 120)
    if(SOAT_NVCC_ARCHES MATCHES "compute_${cc}(\n|$)")
      list(APPEND CMAKE_CUDA_ARCHITECTURES "${cc}-real")
    endif()
  endforeach()

  # PTX for the newest architecture this toolkit knows, so the driver can JIT
  # forward onto cards released after it. Without this the binary runs only on
  # the exact architectures above - which is how the v0.1.1 release shipped
  # Ada-only and refused to start on everything else.
  set(SOAT_NEWEST_VIRTUAL "")
  foreach(cc 90 100 110 120 130)
    if(SOAT_NVCC_ARCHES MATCHES "compute_${cc}(\n|$)")
      set(SOAT_NEWEST_VIRTUAL "${cc}")
    endif()
  endforeach()
  if(SOAT_NEWEST_VIRTUAL)
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${SOAT_NEWEST_VIRTUAL}-virtual")
  endif()

  # FORCE into the cache: a plain set() does not survive the re-configure that
  # `cmake --build` performs, and CMake's own default (52) wins instead.
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CACHE STRING "CUDA architectures to build" FORCE)

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "could not determine any usable CUDA architecture from "
      "`${CMAKE_CUDA_COMPILER} --list-gpu-arch`. Pass one explicitly, e.g. "
      "-DCMAKE_CUDA_ARCHITECTURES=89")
  endif()

  if(NOT SOAT_NVCC_ARCHES MATCHES "compute_120")
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
  src/algos/pearl-pow/algo.cu
  src/algos/sha3-256t/algo.cu
  src/core/run.cpp
  src/core/stratum.cpp
  src/core/stratum_btc.cpp
)

# Push the two .cpp files through nvcc, matching the Makefile's `-x cu`.
set_source_files_properties(
  src/core/run.cpp
  src/core/stratum.cpp
  src/core/stratum_btc.cpp
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

# Offline protocol gate. It uses a checked-in transcript only: no network,
# wallet, pool, GPU, or miner process. Keep it in CTest so the Windows CUDA
# workflow proves that its CMake build includes the BC3 client sources.
include(CTest)
if(BUILD_TESTING)
  find_package(Threads REQUIRED)
  add_executable(test-btc-stratum
    tests/test_btc_stratum.cpp
    src/core/stratum_btc.cpp
  )
  target_include_directories(test-btc-stratum PRIVATE src)
  target_link_libraries(test-btc-stratum PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(test-btc-stratum PRIVATE ws2_32)
  endif()
  add_test(NAME btc-stratum-offline
           COMMAND test-btc-stratum
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/btc_stratum_v1.jsonl)
  add_executable(test-bc3-destination tests/test_bc3_destination.cpp)
  target_include_directories(test-bc3-destination PRIVATE src)
  add_test(NAME bc3-destination-schema COMMAND test-bc3-destination)
endif()
