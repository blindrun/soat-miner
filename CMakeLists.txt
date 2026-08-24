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
# BOTH BACKENDS, and each one optional.
#
# This file used to say `project(... LANGUAGES CXX CUDA)`, which made a CUDA
# toolkit a hard requirement of configuring at all. The effect was that the
# people who cannot use the CUDA binary were also the people who could not run
# cmake: an AMD or Intel card has no CUDA to find, so `cmake -B build` failed
# outright on the only machines the Vulkan binary is for. There was no Vulkan
# target here either, so the Windows Vulkan .exe could only ever be produced by
# cross-compiling it from Linux with mingw - which is not something a Windows
# user can be asked to do to build the binary their own card needs.
#
# So both languages are detected rather than required, and either backend can
# be built alone:
#
#   cmake -B build                      # build whatever this machine can
#   cmake -B build -DSOAT_BUILD_CUDA=OFF
#   cmake -B build -DSOAT_BUILD_VULKAN=OFF
#
# The Vulkan target needs glslangValidator (or glslc) and Python 3 to turn the
# .comp shaders into embedded SPIR-V, exactly as the Makefile does. Both come
# with the Vulkan SDK; if they are missing the Vulkan target is skipped with a
# message rather than failing the configure, so a CUDA-only box is unaffected.
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

project(soat-miner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SOAT_BUILD_CUDA   "Build the CUDA binary (needs a CUDA toolkit)" ON)
option(SOAT_BUILD_VULKAN "Build the Vulkan binary (needs Vulkan + glslang + Python3)" ON)

# --- which backends can this machine actually build? ------------------------
#
# Detected, not required. A box with one vendor's card has no reason to carry
# the other vendor's toolchain, and refusing to configure without it is what
# kept Windows AMD users from building the only binary that runs on their card.
include(CheckLanguage)
set(SOAT_HAVE_CUDA FALSE)
if(SOAT_BUILD_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(SOAT_HAVE_CUDA TRUE)
  else()
    message(STATUS "no CUDA toolkit found - skipping the CUDA binary")
  endif()
endif()

# Vulkan needs three things, and the two shader tools are the ones usually
# missing: the loader/headers, a GLSL compiler, and Python 3 for the same
# embed script the Makefile uses. glslangValidator and glslc are
# interchangeable here; the SDK ships both, some Linux distributions package
# only one.
set(SOAT_HAVE_VULKAN FALSE)
if(SOAT_BUILD_VULKAN)
  find_package(Vulkan QUIET)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  find_program(SOAT_GLSLANG
    NAMES glslangValidator glslang
    HINTS ENV VULKAN_SDK
    PATH_SUFFIXES bin Bin Bin32)
  find_program(SOAT_GLSLC
    NAMES glslc
    HINTS ENV VULKAN_SDK
    PATH_SUFFIXES bin Bin Bin32)
  if(NOT Vulkan_FOUND)
    message(STATUS "no Vulkan SDK found - skipping the Vulkan binary")
  elseif(NOT Python3_FOUND)
    message(STATUS "no Python 3 found - skipping the Vulkan binary "
                   "(it embeds the compiled shaders)")
  elseif(NOT SOAT_GLSLANG AND NOT SOAT_GLSLC)
    message(STATUS "no glslangValidator or glslc found - skipping the Vulkan "
                   "binary. Both ship with the Vulkan SDK.")
  else()
    set(SOAT_HAVE_VULKAN TRUE)
  endif()
endif()

if(NOT SOAT_HAVE_CUDA AND NOT SOAT_HAVE_VULKAN)
  message(FATAL_ERROR
    "neither backend can be built here. The CUDA binary needs a CUDA toolkit; "
    "the Vulkan binary needs the Vulkan SDK (which provides glslangValidator) "
    "and Python 3.")
endif()

# Architecture selection. This has to come after enable_language(CUDA), which
# is what sets CMAKE_CUDA_COMPILER_VERSION - it used to be project() that did,
# before CUDA became optional above.
#
# "-real" emits SASS for that architecture; "-virtual" emits PTX, which the
# driver JITs onto anything newer. Shipping the PTX is what lets a build from
# CUDA 12.4 (whose newest target is sm_90) run on a Blackwell card at all.
if(SOAT_HAVE_CUDA AND NOT SOAT_ARCH_FROM_USER)
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
if(SOAT_HAVE_CUDA)
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
endif()  # SOAT_HAVE_CUDA

# ---------------------------------------------------------------------------
# The Vulkan binary: AMD, Intel, and NVIDIA.
#
# This mirrors the Makefile's `vulkan` target rather than inventing a second
# recipe. Each .comp is compiled to SPIR-V and then embedded as a uint32 array
# by scripts/embed_spirv.py, so the binary carries every shader and needs no
# files beside it at runtime.
#
# The shader table is (tag, source, spv name, embedded symbol, target env,
# include dir), matching the
# tables in src/algos/*/algo.mk. The .spv names are not derivable from the
# source names - autolykos2 already owns kernel.spv, so Pearl's kernel.comp
# becomes kernel_pearl.spv - and neither are the symbols, which is why this is
# a table and not a glob.
# ---------------------------------------------------------------------------
if(SOAT_HAVE_VULKAN)
  set(SOAT_SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SOAT_SHADER_GEN_DIR}")

  # Pearl's shaders need cooperative matrix, which is Vulkan 1.3. Everything
  # else targets 1.2 and must stay there.
  set(SOAT_SHADERS
    "spirv|src/algos/autolykos2/kernel.comp|kernel|kAutolykosSpirv|vulkan1.2|-"
    "spirv_sha3|src/algos/sha3-256t/kernel.comp|kernel_sha3|kSha3Spirv|vulkan1.2|-"
    "spirv_kawpow|src/algos/kawpow/kernel.comp|kawpow|kKawpowSpirv|vulkan1.2|-"
    "spirv_kawpow_spec|src/algos/kawpow/kernel_spec.comp|kawpow_spec|kKawpowSpecSpirv|vulkan1.2|-"
    "spirv_dag|src/algos/progpow/dag.comp|dag|kProgpowDagSpirv|vulkan1.2|-"
    "spirv_pearl_gemm|src/algos/pearl-pow/kernel.comp|kernel_pearl|kPearlGemmSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_chunk|src/algos/pearl-pow/merkle_chunk.comp|merkle_chunk|kPearlMerkleChunkSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_reduce|src/algos/pearl-pow/merkle_reduce.comp|merkle_reduce|kPearlMerkleReduceSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_root|src/algos/pearl-pow/merkle_root.comp|merkle_root|kPearlMerkleRootSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_genmatrix|src/algos/pearl-pow/genmatrix.comp|genmatrix|kPearlGenMatrixSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_transpose|src/algos/pearl-pow/transpose.comp|transpose|kPearlTransposeSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_commitments|src/algos/pearl-pow/commitments.comp|commitments|kPearlCommitmentsSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_noise|src/algos/pearl-pow/noise.comp|noise|kPearlNoiseSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_applynoise|src/algos/pearl-pow/apply_noise.comp|apply_noise|kPearlApplyNoiseSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_powscan|src/algos/pearl-pow/powscan.comp|powscan|kPearlPowScanSpirv|vulkan1.3|src/algos/pearl-pow"
    "spirv_pearl_dp|src/algos/pearl-pow/kernel_dp.comp|kernel_dp|kPearlDpSpirv|vulkan1.3|src/algos/pearl-pow"
  )

  # blake3.glsl has no main(), so it is #included rather than compiled. It is
  # still a real dependency of every Pearl shader: glslang will not tell the
  # build system it changed, so editing it must rebuild them explicitly.
  set(SOAT_SHADER_EXTRA_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/src/algos/pearl-pow/blake3.glsl")

  set(SOAT_SPIRV_SOURCES "")
  foreach(entry IN LISTS SOAT_SHADERS)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 tag)
    list(GET parts 1 src)
    list(GET parts 2 spvname)
    list(GET parts 3 symbol)
    list(GET parts 4 targetenv)
    list(GET parts 5 incdir)

    set(spv "${SOAT_SHADER_GEN_DIR}/${spvname}.spv")
    set(gen "${SOAT_SHADER_GEN_DIR}/${tag}.cpp")

    set(incflag "")
    set(extradep "")
    # "-" rather than an empty field: CMake drops trailing empty elements when
    # it splits a string into a list, so an empty sixth field would make
    # list(GET parts 5 ...) an index-out-of-range error rather than "".
    if(NOT incdir STREQUAL "-")
      # glslangValidator takes no space after -I; glslc accepts either.
      set(incflag "-I${CMAKE_CURRENT_SOURCE_DIR}/${incdir}")
      set(extradep "${SOAT_SHADER_EXTRA_DEPS}")
    endif()

    if(SOAT_GLSLANG)
      set(compile_cmd ${SOAT_GLSLANG} -V --target-env ${targetenv} ${incflag}
                      "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${spv}")
    else()
      # glslc spells the target differently: vulkan1.2, not --target-env
      # vulkan1.2 as a separate word.
      set(compile_cmd ${SOAT_GLSLC} --target-env=${targetenv} ${incflag}
                      "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${spv}")
    endif()

    add_custom_command(
      OUTPUT "${spv}"
      COMMAND ${compile_cmd}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}" ${extradep}
      COMMENT "glsl ${src} -> ${spvname}.spv"
      VERBATIM)

    add_custom_command(
      OUTPUT "${gen}"
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
              "${spv}" "${gen}" ${symbol}
      DEPENDS "${spv}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
      COMMENT "embed ${spvname}.spv as ${symbol}"
      VERBATIM)

    list(APPEND SOAT_SPIRV_SOURCES "${gen}")
  endforeach()

  # Every algorithm with a Vulkan backend, found rather than listed - a list is
  # what drifted twice in the Makefile and broke the cross-build.
  # CONFIGURE_DEPENDS because a plain glob is evaluated once at configure
  # time, where the Makefile's wildcard is re-evaluated every build. Without
  # it a newly added algo_vk.cpp stays invisible until someone reconfigures
  # by hand - a link error rather than a silent omission, since vk_registry
  # references every algorithm unconditionally, but a confusing one.
  file(GLOB SOAT_VK_ALGO_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/algos/*/algo_vk.cpp")

  # The per-algorithm stratum sources are not optional: run.cpp and
  # vk_registry.cpp reference them unconditionally, and leaving one out is a
  # link error rather than a smaller binary.
  set(SOAT_VK_CORE_SOURCES
    src/core/vk_common.cpp
    src/core/vk_registry.cpp
    src/core/miner_vk.cpp
    src/core/run.cpp
    src/core/stratum.cpp
    src/core/stratum_btc.cpp
    src/core/stratum_kawpow.cpp
  )

  add_executable(soat-miner-vk
    ${SOAT_VK_CORE_SOURCES}
    ${SOAT_VK_ALGO_SOURCES}
    ${SOAT_SPIRV_SOURCES}
  )
  target_include_directories(soat-miner-vk PRIVATE src)
  target_link_libraries(soat-miner-vk PRIVATE Vulkan::Vulkan)

  find_package(Threads REQUIRED)
  target_link_libraries(soat-miner-vk PRIVATE Threads::Threads)
  if(UNIX)
    target_link_libraries(soat-miner-vk PRIVATE dl)
  endif()
  if(WIN32)
    target_link_libraries(soat-miner-vk PRIVATE ws2_32)
  endif()

  install(TARGETS soat-miner-vk RUNTIME DESTINATION bin)
endif()  # SOAT_HAVE_VULKAN

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
