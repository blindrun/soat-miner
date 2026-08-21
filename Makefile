# SOAT Miner
#
# Which cards the CUDA binary runs on
# -----------------------------------
# This used to be `-arch=sm_89`, which embeds SASS for Ada and nothing else -
# no PTX, so no JIT fallback. The v0.1.1 release binary therefore refused to
# start on every card that was not a 40-series, with "no kernel image is
# available for execution on the device". A miner that only runs on the
# machine that built it is not a release.
#
# So the default now builds SASS for the common generations, plus PTX for the
# newest virtual architecture the installed toolkit knows about. The driver
# JITs that PTX forward onto anything newer than the toolkit - which is what
# lets a CUDA 12.4 build run at all on Blackwell (RTX 50-series), whose SASS
# needs CUDA 12.8+ to generate.
#
# Override to build one architecture, which compiles about 4x faster:
#   make ARCH=sm_89    # Ada:       RTX 4090/4080/4070
#   make ARCH=sm_120   # Blackwell: RTX 5090/5080/5070 (needs CUDA 12.8+)
#   make ARCH=sm_86    # Ampere:    RTX 3090/3080
#   make ARCH=sm_75    # Turing:    RTX 2080/1660
#
# To add an algorithm: drop it in src/algos/<name>/, add the object to
# ALGO_OBJS below, and add two lines to src/core/registry.cu.

ARCH    ?=
NVCC    ?= nvcc

# Newest virtual arch this toolkit supports: compute_90 on CUDA 12.4,
# compute_120 on 12.8+. This is the one shipped as JIT-forward PTX.
#
# The toolkit is a per-architecture choice, measured on pearl-pow:
#
#   RTX 5080   CUDA 13.1 native sm_120  268.7 M cand/s   vs 253.7 JIT   +5.9%
#   RTX 4090   CUDA 13.1                334.6            vs 350.0 (12.4) -4.4%
#
# Same 110 registers and no spills either way, so the Ada regression is ptxas
# scheduling, not allocation. There is no toolkit that wins both here: Ubuntu
# 26.04 ships only 13-1 and NVIDIA's own repo for 26.04 has only 13-3, so no
# 12.8/12.9 exists for this glibc. Build Ada with 12.4 and Blackwell with 13.x:
#
#   make NVCC=/usr/local/cuda-13.1/bin/nvcc BUILD=build13 BIN=soat-miner-13
#
# CUDA 13.1 needs a one-line fix against glibc 2.43, which declares rsqrt and
# rsqrtf __THROW while CUDA declares them without. Both are marked noexcept in
# crt/math_functions.h on this box; the original is beside it as
# .orig-preglibc243. Without it even an empty .cu file fails to compile.
NVCC_MAX_COMPUTE := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | tail -1)
NVCC_HAS_BLACKWELL := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | grep -c compute_120)

ifeq ($(strip $(ARCH)),)
  GENCODE  = -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89
  ifeq ($(NVCC_HAS_BLACKWELL),1)
    GENCODE += -gencode arch=compute_120,code=sm_120
  endif
  # PTX last, so the driver has something to JIT for future cards.
  GENCODE += -gencode arch=$(NVCC_MAX_COMPUTE),code=$(NVCC_MAX_COMPUTE)
else
  GENCODE  = -arch=$(ARCH)
endif

NVFLAGS  = -O3 $(GENCODE) --std=c++17 -lineinfo
BIN      = soat-miner
BIN_VK   = soat-miner-vk
BUILD    = build
CXX     ?= g++
# -MMD -MP makes the compiler write the dependency list, so a header edit
# rebuilds whatever included it - directly or transitively. Hand-maintained
# prerequisites have now missed a header three times in this repo, and the
# failure is silent in the worst way: a test that did not rebuild reports the
# OLD binary's result, so a mutation looks survived and a real edit looks
# tested. Nothing below needs updating when an include is added.
CXXFLAGS = -O3 --std=c++17 -Isrc -MMD -MP

# The CUDA binary is built whole-program, in one nvcc invocation, rather than
# per-object with -dc. That is deliberate and load-bearing: separable
# compilation runs the device linker, and nvlink resolves to real SASS and
# discards PTX, so a -dc binary carries no JIT fallback and runs on exactly the
# architectures it was compiled for and nothing else. Whole-program keeps the
# PTX, which the driver JITs onto newer cards - that is what lets a build from
# CUDA 12.4 (max sm_90) run on Blackwell at all. Measured: identical hashrate
# (217.6 vs 217.5 MH/s on a 4090) and the same ~60 s build.
CUDA_SRC  = src/core/miner.cu src/core/registry.cu src/algos/autolykos2/algo.cu \
            src/algos/pearl-pow/algo.cu src/algos/sha3-256t/algo.cu
CUDA_SRC_CPP = src/core/run.cpp src/core/stratum.cpp \
               src/core/stratum_btc.cpp
CUDA_DEPS = src/core/algo.h src/core/http.h src/core/run.h src/core/telemetry.h \
            src/core/stratum.h src/core/blake2b.cuh \
            src/core/pearl_pool.h \
            src/core/stratum_btc.h src/core/btc_job.h src/core/sha256.h \
            src/core/btc_protocol.h src/core/bc3_destination.h \
            src/core/json_lite.h \
            src/algos/autolykos2/mine.cuh src/algos/autolykos2/autolykos.cuh \
            src/algos/sha3-256t/sha3.h src/algos/sha3-256t/mine.cuh \
            src/algos/pearl-pow/job.h src/algos/pearl-pow/noisy_gemm.cuh \
            src/algos/pearl-pow/blake3.cuh src/algos/pearl-pow/prepare.cuh

# The Vulkan build carries one object per algorithm plus its embedded SPIR-V
# module, and vk_common.o holds the registry and the shared device handling.
# Adding a Vulkan algorithm is: a .comp, an algo_vk.cpp, the four rules below
# copied, and two lines in src/core/vk_common.cpp.
# The Vulkan binary's sources. ONE list, and BOTH the Linux and the Windows
# builds derive their objects from it. Neither target keeps its own copy.
#
# This is structural, not tidiness. The Windows cross-build used to carry a
# hand-written list of the same files, and adding an algorithm kept forgetting
# it: BC3 broke the mingw link with an undefined BitcoinStratumSource, and the
# next algorithm broke it again the same way, in a session where the comment
# warning about the first was directly above the target. Both times the Linux
# build was perfectly happy, so nothing caught it until someone ran the
# cross-build by hand. Deriving both from one list is what makes a third
# occurrence impossible rather than merely documented.
#
# Adding a Vulkan algorithm is now: add its algo_vk.cpp to VK_SRC_ALGO, its
# generated SPIR-V to VK_SRC_GEN, and any new core source to VK_SRC_CORE. Both
# platforms pick it up.
VK_SRC_CORE = src/core/vk_common.cpp src/core/vk_registry.cpp \
              src/core/miner_vk.cpp src/core/run.cpp \
              src/core/stratum.cpp src/core/stratum_btc.cpp
# Every algorithm that has a Vulkan backend, found rather than listed. A list
# is the thing that drifted twice; a wildcard cannot.
VK_SRC_ALGO = $(wildcard src/algos/*/algo_vk.cpp)

# The generated SPIR-V cannot be a wildcard - it does not exist until it is
# built - so each algorithm declares its own, in its own directory. autolykos2
# and sha3-256t have not moved yet, so theirs stay here for now.
VK_SRC_GEN  = $(BUILD)/spirv.cpp $(BUILD)/spirv_sha3.cpp
include $(wildcard src/algos/*/algo.mk)

# The Vulkan objects' dependency files are included further down, beside the
# rules that generate them. The test binaries' are not, and they are the ones
# that let a stale test report an old result.
-include $(wildcard tests/*.d)

# Only the files that really call into Vulkan. The mingw import library is
# generated by scanning these for vk* symbols, so widening it would export
# names that do not exist.
VK_SRC      = src/core/vk_common.cpp src/core/vk_registry.cpp $(VK_SRC_ALGO)

# Where a given Vulkan source's object lands. Test targets that need to link
# one or two of them use this rather than spelling out the layout, so the
# mirroring stays an implementation detail of this file.
vkobj = $(patsubst %.cpp,$(BUILD)/lin/%.o,$(1))
vkgen = $(BUILD)/lin/gen/$(1).o

# Object paths MIRROR the source path rather than being flattened, because
# several algorithms have a file called algo_vk.cpp and a flat scheme needs
# hand-picked names - which is the drift this change exists to remove.
VK_OBJS  = $(patsubst %.cpp,$(BUILD)/lin/%.o,$(VK_SRC_CORE) $(VK_SRC_ALGO)) \
           $(patsubst $(BUILD)/%.cpp,$(BUILD)/lin/gen/%.o,$(VK_SRC_GEN))

.PHONY: all cuda vulkan clean test test-pearl test-telemetry test-pearl-pool \
        test-pearl-cli test-pearl-e2e test-pearl-parity test-claim-guard test-ergo-pools test-btc-stratum test-bc3-destination \
        test-bc3-host test-bc3-cmake test-bc3-device test-bc3-vulkan \
        bench install dirs package

all: cuda vulkan

cuda: $(BIN)
vulkan: $(BIN_VK)

dirs:
	@mkdir -p $(BUILD)

$(BUILD)/stratum_cl.o: src/core/stratum.cpp src/core/stratum.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -x cu forces the .cpp sources through the CUDA compiler; it applies to every
# file after it, so the .cu sources have to come first.
$(BIN): $(CUDA_SRC) $(CUDA_SRC_CPP) $(CUDA_DEPS) | dirs
	$(NVCC) $(NVFLAGS) -Isrc $(CUDA_SRC) -x cu $(CUDA_SRC_CPP) -o $@

# --- Vulkan build: portable, one SPIR-V module for every vendor ------------
$(BUILD)/kernel.spv: src/algos/autolykos2/kernel.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

# Pearl's shaders are kept on their own rules and their own target env.
# Cooperative matrix needs vulkan1.3; Autolykos builds against 1.2 and must
# stay there.
#
# The three Merkle shaders share src/algos/pearl-pow/blake3.glsl, which is
# .glsl and not .comp ON PURPOSE - it has no main(), so a %.comp pattern rule
# would try to build it standalone and fail. Each .spv depends on it, or
# editing the shared blake3 rebuilds nothing. glslangValidator takes no space
# after -I.
# Pearl's shader, embed and device-gate rules are in src/algos/pearl-pow/algo.mk,
# beside the shaders they build. PEARL_MERKLE_SPV is kept here only because the
# merkle gate's rule below still names it.
PEARL_MERKLE_SPV = $(BUILD)/merkle_chunk.spv $(BUILD)/merkle_reduce.spv \
                   $(BUILD)/merkle_root.spv

$(BUILD)/spirv.cpp: $(BUILD)/kernel.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@

$(BUILD)/kernel_sha3.spv: src/algos/sha3-256t/kernel.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

$(BUILD)/spirv_sha3.cpp: $(BUILD)/kernel_sha3.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@ kSha3Spirv

# One rule per platform instead of one per file. -MMD -MP generates real header
# dependencies, which the hand-written lists had already drifted from.
$(BUILD)/lin/%.o: %.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/lin/gen/%.o: $(BUILD)/%.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(VK_OBJS:.o=.d)

$(BIN_VK): $(VK_OBJS)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -ldl -lpthread -o $@

# --- correctness gates -----------------------------------------------------
# test_hit is the one that matters: it rebuilds the real dataset and
# reproduces a real mainnet block's hit from that block's winning nonce.
#
# Pinned vector, mainnet block f9e0e887ec38c2b95631bfef81becf20025c4cd8d700666
# ecf6cdf80810b260f. Height 500,012 is deliberate: it is Autolykos v2 but below
# the N-increase start, so the dataset is 2.15 GB and this runs on any card
# that can run the miner. Regenerate with:
#   python3 tests/reference.py --vector <blockId>
HIT_MSG    = 21098a45fe5ec2e9b1087e8b248e810a770fe0200a1f8a211ace97faa9813e7b
HIT_HEIGHT = 500012
HIT_NONCE  = 1232002a7deb44c0
HIT_EXPECT = 0000000000001fa822c756eef06389799c10c7c1d06868eb29c75214e0a7aab1

PEARL_VEC  = $(BUILD)/pearl_vectors.bin
PEARL_VEC64 = $(BUILD)/pearl_vectors_r64.bin
PEARL_JOBVEC = $(BUILD)/pearl_job_vectors.bin

test: tests/test_element tests/test_hit tests/test_algo tests/test_vulkan test-pearl
	@echo "--- python reference vs real mainnet blocks ---"
	@python3 tests/reference.py
	@echo "--- device dataset elements vs python reference ---"
	@./tests/test_element 1851437
	@echo "--- device end-to-end: real block's hit from its winning nonce ---"
	@./tests/test_hit $(HIT_MSG) $(HIT_HEIGHT) $(HIT_NONCE) $(HIT_EXPECT)
	@echo "--- the Algorithm object: search, verify and the build-ahead swap ---"
	@./tests/test_algo $(HIT_MSG) $(HIT_HEIGHT) $(HIT_NONCE) $(HIT_EXPECT)
	@echo "--- Lithos stratum protocol, against a mock built from its source ---"
	@python3 tests/test_lithos.py
	@echo "--- vulkan backend against the same pinned vector ---"
	@./tests/test_vulkan $(HIT_MSG) $(HIT_HEIGHT) $(HIT_NONCE) $(HIT_EXPECT)

# --- pearl-pow gates ------------------------------------------------------
# Split out because Pearl needs its own vectors emitted first, and because the
# end-to-end one needs a running pearl-gateway - which a build machine will not
# have. `make test` runs everything that does not; test_pearl_algo is run by
# hand against the regtest rig.
$(PEARL_VEC): tests/pearl_reference.py | dirs
	@python3 tests/pearl_reference.py --emit-vectors $@ 128

$(PEARL_VEC64): tests/pearl_reference.py | dirs
	@python3 tests/pearl_reference.py --emit-vectors $@ 64

$(PEARL_JOBVEC): tests/pearl_job.py tests/pearl_reference.py | dirs
	@python3 tests/pearl_job.py --emit-vectors $@

# PEARL_VK_GATES comes from src/algos/pearl-pow/algo.mk, beside the shaders.
# It was a hand-written list here and it had fallen three gates behind: noise,
# apply_noise and powscan all passed on a 4090 and a 7900 XT and were then
# never built by anything again.
test-pearl: tests/test_pearl tests/test_pearl_job tests/test_pearl_prepare \
            tests/test_pearl_mining_shape $(PEARL_VK_GATES) \
            tests/test_claim_guard \
            $(BUILD)/kernel_pearl.spv \
            $(PEARL_VEC) $(PEARL_VEC64) $(PEARL_JOBVEC)
	@echo "--- the GPU claim guard, before anything touches a card ---"
	@./tests/test_claim_guard
	@echo "--- pearl: the algorithm in numpy, mutation-tested ---"
	@python3 tests/pearl_reference.py
	@echo "--- pearl: the job pipeline, and the proof a node accepts ---"
	@python3 tests/pearl_job.py
	@echo "--- pearl: every CUDA kernel against the reference, rank 128 ---"
	@./tests/test_pearl $(PEARL_VEC)
	@echo "--- pearl: the same at rank 64 ---"
	@./tests/test_pearl $(PEARL_VEC64)
	@echo "--- pearl: the host commitment chain and proof encoding ---"
	@./tests/test_pearl_job $(PEARL_JOBVEC)
	@echo "--- pearl: the device prepare stage, and what it costs ---"
	@./tests/test_pearl_prepare $(PEARL_JOBVEC)
	@echo "--- pearl: cp.async kernels vs dbuf at the real mining shape ---"
	@./tests/test_pearl_mining_shape 1024 1024 2048 128 ptx 64
	@./tests/test_pearl_mining_shape 4096 32768 2048 128 ptx 64
	@./tests/test_pearl_mining_shape 4096 32768 2048 128 ptx 32
	@./tests/test_pearl_mining_shape 4096 32768 2048 128 async
	@echo "--- pearl: the Vulkan GEMM and transcript, byte-identical to CUDA ---"
	@./tests/test_pearl_vk $(PEARL_VEC) $(BUILD)/kernel_pearl.spv
	@echo "--- pearl: Vulkan matrix generation, against the host reference ---"
	@./tests/test_pearl_genmatrix_vk $(BUILD)/genmatrix.spv
	@echo "--- pearl: the two Vulkan layout changes ---"
	@./tests/test_pearl_transpose_vk $(BUILD)/transpose.spv
	@echo "--- pearl: the Vulkan commitment chain, against job.h ---"
	@./tests/test_pearl_commitments_vk $(BUILD)/commitments.spv
	@echo "--- pearl: the Vulkan blake3 Merkle roots, against the CUDA vectors ---"
	@./tests/test_pearl_merkle_vk $(PEARL_JOBVEC) $(BUILD)/merkle_chunk.spv \
	    $(BUILD)/merkle_reduce.spv $(BUILD)/merkle_root.spv
	@echo "--- pearl: Vulkan noise generation, both domains ---"
	@./tests/test_pearl_noise_vk $(BUILD)/noise.spv
	@echo "--- pearl: Vulkan noise application, A / Bt / B ---"
	@./tests/test_pearl_apply_noise_vk $(BUILD)/apply_noise.spv
	@echo "--- pearl: the Vulkan PoW scan, against the CUDA hit list ---"
	@./tests/test_pearl_powscan_vk $(BUILD)/powscan.spv
	@echo "--- pearl: the coopmat-free GEMM, against the same CUDA vectors ---"
	@./tests/test_pearl_dp_vk $(PEARL_VEC) $(BUILD)/kernel_dp.spv

# The only test that runs the miner against a server. Everything else about
# Pearl is a vector check or a device gate, and neither opens a socket - which
# is exactly the coverage shape that hid a share-counting bug in two other
# algorithms here. It needs the CUDA binary and a claimed GPU, so it is its own
# target rather than part of the host-only set.
# The guard every device gate depends on. Fixtures only - no gpulock, no GPU,
# no claim - so it runs anywhere and runs first.
test-claim-guard: tests/test_claim_guard
	@echo "--- the GPU claim guard's own discrimination ---"
	@./tests/test_claim_guard

tests/test_claim_guard: tests/test_claim_guard.cpp tests/vk_claim_guard.h
	$(CXX) $(CXXFLAGS) -Itests $< -o $@

test-pearl-e2e: $(BIN)
	@echo "--- pearl: the real miner against a mock gateway ---"
	@python3 tests/test_pearl_pool_e2e.py $(BIN)

# A display-contract check only; intentionally host-only so it is safe while a
# GPU is reserved for another workload.
test-telemetry: tests/test_telemetry
	@./tests/test_telemetry

tests/test_telemetry: tests/test_telemetry.cpp src/core/telemetry.h
	$(CXX) $(CXXFLAGS) $< -ldl -o $@

# Protocol serialization only; this opens no socket and does not need CUDA.
test-pearl-pool: tests/test_pearl_pool
	@./tests/test_pearl_pool

tests/test_pearl_pool: tests/test_pearl_pool.cpp src/core/pearl_pool.h src/core/http.h \
                       src/core/algo.h src/algos/pearl-pow/job.h
	$(CXX) $(CXXFLAGS) $< -ldl -o $@

# Parses help and rejects malformed endpoints before any CUDA initialization.
test-pearl-cli: $(BIN) tests/test_pearl_cli.sh
	@sh tests/test_pearl_cli.sh "$(BIN)"

# Release-file contract only: fixture data, no GPU, miner, or network access.
test-ergo-pools:
	@python3 tests/test_ergo_pool_scripts.py

tests/test_pearl: tests/test_pearl.cu src/algos/pearl-pow/noisy_gemm.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

# Not part of `make test`: there is no Vulkan Pearl shader to feed it yet.
# Kept so the correctness gate exists the day someone writes one.
tests/test_pearl_commitments_vk: tests/test_pearl_commitments_vk.cpp \
                                src/algos/pearl-pow/job.h $(BUILD)/commitments.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

tests/test_pearl_transpose_vk: tests/test_pearl_transpose_vk.cpp \
                              $(BUILD)/transpose.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

tests/test_pearl_genmatrix_vk: tests/test_pearl_genmatrix_vk.cpp \
                              src/algos/pearl-pow/job.h $(BUILD)/genmatrix.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

tests/test_pearl_merkle_vk: tests/test_pearl_merkle_vk.cpp \
                           src/algos/pearl-pow/job.h $(PEARL_MERKLE_SPV)
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

tests/test_pearl_vk: tests/test_pearl_vk.cpp src/core/vk_common.cpp \
                    $(BUILD)/kernel_pearl.spv
	$(CXX) $(CXXFLAGS) $< src/core/vk_common.cpp -lvulkan -o $@

# Not part of `make test` - it is a measurement, not a gate.
tests/measure_launch_gap: tests/measure_launch_gap.cu \
                          src/algos/pearl-pow/noisy_gemm.cuh \
                          src/algos/pearl-pow/prepare.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

tests/test_pearl_mining_shape: tests/test_pearl_mining_shape.cu \
                               src/algos/pearl-pow/noisy_gemm.cuh \
                               src/algos/pearl-pow/prepare.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

tests/test_pearl_job: tests/test_pearl_job.cpp src/algos/pearl-pow/job.h
	$(CXX) $(CXXFLAGS) $< -o $@

tests/test_pearl_prepare: tests/test_pearl_prepare.cu \
                          src/algos/pearl-pow/prepare.cuh \
                          src/algos/pearl-pow/blake3.cuh \
                          src/algos/pearl-pow/noisy_gemm.cuh \
                          src/algos/pearl-pow/job.h
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

# Needs a pearl-gateway on 127.0.0.1:8455 - see ~/pearl-regtest-gateway.sh.
tests/test_pearl_algo: tests/test_pearl_algo.cu src/algos/pearl-pow/algo.cu \
                       src/core/pearl_gateway.h src/core/algo.h
	$(NVCC) $(NVFLAGS) -Isrc $< src/algos/pearl-pow/algo.cu -o $@

tests/test_pearl_gateway: tests/test_pearl_gateway.cpp src/core/pearl_gateway.h \
                          src/algos/pearl-pow/job.h
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

# The Vulkan gate needs only the algorithm and the embedded SPIR-V, not the
# miner's job/stratum plumbing.
VK_T_AUTO = $(call vkobj,src/algos/autolykos2/algo_vk.cpp) $(call vkgen,spirv) \
            $(call vkobj,src/core/vk_common.cpp)

tests/test_vulkan: tests/test_vulkan.cpp $(VK_T_AUTO) src/core/algo.h
	$(CXX) $(CXXFLAGS) $< $(VK_T_AUTO) -lvulkan -ldl -lpthread -o $@

# The Vulkan BC3 gate. This is the one that decides whether the shader is
# trusted at all: NVIDIA's Vulkan compiler miscompiled the Autolykos kernel in
# this repo once, and the miner mined happily while every share was silently
# rejected. Nothing but a host-vs-device comparison catches that.
#
# It links vk_common.o for the registry but drives the algorithm directly, and
# it also runs its own bare pipeline against the embedded module so a bug in
# algo_vk.cpp's push-constant layout and a bug in the shader cannot mask each
# other.
VK_T_SHA3 = $(call vkobj,src/algos/sha3-256t/algo_vk.cpp) \
            $(call vkgen,spirv_sha3) $(call vkobj,src/core/vk_common.cpp)

tests/test_sha3_vulkan: tests/test_sha3_vulkan.cpp $(VK_T_SHA3) \
                        src/algos/sha3-256t/sha3.h src/core/algo.h
	$(CXX) $(CXXFLAGS) $< $(VK_T_SHA3) -lvulkan -ldl -lpthread -o $@

# Opt-in like test-bc3-device: it puts real load on a GPU, so it runs after a
# purpose-specific gpulock claim rather than as part of a plain `make test`.
test-bc3-vulkan: tests/test_sha3_vulkan
	@./tests/test_sha3_vulkan

tests/test_element: tests/test_element.cu src/algos/autolykos2/autolykos.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

tests/test_hit: tests/test_hit.cu src/algos/autolykos2/mine.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

tests/test_sha3_algo: tests/test_sha3_algo.cu tests/sha3_vectors.h \
                      src/algos/sha3-256t/algo.cu \
                      src/algos/sha3-256t/mine.cuh src/algos/sha3-256t/sha3.h \
                      src/core/algo.h src/core/btc_job.h
	$(NVCC) $(NVFLAGS) -Isrc $< src/algos/sha3-256t/algo.cu -o $@

# Offline Bitcoin-Stratum protocol fixture: no socket, wallet, GPU, or pool.
test-btc-stratum: tests/test_btc_stratum tests/fixtures/btc_stratum_v1.jsonl
	@./tests/test_btc_stratum tests/fixtures/btc_stratum_v1.jsonl

tests/test_btc_stratum: tests/test_btc_stratum.cpp src/core/stratum_btc.cpp \
	                       src/core/stratum_btc.h src/core/btc_protocol.h \
	                       src/core/btc_job.h src/core/sha256.h src/core/json_lite.h
	$(CXX) $(CXXFLAGS) -pthread $< src/core/stratum_btc.cpp -o $@

test-bc3-destination: tests/test_bc3_destination
	@./tests/test_bc3_destination

tests/test_bc3_destination: tests/test_bc3_destination.cpp src/core/bc3_destination.h
	$(CXX) $(CXXFLAGS) $< -o $@

# Phase-separated on purpose. The host gate is safe anywhere; the device gate
# is opt-in and must only run after a purpose-specific GPU claim.
test-bc3-host: test-btc-stratum test-bc3-destination

# CMake/CTest equivalent of the offline host gate.  It configures and builds
# only test-btc-stratum, then runs its one checked-in fixture test.  CUDA is
# still required by the project configure, but this target never executes a
# CUDA binary or opens a socket.
test-bc3-cmake:
	@bash scripts/test-bc3-cmake.sh

test-bc3-device: tests/test_sha3_algo
	@./tests/test_sha3_algo

tests/test_algo: tests/test_algo.cu src/algos/autolykos2/algo.cu \
                 src/algos/autolykos2/mine.cuh src/algos/autolykos2/autolykos.cuh \
                 src/core/algo.h src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< src/algos/autolykos2/algo.cu -o $@

bench: $(BIN)
	./$(BIN) --bench

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)/usr/local/bin/$(BIN)

clean:
	rm -rf $(BUILD) $(BIN) $(BIN_VK) tests/test_element tests/test_hit \
	       tests/test_algo tests/test_vulkan tests/test_pearl \
	       tests/test_pearl_job tests/test_pearl_prepare tests/test_telemetry \
	       tests/test_pearl_pool \
	       tests/test_pearl_algo tests/test_pearl_gateway tests/test_btc_stratum \
	       tests/test_bc3_destination \
	       tests/test_sha3_algo tests/test_sha3_vulkan

# --- release packaging (lolMiner-style flat archive) -----------------------
VERSION ?= 0.2.17
PKGNAME  = soat-miner_v$(VERSION)_Lin64
package: cuda vulkan
	@rm -rf $(BUILD)/$(PKGNAME) && mkdir -p $(BUILD)/$(PKGNAME)
	cp $(BIN) $(BIN_VK) $(BUILD)/$(PKGNAME)/
	cp packaging/*.sh packaging/*.bat packaging/*.json packaging/config.txt packaging/lithos-status \
	   README.md LICENSE $(BUILD)/$(PKGNAME)/
	cp scripts/soat-miner-guard.py scripts/guard.conf.example \
	   scripts/soat-miner.service scripts/soat-miner-guard.service \
	   $(BUILD)/$(PKGNAME)/
	chmod +x $(BUILD)/$(PKGNAME)/*.sh $(BUILD)/$(PKGNAME)/lithos-status
	cd $(BUILD) && tar czf $(PKGNAME).tar.gz $(PKGNAME)
	@sha256sum $(BUILD)/$(PKGNAME).tar.gz | tee $(BUILD)/$(PKGNAME).tar.gz.sha256
	@echo "packaged: $(BUILD)/$(PKGNAME).tar.gz"

# --- Windows cross-build (Vulkan only; CUDA needs MSVC on a Windows host) ---
WINCXX  ?= x86_64-w64-mingw32-g++
WINDIR   = $(BUILD)/win
WINFLAGS = -O3 -std=c++17 -Isrc -I$(WINDIR)/include -static -static-libgcc -static-libstdc++
WINPKG   = soat-miner_v$(VERSION)_Win64

.PHONY: windows windows-package

$(WINDIR)/include/vulkan:
	@mkdir -p $(WINDIR)/include
	cp -r /usr/include/vulkan $(WINDIR)/include/
	-cp -r /usr/include/vk_video $(WINDIR)/include/

# Import library for vulkan-1.dll, generated from the symbols we call.
#
# It depends on $(VK_SRC) deliberately. Without that this is a one-shot target:
# adding a Vulkan call anywhere leaves the stale .def in place, the symbol is
# never exported, and the cross-build fails at link with an undefined reference
# that looks like a toolchain problem rather than a stale generated file.
# Adding vkGetPhysicalDeviceFeatures for the BC3 shaderInt64 check hit exactly
# this.
$(WINDIR)/libvulkan-1.a: $(VK_SRC) | $(WINDIR)/include/vulkan
	@mkdir -p $(WINDIR)
	@{ echo "LIBRARY vulkan-1.dll"; echo "EXPORTS"; \
	   grep -ohE '\bvk[A-Z][A-Za-z0-9]*\s*\(' $(VK_SRC) \
	   | tr -d '( ' | sort -u \
	   | grep -vE '^vk(DeviceMemGB|DeviceName|DriverVersion|ListDevices|PickPhysicalDevice|SetDeviceInfo)$$' \
	   | sed 's/^/    /'; } > $(WINDIR)/vulkan-1.def
	x86_64-w64-mingw32-dlltool -d $(WINDIR)/vulkan-1.def -l $@

# stratum_btc.cpp is here because run.cpp references BitcoinStratumSource
# unconditionally. The Linux Vulkan target already links stratum_btc_vk.o; this
# one did not, so adding BC3 broke the Windows cross-build at link time with an
# undefined symbol and nothing else changed.
WIN_OBJS = $(patsubst %.cpp,$(WINDIR)/%.o,$(VK_SRC_CORE) $(VK_SRC_ALGO)) \
           $(patsubst $(BUILD)/%.cpp,$(WINDIR)/gen/%.o,$(VK_SRC_GEN))

$(WINDIR)/%.o: %.cpp | $(WINDIR)/include/vulkan
	@mkdir -p $(dir $@)
	$(WINCXX) $(WINFLAGS) -MMD -MP -c $< -o $@

$(WINDIR)/gen/%.o: $(BUILD)/%.cpp | $(WINDIR)/include/vulkan
	@mkdir -p $(dir $@)
	$(WINCXX) $(WINFLAGS) -MMD -MP -c $< -o $@

-include $(WIN_OBJS:.o=.d)

# Derived from VK_SRC_* exactly as the Linux objects are, so an algorithm
# cannot be added to one platform and forgotten on the other.
windows: $(VK_SRC_GEN) $(WINDIR)/libvulkan-1.a $(WIN_OBJS)
	$(WINCXX) $(WINFLAGS) $(WIN_OBJS) $(WINDIR)/libvulkan-1.a -lws2_32 \
	    -o $(WINDIR)/soat-miner-vk.exe
	@x86_64-w64-mingw32-objdump -p $(WINDIR)/soat-miner-vk.exe | grep -i "DLL Name" | sort -u

windows-package: windows
	@rm -rf $(BUILD)/$(WINPKG) && mkdir -p $(BUILD)/$(WINPKG)
	cp $(WINDIR)/soat-miner-vk.exe $(BUILD)/$(WINPKG)/
	@# Fold in the CUDA build if a Windows host staged it in build/win/cuda/.
	@# nvcc needs MSVC so it cannot be built here; the windows-cuda CI job (or
	@# a hand build) drops soat-miner.exe plus its VC++ runtime DLLs there.
	@if [ -f $(WINDIR)/cuda/soat-miner.exe ]; then \
	  cp $(WINDIR)/cuda/soat-miner.exe $(WINDIR)/cuda/*.dll $(BUILD)/$(WINPKG)/ && \
	  echo "  + CUDA build folded in ($(WINDIR)/cuda/)"; \
	else \
	  echo "  (no CUDA build staged in $(WINDIR)/cuda/, shipping Vulkan only)"; \
	fi
	cp packaging/*.bat packaging/*.json packaging/config.txt packaging/README-WINDOWS.txt README.md LICENSE $(BUILD)/$(WINPKG)/
	cd $(BUILD) && zip -qr $(WINPKG).zip $(WINPKG)
	@sha256sum $(BUILD)/$(WINPKG).zip | tee $(BUILD)/$(WINPKG).zip.sha256
	@echo "packaged: $(BUILD)/$(WINPKG).zip"
