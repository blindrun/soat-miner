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
CXXFLAGS = -O3 --std=c++17 -Isrc

# The CUDA binary is built whole-program, in one nvcc invocation, rather than
# per-object with -dc. That is deliberate and load-bearing: separable
# compilation runs the device linker, and nvlink resolves to real SASS and
# discards PTX, so a -dc binary carries no JIT fallback and runs on exactly the
# architectures it was compiled for and nothing else. Whole-program keeps the
# PTX, which the driver JITs onto newer cards - that is what lets a build from
# CUDA 12.4 (max sm_90) run on Blackwell at all. Measured: identical hashrate
# (217.6 vs 217.5 MH/s on a 4090) and the same ~60 s build.
PROGPOW_DEPS = src/algos/progpow/keccak.h src/algos/progpow/ethash.h \
               src/algos/progpow/progpow.h

CUDA_SRC  = src/core/miner.cu src/core/registry.cu src/algos/autolykos2/algo.cu \
            src/algos/pearl-pow/algo.cu src/algos/sha3-256t/algo.cu \
            src/algos/kawpow/algo.cu
CUDA_SRC_CPP = src/core/run.cpp src/core/stratum.cpp \
               src/core/stratum_btc.cpp src/core/stratum_kawpow.cpp
CUDA_DEPS = src/core/algo.h src/core/http.h src/core/run.h src/core/telemetry.h \
            src/core/stratum.h src/core/blake2b.cuh \
            src/algos/autolykos2/mine.cuh src/algos/autolykos2/autolykos.cuh \
            src/core/stratum_btc.h src/core/btc_job.h src/core/sha256.h \
            src/core/btc_protocol.h src/core/bc3_destination.h src/core/json_lite.h src/algos/sha3-256t/sha3.h \
            src/algos/sha3-256t/mine.cuh \
            src/algos/kawpow/mine.cuh src/algos/kawpow/kawpow.h \
            src/algos/kawpow/job.h src/algos/progpow/mine.cuh \
            src/algos/progpow/keccak.h src/algos/progpow/ethash.h \
            src/algos/progpow/progpow.h src/core/stratum_kawpow.h \
            src/core/kawpow_address.h

# The Vulkan build carries one object per algorithm plus its embedded SPIR-V
# module, and vk_common.o holds the registry and the shared device handling.
# Adding a Vulkan algorithm is: a .comp, an algo_vk.cpp, the four rules below
# copied, and two lines in src/core/vk_common.cpp.
# The Vulkan binary's sources. ONE list, and BOTH the Linux and the Windows
# builds derive their objects from it. Neither target keeps its own copy.
#
# This is structural, not tidiness. The Windows cross-build used to carry a
# hand-written list of the same files and it was forgotten TWICE: adding BC3
# broke the mingw link with an undefined BitcoinStratumSource, and adding
# KawPow broke it again with makeKawpowVK and KawpowStratumSource. Both times
# the Linux build was perfectly happy, so nothing caught it until someone ran
# the cross-build by hand, and the second time the comment warning about the
# first was directly above the target. Deriving both from one list is what
# makes a third occurrence impossible rather than merely documented.
#
# Adding a Vulkan algorithm is now: add its algo_vk.cpp to VK_SRC_ALGO, its
# generated SPIR-V to VK_SRC_GEN, and any new core source to VK_SRC_CORE. Both
# platforms pick it up.
VK_SRC_CORE = src/core/vk_common.cpp src/core/vk_registry.cpp \
              src/core/miner_vk.cpp src/core/run.cpp \
              src/core/stratum.cpp src/core/stratum_btc.cpp \
              src/core/stratum_kawpow.cpp
VK_SRC_ALGO = src/algos/autolykos2/algo_vk.cpp \
              src/algos/sha3-256t/algo_vk.cpp \
              src/algos/kawpow/algo_vk.cpp
VK_SRC_GEN  = $(BUILD)/spirv.cpp $(BUILD)/spirv_sha3.cpp \
              $(BUILD)/spirv_kawpow.cpp $(BUILD)/spirv_kawpow_spec.cpp \
              $(BUILD)/spirv_dag.cpp

# Only the files that actually call into Vulkan. The mingw import library is
# generated by scanning these for vk* symbols, so widening it would export
# names that do not exist.
VK_SRC      = src/core/vk_common.cpp src/core/vk_registry.cpp $(VK_SRC_ALGO)

# Where a given Vulkan source's object lands. Test targets that link one or two
# of them use this rather than spelling out the layout, so the mirroring stays
# an implementation detail of this file.
vkobj = $(patsubst %.cpp,$(BUILD)/lin/%.o,$(1))
vkgen = $(BUILD)/lin/gen/$(1).o

# Object paths MIRROR the source path rather than being flattened, because
# there are three different files called algo_vk.cpp and a flat scheme needs
# hand-picked names - which is the drift this change exists to remove.
VK_OBJS  = $(patsubst %.cpp,$(BUILD)/lin/%.o,$(VK_SRC_CORE) $(VK_SRC_ALGO)) \
           $(patsubst $(BUILD)/%.cpp,$(BUILD)/lin/gen/%.o,$(VK_SRC_GEN))

.PHONY: all cuda vulkan clean test test-nvrtc test-nvrtc-device test-device test-kawpow test-kawpow-device test-kawpow-vulkan test-kawpow-vulkan-real test-pearl test-btc-stratum test-bc3-destination test-bc3-host test-bc3-cmake test-bc3-device test-bc3-vulkan bench install dirs package

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

$(BUILD)/spirv.cpp: $(BUILD)/kernel.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@

$(BUILD)/kernel_sha3.spv: src/algos/sha3-256t/kernel.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

$(BUILD)/spirv_sha3.cpp: $(BUILD)/kernel_sha3.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@ kSha3Spirv

$(BUILD)/spirv_dag.cpp: $(BUILD)/dag.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@ kProgpowDagSpirv

$(BUILD)/kawpow_spec.spv: src/algos/kawpow/kernel_spec.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

$(BUILD)/spirv_kawpow_spec.cpp: $(BUILD)/kawpow_spec.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@ kKawpowSpecSpirv

$(BUILD)/spirv_kawpow.cpp: $(BUILD)/kawpow.spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $< $@ kKawpowSpirv

# One rule per platform instead of one per file. -MMD -MP generates real header
# dependencies, which the hand-written lists had already drifted from - run_vk.o
# did not list stratum_kawpow.h, so editing it rebuilt nothing.
$(BUILD)/lin/%.o: %.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/lin/gen/%.o: $(BUILD)/%.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(VK_OBJS:.o=.d)

$(BIN_VK): $(VK_OBJS)
	$(CXX) $(CXXFLAGS) $^ -lvulkan -ldl -lpthread -o $@

# --- does everything still LINK? -------------------------------------------
# Builds every test binary and both platform binaries without running any of
# them. No GPU, no pool, no network.
#
# This exists because three bugs in one day were the same shape: a target
# nobody builds. The Windows cross-build broke twice, once for BC3 and once for
# the next algorithm, and three test targets that link individual objects were
# left naming object paths that no longer existed. None of it was visible to
# `make cuda`, `make vulkan` or the offline test gate, because those targets
# link a different set of objects. Each was found by someone running a target
# by hand, which is not a system.
#
# Run this after any change to the build, to a shared header, or to anything
# that adds or renames a source. It is the cheapest thing that catches
# link-level drift across the whole tree.
TEST_BINARIES = $(shell grep -oE '^tests/[a-z0-9_]+:' $(firstword $(MAKEFILE_LIST)) \
                  | tr -d ':' | sort -u)

.PHONY: build-tests
build-tests: cuda vulkan $(TEST_BINARIES)
	@echo "all $(words $(TEST_BINARIES)) test binaries and both platform binaries link"

# The Windows cross-build is separate because it needs mingw, which a GPU box
# may not have. Run it too when you have the toolchain - it is the target with
# the worst track record in this file.
.PHONY: build-all
build-all: build-tests windows

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

test: tests/test_element tests/test_hit tests/test_algo tests/test_vulkan test-pearl test-nvrtc
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

# --- kawpow gates ---------------------------------------------------------
# The host reference is the anchor for everything else in this algorithm: the
# CUDA and Vulkan kernels are checked against it, so it is checked against
# Ravencoin's own published vectors first.
#
# The mutants matter as much as the vectors. Four constants, each of which
# produces a miner that runs at full speed and has every share rejected, are
# overridable at compile time; each broken build must fail the same test. A
# gate nobody has watched fail is not a gate.
KAWPOW_MUTANTS = KAWPOW_PERIOD_LENGTH=50 KAWPOW_EPOCH_LENGTH=30000 \
                 KAWPOW_RAVEN_TAIL=0 KAWPOW_KECCAK_PAD=0x06u

# A real Ravencoin mainnet block, at a height whose DAG is the 6.1 GB one a
# miner actually builds today. Regenerate with:
#   python3 tests/kawpow_block.py <height>
KAWPOW_BLK_HEADER = 8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7
KAWPOW_BLK_HEIGHT = 4501000
KAWPOW_BLK_NONCE  = 329dd53ca526c2cd
KAWPOW_BLK_MIX    = 7b1e003a71d7bf9f81d95c0c68126f72876de28b69b175c84e4c11234ad5d7c7
KAWPOW_BLK_HASH   = 000000000001a5a2da8a4a4e8867948fd5958d98dc61521ea5f9906bf4b9e7b5
KAWPOW_BLK_BITS   = 1b06292b


tests/test_kawpow_hash: tests/test_kawpow_hash.cpp src/algos/kawpow/kawpow.h \
                        $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

tests/test_kawpow_block: tests/test_kawpow_block.cpp src/algos/kawpow/kawpow.h \
                         $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

# --- runtime kernel generation (NVRTC) ------------------------------------
#
# The header list lives in THIS RULE, so the Makefile itself is a prerequisite:
# a generated artifact must depend on whatever determines its content, not just
# on its inputs' timestamps. Adding a header here without that dependency would
# reuse a stale embed and compile against a header set nobody asked for - the
# same shape as the mingw import library that was silently reused after its
# source list changed.
NVRTC_HDRS = keccak.h=src/algos/progpow/keccak.h \
             ethash.h=src/algos/progpow/ethash.h \
             progpow.h=src/algos/progpow/progpow.h \
             kawpow.h=src/algos/kawpow/kawpow.h \
             progpow_mine.cuh=src/algos/progpow/mine.cuh \
             kawpow_mine.cuh=src/algos/kawpow/mine.cuh

$(BUILD)/nvrtc_embed.cpp: scripts/embed_headers.py Makefile \
                          src/algos/progpow/keccak.h \
                          src/algos/progpow/ethash.h \
                          src/algos/progpow/progpow.h \
                          src/algos/kawpow/kawpow.h \
                          src/algos/progpow/mine.cuh \
                          src/algos/kawpow/mine.cuh | dirs
	@python3 scripts/embed_headers.py $@ $(NVRTC_HDRS)

tests/test_nvrtc: tests/test_nvrtc.cpp $(BUILD)/nvrtc_embed.cpp \
                  src/algos/kawpow/nvrtc_source.h \
                  src/algos/kawpow/nvrtc_headers.h $(PROGPOW_DEPS)
	$(NVCC) -O2 -Isrc tests/test_nvrtc.cpp $(BUILD)/nvrtc_embed.cpp \
		-lnvrtc -o $@

tests/test_nvrtc_device: tests/test_nvrtc_device.cu $(BUILD)/nvrtc_embed.cpp \
                        src/algos/kawpow/nvrtc_jit.h \
                        src/algos/kawpow/nvrtc_source.h \
                        src/algos/kawpow/mine.cuh $(PROGPOW_DEPS)
	$(NVCC) $(NVFLAGS) -Isrc tests/test_nvrtc_device.cu \
		$(BUILD)/nvrtc_embed.cpp -o $@

# Runs BOTH directions. The forward check alone cannot tell a faithful kernel
# from a comparison that never fails, so the mutation case - a kernel compiled
# for the wrong period - has to be caught for the forward pass to mean anything.
.PHONY: test-nvrtc-device
test-nvrtc-device: tests/test_nvrtc_device
	@echo "--- nvrtc: generated kernel vs interpreted, on a real card ---"
	@./tests/test_nvrtc_device $(KAWPOW_BLK_HEIGHT)
	@echo "--- nvrtc: and the comparison itself is not vacuous ---"
	@./tests/test_nvrtc_device $(KAWPOW_BLK_HEIGHT) --mutate

.PHONY: test-nvrtc
test-nvrtc: tests/test_nvrtc
	@echo "--- nvrtc: the runtime-compilation chain, offline ---"
	@./tests/test_nvrtc

tests/test_kawpow_device: tests/test_kawpow_device.cu \
                          src/algos/kawpow/mine.cuh src/algos/kawpow/kawpow.h \
                          src/algos/progpow/mine.cuh $(PROGPOW_DEPS)
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

# The device gate needs a card and enough VRAM for the real 6.1 GB DAG, so it
# is its own target rather than part of test-kawpow, which must pass on a build
# machine with no GPU at all. It skips itself politely on a small card.
# Every gate that needs a real GPU, in one place.
#
# These were reachable only by typing their exact names, which meant nobody ran
# them and a segfault in `--bench` - the first command most people type - stayed
# invisible while the offline gate reported 119 checks passing. A gate nobody
# invokes is not a gate.
test-device: test-bench-runs test-kawpow-device test-nvrtc-device
	@echo "device gates: all passed"

test-bench-runs: cuda
	@echo "--- bench mode starts and reports, on every algorithm ---"
	@sh tests/test_bench_runs.sh ./soat-miner

test-kawpow-device: tests/test_kawpow_device test-bench-runs
	@echo "--- kawpow: CUDA kernels vs the host reference, on a real DAG ---"
	@./tests/test_kawpow_device $(KAWPOW_BLK_HEADER) $(KAWPOW_BLK_HEIGHT) \
		$(KAWPOW_BLK_NONCE) $(KAWPOW_BLK_MIX) $(KAWPOW_BLK_HASH) \
		$(KAWPOW_BLK_BITS) 8

$(BUILD)/dag.spv: src/algos/progpow/dag.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

$(BUILD)/kawpow.spv: src/algos/kawpow/kernel.comp | dirs
	glslangValidator -V --target-env vulkan1.2 $< -o $@

tests/test_kawpow_vulkan: tests/test_kawpow_vulkan.cpp $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

# The whole path, against a mock pool built from replies captured off live
# pools: binary, run loop, job parsing, dataset build, search, verify, submit.
# Needs a GPU; uses height 100 so the dataset is ~1 GB rather than 6.11.
#
# This is the one that found the counter bug: 263 shares submitted and the
# readout stuck at "accepted 0", because the run loop reached the counters
# through a dynamic_cast chain that did not name KawPow's source.
.PHONY: test-kawpow-pool
test-kawpow-pool: cuda
	@echo "--- kawpow: the real binary against a mock pool, end to end ---"
	@python3 tests/test_kawpow_pool.py ./soat-miner

# Needs a Vulkan GPU, so it is its own target like the CUDA device gate. It
# skips itself politely when there is no usable device.
test-kawpow-vulkan: tests/test_kawpow_vulkan $(BUILD)/dag.spv $(BUILD)/kawpow.spv \
                    $(BUILD)/kawpow_spec.spv
	@echo "--- kawpow: the Vulkan DAG and mix shaders vs the host reference ---"
	@./tests/test_kawpow_vulkan $(BUILD)/dag.spv $(BUILD)/kawpow.spv \
		$(BUILD)/kawpow_spec.spv

# The Vulkan backend at the geometry the miner really uses - a 6.11 GB DAG in
# 2 GiB device-local buffers - rather than the harness's 16 MB. Needs ~7 GB of
# VRAM; skips politely without it.
VK_T_KAWPOW = $(call vkobj,src/algos/kawpow/algo_vk.cpp) \
              $(call vkgen,spirv_kawpow) $(call vkgen,spirv_kawpow_spec) \
              $(call vkgen,spirv_dag) $(call vkobj,src/core/vk_common.cpp)

tests/test_kawpow_vk_real: tests/test_kawpow_vk_real.cpp $(VK_T_KAWPOW) \
                           src/algos/kawpow/kawpow.h src/algos/kawpow/job.h
	$(CXX) $(CXXFLAGS) $< $(VK_T_KAWPOW) -lvulkan -lpthread -o $@

test-kawpow-vulkan-real: tests/test_kawpow_vk_real
	@echo "--- kawpow: Vulkan at the real 6.11 GB geometry, vs a mainnet block ---"
	@./tests/test_kawpow_vk_real $(KAWPOW_BLK_HEADER) $(KAWPOW_BLK_HEIGHT) \
		$(KAWPOW_BLK_NONCE) $(KAWPOW_BLK_MIX) $(KAWPOW_BLK_BITS)

tests/test_progpow_stock: tests/test_progpow_stock.cpp $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

tests/test_progpow_ops: tests/test_progpow_ops.cpp $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

tests/test_progpow_092: tests/test_progpow_092.cpp $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< -o $@

tests/test_kawpow_stratum: tests/test_kawpow_stratum.cpp \
                           src/core/stratum_kawpow.cpp \
                           src/core/stratum_kawpow.h src/core/json_lite.h \
                           src/algos/kawpow/kawpow.h src/algos/kawpow/job.h \
                           $(PROGPOW_DEPS)
	$(CXX) $(CXXFLAGS) $< src/core/stratum_kawpow.cpp -lpthread -o $@

tests/test_kawpow_address: tests/test_kawpow_address.cpp src/core/kawpow_address.h \
                        src/core/sha256.h
	$(CXX) $(CXXFLAGS) $< -o $@

# test_algo_registry compares the binaries against the launcher, so both
# binaries have to exist for it to say anything.
# NOTE: this gate needs no GPU and therefore does NOT run the device gates.
# It ends by saying so - see the last line of the recipe. An offline gate that
# reports success without naming what it skipped is how "the tests pass" came
# to mean less than it sounded.
test-kawpow: cuda vulkan tests/test_kawpow_hash tests/test_kawpow_block \
             tests/test_progpow_stock tests/test_progpow_ops \
             tests/test_progpow_092 \
             tests/test_kawpow_stratum tests/test_kawpow_address | dirs
	@echo "--- kawpow: host reference vs Ravencoin's 13 published vectors ---"
	@./tests/test_kawpow_hash
	@echo "--- kawpow: four deliberately broken builds, each must fail it ---"
	@for m in $(KAWPOW_MUTANTS); do \
		$(CXX) $(CXXFLAGS) -D$$m tests/test_kawpow_hash.cpp \
		  -o $(BUILD)/mutant_kawpow || exit 1; \
		printf '  %-26s ' "$$m"; \
		$(BUILD)/mutant_kawpow --expect-fail || exit 1; \
	done
	@echo "--- kawpow: end to end against a real Ravencoin mainnet block ---"
	@./tests/test_kawpow_block $(KAWPOW_BLK_HEADER) $(KAWPOW_BLK_HEIGHT) \
		$(KAWPOW_BLK_NONCE) $(KAWPOW_BLK_MIX) $(KAWPOW_BLK_HASH) \
		$(KAWPOW_BLK_BITS)
	@echo "--- progpow family: every operation, named, both revisions ---"
	@./tests/test_progpow_ops
	@echo "--- progpow family: the shared layer vs stock ProgPoW's vectors ---"
	@./tests/test_progpow_stock
	@echo "--- progpow family: and those vectors discriminate the parameters ---"
	@for m in "--period 3" "--period 50" "--math092" "--ops 12 20"; do \
		printf '  %-14s ' "$$m"; \
		./tests/test_progpow_stock $$m --expect-fail || exit 1; \
	done
	@echo "--- progpow 0.9.2: the layers ProgPowZ needs, vs published vectors ---"
	@./tests/test_progpow_092
	@echo "--- kawpow: stratum, against shapes captured from live pools ---"
	@./tests/test_kawpow_stratum
	@echo "--- kawpow: Ravencoin address validation ---"
	@./tests/test_kawpow_address
	@echo "--- launcher: which backend each algorithm actually gets ---"
	@./tests/test_launcher_backend.sh
	@echo "--- registries: do the four lists describing an algorithm agree? ---"
	@./tests/test_algo_registry.sh
	@echo ""
	@echo "NOT RUN by this gate, because they need a real GPU:"
	@echo "  device kernels vs the host reference, and --bench on every algorithm."
	@echo "  Run them with:  make test-device"
	@echo "  A crash in --bench was invisible to CI until 2026-08-21 because"
	@echo "  nothing here said it had not been checked."

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

test-pearl: tests/test_pearl tests/test_pearl_job tests/test_pearl_prepare \
            tests/test_pearl_mining_shape \
            $(PEARL_VEC) $(PEARL_VEC64) $(PEARL_JOBVEC)
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

tests/test_pearl: tests/test_pearl.cu src/algos/pearl-pow/noisy_gemm.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

# Not part of `make test`: there is no Vulkan Pearl shader to feed it yet.
# Kept so the correctness gate exists the day someone writes one.
tests/test_pearl_vk: tests/test_pearl_vk.cpp
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

# Not part of `make test` - it is a measurement, not a gate.
tests/measure_launch_gap: tests/measure_launch_gap.cu \
                          src/algos/pearl-pow/noisy_gemm.cuh
	$(NVCC) $(NVFLAGS) -Isrc $< -o $@

tests/test_pearl_mining_shape: tests/test_pearl_mining_shape.cu \
                               src/algos/pearl-pow/noisy_gemm.cuh
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
                      src/algos/sha3-256t/mine.cuh \
            src/algos/kawpow/mine.cuh src/algos/kawpow/kawpow.h \
            src/algos/kawpow/job.h src/algos/progpow/mine.cuh \
            src/algos/progpow/keccak.h src/algos/progpow/ethash.h \
            src/algos/progpow/progpow.h src/core/stratum_kawpow.h \
            src/core/kawpow_address.h src/algos/sha3-256t/sha3.h \
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
	       tests/test_pearl_job tests/test_pearl_prepare \
	       tests/test_pearl_algo tests/test_pearl_gateway tests/test_btc_stratum \
	       tests/test_bc3_destination \
	       tests/test_sha3_algo tests/test_sha3_vulkan

# --- release packaging (lolMiner-style flat archive) -----------------------
VERSION ?= 0.2.17
PKGNAME  = soat-miner_v$(VERSION)_Lin64
package: cuda vulkan
	@rm -rf $(BUILD)/$(PKGNAME) && mkdir -p $(BUILD)/$(PKGNAME)
	cp $(BIN) $(BIN_VK) $(BUILD)/$(PKGNAME)/
	cp packaging/*.sh packaging/*.bat packaging/config.txt packaging/lithos-status \
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

# EVERY object in the Linux Vulkan build must appear here too, including the
# per-algorithm stratum sources, because run.cpp and vk_registry.cpp reference
# them unconditionally.
#
# This list has now been forgotten TWICE. Adding BC3 broke the cross-build with
# an undefined BitcoinStratumSource; adding KawPow broke it again with
# makeKawpowVK and KawpowStratumSource, in a session where the comment warning
# about the first one was directly above the target. A warning nothing checks
# does not prevent anything.
#
# So when adding an algorithm: build this target. `make windows` is the check,
# it needs only mingw and takes seconds, and it is the ONLY thing that catches
# this - the Linux build is perfectly happy either way.
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
# cannot be added to one platform and forgotten on the other. That happened
# twice before this; see the comment on VK_SRC_CORE.
windows: $(VK_SRC_GEN) $(WINDIR)/libvulkan-1.a $(WIN_OBJS)
	$(WINCXX) $(WINFLAGS) $(WIN_OBJS) $(WINDIR)/libvulkan-1.a -lws2_32 \
	    -o $(WINDIR)/soat-miner-vk.exe
	@echo "built: $(WINDIR)/soat-miner-vk.exe"

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
	cp packaging/*.bat packaging/config.txt packaging/README-WINDOWS.txt README.md LICENSE $(BUILD)/$(WINPKG)/
	cd $(BUILD) && zip -qr $(WINPKG).zip $(WINPKG)
	@sha256sum $(BUILD)/$(WINPKG).zip | tee $(BUILD)/$(WINPKG).zip.sha256
	@echo "packaged: $(BUILD)/$(WINPKG).zip"
