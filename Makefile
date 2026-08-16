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
CUDA_SRC  = src/core/miner.cu src/core/registry.cu src/algos/autolykos2/algo.cu
CUDA_SRC_CPP = src/core/run.cpp src/core/stratum.cpp
CUDA_DEPS = src/core/algo.h src/core/http.h src/core/run.h src/core/telemetry.h \
            src/core/stratum.h src/core/blake2b.cuh \
            src/algos/autolykos2/mine.cuh src/algos/autolykos2/autolykos.cuh

VK_OBJS   = $(BUILD)/miner_vk.o $(BUILD)/algo_vk.o $(BUILD)/spirv.o $(BUILD)/run_vk.o $(BUILD)/stratum_vk.o

.PHONY: all cuda vulkan clean test bench install dirs package

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

$(BUILD)/spirv.o: $(BUILD)/spirv.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/algo_vk.o: src/algos/autolykos2/algo_vk.cpp src/core/algo.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/run_vk.o: src/core/run.cpp src/core/run.h src/core/telemetry.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/stratum_vk.o: src/core/stratum.cpp src/core/stratum.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/miner_vk.o: src/core/miner_vk.cpp src/core/run.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

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

test: tests/test_element tests/test_hit tests/test_vulkan
	@echo "--- python reference vs real mainnet blocks ---"
	@python3 tests/reference.py
	@echo "--- device dataset elements vs python reference ---"
	@./tests/test_element 1851437
	@echo "--- device end-to-end: real block's hit from its winning nonce ---"
	@./tests/test_hit $(HIT_MSG) $(HIT_HEIGHT) $(HIT_NONCE) $(HIT_EXPECT)
	@echo "--- vulkan backend against the same pinned vector ---"
	@./tests/test_vulkan $(HIT_MSG) $(HIT_HEIGHT) $(HIT_NONCE) $(HIT_EXPECT)

# The Vulkan gate needs only the algorithm and the embedded SPIR-V, not the
# miner's job/stratum plumbing.
tests/test_vulkan: tests/test_vulkan.cpp $(BUILD)/algo_vk.o $(BUILD)/spirv.o \
                   src/core/algo.h
	$(CXX) $(CXXFLAGS) $< $(BUILD)/algo_vk.o $(BUILD)/spirv.o \
	    -lvulkan -ldl -lpthread -o $@

tests/test_element: tests/test_element.cu src/algos/autolykos2/autolykos.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

tests/test_hit: tests/test_hit.cu src/algos/autolykos2/mine.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

bench: $(BIN)
	./$(BIN) --bench

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)/usr/local/bin/$(BIN)

clean:
	rm -rf $(BUILD) $(BIN) $(BIN_VK) tests/test_element tests/test_hit tests/test_vulkan

# --- release packaging (lolMiner-style flat archive) -----------------------
VERSION ?= 0.1.2
PKGNAME  = soat-miner_v$(VERSION)_Lin64
package: cuda vulkan
	@rm -rf $(BUILD)/$(PKGNAME) && mkdir -p $(BUILD)/$(PKGNAME)
	cp $(BIN) $(BIN_VK) $(BUILD)/$(PKGNAME)/
	cp packaging/*.sh packaging/*.bat packaging/config.txt \
	   README.md LICENSE $(BUILD)/$(PKGNAME)/
	cp scripts/soat-miner-guard.py scripts/guard.conf.example \
	   scripts/soat-miner.service scripts/soat-miner-guard.service \
	   $(BUILD)/$(PKGNAME)/
	chmod +x $(BUILD)/$(PKGNAME)/*.sh
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
$(WINDIR)/libvulkan-1.a: | $(WINDIR)/include/vulkan
	@mkdir -p $(WINDIR)
	@{ echo "LIBRARY vulkan-1.dll"; echo "EXPORTS"; \
	   grep -ohE '\bvk[A-Z][A-Za-z0-9]*\s*\(' src/algos/autolykos2/algo_vk.cpp \
	   | tr -d '( ' | sort -u \
	   | grep -vE '^vk(DeviceMemGB|DeviceName|DriverVersion|ListDevices)$$' \
	   | sed 's/^/    /'; } > $(WINDIR)/vulkan-1.def
	x86_64-w64-mingw32-dlltool -d $(WINDIR)/vulkan-1.def -l $@

windows: $(BUILD)/spirv.cpp $(WINDIR)/libvulkan-1.a
	$(WINCXX) $(WINFLAGS) -c src/algos/autolykos2/algo_vk.cpp -o $(WINDIR)/algo_vk.o
	$(WINCXX) $(WINFLAGS) -c $(BUILD)/spirv.cpp              -o $(WINDIR)/spirv.o
	$(WINCXX) $(WINFLAGS) -c src/core/run.cpp                -o $(WINDIR)/run.o
	$(WINCXX) $(WINFLAGS) -c src/core/stratum.cpp            -o $(WINDIR)/stratum.o
	$(WINCXX) $(WINFLAGS) -c src/core/miner_vk.cpp           -o $(WINDIR)/miner_vk.o
	$(WINCXX) $(WINFLAGS) $(WINDIR)/miner_vk.o $(WINDIR)/algo_vk.o $(WINDIR)/spirv.o \
	    $(WINDIR)/run.o $(WINDIR)/stratum.o $(WINDIR)/libvulkan-1.a -lws2_32 \
	    -o $(WINDIR)/soat-miner-vk.exe
	@x86_64-w64-mingw32-objdump -p $(WINDIR)/soat-miner-vk.exe | grep -i "DLL Name" | sort -u

windows-package: windows
	@rm -rf $(BUILD)/$(WINPKG) && mkdir -p $(BUILD)/$(WINPKG)
	cp $(WINDIR)/soat-miner-vk.exe $(BUILD)/$(WINPKG)/
	cp packaging/*.bat packaging/config.txt packaging/README-WINDOWS.txt README.md LICENSE $(BUILD)/$(WINPKG)/
	cd $(BUILD) && zip -qr $(WINPKG).zip $(WINPKG)
	@sha256sum $(BUILD)/$(WINPKG).zip | tee $(BUILD)/$(WINPKG).zip.sha256
	@echo "packaged: $(BUILD)/$(WINPKG).zip"
