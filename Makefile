# SOAT Miner
#
# ARCH defaults to sm_89 (Ada: RTX 4090/4080/4070). Override for other cards:
#   make ARCH=sm_86    # Ampere: RTX 3090/3080
#   make ARCH=sm_75    # Turing: RTX 2080/1660
#
# To add an algorithm: drop it in src/algos/<name>/, add the object to
# ALGO_OBJS below, and add two lines to src/core/registry.cu.

ARCH    ?= sm_89
NVCC    ?= nvcc
NVFLAGS  = -O3 -arch=$(ARCH) --std=c++17 -lineinfo
BIN      = soat-miner
BIN_VK   = soat-miner-vk
BUILD    = build
CXX     ?= g++
CXXFLAGS = -O3 --std=c++17 -Isrc

ALGO_OBJS = $(BUILD)/autolykos2.o
CORE_OBJS = $(BUILD)/miner.o $(BUILD)/registry.o $(BUILD)/run_cuda.o $(BUILD)/stratum_cuda.o
VK_OBJS   = $(BUILD)/miner_vk.o $(BUILD)/algo_vk.o $(BUILD)/spirv.o $(BUILD)/run_vk.o $(BUILD)/stratum_vk.o

.PHONY: all cuda vulkan clean test bench install dirs package

all: cuda vulkan

cuda: $(BIN)
vulkan: $(BIN_VK)

dirs:
	@mkdir -p $(BUILD)

$(BUILD)/miner.o: src/core/miner.cu src/core/algo.h src/core/http.h | dirs
	$(NVCC) $(NVFLAGS) -dc $< -o $@

$(BUILD)/registry.o: src/core/registry.cu src/core/algo.h | dirs
	$(NVCC) $(NVFLAGS) -dc $< -o $@

$(BUILD)/autolykos2.o: src/algos/autolykos2/algo.cu \
                       src/algos/autolykos2/mine.cuh \
                       src/algos/autolykos2/autolykos.cuh \
                       src/core/blake2b.cuh src/core/algo.h | dirs
	$(NVCC) $(NVFLAGS) -dc $< -o $@

$(BUILD)/stratum_cuda.o: src/core/stratum.cpp src/core/stratum.h | dirs
	$(NVCC) $(NVFLAGS) -dc -x cu $< -o $@

$(BUILD)/stratum_cl.o: src/core/stratum.cpp src/core/stratum.h | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/run_cuda.o: src/core/run.cpp src/core/run.h src/core/telemetry.h | dirs
	$(NVCC) $(NVFLAGS) -dc -x cu $< -o $@

$(BIN): $(CORE_OBJS) $(ALGO_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

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
test: tests/test_element tests/test_hit
	@echo "--- python reference vs real mainnet blocks ---"
	@python3 tests/reference.py
	@echo "--- device dataset elements vs python reference ---"
	@./tests/test_element 1851437

tests/test_element: tests/test_element.cu src/algos/autolykos2/autolykos.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

tests/test_hit: tests/test_hit.cu src/algos/autolykos2/mine.cuh src/core/blake2b.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

bench: $(BIN)
	./$(BIN) --bench

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)/usr/local/bin/$(BIN)

clean:
	rm -rf $(BUILD) $(BIN) $(BIN_VK) tests/test_element tests/test_hit tests/test_opencl

# --- release packaging (lolMiner-style flat archive) -----------------------
VERSION ?= 0.1.1
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
