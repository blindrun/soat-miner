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
VERSION ?= 0.1.0
PKGNAME  = soat-miner_v$(VERSION)_Lin64
package: cuda vulkan
	@rm -rf $(BUILD)/$(PKGNAME) && mkdir -p $(BUILD)/$(PKGNAME)
	cp $(BIN) $(BIN_VK) $(BUILD)/$(PKGNAME)/
	cp packaging/soat-miner.sh packaging/soat-miner.bat packaging/config.txt \
	   README.md LICENSE $(BUILD)/$(PKGNAME)/
	cp scripts/soat-miner-guard.py scripts/guard.conf.example \
	   scripts/soat-miner.service scripts/soat-miner-guard.service \
	   $(BUILD)/$(PKGNAME)/
	chmod +x $(BUILD)/$(PKGNAME)/soat-miner.sh
	cd $(BUILD) && tar czf $(PKGNAME).tar.gz $(PKGNAME)
	@sha256sum $(BUILD)/$(PKGNAME).tar.gz | tee $(BUILD)/$(PKGNAME).tar.gz.sha256
	@echo "packaged: $(BUILD)/$(PKGNAME).tar.gz"
