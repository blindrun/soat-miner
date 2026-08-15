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
BUILD    = build

ALGO_OBJS = $(BUILD)/autolykos2.o
CORE_OBJS = $(BUILD)/miner.o $(BUILD)/registry.o

.PHONY: all clean test bench install dirs

all: $(BIN)

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

$(BIN): $(CORE_OBJS) $(ALGO_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

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
	rm -rf $(BUILD) $(BIN) tests/test_element tests/test_hit
