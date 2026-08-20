# Pearl's own build rules. The Makefile does `include $(wildcard
# src/algos/*/algo.mk)`, so everything an algorithm needs lives in the
# algorithm's own directory and adding one edits no list anywhere else.
#
# This exists because the alternative kept failing, twice in the same way and
# once in a new one:
#
#   - VK_SRC_ALGO and VK_SRC_GEN were hand-maintained lists in the Makefile,
#     and twice a new algorithm went into the Linux half and not the Windows
#     half. Both times the Linux build was perfectly happy and nothing caught
#     it until someone ran the cross-build by hand. VK_SRC_ALGO is now a
#     wildcard, which fixes that half.
#   - The generated SPIR-V could not be a wildcard: it does not exist until it
#     is built, and the names are not derivable because each algorithm chose
#     its own. Hence this file.
#   - Pearl grew ten shaders and ten device gates over two sessions, and only
#     four shaders and four gates reached the Makefile. noise, apply_noise and
#     powscan passed on real hardware and then were never built again by
#     anything. That is the same class of bug and it is why the list below is
#     one list feeding rules, gates and the embedded modules together.
#
# A new algorithm should copy the shape of this file rather than touch the
# Makefile. autolykos2 and sha3-256t still have their rules in the Makefile;
# they should move here too, but that is their lanes' change, not one to slip
# into a Pearl commit.

PEARL_DIR = src/algos/pearl-pow
PEARL_SHADER_INC = -I$(PEARL_DIR)

# tag : source .comp : output .spv : embedded symbol
#
# The .spv names are the ones the device gates already pass on the command
# line, so they are kept rather than regularised. kernel.comp becomes
# kernel_pearl.spv because autolykos2 already owns kernel.spv.
PEARL_SHADERS = \
    gemm:kernel:kernel_pearl:kPearlGemmSpirv \
    chunk:merkle_chunk:merkle_chunk:kPearlMerkleChunkSpirv \
    reduce:merkle_reduce:merkle_reduce:kPearlMerkleReduceSpirv \
    root:merkle_root:merkle_root:kPearlMerkleRootSpirv \
    genmatrix:genmatrix:genmatrix:kPearlGenMatrixSpirv \
    transpose:transpose:transpose:kPearlTransposeSpirv \
    commitments:commitments:commitments:kPearlCommitmentsSpirv \
    noise:noise:noise:kPearlNoiseSpirv \
    applynoise:apply_noise:apply_noise:kPearlApplyNoiseSpirv \
    powscan:powscan:powscan:kPearlPowScanSpirv

pearl_field = $(word $(1),$(subst :, ,$(2)))

PEARL_SPV = $(foreach s,$(PEARL_SHADERS),$(BUILD)/$(call pearl_field,3,$(s)).spv)

VK_SRC_GEN += $(foreach s,$(PEARL_SHADERS),\
                $(BUILD)/spirv_pearl_$(call pearl_field,1,$(s)).cpp)

# One pair of rules per shader, generated rather than written out, so a shader
# added above needs nothing further. blake3.glsl is a prerequisite of all of
# them: it is #included, and glslang will not tell make it changed. It is .glsl
# and not .comp on purpose - it has no main(), so a %.comp pattern rule would
# try to build it standalone and fail.
# One pair of rules per shader, generated rather than written out, so a shader
# added above needs nothing further. blake3.glsl is a prerequisite of all of
# them: it is #included, and glslang will not tell make it changed. It is .glsl
# and not .comp on purpose - it has no main(), so a %.comp pattern rule would
# try to build it standalone and fail.
#
# The rule takes the whole colon-tuple as ONE argument and splits it inside.
# Passing four separate arguments does not work: $(call) splits its argument
# text on commas BEFORE expanding it, so a variable holding "a,b,c,d" arrives
# as argument 1 in its entirety and 2 through 4 are empty. Every rule was then
# generated for the target `build/.spv`, which make reports only as a
# duplicate-recipe warning - and `make -n | head` hid even that.
define PEARL_SHADER_RULE
$(BUILD)/$(call pearl_field,3,$(1)).spv: $(PEARL_DIR)/$(call pearl_field,2,$(1)).comp $(PEARL_DIR)/blake3.glsl | dirs
	glslangValidator -V --target-env vulkan1.3 $(PEARL_SHADER_INC) $$< -o $$@

$(BUILD)/spirv_pearl_$(call pearl_field,1,$(1)).cpp: $(BUILD)/$(call pearl_field,3,$(1)).spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $$< $$@ $(call pearl_field,4,$(1))
endef

$(foreach s,$(PEARL_SHADERS),$(eval $(call PEARL_SHADER_RULE,$(s))))

# The device gates. Every one of these must run on a real card of each vendor
# before a shader counts as done, so they are listed here beside the shader
# they check rather than in a separate list that can fall behind - which is
# exactly what happened to the last three.
PEARL_VK_GATES = tests/test_pearl_vk tests/test_pearl_merkle_vk \
                 tests/test_pearl_genmatrix_vk tests/test_pearl_transpose_vk \
                 tests/test_pearl_commitments_vk tests/test_pearl_noise_vk \
                 tests/test_pearl_apply_noise_vk tests/test_pearl_powscan_vk

tests/test_pearl_noise_vk: tests/test_pearl_noise_vk.cpp \
                           $(PEARL_DIR)/job.h $(BUILD)/noise.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

tests/test_pearl_apply_noise_vk: tests/test_pearl_apply_noise_vk.cpp \
                                 $(PEARL_DIR)/job.h $(BUILD)/apply_noise.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@

# One argument: the .spv. Wiring it in with the vector file first made it read
# 24 MB of test vectors as SPIR-V and fail with VK_ERROR_INVALID_SHADER, which
# reads exactly like a broken shader. Check a gate's usage line before adding
# it to a target; do not infer its arguments from its neighbours.
tests/test_pearl_powscan_vk: tests/test_pearl_powscan_vk.cpp \
                             $(PEARL_DIR)/job.h $(BUILD)/powscan.spv
	$(CXX) $(CXXFLAGS) $< -lvulkan -o $@
