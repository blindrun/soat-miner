# KawPow's own build rules. The Makefile does `include $(wildcard
# src/algos/*/algo.mk)`, so an algorithm's build lives in the algorithm's own
# directory and adding one edits no list anywhere else.
#
# This file exists because the alternative failed here specifically: VK_SRC_GEN
# was a hand-maintained list in the Makefile, KawPow was added to the Linux half
# and not the Windows half, and the mingw link broke on makeKawpowVK and
# KawpowStratumSource. The Linux build was perfectly happy throughout. BC3 had
# broken the same list the same way one algorithm earlier, with the comment
# warning about it directly above the target - a warning nothing checks does not
# prevent anything.
#
# The generated SPIR-V cannot be a wildcard: it does not exist until it is
# built, and the embedded symbol names are not derivable. Hence the table.

KAWPOW_DIR = src/algos/kawpow

# tag : source .comp : output .spv : embedded symbol
#
# dag.comp lives under progpow/ because the DAG build is shared with ProgPowZ,
# not KawPow-specific - the .spv name is kept as dag.spv since the device gates
# already pass it on the command line.
KAWPOW_SHADERS = \
    kawpow:$(KAWPOW_DIR)/kernel:kawpow:kKawpowSpirv \
    kawpow_spec:$(KAWPOW_DIR)/kernel_spec:kawpow_spec:kKawpowSpecSpirv \
    dag:src/algos/progpow/dag:dag:kProgpowDagSpirv

kawpow_field = $(word $(1),$(subst :, ,$(2)))

VK_SRC_GEN += $(foreach s,$(KAWPOW_SHADERS),\
                $(BUILD)/spirv_$(call kawpow_field,1,$(s)).cpp)

# One pair of rules per shader, generated rather than written out.
#
# The rule takes the whole colon-tuple as ONE argument and splits it inside.
# Passing four separate arguments does not work: $(call) splits its argument
# text on commas BEFORE expanding it, so a variable holding "a,b,c,d" arrives as
# argument 1 entire and 2 through 4 are empty - and every rule is then generated
# for the target `build/.spv`, which make reports only as a duplicate-recipe
# warning.
define KAWPOW_SHADER_RULE
$(BUILD)/$(call kawpow_field,3,$(1)).spv: $(call kawpow_field,2,$(1)).comp | dirs
	glslangValidator -V --target-env vulkan1.2 $$< -o $$@

$(BUILD)/spirv_$(call kawpow_field,1,$(1)).cpp: $(BUILD)/$(call kawpow_field,3,$(1)).spv scripts/embed_spirv.py | dirs
	python3 scripts/embed_spirv.py $$< $$@ $(call kawpow_field,4,$(1))
endef

$(foreach s,$(KAWPOW_SHADERS),$(eval $(call KAWPOW_SHADER_RULE,$(s))))
