#!/usr/bin/env python3
"""Embeds a compiled SPIR-V module as a uint32 array so the binary is self-contained."""
import sys, struct
src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
assert len(data) % 4 == 0, "SPIR-V must be a multiple of 4 bytes"
words = struct.unpack(f"<{len(data)//4}I", data)
with open(dst, "w") as f:
    f.write("// Generated from kernel.comp - do not edit.\n")
    f.write("#include <stdint.h>\n#include <stddef.h>\n")
    f.write("extern const uint32_t kAutolykosSpirv[];\n")
    f.write("extern const size_t kAutolykosSpirvWords;\n")
    f.write(f"const uint32_t kAutolykosSpirv[] = {{\n")
    for i in range(0, len(words), 8):
        f.write("    " + ",".join(f"0x{w:08x}u" for w in words[i:i+8]) + ",\n")
    f.write("};\n")
    f.write(f"const size_t kAutolykosSpirvWords = {len(words)};\n")
print(f"embedded {len(data)} bytes ({len(words)} words) -> {dst}")
