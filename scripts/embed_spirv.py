#!/usr/bin/env python3
"""Embeds a compiled SPIR-V module as a uint32 array so the binary is self-contained.

usage: embed_spirv.py <in.spv> <out.cpp> [symbol]

The symbol defaults to kAutolykosSpirv for the original single-shader build.
A second algorithm passes its own, e.g. kSha3Spirv, so both modules can live in
the same binary - which they must, now that the Vulkan build has a registry
rather than one hardcoded algorithm.
"""
import sys, struct

src, dst = sys.argv[1], sys.argv[2]
sym = sys.argv[3] if len(sys.argv) > 3 else "kAutolykosSpirv"

data = open(src, "rb").read()
assert len(data) % 4 == 0, "SPIR-V must be a multiple of 4 bytes"
words = struct.unpack(f"<{len(data)//4}I", data)
with open(dst, "w") as f:
    f.write(f"// Generated from a .comp shader - do not edit.\n")
    f.write("#include <stdint.h>\n#include <stddef.h>\n")
    f.write(f"extern const uint32_t {sym}[];\n")
    f.write(f"extern const size_t {sym}Words;\n")
    f.write(f"const uint32_t {sym}[] = {{\n")
    for i in range(0, len(words), 8):
        f.write("    " + ",".join(f"0x{w:08x}u" for w in words[i:i+8]) + ",\n")
    f.write("};\n")
    f.write(f"const size_t {sym}Words = {len(words)};\n")
print(f"embedded {len(data)} bytes ({len(words)} words) -> {dst} as {sym}")
