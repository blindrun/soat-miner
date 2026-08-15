#!/usr/bin/env python3
"""Embeds kernel.cl into a C++ string literal so the binary is self-contained."""
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    text = f.read()
with open(dst, "w") as f:
    f.write("// Generated from kernel.cl - do not edit.\n")
    f.write('extern const char *kAutolykosKernelSource;\n')
    f.write('const char *kAutolykosKernelSource =\n')
    for line in text.splitlines():
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        f.write(f'    "{esc}\\n"\n')
    f.write(";\n")
print(f"embedded {len(text)} bytes -> {dst}")
