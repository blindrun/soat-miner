#!/usr/bin/env python3
r"""Emits the Keccak-f[1600] round body used by src/algos/sha3-256t/kernel.comp.

The round is straight-line GLSL with literal array indices, which is what keeps
the 25 lanes in registers - a dynamic index into a local array is the GLSL trap
that cost this repo a 75x slowdown in the Blake2b kernel.

It is generated rather than hand-written because the rho/pi step is 24
assignments driven by two permutation tables, and a single transposed entry
produces digests that look perfectly plausible and are wrong. The tables below
are the same ones in src/algos/sha3-256t/sha3.h, which is the host and CUDA
source of truth.

The generated text is checked in as part of kernel.comp; this script exists so
the derivation is reproducible and reviewable, not as a build step. Regenerate
and diff against the kernel with:

    python3 scripts/gen_keccak_round.py | diff - <(sed -n '/Theta/,/RC\[r\]/p' \
        src/algos/sha3-256t/kernel.comp)
"""

# Rotation offsets and the pi lane permutation, in rho/pi step order.
ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
        27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]

out = []
out.append("        // Theta")
for i in range(5):
    out.append("        bc%d = a[%d] ^ a[%d] ^ a[%d] ^ a[%d] ^ a[%d];"
               % (i, i, i + 5, i + 10, i + 15, i + 20))
for i in range(5):
    out.append("        t = bc%d ^ rotl64(bc%d, 1);" % ((i + 4) % 5, (i + 1) % 5))
    for j in range(0, 25, 5):
        out.append("        a[%d] ^= t;" % (j + i))
out.append("")
out.append("        // Rho and Pi")
out.append("        t = a[1];")
for i in range(24):
    j = PILN[i]
    out.append("        bc0 = a[%d]; a[%d] = rotl64(t, %d); t = bc0;"
               % (j, j, ROTC[i]))
out.append("")
out.append("        // Chi")
for j in range(0, 25, 5):
    for i in range(5):
        out.append("        bc%d = a[%d];" % (i, j + i))
    for i in range(5):
        out.append("        a[%d] ^= (~bc%d) & bc%d;" % (j + i, (i + 1) % 5, (i + 2) % 5))
out.append("")
out.append("        // Iota")
out.append("        a[0] ^= RC[r];")

print("\n".join(out))
