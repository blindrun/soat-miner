#!/usr/bin/env python3
r"""Emits the Keccak-f[800] round body used by the ProgPoW mix shaders.

The sibling script gen_keccak_round.py does the same for Keccak-f[1600], which
the Ethash DAG needs. This one is the 32-bit-lane permutation ProgPoW hashes
with: same 5x5 state, same rho/pi tables, 22 rounds instead of 24, and the
rotation offsets taken modulo 32 because the lanes are half as wide.

That modulo is the whole reason this is generated rather than typed. Eleven of
the twenty-four offsets change when you narrow the lane - 36 becomes 4, 62
becomes 30 - and a single one left at its 64-bit value produces digests that
look perfectly plausible and are wrong. The tables below are the same ones in
src/algos/progpow/keccak.h, which is the host and CUDA source of truth.

The generated text is checked in as part of the shader; this script exists so
the derivation is reproducible and reviewable, not as a build step.
"""

# Rotation offsets and the pi lane permutation, in rho/pi step order. These are
# the f[1600] tables; the modulo below is what makes them f[800]'s.
ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
        27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]

# No offset is a multiple of 32, so no rotation degenerates to zero - which
# would need a special case, GLSL's shift by 32 being undefined.
assert all(r % 32 != 0 for r in ROTC), "a rotation reduced to zero mod 32"

out = []
out.append("        // Theta")
for i in range(5):
    terms = " ^ ".join(f"a[{i + 5 * j}]" for j in range(5))
    out.append(f"        bc{i} = {terms};")
for i in range(5):
    out.append(f"        t = bc{(i + 4) % 5} ^ rotl32(bc{(i + 1) % 5}, 1);")
    for j in range(0, 25, 5):
        out.append(f"        a[{j + i}] ^= t;")

out.append("")
out.append("        // Rho and Pi")
out.append("        t = a[1];")
for i in range(24):
    j = PILN[i]
    out.append(f"        bc0 = a[{j}];")
    out.append(f"        a[{j}] = rotl32(t, {ROTC[i] % 32});")
    out.append("        t = bc0;")

out.append("")
out.append("        // Chi")
for j in range(0, 25, 5):
    for i in range(5):
        out.append(f"        bc{i} = a[{j + i}];")
    for i in range(5):
        out.append(f"        a[{j + i}] = bc{i} ^ ((~bc{(i + 1) % 5}) & bc{(i + 2) % 5});")

out.append("")
out.append("        // Iota")
out.append("        a[0] ^= RC800[r];")

print("\n".join(out))
