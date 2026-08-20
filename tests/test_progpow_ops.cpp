// ProgPoW's random_math and random_merge, against the spec's own per-operation
// vectors - and the 0.9.2 rotation derived from them.
//
// Why this is worth a file of its own
// -----------------------------------
// The hash-level vectors in test_progpow_stock and test_kawpow_hash exercise
// these two functions millions of times, but only in aggregate: a wrong
// operation shows up as a wrong 256-bit digest with no indication of which of
// the eleven was wrong. These vectors name the operation each one exercises, so
// a failure here says "mul_hi32" instead of "something".
//
// The part that matters for the next coin
// ---------------------------------------
// ProgPowZ is revision 0.9.2, which labels the SAME eleven operations in a
// different order - selector % 11 == 0 means clz(a)+clz(b) there and a+b in
// KawPow's 0.9.4. The shared layer implements that as a rotation of the
// selector, and that one line is the difference between ProgPowZ working and
// producing perfect-looking rejected shares. Nothing else in the repository
// checks it, because no published vector covers 0.9.2.
//
// So each vector is used twice: once to verify 0.9.4 directly, and once to
// verify that the 0.9.2 variant produces the operation the rotation predicts -
// computed here from the operation NAMES rather than by calling the same switch
// the test is checking.
//
// Vectors from ifdefelse/ProgPOW, test-vectors.md.

#include <cstdio>
#include <cstring>
#include <string>

#include "algos/progpow/progpow.h"

using namespace om::progpow;

namespace {

int g_fail = 0;

void check(bool ok, const std::string &what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) g_fail++;
}

/** The eleven operations in 0.9.4's order, written out independently of the
 *  production switch so this test cannot agree with a bug by construction. */
uint32_t opAt(int index, uint32_t a, uint32_t b) {
    switch (index) {
        case 0: return a + b;
        case 1: return a * b;
        case 2: return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
        case 3: return a < b ? a : b;
        case 4: return rotl32(a, b);
        case 5: return rotr32(a, b);
        case 6: return a & b;
        case 7: return a | b;
        case 8: return a ^ b;
        case 9: {
            uint32_t ca = 0, cb = 0, x = a, y = b;
            if (x == 0) ca = 32; else while (!(x & 0x80000000u)) { x <<= 1; ca++; }
            if (y == 0) cb = 32; else while (!(y & 0x80000000u)) { y <<= 1; cb++; }
            return ca + cb;
        }
        default: {
            uint32_t na = 0, nb = 0, x = a, y = b;
            while (x) { na += x & 1u; x >>= 1; }
            while (y) { nb += y & 1u; y >>= 1; }
            return na + nb;
        }
    }
}

struct MathVec {
    uint32_t a, b, r, result;
    int index;              ///< which 0.9.4 operation the spec says this is
    const char *name;
};

const MathVec kMath[] = {
    {0x8626BB1Fu, 0xBBDFBC4Eu, 0x883E5B49u, 0x4206776Du, 0, "add"},
    {0x3F4BDFACu, 0xD79E414Fu, 0x36B71236u, 0x4C5CB214u, 1, "mul"},
    {0x6D175B7Eu, 0xC4E89D4Cu, 0x944ECABBu, 0x53E9023Fu, 2, "mul_hi32"},
    {0x2EDDD94Cu, 0x7E70CB54u, 0x3F472A85u, 0x2EDDD94Cu, 3, "min"},
    {0x61AE0E62u, 0xe0596b32u, 0x3F472A85u, 0x61AE0E62u, 3, "min, unsigned"},
    {0x8A81E396u, 0x3F4BDFACu, 0xCEC46E67u, 0x1E3968A8u, 4, "rotl32"},
    {0x8A81E396u, 0x7E70CB54u, 0xDBE71FF7u, 0x1E3968A8u, 5, "rotr32"},
    {0xA7352F36u, 0xA0EB7045u, 0x59E7B9D8u, 0xA0212004u, 6, "and"},
    {0xC89805AFu, 0x64291E2Fu, 0x1BDC84A9u, 0xECB91FAFu, 7, "or"},
    {0x760726D3u, 0x79FC6A48u, 0xC675CAC5u, 0x0FFB4C9Bu, 8, "xor"},
    {0x75551D43u, 0x3383BA34u, 0x2863AD31u, 0x00000003u, 9, "clz"},
    {0xEA260841u, 0xE92C44B7u, 0xF83FFE7Du, 0x0000001Bu, 10, "popcount"},
};

struct MergeVec {
    uint32_t a, b, r, result;
    const char *path;
};

const MergeVec kMerge[] = {
    {0x3B0BB37Du, 0xA0212004u, 0x9BD26AB0u, 0x3CA34321u, "mul/add"},
    {0x10C02F0Du, 0x870FA227u, 0xD4F45515u, 0x91C1326Au, "xor/mul"},
    {0x24D2BAE4u, 0x0FFB4C9Bu, 0x7FDBC2F2u, 0x2EDDD94Cu, "rotl/xor"},
    {0xDA39E821u, 0x089C4008u, 0x8B6CD8C3u, 0x8A81E396u, "rotr/xor"},
};

}  // namespace

int main(int argc, char **argv) {
    const bool expectFail = argc > 1 && std::string(argv[1]) == "--expect-fail";
    int quiet094 = 0, quiet092 = 0, quietSel = 0, quietMerge = 0;

    for (const MathVec &v : kMath) {
        // 1. The production function reproduces the spec's result.
        const uint32_t got = randomMath(v.a, v.b, v.r, kMath094);
        if (got == v.result) quiet094++;
        else if (!expectFail)
            printf("  FAIL  0.9.4 %-14s want %08x got %08x\n", v.name, v.result, got);

        // 2. The spec's own selector agrees with the operation it names, which
        //    is what pins the ORDER rather than just the arithmetic.
        if ((int)(v.r % 11u) == v.index) quietSel++;
        else if (!expectFail)
            printf("  FAIL  %-14s selector %%11 is %u, spec labels it %d\n",
                   v.name, v.r % 11u, v.index);

        // 3. Under 0.9.2 the same selector must select the operation nine
        //    positions along. Expected value computed from the operation table
        //    above, not from the function under test.
        const uint32_t want092 = opAt((v.index + 9) % 11, v.a, v.b);
        const uint32_t got092 = randomMath(v.a, v.b, v.r, kMath092);
        if (got092 == want092) quiet092++;
        else if (!expectFail)
            printf("  FAIL  0.9.2 %-14s want %08x got %08x\n", v.name, want092, got092);
    }

    for (const MergeVec &v : kMerge) {
        uint32_t a = v.a;
        randomMerge(&a, v.b, v.r);
        if (a == v.result) quietMerge++;
        else if (!expectFail)
            printf("  FAIL  merge %-10s want %08x got %08x\n", v.path, v.result, a);
    }

    const int total = (int)(sizeof(kMath) / sizeof(kMath[0]));
    const int mtotal = (int)(sizeof(kMerge) / sizeof(kMerge[0]));
    const int failures = (total - quiet094) + (total - quietSel) +
                         (total - quiet092) + (mtotal - quietMerge);

    if (expectFail) {
        if (failures > 0) {
            printf("  ok    mutant rejected: %d checks wrong\n", failures);
            return 0;
        }
        printf("  FAIL  mutant reproduced every vector\n");
        return 1;
    }

    check(quiet094 == total, "all 11 operations reproduce the spec's vectors (0.9.4)");
    check(quietSel == total, "the spec's selectors agree with the operations it names");
    check(quiet092 == total,
          "0.9.2 selects the operation nine positions along, for every one");
    check(quietMerge == mtotal, "all four random_merge paths reproduce their vectors");

    if (g_fail == 0) printf("ProgPoW operation vectors: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
