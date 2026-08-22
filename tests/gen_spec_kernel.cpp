// Emits a KawPow search kernel with one period's program as literal constants.
//
// This is what NVRTC produces at runtime. Generating it offline and compiling
// with nvcc measures the same codegen on the REAL kernel - with the DAG gather
// in the picture - without building the runtime-compilation plumbing first.
// The difference between this and NVRTC is when compilation happens, not what
// it emits.
//
// Usage: gen_spec_kernel <blockHeight> > spec_kernel.cuh
#include <cstdio>
#include <cstdlib>

#include "algos/kawpow/kawpow.h"

using namespace om::progpow;

int main(int argc, char **argv) {
    const int height = argc > 1 ? atoi(argv[1]) : 4501000;
    Params p = om::kawpow::params();
    const Program g = buildProgram(height, p);

    printf("// GENERATED for block height %d - do not edit.\n", height);
    printf("// Program is a function of height/period, so this kernel is valid\n");
    printf("// ONLY for that period. algo.cu checks before using it.\n");
    printf("#pragma once\n#include \"../src/algos/kawpow/mine.cuh\"\n\n");
    printf("namespace om { namespace kawpow {\n\n");
    printf("static const int kSpecHeight = %d;\n\n", height);

    // The specialised lane: every index and selector a literal.
    printf("__device__ __forceinline__ void specLane(uint32_t *mix,\n");
    printf("        const uint32_t *l1, const uint32_t dagWords[4]) {\n");
    const int maxOps = g.cacheAccesses > g.mathOps ? g.cacheAccesses : g.mathOps;
    for (int i = 0; i < maxOps; i++) {
        if (i < g.cacheAccesses)
            printf("    om::progpow::randomMerge(&mix[%u], "
                   "l1[mix[%u] %% om::progpow::kL1CacheWords], %uu);\n",
                   g.cacheDst[i], g.cacheSrc[i], g.cacheSel[i]);
        if (i < g.mathOps)
            printf("    om::progpow::randomMerge(&mix[%u], "
                   "om::progpow::randomMath(mix[%u], mix[%u], %uu, %d), %uu);\n",
                   g.mathDst[i], g.mathSrc1[i], g.mathSrc2[i], g.mathSel1[i],
                   g.mathVariant, g.mathSel2[i]);
    }
    for (uint32_t i = 0; i < om::progpow::kDagLoads; i++)
        printf("    om::progpow::randomMerge(&mix[%u], dagWords[%u], %uu);\n",
               g.dagDst[i], i, g.dagSel[i]);
    printf("}\n\n");
    printf("}}  // namespace\n");
    return 0;
}
