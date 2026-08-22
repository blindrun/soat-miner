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
#include "algos/kawpow/nvrtc_source.h"

using namespace om::progpow;
using om::kawpow::emitSpecLane;

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

    // The lane body comes from the SHARED emitter in nvrtc_source.h, which is
    // the same one the runtime NVRTC path uses. It used to be a second copy
    // here, and a second copy is how the kernel that gets measured stops being
    // the kernel that ships.
    fputs(emitSpecLane(g).c_str(), stdout);
    printf("\n");
    printf("}}  // namespace\n");
    return 0;
}
