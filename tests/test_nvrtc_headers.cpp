// Can NVRTC compile a kernel that includes our real headers?
//
// This is the foundation of runtime kernel generation: if NVRTC cannot compile
// the headers the interpreted kernel uses, the generated kernel would need its
// own copy of keccak and the mix primitives - a second definition of a hash
// function, which is how a miner produces shares that are confidently wrong.
//
// It passes, and four things were needed to make it pass. Each would have been
// a blocker found later and harder:
//
//   1. NVRTC has no C library, so `stdint.h` is supplied as a stub.
//   2. NVRTC matches in-memory headers by EXACT name and does not resolve
//      relative paths.
//   3. `#pragma once` is per-path, so registering one header under two
//      spellings includes it twice and redefines every function. The include
//      lines are canonicalised instead.
//   4. `-default-device` is required: ethash.h has host-only `inline`
//      functions and JIT mode rejects unannotated ones.
//
// Kept as a test because it is a standing property. Editing a device header to
// add a standard-library include, or a host-only construct NVRTC rejects,
// silently breaks runtime generation - and this catches it at build time
// rather than when a generated kernel fails to compile on a user's machine.
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <nvrtc.h>
#include <cstring>

static std::string slurp(const char *p) {
    std::ifstream f(p);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main() {
    // NVRTC has no C library. Our device headers need only fixed-width
    // integer types, so a stub is enough - and keeping it minimal means it
    // cannot quietly diverge from the real one in ways that matter.
    static const char *kStdint =
        "#pragma once\n"
        "typedef unsigned char uint8_t;\n"
        "typedef unsigned short uint16_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "typedef signed char int8_t;\n"
        "typedef short int16_t;\n"
        "typedef int int32_t;\n"
        "typedef long long int64_t;\n";
    // NVRTC matches in-memory headers by EXACT name and does not resolve
    // relative paths, so each header is registered under every spelling that
    // references it: kawpow.h asks for "../progpow/ethash.h" while progpow.h
    // asks for "ethash.h", and both must resolve to the same text.
    const char *names[] = {"stdint.h", "keccak.h", "ethash.h", "progpow.h",
                           "kawpow.h"};
    const char *paths[] = {
        nullptr,   // stdint is the stub above, not a file
        "/home/blindrun/Projects/soat-kawpow/src/algos/progpow/keccak.h",
        "/home/blindrun/Projects/soat-kawpow/src/algos/progpow/ethash.h",
        "/home/blindrun/Projects/soat-kawpow/src/algos/progpow/progpow.h",
        "/home/blindrun/Projects/soat-kawpow/src/algos/kawpow/kawpow.h"};
    std::vector<std::string> bodies;
    std::vector<const char *> hsrc, hname;
    // Canonicalise the include spellings so each header is registered ONCE.
    // `#pragma once` is per-path, so the same text registered as both
    // "keccak.h" and "../progpow/keccak.h" is included twice and every
    // function is redefined.
    auto canon = [](std::string t) {
        const char *from[] = {"\"../progpow/keccak.h\"", "\"../progpow/ethash.h\"",
                              "\"../progpow/progpow.h\""};
        const char *to[] = {"\"keccak.h\"", "\"ethash.h\"", "\"progpow.h\""};
        for (int i = 0; i < 3; i++) {
            size_t p;
            while ((p = t.find(from[i])) != std::string::npos)
                t.replace(p, strlen(from[i]), to[i]);
        }
        return t;
    };
    bodies = {kStdint, canon(slurp(paths[1])), canon(slurp(paths[2])),
              canon(slurp(paths[3])), canon(slurp(paths[4]))};
    for (size_t i = 0; i < bodies.size(); i++) {
        hsrc.push_back(bodies[i].c_str());
        hname.push_back(names[i]);
    }

    const std::string src = R"CU(
#include "kawpow.h"
extern "C" __global__ void probe(unsigned *out) {
    // Touch one function from each header so nothing is optimised away.
    unsigned st[25]; for (int i = 0; i < 25; i++) st[i] = i;
    om::progpow::keccakf800(st);
    unsigned m = om::progpow::fnv1a(2166136261u, st[0]);
    m = om::progpow::randomMath(m, st[1], 3u, om::progpow::kMath094);
    out[0] = m;
}
)CU";

    nvrtcProgram prog;
    nvrtcResult r = nvrtcCreateProgram(&prog, src.c_str(), "probe.cu",
                                       (int)hsrc.size(), hsrc.data(), hname.data());
    if (r != NVRTC_SUCCESS) { printf("create failed: %s\n", nvrtcGetErrorString(r)); return 1; }
    const char *opts[] = {"--std=c++17", "--gpu-architecture=compute_89",
                          "-default-device"};  // host-only inlines in ethash.h
    r = nvrtcCompileProgram(prog, 3, opts);
    size_t logSize = 0; nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    if (logSize > 1) nvrtcGetProgramLog(prog, &log[0]);
    if (r != NVRTC_SUCCESS) {
        printf("COMPILE FAILED: %s\n%s\n", nvrtcGetErrorString(r), log.c_str());
        return 1;
    }
    size_t ptxSize = 0; nvrtcGetPTXSize(prog, &ptxSize);
    printf("NVRTC COMPILED OUR HEADERS: %zu bytes of PTX\n", ptxSize);
    if (logSize > 1) printf("log: %s\n", log.c_str());
    return 0;
}
