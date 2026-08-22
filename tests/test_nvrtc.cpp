// Does the runtime-compilation chain actually work, end to end, offline?
//
// Runtime kernel generation has three links and each can fail on its own:
// the embedded headers must satisfy NVRTC, the generated source must be valid
// CUDA, and the compiled module must expose the kernels by the names the
// launcher looks up. A GPU is needed only for the last of those, so all three
// are checked here without one - which means a broken generator is caught by
// `make test` rather than on a user's machine.
//
// What this test does NOT establish is that the generated kernel computes the
// same hash as the interpreted one. That needs a device, and it lives in the
// device gate. Saying so here matters: a green result from this file means
// "it compiles", and compiling is the cheap half.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <nvrtc.h>

#include "algos/kawpow/kawpow.h"
#include "algos/kawpow/nvrtc_headers.h"
#include "algos/kawpow/nvrtc_source.h"

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char *what, const std::string &detail = "") {
    checks++;
    if (ok) {
        printf("  ok    %s\n", what);
    } else {
        failures++;
        printf("  FAIL  %s\n", what);
        if (!detail.empty()) printf("        %s\n", detail.c_str());
    }
}

/// Compile one translation unit with the embedded headers, returning PTX.
static bool compile(const std::string &src, std::string *ptx,
                    std::string *log) {
    int n = 0;
    const om::kawpow::NvrtcHeader *hdrs = om::kawpow::nvrtcHeaders(&n);
    std::vector<const char *> names, sources;
    for (int i = 0; i < n; i++) {
        names.push_back(hdrs[i].name);
        sources.push_back(hdrs[i].source);
    }

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "kawpow_spec.cu", n,
                           sources.data(), names.data()) != NVRTC_SUCCESS)
        return false;

    // -default-device: ethash.h has host-only `inline` functions and JIT mode
    // rejects unannotated ones. Without it every one of them is an error.
    const char *opts[] = {"-default-device", "--std=c++14"};
    const nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);

    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        log->resize(logSize);
        nvrtcGetProgramLog(prog, &(*log)[0]);
    }
    if (r != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    ptx->resize(ptxSize);
    nvrtcGetPTX(prog, &(*ptx)[0]);
    nvrtcDestroyProgram(&prog);
    return true;
}

int main() {
    printf("NVRTC runtime-compilation chain\n");

    // 1. The embedded header table must be complete and uniquely named. A
    //    duplicate name resolves silently to the wrong header, so this is
    //    checked rather than assumed - see scripts/embed_headers.py.
    int n = 0;
    const om::kawpow::NvrtcHeader *hdrs = om::kawpow::nvrtcHeaders(&n);
    check(n >= 7, "every device header is embedded",
          "got " + std::to_string(n));
    bool unique = true;
    for (int i = 0; i < n && unique; i++)
        for (int j = i + 1; j < n; j++)
            if (strcmp(hdrs[i].name, hdrs[j].name) == 0) unique = false;
    check(unique, "no two headers share an include name");

    // 2. The real, unspecialised kernel compiles under NVRTC. This is the
    //    foundation: the generated kernel includes it rather than restating
    //    the mix, so if this fails nothing above it can work.
    const std::string plain = std::string("#include \"") +
                              om::kawpow::kDeviceHeaderName() +
                              "\"\nextern \"C\" __global__ void t(unsigned *o) "
                              "{ *o = om::progpow::kL1CacheWords; }\n";
    std::string ptx, log;
    check(compile(plain, &ptx, &log), "the real device header compiles", log);

    // 3. The generated source for a real period compiles.
    const int height = 4501000;
    const om::progpow::Program g =
        om::progpow::buildProgram(height, om::kawpow::params());
    const std::string src = om::kawpow::specialisedSource(g);
    log.clear();
    const bool built = compile(src, &ptx, &log);
    check(built, "the generated specialised kernel compiles", log);

    if (built) {
        // 4. Both kernels must be present under the exact names the launcher
        //    asks cuModuleGetFunction for. A rename in the generator would
        //    otherwise surface as a null function pointer at mining time.
        check(ptx.find(".entry kpSearchSpec") != std::string::npos,
              "kpSearchSpec is exported under its C name");
        check(ptx.find(".entry kpHashSpec") != std::string::npos,
              "kpHashSpec is exported under its C name");

        // 5. The point of the exercise: the program's indices are literals, so
        //    mix[] can live in registers. If the generator emitted a loop over
        //    the Program struct instead, this would still compile and still be
        //    slow - and the only visible difference would be a hashrate nobody
        //    had a baseline for.
        size_t merges = 0;
        for (size_t p = src.find("randomMerge"); p != std::string::npos;
             p = src.find("randomMerge", p + 1))
            merges++;
        const size_t want = (size_t)g.cacheAccesses + (size_t)g.mathOps +
                            om::progpow::kDagLoads;
        check(merges == want, "every operation is emitted as a literal",
              "emitted " + std::to_string(merges) + " merges, expected " +
                  std::to_string(want));
    }

    printf("%d check(s), %d failure(s)\n", checks, failures);
    // A gate that skips everything reports success too, so zero checks is a
    // failure in its own right.
    if (checks == 0) {
        printf("FAIL: no checks ran\n");
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
