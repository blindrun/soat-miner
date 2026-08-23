// Runtime compilation of the specialised KawPow kernel.
//
// # What this buys, and what it costs
//
// The interpreted kernel reads each operation's indices out of a `Program`
// struct at runtime. Indexing `mix[32]` by a runtime value means it cannot live
// in registers, so it spills: a 128-byte stack frame per lane, every round.
// Compiling the period's program into the kernel as literals removes the
// indexing entirely and the array stays in registers.
//
// The cost is that compilation happens while mining, once per period. That
// latency is NOT measured yet - stated here rather than estimated, because an
// estimate in a comment becomes a fact in a summary two hands later.
//
// # Why both libraries are dlopen'd
//
// NVRTC ships with the CUDA toolkit, not the driver, so a machine that can run
// this binary may well not have it. Linking it would turn a missing optional
// optimisation into a binary that will not start. The same pattern as
// `telemetry.h` uses for NVML, for the same reason.
//
// # Why this is off by default
//
// Generating and compiling code at runtime is behaviour that endpoint security
// products react to, and that reaction has not been tested on Windows. Whether
// Defender or SmartScreen objects is an open question, not an assumed answer,
// so the interpreted path stays the default and this is opt-in until somebody
// has actually run it on a quarantined machine.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "kawpow.h"
#include "nvrtc_headers.h"
#include "nvrtc_source.h"

namespace om {
namespace kawpow {

/// Compiles and holds one period's specialised kernels.
///
/// A single instance is reused across periods: `build()` replaces the module,
/// so a long mining run does not leak a module per period.
class NvrtcJit {
  public:
    ~NvrtcJit() { unload(); }

    NvrtcJit() = default;
    NvrtcJit(const NvrtcJit &) = delete;
    NvrtcJit &operator=(const NvrtcJit &) = delete;

    /// Are both libraries present and usable?
    ///
    /// Answered once and cached, because the answer cannot change while the
    /// process runs and dlopen on every period would be noise.
    bool available() {
        if (probed_) return ok_;
        probed_ = true;
        ok_ = loadLibs();
        if (!ok_ && !quiet_) {
            fprintf(stderr,
                    "[kawpow] runtime kernel compilation unavailable: %s\n",
                    why_.c_str());
        }
        return ok_;
    }

    /// Compile the kernels for `program`, replacing whatever was loaded.
    ///
    /// Returns false and leaves the object unloaded on any failure, so a caller
    /// that ignores the return value falls back rather than launching a null
    /// function.
    bool build(const om::progpow::Program &program, int major, int minor) {
        if (!available()) return false;
        unload();

        const std::string src = specialisedSource(program);
        std::string ptx, log;
        const auto tc = std::chrono::steady_clock::now();
        const bool compiled = compile(src, major, minor, &ptx, &log);
        compileMs_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - tc)
                         .count();
        if (!compiled) {
            fprintf(stderr, "[kawpow] kernel compile failed; using the "
                            "interpreted kernel\n");
            if (!log.empty()) fprintf(stderr, "%s\n", log.c_str());
            return false;
        }
        return loadPtx(ptx);
    }

    /// Load already-compiled PTX, replacing whatever was loaded.
    ///
    /// Split out from `build()` so a kernel compiled ahead of time - on another
    /// thread, during the previous period - can be adopted without recompiling.
    /// This half needs the device; the compile half does not, which is the
    /// whole reason the split is worth having.
    bool loadPtx(const std::string &ptx) {
        if (!available()) return false;
        if (module_) unload();

        // The driver API needs a current context and the runtime API only
        // creates the primary context lazily - on the first allocation, not on
        // cudaGetDeviceProperties. Without this, cuModuleLoadData fails with
        // INVALID_CONTEXT on a perfectly good PTX, which reads as a compiler
        // problem and is not one.
        cudaFree(nullptr);

        const auto tl = std::chrono::steady_clock::now();
        const CUresult lr = pCuModuleLoadData(&module_, ptx.c_str());
        loadMs_ = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - tl)
                      .count();
        if (lr != CUDA_SUCCESS) {
            // Print the code. "Could not load the compiled module" sent me
            // looking at the generated source for a fault that was in the
            // caller's setup, and a message that cannot distinguish those two
            // costs more than it saves.
            const char *name = nullptr;
            if (pCuGetErrorName) pCuGetErrorName(lr, &name);
            fprintf(stderr,
                    "[kawpow] could not load the compiled module (%s); "
                    "using the interpreted kernel\n",
                    name ? name : "unknown driver error");
            module_ = nullptr;
            return false;
        }
        if (pCuModuleGetFunction(&search_, module_, "kpSearchSpec") !=
                CUDA_SUCCESS ||
            pCuModuleGetFunction(&hash_, module_, "kpHashSpec") !=
                CUDA_SUCCESS) {
            // The names are asserted offline by tests/test_nvrtc, so reaching
            // here means the generator and the launcher have diverged.
            fprintf(stderr, "[kawpow] compiled module is missing a kernel\n");
            unload();
            return false;
        }
        return true;
    }

    /// Take ownership of another instance's already-loaded module.
    ///
    /// The point of the exercise: cuModuleLoadData costs 0.4 ms on an idle GPU
    /// and ~400 ms while mining - the same call, contending with the in-flight
    /// kernel. So the load happens on the background thread during the previous
    /// period, and the period boundary only moves three pointers.
    void takeModuleFrom(NvrtcJit &other) {
        unload();
        module_ = other.module_;
        search_ = other.search_;
        hash_ = other.hash_;
        compileMs_ = other.compileMs_;
        loadMs_ = other.loadMs_;
        other.module_ = nullptr;
        other.search_ = nullptr;
        other.hash_ = nullptr;
    }

    /// True once `build()` has succeeded and the kernels are launchable.
    bool loaded() const { return module_ != nullptr && search_ && hash_; }

    CUfunction searchKernel() const { return search_; }
    CUfunction hashKernel() const { return hash_; }

    /// Launch a compiled kernel. Thin wrapper so callers need no driver-API
    /// declarations of their own.
    bool launch(CUfunction f, unsigned blocks, unsigned threads,
                unsigned sharedBytes, void *stream, void **args) {
        if (!f) return false;
        return pCuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, sharedBytes,
                               (CUstream)stream, args, nullptr) == CUDA_SUCCESS;
    }

    /// Compile only, no module load and no CUDA context.
    ///
    /// NVRTC is a compiler library and needs no device, so this measures the
    /// dominant part of the per-period cost on a machine whose GPU is busy -
    /// which is not a hypothetical: the one local card is shared with three
    /// other lanes. Excludes module load and verification, both of which do
    /// need the device.
    bool compileOnly(const om::progpow::Program &program, int major, int minor,
                     std::string *ptx) {
        if (!available()) return false;
        std::string log;
        // Timed here as well as in build(). Without this the background
        // thread's compile is reported as "compile 0 ms", which is not a
        // measurement, it is a field nobody filled in.
        const auto t = std::chrono::steady_clock::now();
        const bool ok = compile(specialisedSource(program), major, minor, ptx,
                                &log);
        compileMs_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t).count();
        return ok;
    }

    /// How long the last successful build() spent compiling, and loading.
    ///
    /// Separate because they are different costs with different fixes: a slow
    /// compile is the generated source, a slow load is the driver.
    double lastCompileMs() const { return compileMs_; }
    double lastLoadMs() const { return loadMs_; }

    /// Suppress the one-time "unavailable" message, for probes that expect it.
    void setQuiet(bool q) { quiet_ = q; }

    /// Why the libraries could not be used, for diagnostics.
    const std::string &unavailableReason() const { return why_; }

  private:
    void unload() {
        if (module_ && pCuModuleUnload) pCuModuleUnload(module_);
        module_ = nullptr;
        search_ = nullptr;
        hash_ = nullptr;
    }

    bool compile(const std::string &src, int major, int minor, std::string *ptx,
                 std::string *log) {
        int n = 0;
        const NvrtcHeader *hdrs = nvrtcHeaders(&n);
        std::vector<const char *> names, sources;
        for (int i = 0; i < n; i++) {
            names.push_back(hdrs[i].name);
            sources.push_back(hdrs[i].source);
        }

        void *prog = nullptr;
        if (pNvrtcCreateProgram(&prog, src.c_str(), "kawpow_spec.cu", n,
                                sources.data(), names.data()) != 0)
            return false;

        // sm_XX, not compute_XX, and that distinction is worth 375 ms per
        // period.
        //
        // compute_XX yields PTX, and cuModuleLoadData on PTX makes the DRIVER
        // JIT it to SASS - on the critical path, while mining. That cost hid
        // from an offline harness because CUDA caches driver JIT results in
        // ~/.nv/ComputeCache and the harness kept re-loading the same few
        // programs; in a real run every period is a program the cache has
        // never seen. sm_XX makes NVRTC run ptxas itself, so what reaches the
        // driver is a finished cubin with nothing left to compile.
        //
        // Still no device involved: ptxas inside NVRTC is host code, so this
        // stays on the background thread where it belongs.
        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major,
                 minor);
        // -default-device: ethash.h has host-only `inline` functions and JIT
        // mode rejects unannotated ones.
        const char *opts[] = {arch, "-default-device", "--std=c++14"};
        const int r = pNvrtcCompileProgram(prog, 3, opts);

        size_t logSize = 0;
        pNvrtcGetProgramLogSize(prog, &logSize);
        if (logSize > 1) {
            log->resize(logSize);
            pNvrtcGetProgramLog(prog, &(*log)[0]);
        }
        if (r != 0) {
            pNvrtcDestroyProgram(&prog);
            return false;
        }
        // Prefer the cubin. Fall back to PTX where NVRTC is too old to
        // provide one (nvrtcGetCUBIN arrived in 11.2) - correct, just with the
        // driver-JIT cost back at each period boundary.
        bool got = false;
        if (pNvrtcGetCUBINSize && pNvrtcGetCUBIN) {
            size_t cubinSize = 0;
            if (pNvrtcGetCUBINSize(prog, &cubinSize) == 0 && cubinSize > 0) {
                ptx->resize(cubinSize);
                got = pNvrtcGetCUBIN(prog, &(*ptx)[0]) == 0;
            }
        }
        if (!got) {
            size_t ptxSize = 0;
            pNvrtcGetPTXSize(prog, &ptxSize);
            ptx->resize(ptxSize);
            got = pNvrtcGetPTX(prog, &(*ptx)[0]) == 0;
        }
        pNvrtcDestroyProgram(&prog);
        return got;
    }

    bool loadLibs() {
        nvrtcLib_ = dlopen("libnvrtc.so", RTLD_LAZY);
        if (!nvrtcLib_) nvrtcLib_ = dlopen("libnvrtc.so.12", RTLD_LAZY);
        if (!nvrtcLib_) nvrtcLib_ = dlopen("libnvrtc.so.11.2", RTLD_LAZY);
        if (!nvrtcLib_) {
            why_ = "libnvrtc not found (it ships with the CUDA toolkit, not "
                   "the driver)";
            return false;
        }
        cudaLib_ = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!cudaLib_) cudaLib_ = dlopen("libcuda.so", RTLD_LAZY);
        if (!cudaLib_) {
            why_ = "libcuda not found";
            return false;
        }

        pNvrtcCreateProgram = (int (*)(void **, const char *, const char *, int,
                                       const char **, const char **))
            dlsym(nvrtcLib_, "nvrtcCreateProgram");
        pNvrtcCompileProgram =
            (int (*)(void *, int, const char **))dlsym(nvrtcLib_,
                                                       "nvrtcCompileProgram");
        pNvrtcGetPTXSize =
            (int (*)(void *, size_t *))dlsym(nvrtcLib_, "nvrtcGetPTXSize");
        pNvrtcGetPTX = (int (*)(void *, char *))dlsym(nvrtcLib_, "nvrtcGetPTX");
        pNvrtcGetProgramLogSize = (int (*)(void *, size_t *))dlsym(
            nvrtcLib_, "nvrtcGetProgramLogSize");
        pNvrtcGetProgramLog =
            (int (*)(void *, char *))dlsym(nvrtcLib_, "nvrtcGetProgramLog");
        pNvrtcDestroyProgram =
            (int (*)(void **))dlsym(nvrtcLib_, "nvrtcDestroyProgram");
        // Optional: absent before NVRTC 11.2, checked at the call site rather
        // than in the all-symbols test below.
        pNvrtcGetCUBINSize =
            (int (*)(void *, size_t *))dlsym(nvrtcLib_, "nvrtcGetCUBINSize");
        pNvrtcGetCUBIN =
            (int (*)(void *, char *))dlsym(nvrtcLib_, "nvrtcGetCUBIN");

        pCuModuleLoadData = (CUresult (*)(CUmodule *, const void *))dlsym(
            cudaLib_, "cuModuleLoadData");
        pCuModuleGetFunction =
            (CUresult (*)(CUfunction *, CUmodule, const char *))dlsym(
                cudaLib_, "cuModuleGetFunction");
        pCuModuleUnload =
            (CUresult (*)(CUmodule))dlsym(cudaLib_, "cuModuleUnload");
        pCuLaunchKernel =
            (CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                          unsigned, unsigned, unsigned, CUstream, void **,
                          void **))dlsym(cudaLib_, "cuLaunchKernel");
        // Diagnostics only, so a missing one is not fatal - it is checked at
        // the call site rather than in the all-symbols test below.
        pCuGetErrorName = (CUresult (*)(CUresult, const char **))dlsym(
            cudaLib_, "cuGetErrorName");

        // Every symbol checked, not just the first. A partial load would fail
        // later at a call site with no context, which is a worse place to find
        // out than here.
        if (!pNvrtcCreateProgram || !pNvrtcCompileProgram || !pNvrtcGetPTXSize ||
            !pNvrtcGetPTX || !pNvrtcGetProgramLogSize || !pNvrtcGetProgramLog ||
            !pNvrtcDestroyProgram || !pCuModuleLoadData ||
            !pCuModuleGetFunction || !pCuModuleUnload || !pCuLaunchKernel) {
            why_ = "libnvrtc or libcuda is missing an expected symbol";
            return false;
        }
        return true;
    }

    void *nvrtcLib_ = nullptr;
    void *cudaLib_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction search_ = nullptr;
    CUfunction hash_ = nullptr;
    bool probed_ = false, ok_ = false, quiet_ = false;
    double compileMs_ = 0, loadMs_ = 0;
    std::string why_;

    int (*pNvrtcCreateProgram)(void **, const char *, const char *, int,
                               const char **, const char **) = nullptr;
    int (*pNvrtcCompileProgram)(void *, int, const char **) = nullptr;
    int (*pNvrtcGetPTXSize)(void *, size_t *) = nullptr;
    int (*pNvrtcGetPTX)(void *, char *) = nullptr;
    int (*pNvrtcGetProgramLogSize)(void *, size_t *) = nullptr;
    int (*pNvrtcGetProgramLog)(void *, char *) = nullptr;
    int (*pNvrtcDestroyProgram)(void **) = nullptr;
    int (*pNvrtcGetCUBINSize)(void *, size_t *) = nullptr;
    int (*pNvrtcGetCUBIN)(void *, char *) = nullptr;

    CUresult (*pCuModuleLoadData)(CUmodule *, const void *) = nullptr;
    CUresult (*pCuModuleGetFunction)(CUfunction *, CUmodule,
                                     const char *) = nullptr;
    CUresult (*pCuModuleUnload)(CUmodule) = nullptr;
    CUresult (*pCuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned, unsigned, CUstream,
                                void **, void **) = nullptr;
    CUresult (*pCuGetErrorName)(CUresult, const char **) = nullptr;
};

}  // namespace kawpow
}  // namespace om
