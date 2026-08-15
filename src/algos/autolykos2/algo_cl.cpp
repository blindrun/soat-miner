// Autolykos v2 - OpenCL backend (AMD, Intel, NVIDIA).
//
// Verified to produce hits byte-identical to the CUDA backend and to the
// Python reference; see tests/test_opencl.cpp.
//
// The dataset is split across up to four buffers because OpenCL only
// guarantees CL_DEVICE_MAX_MEM_ALLOC_SIZE per allocation, which is commonly a
// quarter of VRAM - 6.3 GB on a 24 GB card, against a 7.27 GB dataset.

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/algo.h"

// Generated from kernel.cl at build time (see Makefile).
extern const char *kAutolykosKernelSource;

namespace om {
namespace {

uint32_t calcN(uint64_t height) {
    const uint32_t NBase = 1u << 26, Start = 600u * 1024u, Period = 50u * 1024u,
                   MaxH = 4198400u;
    uint32_t h = (height < MaxH) ? (uint32_t)height : MaxH;
    if (h < Start) return NBase;
    uint32_t iters = (h - Start) / Period + 1, N = NBase;
    for (uint32_t i = 0; i < iters; i++) N = N / 100 * 105;
    return N;
}

class Autolykos2CL : public Algorithm {
   public:
    const char *name() const override { return "autolykos2"; }

    size_t memoryBytes(const Job &job) const override {
        return (size_t)calcN(job.epoch) * 32ULL;
    }

    /** Every GPU across every OpenCL platform, in enumeration order. */
    static std::vector<std::pair<cl_platform_id, cl_device_id>> enumerateGpus() {
        std::vector<std::pair<cl_platform_id, cl_device_id>> out;
        cl_uint np = 0;
        if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return out;
        std::vector<cl_platform_id> plats(np);
        clGetPlatformIDs(np, plats.data(), nullptr);
        for (auto p : plats) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS ||
                nd == 0)
                continue;
            std::vector<cl_device_id> devs(nd);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
            for (auto d : devs) out.push_back({p, d});
        }
        return out;
    }

    /**
     * Picks a device. With no explicit index, the one with the most VRAM wins -
     * on a machine with an iGPU and a discrete card that is reliably the right
     * answer, and it is what "autodetect the card" has to mean in practice.
     */
    bool init(int requestedIndex = -1) {
        const auto gpus = enumerateGpus();
        if (gpus.empty()) {
            fprintf(stderr,
                    "no OpenCL GPU found.\n"
                    "  NVIDIA: install the driver (OpenCL is included)\n"
                    "  AMD:    install an OpenCL runtime - ROCm, amdgpu-pro, or "
                    "Mesa rusticl\n"
                    "  check with: clinfo -l\n");
            return false;
        }

        size_t pick = 0;
        if (requestedIndex >= 0) {
            if ((size_t)requestedIndex >= gpus.size()) {
                fprintf(stderr, "device %d requested but only %zu GPU(s) found\n",
                        requestedIndex, gpus.size());
                return false;
            }
            pick = (size_t)requestedIndex;
        } else {
            cl_ulong best = 0;
            for (size_t i = 0; i < gpus.size(); i++) {
                cl_ulong mem = 0;
                clGetDeviceInfo(gpus[i].second, CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(mem), &mem, nullptr);
                if (mem > best) { best = mem; pick = i; }
            }
        }
        plat_ = gpus[pick].first;
        dev_ = gpus[pick].second;

        clGetDeviceInfo(dev_, CL_DEVICE_NAME, sizeof(devName_), devName_, nullptr);
        clGetDeviceInfo(dev_, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc_),
                        &maxAlloc_, nullptr);
        clGetDeviceInfo(dev_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(totalMem_),
                        &totalMem_, nullptr);
        clGetDeviceInfo(dev_, CL_DRIVER_VERSION, sizeof(driver_), driver_, nullptr);

        cl_int err = 0;
        ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
#if defined(CL_VERSION_2_0)
        queue_ = clCreateCommandQueueWithProperties(ctx_, dev_, nullptr, &err);
#else
        queue_ = clCreateCommandQueue(ctx_, dev_, 0, &err);
#endif
        if (err != CL_SUCCESS) return false;

        const char *src = kAutolykosKernelSource;
        size_t len = strlen(src);
        prog_ = clCreateProgramWithSource(ctx_, 1, &src, &len, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(prog_, 1, &dev_, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t logSize = 0;
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize + 1, 0);
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, logSize,
                                  log.data(), nullptr);
            fprintf(stderr, "OpenCL build failed:\n%s\n", log.data());
            return false;
        }

        kBuild_ = clCreateKernel(prog_, "build_dataset", &err);
        if (err != CL_SUCCESS) return false;
        kSearch_ = clCreateKernel(prog_, "search", &err);
        if (err != CL_SUCCESS) return false;
        kVerify_ = clCreateKernel(prog_, "verify_one", &err);
        if (err != CL_SUCCESS) return false;

        dMsg_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, 32, nullptr, &err);
        dTarget_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, 32, nullptr, &err);
        dOut_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                               sizeof(cl_ulong) * 5 * kMaxSolutions, nullptr, &err);
        dCount_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
        dVerify_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, 32, nullptr, &err);
        return err == CL_SUCCESS;
    }

    const char *deviceName() const { return devName_; }
    double deviceMemGB() const { return totalMem_ / 1e9; }
    const char *driverVersion() const { return driver_; }

    bool prepare(const Job &job) override {
        const uint32_t N = calcN(job.epoch);
        const uint64_t total = (uint64_t)N * 32ULL;

        uint32_t chunks = 1;
        while ((total + chunks - 1) / chunks > (uint64_t)(maxAlloc_ * 0.9)) chunks++;
        if (chunks > 4) {
            fprintf(stderr,
                    "dataset needs %u chunks but the kernel addresses 4; "
                    "device max alloc is %.2f GB\n",
                    chunks, maxAlloc_ / 1e9);
            return false;
        }

        if (N != n_ || chunks != chunks_) {
            release();
            chunkElems_ = (N + chunks - 1) / chunks;
            cl_int err = 0;
            for (uint32_t c = 0; c < 4; c++) {
                const uint32_t count =
                    (c < chunks) ? ((c == chunks - 1) ? (N - c * chunkElems_)
                                                      : chunkElems_)
                                 : 1;
                ds_[c] = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (size_t)count * 32ULL, nullptr, &err);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "chunk %u alloc failed (%.2f GB)\n", c,
                            count * 32.0 / 1e9);
                    return false;
                }
            }
            n_ = N;
            chunks_ = chunks;
        }

        const uint32_t height = (uint32_t)job.epoch;
        for (uint32_t c = 0; c < chunks_; c++) {
            const uint32_t first = c * chunkElems_;
            const uint32_t count = (c == chunks_ - 1) ? (N - first) : chunkElems_;
            clSetKernelArg(kBuild_, 0, sizeof(cl_mem), &ds_[c]);
            clSetKernelArg(kBuild_, 1, sizeof(uint32_t), &first);
            clSetKernelArg(kBuild_, 2, sizeof(uint32_t), &count);
            clSetKernelArg(kBuild_, 3, sizeof(uint32_t), &height);
            size_t local = 256;
            size_t global = ((count + local - 1) / local) * local;
            if (clEnqueueNDRangeKernel(queue_, kBuild_, 1, nullptr, &global, &local,
                                       0, nullptr, nullptr) != CL_SUCCESS) {
                fprintf(stderr, "dataset build enqueue failed\n");
                return false;
            }
        }
        if (clFinish(queue_) != CL_SUCCESS) return false;
        height_ = height;
        return true;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!ds_[0]) return false;

        clEnqueueWriteBuffer(queue_, dMsg_, CL_TRUE, 0, 32, job.msg, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue_, dTarget_, CL_TRUE, 0, 32, job.target, 0, nullptr,
                             nullptr);
        cl_uint zero = 0;
        clEnqueueWriteBuffer(queue_, dCount_, CL_TRUE, 0, sizeof(cl_uint), &zero, 0,
                             nullptr, nullptr);

        int a = 0;
        for (int c = 0; c < 4; c++) clSetKernelArg(kSearch_, a++, sizeof(cl_mem), &ds_[c]);
        clSetKernelArg(kSearch_, a++, sizeof(uint32_t), &chunkElems_);
        clSetKernelArg(kSearch_, a++, sizeof(cl_mem), &dMsg_);
        cl_ulong base = nonceBase;
        clSetKernelArg(kSearch_, a++, sizeof(cl_ulong), &base);
        clSetKernelArg(kSearch_, a++, sizeof(uint32_t), &n_);
        clSetKernelArg(kSearch_, a++, sizeof(uint32_t), &height_);
        clSetKernelArg(kSearch_, a++, sizeof(cl_mem), &dTarget_);
        clSetKernelArg(kSearch_, a++, sizeof(cl_mem), &dOut_);
        clSetKernelArg(kSearch_, a++, sizeof(cl_mem), &dCount_);
        cl_uint maxOut = kMaxSolutions;
        clSetKernelArg(kSearch_, a++, sizeof(cl_uint), &maxOut);

        size_t local = 256;
        size_t global = ((size_t)count / local) * local;
        if (global == 0) global = local;
        if (clEnqueueNDRangeKernel(queue_, kSearch_, 1, nullptr, &global, &local, 0,
                                   nullptr, nullptr) != CL_SUCCESS)
            return false;
        if (clFinish(queue_) != CL_SUCCESS) return false;

        cl_uint found = 0;
        clEnqueueReadBuffer(queue_, dCount_, CL_TRUE, 0, sizeof(cl_uint), &found, 0,
                            nullptr, nullptr);
        if (found == 0) return true;
        if (found > kMaxSolutions) found = kMaxSolutions;

        std::vector<cl_ulong> raw(5 * found);
        clEnqueueReadBuffer(queue_, dOut_, CL_TRUE, 0, sizeof(cl_ulong) * 5 * found,
                            raw.data(), 0, nullptr, nullptr);
        for (cl_uint i = 0; i < found; i++) {
            Solution s;
            s.nonce = raw[i * 5];
            for (int l = 0; l < 4; l++) s.hit[l] = raw[i * 5 + 1 + l];
            out->push_back(s);
        }
        return true;
    }

    bool verify(const Job &job, const Solution &sol) const override {
        if (!ds_[0]) return false;
        clEnqueueWriteBuffer(queue_, dMsg_, CL_TRUE, 0, 32, job.msg, 0, nullptr, nullptr);
        int a = 0;
        for (int c = 0; c < 4; c++) clSetKernelArg(kVerify_, a++, sizeof(cl_mem), &ds_[c]);
        clSetKernelArg(kVerify_, a++, sizeof(uint32_t), &chunkElems_);
        clSetKernelArg(kVerify_, a++, sizeof(cl_mem), &dMsg_);
        cl_ulong nonce = sol.nonce;
        clSetKernelArg(kVerify_, a++, sizeof(cl_ulong), &nonce);
        clSetKernelArg(kVerify_, a++, sizeof(uint32_t), &n_);
        clSetKernelArg(kVerify_, a++, sizeof(uint32_t), &height_);
        clSetKernelArg(kVerify_, a++, sizeof(cl_mem), &dVerify_);

        size_t one = 1;
        if (clEnqueueNDRangeKernel(queue_, kVerify_, 1, nullptr, &one, &one, 0,
                                   nullptr, nullptr) != CL_SUCCESS)
            return false;
        clFinish(queue_);

        uint64_t hit[4];
        clEnqueueReadBuffer(queue_, dVerify_, CL_TRUE, 0, 32, hit, 0, nullptr, nullptr);
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        for (int i = 3; i >= 0; i--) {
            if (hit[i] != job.target[i]) return hit[i] < job.target[i];
        }
        return false;
    }

    void release() override {
        for (int c = 0; c < 4; c++) {
            if (ds_[c]) { clReleaseMemObject(ds_[c]); ds_[c] = nullptr; }
        }
        n_ = 0;
        chunks_ = 0;
    }

    ~Autolykos2CL() override { release(); }

   private:
    static const cl_uint kMaxSolutions = 32;

    cl_platform_id plat_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_context ctx_ = nullptr;
    mutable cl_command_queue queue_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel kBuild_ = nullptr, kSearch_ = nullptr;
    mutable cl_kernel kVerify_ = nullptr;
    cl_mem ds_[4] = {nullptr, nullptr, nullptr, nullptr};
    mutable cl_mem dMsg_ = nullptr;
    cl_mem dTarget_ = nullptr, dOut_ = nullptr, dCount_ = nullptr;
    mutable cl_mem dVerify_ = nullptr;
    cl_ulong maxAlloc_ = 0, totalMem_ = 0;
    char devName_[256] = {0};
    char driver_[128] = {0};
    uint32_t n_ = 0, chunks_ = 0, chunkElems_ = 0, height_ = 0;
};

Autolykos2CL *g_instance = nullptr;

}  // namespace

Algorithm *makeAutolykos2CL(int deviceIndex) {
    auto *a = new Autolykos2CL();
    if (!a->init(deviceIndex)) {
        delete a;
        return nullptr;
    }
    g_instance = a;
    return a;
}

const char *clDeviceName() { return g_instance ? g_instance->deviceName() : "unknown"; }
double clDeviceMemGB() { return g_instance ? g_instance->deviceMemGB() : 0.0; }
const char *clDriverVersion() {
    return g_instance ? g_instance->driverVersion() : "";
}

/** Prints every OpenCL GPU the machine exposes, for --list-devices. */
void clListDevices() {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        printf("no OpenCL platforms found\n");
        return;
    }
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    int idx = 0;
    for (auto p : plats) {
        char pname[256] = {0};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS ||
            nd == 0)
            continue;
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (auto d : devs) {
            char dn[256] = {0};
            cl_ulong mem = 0, alloc = 0;
            cl_uint cu = 0;
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(alloc), &alloc,
                            nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            printf("  [%d] %-38s %5.1f GB  max alloc %4.1f GB  %3u CUs  (%s)\n",
                   idx++, dn, mem / 1e9, alloc / 1e9, cu, pname);
        }
    }
    if (idx == 0) printf("no OpenCL GPU devices found\n");
}

}  // namespace om
