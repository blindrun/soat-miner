// Verifies the OpenCL backend against the same real mainnet block the CUDA
// backend is checked with. Both must produce a byte-identical hit.
//
// usage: test_opencl <msg_hex64> <height> <nonce_hex16>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static uint32_t calcN(uint32_t height) {
    const uint32_t NBase = 1u << 26, Start = 600u * 1024u, Period = 50u * 1024u,
                   MaxH = 4198400u;
    uint32_t h = height < MaxH ? height : MaxH;
    if (h < Start) return NBase;
    uint32_t iters = (h - Start) / Period + 1, N = NBase;
    for (uint32_t i = 0; i < iters; i++) N = N / 100 * 105;
    return N;
}

static void hex2bin(const char *hex, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v;
        sscanf(hex + i * 2, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

#define CHECK(x)                                                          \
    do {                                                                  \
        cl_int _e = (x);                                                  \
        if (_e != CL_SUCCESS) {                                           \
            printf("OpenCL error %d at line %d (%s)\n", _e, __LINE__, #x); \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s <msg_hex64> <height> <nonce_hex16>\n", argv[0]);
        return 1;
    }
    uint8_t msg[32];
    hex2bin(argv[1], msg, 32);
    uint32_t height = (uint32_t)strtoul(argv[2], nullptr, 10);
    uint64_t nonce = strtoull(argv[3], nullptr, 16);
    uint32_t N = calcN(height);

    cl_platform_id plat;
    cl_device_id dev;
    cl_uint n = 0;
    CHECK(clGetPlatformIDs(1, &plat, &n));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &n));

    char devName[256];
    cl_ulong maxAlloc = 0, totalMem = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devName), devName, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(totalMem), &totalMem, nullptr);
    printf("device: %s | %.1f GB | max alloc %.2f GB\n", devName, totalMem / 1e9,
           maxAlloc / 1e9);

    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    CHECK(err);

    std::ifstream f("src/algos/autolykos2/kernel.cl");
    if (!f) { printf("cannot open kernel.cl (run from repo root)\n"); return 1; }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    const char *srcPtr = src.c_str();
    size_t srcLen = src.size();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcPtr, &srcLen, &err);
    CHECK(err);
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize + 1, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        printf("build failed:\n%s\n", log.data());
        return 1;
    }
    printf("kernel compiled\n");

    // --- chunk the dataset to respect CL_DEVICE_MAX_MEM_ALLOC_SIZE --------
    const uint64_t total = (uint64_t)N * 32ULL;
    uint32_t chunks = 1;
    while ((total + chunks - 1) / chunks > (uint64_t)(maxAlloc * 0.9)) chunks++;
    if (chunks > 4) { printf("needs %u chunks, kernel supports 4\n", chunks); return 1; }
    const uint32_t chunkElems = (N + chunks - 1) / chunks;
    printf("dataset %.2f GB in %u chunk(s) of %u elements\n", total / 1e9, chunks,
           chunkElems);

    cl_mem buf[4] = {nullptr, nullptr, nullptr, nullptr};
    for (uint32_t c = 0; c < 4; c++) {
        uint32_t count = (c < chunks)
                             ? ((c == chunks - 1) ? (N - c * chunkElems) : chunkElems)
                             : 1;
        buf[c] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)count * 32ULL,
                                nullptr, &err);
        CHECK(err);
    }

    cl_kernel kBuild = clCreateKernel(prog, "build_dataset", &err);
    CHECK(err);
    for (uint32_t c = 0; c < chunks; c++) {
        uint32_t first = c * chunkElems;
        uint32_t count = (c == chunks - 1) ? (N - first) : chunkElems;
        CHECK(clSetKernelArg(kBuild, 0, sizeof(cl_mem), &buf[c]));
        CHECK(clSetKernelArg(kBuild, 1, sizeof(uint32_t), &first));
        CHECK(clSetKernelArg(kBuild, 2, sizeof(uint32_t), &count));
        CHECK(clSetKernelArg(kBuild, 3, sizeof(uint32_t), &height));
        size_t local = 256;
        size_t global = ((count + local - 1) / local) * local;
        CHECK(clEnqueueNDRangeKernel(q, kBuild, 1, nullptr, &global, &local, 0,
                                     nullptr, nullptr));
    }
    CHECK(clFinish(q));
    printf("dataset built\n");

    // --- verify a known nonce --------------------------------------------
    cl_mem dMsg = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32,
                                 msg, &err);
    CHECK(err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, nullptr, &err);
    CHECK(err);

    cl_kernel kVer = clCreateKernel(prog, "verify_one", &err);
    CHECK(err);
    int a = 0;
    for (int c = 0; c < 4; c++) CHECK(clSetKernelArg(kVer, a++, sizeof(cl_mem), &buf[c]));
    CHECK(clSetKernelArg(kVer, a++, sizeof(uint32_t), &chunkElems));
    CHECK(clSetKernelArg(kVer, a++, sizeof(cl_mem), &dMsg));
    CHECK(clSetKernelArg(kVer, a++, sizeof(uint64_t), &nonce));
    CHECK(clSetKernelArg(kVer, a++, sizeof(uint32_t), &N));
    CHECK(clSetKernelArg(kVer, a++, sizeof(uint32_t), &height));
    CHECK(clSetKernelArg(kVer, a++, sizeof(cl_mem), &dOut));

    size_t one = 1;
    CHECK(clEnqueueNDRangeKernel(q, kVer, 1, nullptr, &one, &one, 0, nullptr, nullptr));
    CHECK(clFinish(q));

    uint64_t hit[4];
    CHECK(clEnqueueReadBuffer(q, dOut, CL_TRUE, 0, 32, hit, 0, nullptr, nullptr));

    printf("nonce=%016llx\nhit  =", (unsigned long long)nonce);
    for (int l = 3; l >= 0; l--) printf("%016llx", (unsigned long long)hit[l]);
    printf("\n");
    return 0;
}
