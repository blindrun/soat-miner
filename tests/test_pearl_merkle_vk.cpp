// Pearl's blake3 Merkle roots on Vulkan, against the same reference vectors as
// the CUDA test. Every backend must produce byte-identical roots.
//
// This is the gate for chunkCvs / reduceTree / rootCv, ported to GLSL as
// merkle_chunk.comp / merkle_reduce.comp / merkle_root.comp. It mirrors
// sections 2 and 2b of tests/test_pearl_prepare.cu assertion for assertion and
// reads the SAME vector file, so "byte-identical to CUDA" is literal rather
// than transitive:
//
//   section 2   the A and B^t roots from pearl_job_vectors.bin (PRLJ0002),
//               which are the exact bytes the CUDA test asserts against
//   section 2b  a sweep from 2 to 131072 chunks against the host blake3 in
//               job.h, which is itself pinned to the blake3 library and to
//               Pearl's Rust
//
// The sweep going all the way to 131072 is not thoroughness for its own sake.
// That is the count that exposed the shared-memory race in the CUDA: A at 8192
// chunks came out right and B at 131072 came out wrong ON THE SAME CARD,
// because the race scaled with block count. A sweep that stops at 8192 proves
// nothing about the kernel this file exists to check.
//
// RUN THIS ON THE 7900 XT. Measured 2026-08-20 by building a deliberately racy
// merkle_reduce - the read barrier deleted, the original one-barrier-per-level
// bug - and putting it through this gate:
//
//     RTX 4090   (subgroup 32)  broken shader PASSES 5/5   sees nothing
//     RX 7900 XT (subgroup 64)  broken shader FAILS  2/3   catches it
//     llvmpipe   (subgroup 8)   broken shader PASSES        sees nothing
//
// and it failed at 131072 chunks only - 65536 and every smaller size passed the
// broken shader on every card. So a clean run on the 4090 alone means this file
// has proved nothing, and neither does a single run on the 7900 XT: the failure
// is non-deterministic and gave a different wrong root each time. That is why
// the largest case runs five times below.
//
// The intuition that a wider subgroup is a wider hiding place, and that AMD
// would therefore be the lenient card, was written down before the measurement
// and was exactly backwards. It is workgroup scheduling, not subgroup width,
// and it is not predictable from the shader.
//
// usage: test_pearl_merkle_vk <job_vectors.bin> <chunk.spv> <reduce.spv>
//                             <root.spv> [device_index]
//
// Deliberately needs NO cooperative matrix, no int8 storage and no 64-bit
// types - it is plain uint compute. So unlike test_pearl_vk it also runs on
// llvmpipe, which is a free third implementation to disagree with.

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <string>
#include <vector>

#include "../src/algos/pearl-pow/job.h"
#include "vk_claim_guard.h"
#include "vk_spv.h"

#define VKCHECK(x)                                                        \
    do {                                                                  \
        VkResult r_ = (x);                                                \
        if (r_ != VK_SUCCESS) {                                           \
            fprintf(stderr, "%s:%d %s -> VkResult %d\n", __FILE__,        \
                    __LINE__, #x, (int)r_);                               \
            return 2;                                                     \
        }                                                                 \
    } while (0)

namespace {

using namespace om::pearl;

int failures = 0;
int checksRun = 0;

void check(const std::string &name, bool ok, const std::string &detail = "") {
    printf("  [%s] %s%s%s\n", ok ? "ok" : "FAIL", name.c_str(),
           detail.empty() ? "" : " - ", detail.c_str());
    checksRun++;
    if (!ok) failures++;
}

std::string hex(const uint8_t *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}

/** Identical to synthMatrix in tests/test_pearl_prepare.cu. */
std::vector<int8_t> synthMatrix(size_t rows, size_t cols, int64_t salt) {
    std::vector<int8_t> m(rows * cols);
    for (size_t i = 0; i < m.size(); i++)
        m[i] = (int8_t)((int64_t)(((int64_t)i * 37 + salt) & 0x7F) - 64);
    return m;
}

struct Reader {
    const uint8_t *p, *end;
    bool ok = true;
    const uint8_t *take(size_t n) {
        if (p + n > end) { ok = false; return nullptr; }
        const uint8_t *r = p;
        p += n;
        return r;
    }
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *mapped = nullptr;
};

uint32_t findMemType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool readFile(const char *path, std::vector<uint8_t> *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize((size_t)sz);
    bool ok = fread(out->data(), 1, (size_t)sz, f) == (size_t)sz;
    fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <job_vectors.bin> <chunk.spv> <reduce.spv> "
                "<root.spv> [device_index]\n",
                argv[0]);
        return 2;
    }
    const char *vecPath = argv[1];
    const char *spvPaths[3] = {argv[2], argv[3], argv[4]};
    const int wantDevice = argc > 5 ? atoi(argv[5]) : 0;

    // ---- vectors: the same PRLJ0002 file the CUDA prepare test reads ----
    std::vector<uint8_t> vec;
    if (!readFile(vecPath, &vec)) {
        fprintf(stderr, "cannot open %s\n", vecPath);
        return 2;
    }
    Reader rd{vec.data(), vec.data() + vec.size()};
    const uint8_t *magic = rd.take(8);
    if (!magic || memcmp(magic, "PRLJ0002", 8) != 0) {
        fprintf(stderr, "bad vector magic\n");
        return 2;
    }
    const int32_t *dims = (const int32_t *)rd.take(16);
    if (!dims) { fprintf(stderr, "vector file truncated\n"); return 2; }
    const int m = dims[0], n = dims[1], k = dims[2];
    rd.take(76);                                  // header
    rd.take(MiningConfig::kSerializedSize);
    const uint8_t *wantKey = rd.take(32);
    const uint8_t *wantARoot = rd.take(32);
    const uint8_t *wantBtRoot = rd.take(32);
    if (!rd.ok) { fprintf(stderr, "vector file truncated\n"); return 2; }
    printf("vectors: m=%d n=%d k=%d  (job key %s)\n", m, n, k,
           hex(wantKey, 8).c_str());

    // ---- spirv ----
    std::vector<uint8_t> spvBytes[3];
    for (int i = 0; i < 3; i++)
        if (!om::loadSpirv(spvPaths[i], &spvBytes[i])) return 2;

    // ---- instance and device ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "test_pearl_merkle_vk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> pds(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, pds.data());
    if (wantDevice >= (int)ndev) {
        fprintf(stderr, "no device %d (%u present)\n", wantDevice, ndev);
        return 2;
    }
    VkPhysicalDevice pd = pds[wantDevice];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    // This gate had no claim guard at all: it ran a full blake3 tree over
    // 32768 chunks on whatever card it found, claimed or not.
    om::requireGpuClaim(props.deviceName);
    VkPhysicalDeviceSubgroupProperties sgp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sgp;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("device: %s  subgroup %u  sharedMem %u  storageRange %u\n",
           props.deviceName, sgp.subgroupSize,
           props.limits.maxComputeSharedMemorySize,
           props.limits.maxStorageBufferRange);

    // The reduction needs 16 KB of shared memory. The 4090 reports 49152 and
    // the 7900 XT 65536, so this is a wide margin - but it is the limit that
    // would bite first if anyone widened the tile, and NVIDIA is the tighter
    // of the two, which is not the intuitive way round.
    //
    // This is a FAILURE, not a skip, and the distinction is deliberate. 16384 is
    // the Vulkan spec MINIMUM for maxComputeSharedMemorySize, so a conformant
    // device cannot report less. A device that does is broken, not unsupported,
    // and returning 0 here would have been a silent pass on a path no real
    // device can reach - a gate that cannot fail dressed as a graceful skip.
    if (props.limits.maxComputeSharedMemorySize < 512u * 8u * 4u) {
        printf("  [FAIL] %s reports %u bytes of shared memory, below the Vulkan "
               "minimum of 16384\n",
               props.deviceName, props.limits.maxComputeSharedMemorySize);
        vkDestroyInstance(inst, nullptr);
        return 1;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfs.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 2; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    // No optional features requested at all. These shaders are plain uint
    // compute, so there is nothing to negotiate and nothing to fail on.
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    // SOAT_VK_DEVICE_LOCAL=1 puts the big data buffer in VRAM instead of
    // host-visible system memory. Off by default so the gate keeps behaving
    // exactly as it did when it was validated; on, it answers the question of
    // how much of the measured cost is PCIe rather than shader.
    const bool useDeviceLocal = getenv("SOAT_VK_DEVICE_LOCAL") != nullptr;

    auto mkBufEx = [&](VkDeviceSize size, Buf *out, bool deviceLocal) -> VkResult {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r = vkCreateBuffer(dev, &bci, nullptr, &out->buf);
        if (r != VK_SUCCESS) return r;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, out->buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        const VkMemoryPropertyFlags want =
            deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits, want);
        if (mai.memoryTypeIndex == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;
        r = vkAllocateMemory(dev, &mai, nullptr, &out->mem);
        if (r != VK_SUCCESS) return r;
        vkBindBufferMemory(dev, out->buf, out->mem, 0);
        if (deviceLocal) { out->mapped = nullptr; return VK_SUCCESS; }
        return vkMapMemory(dev, out->mem, 0, size, 0, &out->mapped);
    };
    auto mkBuf = [&](VkDeviceSize size, Buf *out) -> VkResult {
        return mkBufEx(size, out, false);
    };
    auto freeBuf = [&](Buf *b) {
        if (b->mapped) vkUnmapMemory(dev, b->mem);
        if (b->buf) vkDestroyBuffer(dev, b->buf, nullptr);
        if (b->mem) vkFreeMemory(dev, b->mem, nullptr);
        *b = Buf{};
    };

    // ---- one descriptor set layout serves all three shaders ----
    // They all take (input, key, output) at bindings 0..2. The push constant
    // range is declared once at 4 bytes; merkle_root simply does not use it.
    VkDescriptorSetLayoutBinding binds[3]{};
    for (int i = 0; i < 3; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkPipeline pipes[3];
    for (int i = 0; i < 3; i++) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spvBytes[i].size();
        smci.pCode = (const uint32_t *)spvBytes[i].data();
        VkShaderModule sm;
        VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkComputePipelineCreateInfo cpci{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = pl;
        VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                         &pipes[i]));
        vkDestroyShaderModule(dev, sm, nullptr);
    }
    enum { P_CHUNK = 0, P_REDUCE = 1, P_ROOT = 2 };

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 5};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 5;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VkDescriptorPool dp;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qfi;
    cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool;
    VKCHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cmdPool));

    // -----------------------------------------------------------------
    /**
     * The whole Merkle root on the device, mirroring merkleRoot() in
     * src/algos/pearl-pow/algo.cu line by line.
     *
     * THE PIPELINE BARRIER IS THE POINT OF THIS FUNCTION. The reduction
     * ping-pongs its two scratch buffers, so level L+1 writes the buffer level
     * L read. In CUDA every level is on one stream and ordering is free; here
     * nothing orders two dispatches unless a barrier says so, and because of
     * the ping-pong the hazard is write-after-read as well as read-after-write.
     * So the source access mask carries SHADER_READ, not just SHADER_WRITE.
     *
     * Getting this wrong reproduces the shared-memory race the GLSL is written
     * to avoid, one level up and equally invisible on a lightly loaded card.
     */
    auto runMerkle = [&](const uint8_t *data, uint32_t chunks,
                         const uint8_t *key, uint8_t out[32], int timedReps,
                         double *msOut) -> int {
        const VkDeviceSize dataBytes = (VkDeviceSize)chunks * 1024;
        const VkDeviceSize cvBytes = (VkDeviceSize)chunks * 32;

        Buf bData{}, bKey{}, bCv[2], bRoot{};
        bCv[0] = Buf{};
        bCv[1] = Buf{};
        if (mkBufEx(dataBytes, &bData, useDeviceLocal) != VK_SUCCESS) return -1;
        if (mkBuf(32, &bKey) != VK_SUCCESS) return -1;
        if (mkBuf(cvBytes, &bCv[0]) != VK_SUCCESS) return -1;
        if (mkBuf(cvBytes < 64 ? 64 : cvBytes, &bCv[1]) != VK_SUCCESS) return -1;
        if (mkBuf(32, &bRoot) != VK_SUCCESS) return -1;
        if (bData.mapped) {
            memcpy(bData.mapped, data, (size_t)dataBytes);
        } else {
            // Device-local: stage through a host-visible buffer once. This is
            // setup, not measured - the timing pass below submits an already
            // recorded command buffer.
            Buf stage{};
            if (mkBufEx(dataBytes, &stage, false) != VK_SUCCESS) return -1;
            memcpy(stage.mapped, data, (size_t)dataBytes);
            VkCommandBufferAllocateInfo scb{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            scb.commandPool = cmdPool;
            scb.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            scb.commandBufferCount = 1;
            VkCommandBuffer scmd;
            if (vkAllocateCommandBuffers(dev, &scb, &scmd) != VK_SUCCESS) return -1;
            VkCommandBufferBeginInfo sbi{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            sbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(scmd, &sbi);
            VkBufferCopy bc{0, 0, dataBytes};
            vkCmdCopyBuffer(scmd, stage.buf, bData.buf, 1, &bc);
            vkEndCommandBuffer(scmd);
            VkSubmitInfo ssi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            ssi.commandBufferCount = 1;
            ssi.pCommandBuffers = &scmd;
            vkQueueSubmit(queue, 1, &ssi, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            vkFreeCommandBuffers(dev, cmdPool, 1, &scmd);
            freeBuf(&stage);
        }
        memcpy(bKey.mapped, key, 32);
        memset(bCv[0].mapped, 0, (size_t)cvBytes);
        memset(bCv[1].mapped, 0, (size_t)(cvBytes < 64 ? 64 : cvBytes));
        memset(bRoot.mapped, 0, 32);

        // 5 sets: chunk, reduce(0->1), reduce(1->0), root(from 0), root(from 1)
        VkDescriptorSetLayout layouts[5] = {dsl, dsl, dsl, dsl, dsl};
        VkDescriptorSetAllocateInfo dsai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dp;
        dsai.descriptorSetCount = 5;
        dsai.pSetLayouts = layouts;
        VkDescriptorSet sets[5];
        if (vkAllocateDescriptorSets(dev, &dsai, sets) != VK_SUCCESS) return -1;

        auto wire = [&](VkDescriptorSet s, VkBuffer in, VkBuffer outb) {
            VkDescriptorBufferInfo dbi[3] = {{in, 0, VK_WHOLE_SIZE},
                                             {bKey.buf, 0, VK_WHOLE_SIZE},
                                             {outb, 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet w[3]{};
            for (int i = 0; i < 3; i++) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = s;
                w[i].dstBinding = (uint32_t)i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &dbi[i];
            }
            vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);
        };
        wire(sets[0], bData.buf, bCv[0].buf);
        wire(sets[1], bCv[0].buf, bCv[1].buf);
        wire(sets[2], bCv[1].buf, bCv[0].buf);
        wire(sets[3], bCv[0].buf, bRoot.buf);
        wire(sets[4], bCv[1].buf, bRoot.buf);

        VkCommandBufferAllocateInfo cbai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb;
        if (vkAllocateCommandBuffers(dev, &cbai, &cb) != VK_SUCCESS) return -1;
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        // NOT one-time-submit: the timing pass below resubmits this same
        // command buffer, which that flag would make invalid.
        vkBeginCommandBuffer(cb, &cbbi);

        // Read-after-write AND write-after-read, in both directions, because
        // the ping-pong makes each level's destination the next level's source
        // and vice versa.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        auto barrier = [&]() {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                                 0, nullptr, 0, nullptr);
        };

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[P_CHUNK]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                                &sets[0], 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &chunks);
        vkCmdDispatch(cb, (chunks + 255) / 256, 1, 1);
        barrier();

        uint32_t count = chunks;
        int src = 0;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[P_REDUCE]);
        while (count > 2) {
            // per == the CUDA's 2*threads, from
            //   threads = 256; while (threads>1 && 2*threads > count/2) threads >>= 1;
            // Checked against that expression at 4, 8, 1024 and 131072.
            uint32_t per = count / 2;
            if (per > 512) per = 512;
            const uint32_t groups = count / per;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                                    &sets[1 + src], 0, nullptr);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &per);
            vkCmdDispatch(cb, groups, 1, 1);
            barrier();
            count = groups;
            src ^= 1;
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[P_ROOT]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                                &sets[3 + src], 0, nullptr);
        vkCmdDispatch(cb, 1, 1, 1);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vkCreateFence(dev, &fci, nullptr, &fence);

        auto submitOnce = [&]() -> VkResult {
            vkResetFences(dev, 1, &fence);
            VkResult r = vkQueueSubmit(queue, 1, &si, fence);
            if (r != VK_SUCCESS) return r;
            return vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull);
        };

        VkResult wr = submitOnce();
        if (wr != VK_SUCCESS) {
            vkDestroyFence(dev, fence, nullptr);
            fprintf(stderr, "fence wait -> %d\n", (int)wr);
            return -1;
        }
        memcpy(out, bRoot.mapped, 32);

        if (timedReps > 0 && msOut) {
            // One warm-up submit already happened above, so the pipeline and
            // the memory are resident before the clock starts.
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < timedReps; i++)
                if (submitOnce() != VK_SUCCESS) {
                    vkDestroyFence(dev, fence, nullptr);
                    return -1;
                }
            const auto t1 = std::chrono::steady_clock::now();
            *msOut = std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     timedReps;
        }
        vkDestroyFence(dev, fence, nullptr);

        vkFreeCommandBuffers(dev, cmdPool, 1, &cb);
        vkFreeDescriptorSets(dev, dp, 5, sets);
        freeBuf(&bData); freeBuf(&bKey); freeBuf(&bCv[0]); freeBuf(&bCv[1]);
        freeBuf(&bRoot);
        return 0;
    };

    auto merkleRoot = [&](const uint8_t *data, uint32_t chunks,
                          const uint8_t *key, uint8_t out[32]) -> int {
        return runMerkle(data, chunks, key, out, 0, nullptr);
    };
    auto timeMerkleRoot = [&](const uint8_t *data, uint32_t chunks,
                              const uint8_t *key, uint8_t out[32], int reps,
                              double *msOut) -> int {
        return runMerkle(data, chunks, key, out, reps, msOut);
    };

    // -----------------------------------------------------------------
    printf("1. the A and B^t roots, against the CUDA vectors\n");
    {
        const std::vector<int8_t> A = synthMatrix(m, k, 11);
        const std::vector<int8_t> Bt = synthMatrix(n, k, 91);
        const size_t kChunkLenL = 1024;
        const size_t aPad = ((A.size() + kChunkLenL - 1) / kChunkLenL) * kChunkLenL;
        const size_t btPad = ((Bt.size() + kChunkLenL - 1) / kChunkLenL) * kChunkLenL;
        const uint32_t aChunks = (uint32_t)(aPad / kChunkLenL);
        const uint32_t btChunks = (uint32_t)(btPad / kChunkLenL);

        check("chunk counts are powers of two, as the reduction requires",
              (aChunks & (aChunks - 1)) == 0 && (btChunks & (btChunks - 1)) == 0,
              std::to_string(aChunks) + " and " + std::to_string(btChunks));

        std::vector<uint8_t> aPadded(aPad, 0), btPadded(btPad, 0);
        memcpy(aPadded.data(), A.data(), A.size());
        memcpy(btPadded.data(), Bt.data(), Bt.size());

        uint8_t gotA[32], gotB[32];
        if (merkleRoot(aPadded.data(), aChunks, wantKey, gotA) != 0) return 2;
        if (merkleRoot(btPadded.data(), btChunks, wantKey, gotB) != 0) return 2;
        check("A root matches the CUDA vector", memcmp(gotA, wantARoot, 32) == 0,
              hex(gotA, 32) + " vs " + hex(wantARoot, 32));
        check("B^t root matches the CUDA vector", memcmp(gotB, wantBtRoot, 32) == 0,
              hex(gotB, 32) + " vs " + hex(wantBtRoot, 32));
    }

    // -----------------------------------------------------------------
    printf("2. the tree reduction at sizes the vectors do not reach\n");
    {
        // Mirrors section 2b of tests/test_pearl_prepare.cu, including its
        // reason for existing: the vectors are 256 chunks, which the driver
        // handles in one reduce dispatch, and a reduction that is right for
        // one dispatch and wrong for two is exactly the bug this catches.
        //
        // 131072 is not optional. That is where the CUDA race showed: A at
        // 8192 chunks was right and B at 131072 was wrong on the same card,
        // because it scaled with block count.
        //
        // The data is generated with the HOST fillMatrix rather than the
        // genMatrix kernel, which keeps this test inside the three kernels it
        // is meant to gate. Same bytes either way - that equivalence is
        // already asserted by section 1 of the CUDA prepare test.
        static const uint32_t kChunkCounts[] = {2,    4,    8,     64,    1024,
                                                4096, 8192, 32768, 65536, 131072};
        for (size_t c = 0; c < sizeof(kChunkCounts) / sizeof(kChunkCounts[0]); c++) {
            const uint32_t chunks = kChunkCounts[c];
            const size_t bytes = (size_t)chunks * 1024;

            std::vector<uint8_t> data(bytes);
            fillMatrix((int8_t *)data.data(), bytes, matrixSeed(chunks, false));

            uint8_t want[32];
            b3::hash(wantKey, data.data(), data.size(), want);

            // The largest case is repeated, and it is worth being precise
            // about how little that buys, because the obvious arithmetic is
            // wrong. A deliberately racy merkle_reduce fails about 2 runs in 3
            // on the 7900 XT, so five independent runs should miss it 0.4% of
            // the time. Measured, five runs INSIDE ONE PROCESS miss it about
            // 25% of the time - one gate trial in four had all five pass.
            // Under independence that event should occur 0.016 times in four
            // trials, so the runs are plainly correlated within a process:
            // whatever the driver settles into, it settles into for the run.
            //
            // Net effect of this loop is 67% -> 75% detection. Nearly nothing.
            // WHAT ACTUALLY HELPS IS SEPARATE INVOCATIONS - run the whole
            // binary several times, which is a decision for the harness rather
            // than for this loop.
            //
            // The loop stays because it costs one 128 MB pass and does catch
            // the uncorrelated cases, but do not read a green "5 runs" as
            // proof that no race exists. It is a regression check on a shader
            // already proven correct, not a race detector.
            //
            // Only the top of the sweep is repeated because only the top of
            // the sweep ever failed - 65536 and below passed the broken shader
            // on every card.
            const int runs = (chunks == kChunkCounts[
                                  sizeof(kChunkCounts) / sizeof(kChunkCounts[0]) - 1])
                                 ? 5
                                 : 1;
            bool allOk = true;
            std::string firstBad;
            for (int r = 0; r < runs; r++) {
                uint8_t got[32];
                if (merkleRoot(data.data(), chunks, wantKey, got) != 0) return 2;
                if (memcmp(got, want, 32) != 0) {
                    allOk = false;
                    if (firstBad.empty())
                        firstBad = "run " + std::to_string(r + 1) + " gave " +
                                   hex(got, 32);
                }
            }

            char nm[128];
            if (runs > 1)
                snprintf(nm, sizeof(nm),
                         "%u-chunk root (%zu KB) matches the host, %d runs",
                         chunks, bytes / 1024, runs);
            else
                snprintf(nm, sizeof(nm), "%u-chunk root (%zu KB) matches the host",
                         chunks, bytes / 1024);
            check(nm, allOk,
                  allOk ? hex(want, 32) : firstBad + " vs " + hex(want, 32));
        }
    }

    // -----------------------------------------------------------------
    printf("3. the gate can actually fail\n");
    {
        // An all-clear from a check that cannot fail looks exactly like a
        // clean result. Two cheap proofs that this one discriminates:
        // a different key must give a different root, and so must a single
        // flipped bit deep inside the data.
        std::vector<uint8_t> data(4096 * 1024);
        fillMatrix((int8_t *)data.data(), data.size(), matrixSeed(4096, false));
        uint8_t base[32], other[32], flipped[32];
        if (merkleRoot(data.data(), 4096, wantKey, base) != 0) return 2;

        uint8_t altKey[32];
        memcpy(altKey, wantKey, 32);
        altKey[0] ^= 1;
        if (merkleRoot(data.data(), 4096, altKey, other) != 0) return 2;
        check("a one-bit key change changes the root",
              memcmp(base, other, 32) != 0, hex(other, 8));

        // Byte 3,000,000 sits in chunk 2929, deep enough that it only reaches
        // the root through several reduction levels.
        data[3000000] ^= 0x01;
        if (merkleRoot(data.data(), 4096, wantKey, flipped) != 0) return 2;
        check("a one-bit data change deep in the tree changes the root",
              memcmp(base, flipped, 32) != 0, hex(flipped, 8));
    }

    // -----------------------------------------------------------------
    printf("4. what it costs (a measurement, not a gate)  [data buffer: %s]\n",
           useDeviceLocal ? "DEVICE_LOCAL" : "HOST_VISIBLE");
    {
        // The only question left about these three kernels once they are
        // correct: is GLSL blake3 in the same ballpark as the CUDA? If it were
        // several times slower the prepare stage would dominate and the whole
        // Vulkan port would be pointless, so this is a feasibility number and
        // deliberately not an optimisation target - Pearl on AMD is a parity
        // feature and loses money on every AMD card at 11.4 c/kWh.
        //
        // CUDA's own figures, from the header of prepare.cuh at m=n=4096
        // k=2048 on a 4090: hash A 0.124 ms (8192 chunks), hash B^t 0.123 ms.
        //
        // Timed over the SUBMIT ONLY. Buffer creation, the 128 MB upload and
        // descriptor setup are excluded, because none of them recur per
        // attempt in the real miner - the matrices are already on the device.
        struct Shape { const char *name; uint32_t chunks; } shapes[] = {
            {"A   at m=4096 k=2048", 8192},
            {"B^t at n=32768 k=2048", 65536},
        };
        for (const Shape &sh : shapes) {
            std::vector<uint8_t> data((size_t)sh.chunks * 1024);
            fillMatrix((int8_t *)data.data(), data.size(), matrixSeed(7, false));
            uint8_t root[32];
            double ms = 0.0;
            if (timeMerkleRoot(data.data(), sh.chunks, wantKey, root, 20, &ms) != 0)
                return 2;
            printf("  %s  %6u chunks  %7.3f ms/root\n", sh.name, sh.chunks, ms);
        }
    }

    // A gate that runs nothing must not report success. wF:p2 lost a coopmat
    // capability check to exactly this: an instance-level function looked up
    // through vkGetInstanceProcAddr(VK_NULL_HANDLE, ...) returns null, so every
    // device would have reported "no coopmat" and every test would have
    // "passed" by skipping. A skip path needs something proving it does not
    // always skip, and this is that something.
    if (checksRun == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing. A gate that executed nothing "
               "is not a pass.\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, checksRun,
           failures);

    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    for (int i = 0; i < 3; i++) vkDestroyPipeline(dev, pipes[i], nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return failures ? 1 : 0;
}
