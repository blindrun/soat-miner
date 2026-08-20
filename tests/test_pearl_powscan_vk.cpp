// Pearl's powScan on Vulkan, against the host reference in job.h.
//
// The gate for src/algos/pearl-pow/powscan.comp. The host twin is
// om::pearl::powDigest, which is the same function algo.cu's recheck() and
// digestFromProof() use, so agreeing with it is agreeing with what a pool will
// recompute.
//
// usage: test_pearl_powscan_vk <powscan.spv> [device_index]
//
// Needs no vectors file: powScan's input is a transcript and its reference is a
// pure function of it, so the transcripts are synthesised here and the expected
// answer computed on the host. Deliberately needs no cooperative matrix, so it
// also runs on llvmpipe.
//
// WHAT THIS GATE IS ACTUALLY WATCHING, because three of the four are silent:
//
//   * The key is COMMITMENT A, not the job key. Keying wrong produces a shader
//     that runs at the right hit rate and finds nothing a pool accepts.
//   * Equality is a WIN. The limb walk runs 7 down to 0 and falling out of it
//     means every limb matched. Reversing that walk was mutation-tested on the
//     CUDA side at 1003 hits where 8 were right - a bug shaped like good luck.
//   * hitCount counts EVERY hit including ones dropped for want of space, so
//     the host can distinguish "8 hits" from "maxHits and an unknown number
//     lost". Section 4 tests that overflow case specifically.
//   * The digest is written in storeCv word order. A byte-order slip here
//     fails as a host-verification mismatch rather than a wrong answer.
//
// TRANSCRIPTS ARE SCRAMBLED, NOT PATTERNED. A low-period fill makes digests
// collide and the hit count meaningless - the same data-symmetry trap that
// made an ablated GEMM's XOR fold cancel to zero and report 403 TOPS. Warned
// about by the lane that hit it.

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/algos/pearl-pow/job.h"
#include "vk_claim_guard.h"

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

int failures = 0, checksRun = 0;

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

/**
 * Scrambled transcripts. splitmix64 per word rather than a counter or a low
 * period fill, so no two digests collide by construction of the input.
 */
uint64_t sm64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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
    if (argc < 2) {
        fprintf(stderr, "usage: %s <powscan.spv> [device_index]\n", argv[0]);
        return 2;
    }
    const int wantDevice = argc > 2 ? atoi(argv[2]) : 0;

    std::vector<uint8_t> spv;
    if (!readFile(argv[1], &spv)) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    // ---- the problem, built on the host first ----
    const uint32_t kCount = 4096;
    const uint32_t kWords = (uint32_t)kTranscriptWords;   // 16
    std::vector<uint32_t> transcripts((size_t)kCount * kWords);
    for (size_t i = 0; i < transcripts.size(); i++)
        transcripts[i] = (uint32_t)(sm64(0xC0FFEEull + i) >> 17);

    uint8_t commitA[32];
    for (int i = 0; i < 32; i++) commitA[i] = (uint8_t)(sm64(0xA11CEull + i) >> 24);

    std::vector<uint8_t> hostDigest((size_t)kCount * 32);
    for (uint32_t i = 0; i < kCount; i++)
        powDigest(&transcripts[(size_t)i * kWords], commitA, &hostDigest[(size_t)i * 32]);

    // Host-side win test, mirroring the shader: limb 7 down to 0, equality
    // continues, falling out of the loop is a win.
    auto limb = [&](uint32_t i, int j) {
        const uint8_t *d = &hostDigest[(size_t)i * 32 + (size_t)j * 4];
        return (uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16) |
               ((uint32_t)d[3] << 24);
    };
    auto hostWins = [&](uint32_t i, const uint32_t target[8]) {
        for (int j = 7; j >= 0; --j) {
            if (limb(i, j) < target[j]) return true;
            if (limb(i, j) > target[j]) return false;
        }
        return true;   // all limbs equal: the bound is <=
    };

    // ---- Vulkan ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "test_pearl_powscan_vk";
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

    // Inside the binary, before a device exists. Not a wrapper.
    om::requireGpuClaim(props.deviceName);

    printf("device: %s\n", props.deviceName);

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
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    auto mkBuf = [&](VkDeviceSize size, Buf *out, bool deviceLocal) -> VkResult {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
        if (deviceLocal) return VK_SUCCESS;
        return vkMapMemory(dev, out->mem, 0, size, 0, &out->mapped);
    };
    auto freeBuf = [&](Buf *b) {
        if (b->mapped) vkUnmapMemory(dev, b->mem);
        if (b->buf) vkDestroyBuffer(dev, b->buf, nullptr);
        if (b->mem) vkFreeMemory(dev, b->mem, nullptr);
        *b = Buf{};
    };

    VkDescriptorSetLayoutBinding binds[6]{};
    for (int i = 0; i < 6; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 6;
    dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size();
    smci.pCode = (const uint32_t *)spv.data();
    VkShaderModule sm;
    VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
    vkDestroyShaderModule(dev, sm, nullptr);

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    VkDescriptorPool dp;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qfi;
    cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool;
    VKCHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cmdPool));

    // The transcript buffer is DEVICE_LOCAL with a staged upload, by default
    // and not behind a flag. A host-visible default is what turned a Merkle
    // measurement into a 27x figure that was really PCIe.
    Buf bT{}, bKey{}, bTgt{}, bHI{}, bHD{}, bHC{};
    const VkDeviceSize tBytes = (VkDeviceSize)transcripts.size() * 4;
    if (mkBuf(tBytes, &bT, true) != VK_SUCCESS) { fprintf(stderr, "alloc\n"); return 2; }
    if (mkBuf(32, &bKey, false) != VK_SUCCESS) return 2;
    if (mkBuf(32, &bTgt, false) != VK_SUCCESS) return 2;

    {   // stage the transcripts into VRAM once
        Buf stage{};
        if (mkBuf(tBytes, &stage, false) != VK_SUCCESS) return 2;
        memcpy(stage.mapped, transcripts.data(), (size_t)tBytes);
        VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        a.commandPool = cmdPool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        a.commandBufferCount = 1;
        VkCommandBuffer c;
        vkAllocateCommandBuffers(dev, &a, &c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(c, &bi);
        VkBufferCopy bc{0, 0, tBytes};
        vkCmdCopyBuffer(c, stage.buf, bT.buf, 1, &bc);
        vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &c;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev, cmdPool, 1, &c);
        freeBuf(&stage);
    }
    memcpy(bKey.mapped, commitA, 32);

    /** Run powScan for one target and maxHits. Returns the raw hitCount. */
    auto run = [&](const uint32_t target[8], uint32_t maxHits,
                   std::vector<uint32_t> *idx, std::vector<uint32_t> *dig) -> uint32_t {
        freeBuf(&bHI); freeBuf(&bHD); freeBuf(&bHC);
        if (mkBuf((VkDeviceSize)maxHits * 4, &bHI, false) != VK_SUCCESS) return UINT32_MAX;
        if (mkBuf((VkDeviceSize)maxHits * 32, &bHD, false) != VK_SUCCESS) return UINT32_MAX;
        if (mkBuf(4, &bHC, false) != VK_SUCCESS) return UINT32_MAX;
        memcpy(bTgt.mapped, target, 32);
        memset(bHI.mapped, 0, (size_t)maxHits * 4);
        memset(bHD.mapped, 0, (size_t)maxHits * 32);
        *(uint32_t *)bHC.mapped = 0;   // the driver loop clears it, not the shader

        VkDescriptorBufferInfo dbi[6] = {
            {bT.buf, 0, VK_WHOLE_SIZE},   {bKey.buf, 0, VK_WHOLE_SIZE},
            {bTgt.buf, 0, VK_WHOLE_SIZE}, {bHI.buf, 0, VK_WHOLE_SIZE},
            {bHD.buf, 0, VK_WHOLE_SIZE},  {bHC.buf, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]{};
        for (int i = 0; i < 6; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds; w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, 6, w, 0, nullptr);

        VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        a.commandPool = cmdPool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        a.commandBufferCount = 1;
        VkCommandBuffer c;
        vkAllocateCommandBuffers(dev, &a, &c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(c, &bi);
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        const uint32_t push[2] = {kCount, maxHits};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(c, (kCount + 255) / 256, 1, 1);
        vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &c;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vkCreateFence(dev, &fci, nullptr, &fence);
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull);
        vkDestroyFence(dev, fence, nullptr);
        vkFreeCommandBuffers(dev, cmdPool, 1, &c);

        const uint32_t n = *(uint32_t *)bHC.mapped;
        const uint32_t kept = n < maxHits ? n : maxHits;
        idx->assign((uint32_t *)bHI.mapped, (uint32_t *)bHI.mapped + kept);
        dig->assign((uint32_t *)bHD.mapped, (uint32_t *)bHD.mapped + (size_t)kept * 8);
        return n;
    };

    std::vector<uint32_t> idx, dig;

    // -----------------------------------------------------------------
    printf("1. every digest matches the host reference\n");
    {
        // A target of all-ones lets everything through, so every transcript is
        // recorded and every digest is checked, not just the rare winners.
        uint32_t target[8];
        for (int j = 0; j < 8; j++) target[j] = 0xFFFFFFFFu;
        const uint32_t n = run(target, kCount, &idx, &dig);
        check("all 4096 transcripts win against an all-ones target",
              n == kCount, std::to_string(n));

        size_t bad = 0;
        std::vector<int> seen(kCount, 0);
        for (size_t s = 0; s < idx.size(); s++) {
            const uint32_t i = idx[s];
            if (i >= kCount) { bad++; continue; }
            seen[i]++;
            for (int j = 0; j < 8; j++)
                if (dig[s * 8 + j] != limb(i, j)) { bad++; break; }
        }
        check("every recorded digest is byte-identical to powDigest", bad == 0,
              std::to_string(bad) + " mismatched");

        size_t missing = 0;
        for (uint32_t i = 0; i < kCount; i++) if (seen[i] != 1) missing++;
        check("every transcript recorded exactly once", missing == 0,
              std::to_string(missing) + " wrong");
    }

    // -----------------------------------------------------------------
    printf("2. the target comparison, including the edges\n");
    {
        uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint32_t n = run(zero, 64, &idx, &dig);
        check("an all-zero target admits nothing", n == 0, std::to_string(n));

        // Equality is a win: use one transcript's own digest as the target.
        //
        // maxHits MUST be large enough to hold every winner here. The first
        // version of this check used 64, and with 1103 winners the one index
        // it was looking for simply had not won the atomic race into the
        // buffer - so it failed against a correct shader. Whether a specific
        // index is recorded is only a meaningful question when nothing is
        // dropped.
        const uint32_t pick = 1234;
        uint32_t eq[8];
        for (int j = 0; j < 8; j++) eq[j] = limb(pick, j);
        const uint32_t n2 = run(eq, kCount, &idx, &dig);
        bool found = false;
        for (uint32_t v : idx) if (v == pick) found = true;
        check("a digest exactly equal to the target WINS (the bound is <=)",
              found, "hits=" + std::to_string(n2) + " of " + std::to_string(kCount));
        check("the host agrees equality wins, so the check is not one-sided",
              hostWins(pick, eq), "");

        // A realistic target: whatever the host says, the device must agree
        // exactly, both in count and in membership.
        uint32_t tgt[8];
        for (int j = 0; j < 8; j++) tgt[j] = 0xFFFFFFFFu;
        tgt[7] = 0x04000000u;   // ~1 in 64
        uint32_t want = 0;
        for (uint32_t i = 0; i < kCount; i++) if (hostWins(i, tgt)) want++;
        const uint32_t got = run(tgt, kCount, &idx, &dig);
        check("hit count matches the host at a selective target",
              got == want, std::to_string(got) + " vs " + std::to_string(want));

        std::vector<int> mark(kCount, 0);
        for (uint32_t v : idx) if (v < kCount) mark[v] = 1;
        size_t wrong = 0;
        for (uint32_t i = 0; i < kCount; i++)
            if (mark[i] != (hostWins(i, tgt) ? 1 : 0)) wrong++;
        check("the same transcripts win on device as on host", wrong == 0,
              std::to_string(wrong) + " differ");
    }

    // -----------------------------------------------------------------
    printf("3. overflow: hitCount is the TRUE total, not the buffer size\n");
    {
        // The load-bearing case. maxHits is smaller than the number of hits,
        // so the counter must report how many there really were while only
        // maxHits are recorded. Clamping it would present a full buffer as the
        // total and the host would never know it had lost any.
        uint32_t target[8];
        for (int j = 0; j < 8; j++) target[j] = 0xFFFFFFFFu;
        const uint32_t maxHits = 100;
        const uint32_t n = run(target, maxHits, &idx, &dig);
        check("hitCount reports all 4096 hits though only 100 fit",
              n == kCount, std::to_string(n));
        check("exactly maxHits entries were written", idx.size() == maxHits,
              std::to_string(idx.size()));
        size_t bad = 0;
        for (size_t s = 0; s < idx.size(); s++) {
            const uint32_t i = idx[s];
            if (i >= kCount) { bad++; continue; }
            for (int j = 0; j < 8; j++)
                if (dig[s * 8 + j] != limb(i, j)) { bad++; break; }
        }
        check("the entries that did fit are still correct", bad == 0,
              std::to_string(bad) + " mismatched");
    }

    // -----------------------------------------------------------------
    printf("4. the gate can fail\n");
    {
        // Wrong key must change every digest. This is the check that would
        // catch keying on the job key instead of commitment A - the failure
        // that runs at the right hit rate and finds nothing a pool accepts.
        uint8_t altKey[32];
        memcpy(altKey, commitA, 32);
        altKey[0] ^= 1;
        std::vector<uint8_t> altDigest(32);
        powDigest(&transcripts[0], altKey, altDigest.data());
        check("a one-bit key change changes the host digest",
              memcmp(altDigest.data(), &hostDigest[0], 32) != 0,
              hex(altDigest.data(), 8));

        uint32_t target[8];
        for (int j = 0; j < 8; j++) target[j] = 0xFFFFFFFFu;
        memcpy(bKey.mapped, altKey, 32);
        run(target, 8, &idx, &dig);
        bool differs = false;
        for (int j = 0; j < 8; j++) if (dig[j] != limb(idx[0], j)) differs = true;
        check("the device disagrees with powDigest when keyed wrong", differs,
              "so section 1 was not vacuous");
        memcpy(bKey.mapped, commitA, 32);
    }

    if (checksRun == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing.\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, checksRun,
           failures);

    freeBuf(&bT); freeBuf(&bKey); freeBuf(&bTgt);
    freeBuf(&bHI); freeBuf(&bHD); freeBuf(&bHC);
    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return failures ? 1 : 0;
}
