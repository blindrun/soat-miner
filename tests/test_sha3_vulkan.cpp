// SHA3-256t on Vulkan, against the host reference and against real BC3
// mainnet blocks.
//
// This is the gate that decides whether the Vulkan BC3 shader is trusted at
// all, and it exists because of a specific failure this project has already
// had: NVIDIA's Vulkan compiler miscompiled the Autolykos kernel in this repo,
// the miner mined happily at a plausible hashrate, and every single share was
// silently rejected by the pool. Nothing in the miner noticed. A host-vs-device
// comparison is the only thing that catches that class of bug, because a wrong
// digest is indistinguishable from a right one until a pool disagrees.
//
// Two checks, and both are needed:
//
//   1. RAW SHADER vs sha3.h. Its own bare Vulkan pipeline drives the embedded
//      module in dump mode, which writes every digest for every nonce with no
//      atomic and no target test, so nothing can hide a disagreement behind a
//      filter. Compared bit for bit against om::s3::hash() on the host. This
//      catches a miscompiled or mistranscribed permutation.
//
//   2. THE REAL Algorithm vs real blocks. The same six mainnet vectors the
//      CUDA gate uses, driven through makeSha3_256tVK() exactly as the miner
//      drives it. The target is set to each block's own hash - the tightest
//      target that block satisfies - so a kernel that is subtly wrong cannot
//      pass by finding some other nonce under a loose target. This catches a
//      push-constant layout error, a wrong solution stride, a byte-swapped
//      comparison: bugs that live in algo_vk.cpp rather than in the shader.
//
// Check 1 alone would pass a backend whose plumbing is wrong. Check 2 alone
// would pass a shader and a host that were wrong in the same way only if the
// same mistake were made twice independently, which is exactly why the vectors
// are real chain data rather than self-generated.
//
// usage: test_sha3_vulkan [device_index]

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/algos/sha3-256t/sha3.h"
#include "../src/core/algo.h"
#include "../src/core/btc_job.h"
#include "../src/core/vk_common.h"
#include "sha3_vectors.h"

extern const uint32_t kSha3Spirv[];
extern const size_t kSha3SpirvWords;

namespace om {
Algorithm *makeSha3_256tVK(int deviceIndex);
}

namespace {

#define VKREQ(x)                                                          \
    do {                                                                  \
        VkResult r_ = (x);                                                \
        if (r_ != VK_SUCCESS) {                                           \
            fprintf(stderr, "%s:%d %s -> VkResult %d\n", __FILE__,        \
                    __LINE__, #x, (int)r_);                               \
            return false;                                                 \
        }                                                                 \
    } while (0)

bool hexToBytes(const char *hex, std::vector<uint8_t> *out) {
    const size_t n = strlen(hex);
    if (n % 2) return false;
    out->clear();
    for (size_t i = 0; i < n; i += 2) {
        char b[3] = {hex[i], hex[i + 1], 0};
        char *end = nullptr;
        const long v = strtol(b, &end, 16);
        if (end != b + 2) return false;
        out->push_back((uint8_t)v);
    }
    return true;
}

// --------------------------------------------------------------------------
// Check 1: a bare pipeline over the embedded module, in dump mode.
//
// Deliberately does not reuse algo_vk.cpp's Vulkan setup. That file keeps its
// boilerplate private, and more importantly a harness that shared its
// push-constant struct could not detect a bug in it.
// --------------------------------------------------------------------------

struct Push {
    uint64_t h[10];
    uint64_t t[4];
    uint32_t base, count, mode;
};

class DumpHarness {
   public:
    static const uint32_t kLocal = 256;

    bool init(int deviceIndex, uint32_t maxNonces) {
        maxNonces_ = maxNonces;

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "test_sha3_vulkan";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VKREQ(vkCreateInstance(&ici, nullptr, &inst_));

        if (!om::vkPickPhysicalDevice(inst_, deviceIndex, &phys_)) return false;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        name_ = props.deviceName;

        VkPhysicalDeviceFeatures have{};
        vkGetPhysicalDeviceFeatures(phys_, &have);
        if (!have.shaderInt64) {
            fprintf(stderr, "%s has no shaderInt64; SHA3-256t needs it\n",
                    name_.c_str());
            return false;
        }

        vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qs.data());
        qf_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf_ = i; break; }
        if (qf_ == UINT32_MAX) return false;

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures feats{};
        feats.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &feats;
        VKREQ(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, qf_, 0, &queue_);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = kSha3SpirvWords * sizeof(uint32_t);
        smci.pCode = kSha3Spirv;
        VKREQ(vkCreateShaderModule(dev_, &smci, nullptr, &shader_));

        VkDescriptorSetLayoutBinding binds[2]{};
        for (uint32_t i = 0; i < 2; i++) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 2;
        dslci.pBindings = binds;
        VKREQ(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &setLayout_));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(Push);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VKREQ(vkCreatePipelineLayout(dev_, &plci, nullptr, &pipeLayout_));

        const uint32_t specData[2] = {kLocal, 256u};
        VkSpecializationMapEntry se[2]{};
        se[0].constantID = 0; se[0].offset = 0; se[0].size = 4;
        se[1].constantID = 1; se[1].offset = 4; se[1].size = 4;
        VkSpecializationInfo spec{};
        spec.mapEntryCount = 2;
        spec.pMapEntries = se;
        spec.dataSize = sizeof(specData);
        spec.pData = specData;

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader_;
        stage.pName = "main";
        stage.pSpecializationInfo = &spec;
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pipeLayout_;
        VKREQ(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                       &pipeline_));

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VKREQ(vkCreateDescriptorPool(dev_, &dpci, nullptr, &descPool_));
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &setLayout_;
        VKREQ(vkAllocateDescriptorSets(dev_, &dsai, &descSet_));

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = qf_;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VKREQ(vkCreateCommandPool(dev_, &cpi, nullptr, &cmdPool_));
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKREQ(vkAllocateCommandBuffers(dev_, &cbai, &cmd_));
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VKREQ(vkCreateFence(dev_, &fci, nullptr, &fence_));

        if (!makeBuffer((VkDeviceSize)maxNonces_ * 4 * sizeof(uint64_t), &out_,
                        &outMem_, &outMapped_))
            return false;
        if (!makeBuffer(sizeof(uint32_t), &cnt_, &cntMem_, &cntMapped_))
            return false;

        VkDescriptorBufferInfo bi[2]{};
        bi[0].buffer = out_; bi[0].range = VK_WHOLE_SIZE;
        bi[1].buffer = cnt_; bi[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[2]{};
        for (uint32_t i = 0; i < 2; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = descSet_;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev_, 2, w, 0, nullptr);
        return true;
    }

    const std::string &deviceName() const { return name_; }

    /** Hashes `count` nonces from `base` and returns every digest, 4 limbs
     *  each, in nonce order. */
    bool dump(const om::s3::Header &hdr, uint32_t base, uint32_t count,
              const uint64_t **digests) {
        if (count > maxNonces_) return false;
        Push p{};
        for (int i = 0; i < 10; i++) p.h[i] = hdr.lane[i];
        p.base = base;
        p.count = count;
        p.mode = 1;  // dump

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKREQ(vkBeginCommandBuffer(cmd_, &bi));
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cmd_, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(Push), &p);
        vkCmdDispatch(cmd_, (count + kLocal - 1) / kLocal, 1, 1);
        VKREQ(vkEndCommandBuffer(cmd_));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VKREQ(vkResetFences(dev_, 1, &fence_));
        VKREQ(vkQueueSubmit(queue_, 1, &si, fence_));
        VKREQ(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        *digests = (const uint64_t *)outMapped_;
        return true;
    }

    ~DumpHarness() {
        if (dev_) {
            vkDeviceWaitIdle(dev_);
            if (outMapped_) vkUnmapMemory(dev_, outMem_);
            if (cntMapped_) vkUnmapMemory(dev_, cntMem_);
            if (out_) vkDestroyBuffer(dev_, out_, nullptr);
            if (outMem_) vkFreeMemory(dev_, outMem_, nullptr);
            if (cnt_) vkDestroyBuffer(dev_, cnt_, nullptr);
            if (cntMem_) vkFreeMemory(dev_, cntMem_, nullptr);
            if (fence_) vkDestroyFence(dev_, fence_, nullptr);
            if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
            if (descPool_) vkDestroyDescriptorPool(dev_, descPool_, nullptr);
            if (pipeline_) vkDestroyPipeline(dev_, pipeline_, nullptr);
            if (pipeLayout_) vkDestroyPipelineLayout(dev_, pipeLayout_, nullptr);
            if (setLayout_) vkDestroyDescriptorSetLayout(dev_, setLayout_, nullptr);
            if (shader_) vkDestroyShaderModule(dev_, shader_, nullptr);
            vkDestroyDevice(dev_, nullptr);
        }
        if (inst_) vkDestroyInstance(inst_, nullptr);
    }

   private:
    bool makeBuffer(VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem,
                    void **mapped) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKREQ(vkCreateBuffer(dev_, &bci, nullptr, buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, *buf, &req);
        uint32_t type = UINT32_MAX;
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) &&
                (memProps_.memoryTypes[i].propertyFlags & want) == want) {
                type = i;
                break;
            }
        if (type == UINT32_MAX) {
            fprintf(stderr, "no host-visible memory type\n");
            return false;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        VKREQ(vkAllocateMemory(dev_, &mai, nullptr, mem));
        VKREQ(vkBindBufferMemory(dev_, *buf, *mem, 0));
        VKREQ(vkMapMemory(dev_, *mem, 0, size, 0, mapped));
        return true;
    }

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qf_ = UINT32_MAX;
    VkShaderModule shader_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};
    VkBuffer out_ = VK_NULL_HANDLE, cnt_ = VK_NULL_HANDLE;
    VkDeviceMemory outMem_ = VK_NULL_HANDLE, cntMem_ = VK_NULL_HANDLE;
    void *outMapped_ = nullptr, *cntMapped_ = nullptr;
    uint32_t maxNonces_ = 0;
    std::string name_;
};

int runDumpComparison(int deviceIndex) {
    // Enough nonces per header to cross many workgroups and every subgroup
    // lane, small enough that the host side stays a few seconds.
    const uint32_t kSweep = 1u << 18;  // 262,144

    DumpHarness h;
    if (!h.init(deviceIndex, kSweep)) {
        printf("FAIL: could not set up the dump harness\n");
        return 1;
    }
    printf("device: %s\n", h.deviceName().c_str());

    int failures = 0;
    uint64_t compared = 0;

    for (int vi = 0; vi < kSha3VectorCount; vi++) {
        const Sha3Vector &v = kSha3Vectors[vi];
        std::vector<uint8_t> hdrBytes;
        if (!hexToBytes(v.header, &hdrBytes) || hdrBytes.size() != 80) {
            printf("FAIL %s: bad vector\n", v.name);
            failures++;
            continue;
        }
        om::s3::Header hdr;
        om::s3::loadHeader(hdrBytes.data(), &hdr);

        const uint32_t winner =
            (uint32_t)hdrBytes[76] | ((uint32_t)hdrBytes[77] << 8) |
            ((uint32_t)hdrBytes[78] << 16) | ((uint32_t)hdrBytes[79] << 24);

        // Three windows per header: from zero, straddling the real winning
        // nonce, and butted against the top of the 32-bit space so the wrap
        // boundary is covered rather than assumed.
        const uint32_t bases[3] = {0u, winner - kSweep / 2,
                                   0xffffffffu - kSweep + 1};

        for (int bi = 0; bi < 3; bi++) {
            const uint32_t base = bases[bi];
            const uint64_t *dev = nullptr;
            if (!h.dump(hdr, base, kSweep, &dev)) {
                printf("FAIL %s: dispatch failed at base %08x\n", v.name, base);
                failures++;
                break;
            }
            uint32_t bad = 0;
            for (uint32_t i = 0; i < kSweep; i++) {
                uint64_t want[4];
                om::s3::hash(hdr, base + i, want);
                if (memcmp(want, dev + (size_t)i * 4, 32) != 0) {
                    if (bad == 0) {
                        printf("FAIL %s: nonce %08x host/device disagree\n"
                               "     host   %016llx %016llx %016llx %016llx\n"
                               "     device %016llx %016llx %016llx %016llx\n",
                               v.name, base + i,
                               (unsigned long long)want[0],
                               (unsigned long long)want[1],
                               (unsigned long long)want[2],
                               (unsigned long long)want[3],
                               (unsigned long long)dev[(size_t)i * 4 + 0],
                               (unsigned long long)dev[(size_t)i * 4 + 1],
                               (unsigned long long)dev[(size_t)i * 4 + 2],
                               (unsigned long long)dev[(size_t)i * 4 + 3]);
                    }
                    bad++;
                }
            }
            compared += kSweep;
            if (bad) {
                printf("FAIL %s: %u of %u digests wrong from base %08x\n",
                       v.name, bad, kSweep, base);
                failures++;
            }
        }
        if (!failures)
            printf("ok   %s: %u digests match sha3.h across 3 nonce windows\n",
                   v.name, kSweep * 3);
    }

    // Prove the comparison can actually fail. A check that has never been seen
    // to reject anything is not evidence of agreement - the Pearl work in this
    // repo shipped a test that reported PASS against an empty fixture.
    {
        om::s3::Header hdr{};
        uint64_t want[4];
        om::s3::hash(hdr, 0, want);
        uint64_t tampered[4];
        memcpy(tampered, want, 32);
        tampered[0] ^= 1;
        if (memcmp(want, tampered, 32) == 0) {
            printf("FAIL: the comparison itself does not discriminate\n");
            failures++;
        } else {
            printf("ok   negative control: a one-bit change is detected\n");
        }
    }

    printf("%llu host/device digest comparisons, %d failure(s)\n",
           (unsigned long long)compared, failures);
    return failures;
}

// --------------------------------------------------------------------------
// Check 2: the real Algorithm against real mainnet blocks. Mirrors
// tests/test_sha3_algo.cu, which does the same to the CUDA backend.
// --------------------------------------------------------------------------

int runVectorGate(int deviceIndex) {
    om::Algorithm *algo = om::makeSha3_256tVK(deviceIndex);
    if (!algo) {
        printf("FAIL: could not construct the Vulkan sha3-256t backend\n");
        return 1;
    }
    int failures = 0;

    for (int vi = 0; vi < kSha3VectorCount; vi++) {
        const Sha3Vector &v = kSha3Vectors[vi];
        std::vector<uint8_t> hdr, hit;
        if (!hexToBytes(v.header, &hdr) || hdr.size() != 80 ||
            !hexToBytes(v.hitLE, &hit) || hit.size() != 32) {
            printf("FAIL %s: bad vector\n", v.name);
            failures++;
            continue;
        }

        const uint32_t winner = (uint32_t)hdr[76] | ((uint32_t)hdr[77] << 8) |
                                ((uint32_t)hdr[78] << 16) |
                                ((uint32_t)hdr[79] << 24);

        om::Job job;
        // The job's header must carry a zero nonce: the whole point is that
        // the search supplies it. Leaving the winner in place would still pass
        // on a kernel that ignored `base` entirely.
        uint8_t tmpl[80];
        memcpy(tmpl, hdr.data(), 80);
        memset(tmpl + 76, 0, 4);
        job.extra = om::encodeBtcJobExtra(tmpl, std::string(8, '\0'));
        memcpy(job.msg, tmpl + 36, 32);
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            for (int b = 7; b >= 0; b--) limb = (limb << 8) | hit[i * 8 + b];
            job.target[i] = limb;
        }
        job.epoch = 0;
        job.valid = true;

        if (!algo->prepare(job)) {
            printf("FAIL %s: prepare\n", v.name);
            failures++;
            continue;
        }

        // A window that starts below the winner, so finding it proves the base
        // offset is applied rather than the thread index being used raw.
        const uint32_t base = winner - 4096;
        std::vector<om::Solution> sols;
        if (!algo->search(job, base, 8192, &sols)) {
            printf("FAIL %s: search returned false\n", v.name);
            failures++;
            continue;
        }

        bool found = false;
        for (const om::Solution &s : sols) {
            if ((uint32_t)s.nonce != winner) continue;
            found = true;
            if (memcmp(s.hit, job.target, 32) != 0) {
                printf("FAIL %s: hit does not match the block's hash\n", v.name);
                failures++;
            } else if (!algo->verify(job, s)) {
                printf("FAIL %s: verify() rejected the real winning nonce\n",
                       v.name);
                failures++;
            } else {
                printf("ok   %s: found nonce %08x, hit matches, verify passed\n",
                       v.name, winner);
            }
            break;
        }
        if (!found) {
            printf("FAIL %s: winning nonce %08x not found in %zu solutions\n",
                   v.name, winner, sols.size());
            failures++;
            continue;
        }

        // A tampered hit must not survive verification: this is the guard that
        // stops an unstable GPU's wrong answer reaching the pool.
        om::Solution bad;
        bad.nonce = winner;
        memcpy(bad.hit, job.target, 32);
        bad.hit[0] ^= 1;
        if (algo->verify(job, bad)) {
            printf("FAIL %s: verify() accepted a corrupted hit\n", v.name);
            failures++;
        } else {
            printf("ok   %s: verify() rejects a corrupted hit\n", v.name);
        }

        // A nonce that did not win must not verify either.
        om::Solution wrong;
        wrong.nonce = winner + 1;
        memcpy(wrong.hit, job.target, 32);
        if (algo->verify(job, wrong)) {
            printf("FAIL %s: verify() accepted the wrong nonce\n", v.name);
            failures++;
        } else {
            printf("ok   %s: verify() rejects the wrong nonce\n", v.name);
        }
    }

    // A batch larger than one dispatch slice must still find a winner, so the
    // multi-dispatch split in search() is covered rather than assumed. 5 M
    // nonces crosses the 4,194,304 boundary.
    {
        const Sha3Vector &v = kSha3Vectors[0];
        std::vector<uint8_t> hdr, hit;
        hexToBytes(v.header, &hdr);
        hexToBytes(v.hitLE, &hit);
        const uint32_t winner = (uint32_t)hdr[76] | ((uint32_t)hdr[77] << 8) |
                                ((uint32_t)hdr[78] << 16) |
                                ((uint32_t)hdr[79] << 24);
        uint8_t tmpl[80];
        memcpy(tmpl, hdr.data(), 80);
        memset(tmpl + 76, 0, 4);
        om::Job job;
        job.extra = om::encodeBtcJobExtra(tmpl, std::string(8, '\0'));
        memcpy(job.msg, tmpl + 36, 32);
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            for (int b = 7; b >= 0; b--) limb = (limb << 8) | hit[i * 8 + b];
            job.target[i] = limb;
        }
        job.valid = true;
        // Start far enough below the winner that the winner lands in the
        // second dispatch slice, not the first.
        const uint64_t base = (uint64_t)winner - 4500000ULL;
        std::vector<om::Solution> sols;
        if (base > winner) {
            printf("skip  multi-dispatch: vector's nonce is too low to test\n");
        } else if (!algo->search(job, base, 5000000, &sols)) {
            printf("FAIL multi-dispatch: search returned false\n");
            failures++;
        } else {
            bool ok = false;
            for (const om::Solution &s : sols)
                if ((uint32_t)s.nonce == winner) ok = true;
            if (ok) {
                printf("ok   multi-dispatch: winner found in slice 2 of a "
                       "5,000,000-nonce batch\n");
            } else {
                printf("FAIL multi-dispatch: winner lost when the batch was "
                       "split across dispatches\n");
                failures++;
            }
        }
    }

    algo->release();
    delete algo;
    return failures;
}

}  // namespace

int main(int argc, char **argv) {
    const int deviceIndex = (argc > 1) ? atoi(argv[1]) : -1;

    printf("=== sha3-256t Vulkan: raw shader vs the host reference ===\n");
    const int a = runDumpComparison(deviceIndex);

    printf("\n=== sha3-256t Vulkan: the Algorithm vs real BC3 mainnet blocks ===\n");
    const int b = runVectorGate(deviceIndex);

    const int failures = a + b;
    if (failures) {
        printf("\n%d FAILURE(S) - the Vulkan sha3-256t backend is NOT correct\n",
               failures);
        return 1;
    }
    printf("\nall sha3-256t Vulkan checks passed\n");
    return 0;
}
