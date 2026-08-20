// SHA3-256t (BitcoinIII / BC3) - Vulkan compute backend (AMD, NVIDIA, Intel).
//
// The AMD half of the ship gate: an algorithm is not shippable while it is
// CUDA-only, however well it works on NVIDIA.
//
// Much smaller than the Autolykos backend for one structural reason: there is
// no dataset. prepare() has nothing to build, so there is no chunking, no
// build slices, no TDR watchdog problem and no dedicated-allocation trick -
// all of that exists in algo_vk.cpp next door to move 7.27 GB around. Here the
// entire input is 80 bytes of header travelling in push constants, and the
// only device memory is a 256-slot solution buffer and a counter.
//
// verify() runs on the HOST, not on the device, and that is deliberate. It is
// the same call the CUDA backend makes: three Keccak permutations cost nothing
// on a CPU, and asking the same possibly-unstable GPU to confirm its own
// answer is how a wrong hit gets confirmed twice and submitted to a pool. It
// also means the shader and sha3.h are cross-checked on every single share the
// miner ever finds, not only in the test suite.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/algo.h"
#include "../../core/btc_job.h"
#include "../../core/vk_common.h"
#include "sha3.h"

extern const uint32_t kSha3Spirv[];
extern const size_t kSha3SpirvWords;

namespace om {
namespace {

/**
 * Must match the push_constant block in kernel.comp exactly.
 *
 * Checked against the compiled module rather than assumed: glslangValidator
 * lays these out at offsets 0,8..72 (header lanes), 80..104 (target limbs),
 * 112/116/120 (base/count/mode), which is what this struct produces on every
 * ABI the miner builds for.
 */
struct Push {
    uint64_t h[10];   ///< the 80-byte header as ten little-endian lanes
    uint64_t t[4];    ///< 256-bit target, t[0] least significant
    uint32_t base;    ///< first nonce in this batch
    uint32_t count;   ///< nonces in this batch
    uint32_t mode;    ///< 0 = search, 1 = dump every digest (tests only)
};
static_assert(sizeof(Push) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(offsetof(Push, t) == 80, "target limbs must land at offset 80");
static_assert(offsetof(Push, base) == 112, "base must land at offset 112");

#define VKCHECK(x)                                                     \
    do {                                                               \
        VkResult _r = (x);                                             \
        if (_r != VK_SUCCESS) {                                        \
            fprintf(stderr, "vulkan error %d at %s:%d\n", (int)_r,     \
                    __FILE__, __LINE__);                               \
            return false;                                              \
        }                                                              \
    } while (0)

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;
};

/** Workgroup size, fed to the shader as specialization constant 0 so the host
 *  dispatch and the shader cannot drift apart. */
const uint32_t kLocalSize = 256;

/** Solution buffer capacity, specialization constant 1. Same 256 as the CUDA
 *  path and for the same reason: a deliberately low test or pool difficulty
 *  can find far more than a handful in one batch. */
const uint32_t kMaxSolutions = 256;

/**
 * Nonces per dispatch.
 *
 * BC3's nonce is 32 bits, so a whole batch is bounded anyway, but a single
 * dispatch still has to stay inside Windows' 2-second GPU watchdog (TDR),
 * which resets the device rather than waiting. Autolykos hit this on its
 * dataset build; the same ceiling applies to any long-running dispatch, and a
 * 4 M-nonce batch on a slow card is well inside it while a full 2^32 sweep
 * from --bench would not be.
 */
const uint32_t kMaxPerDispatch = 1u << 22;

class Sha3_256tVK : public Algorithm {
   public:
    const char *name() const override { return "sha3-256t"; }

    size_t memoryBytes(const Job &) const override {
        return sizeof(uint64_t) * 5 * kMaxSolutions + sizeof(uint32_t);
    }

    // ------------------------------------------------------------- setup --
    bool init(int requestedIndex) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "soat-miner";
        app.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VKCHECK(vkCreateInstance(&ici, nullptr, &inst_));

        if (!vkPickPhysicalDevice(inst_, requestedIndex, &phys_)) return false;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        snprintf(devName_, sizeof(devName_), "%s", props.deviceName);
        snprintf(driver_, sizeof(driver_), "Vulkan %u.%u.%u",
                 VK_VERSION_MAJOR(props.apiVersion),
                 VK_VERSION_MINOR(props.apiVersion),
                 VK_VERSION_PATCH(props.apiVersion));

        // Keccak-f[1600] is 25 lanes of 64-bit rotates and XORs, so 64-bit
        // integers are not an optimisation here, they are the algorithm.
        // shaderInt64 is true on every card this miner targets - RTX 4090, RX
        // 7900 XT (RDNA3) and RX 6700 XT (RDNA2) were all checked - so there is
        // no lane-pair fallback to fall back to. Refuse honestly rather than
        // create the device and produce wrong digests.
        VkPhysicalDeviceFeatures have{};
        vkGetPhysicalDeviceFeatures(phys_, &have);
        if (!have.shaderInt64) {
            fprintf(stderr,
                    "%s does not support 64-bit shader integers (shaderInt64), "
                    "which SHA3-256t requires.\n"
                    "  Use the CUDA build on this card, or an AMD/Intel GPU "
                    "with a current Vulkan driver.\n",
                    devName_);
            return false;
        }

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        memProps_ = mp;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                deviceLocalBytes_ =
                    std::max(deviceLocalBytes_, mp.memoryHeaps[h].size);

        // --- queue ---------------------------------------------------------
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, qs.data());
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < qcount; i++)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamily_ = i;
                break;
            }
        if (queueFamily_ == UINT32_MAX) {
            fprintf(stderr, "no compute queue family\n");
            return false;
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkPhysicalDeviceFeatures feats{};
        feats.shaderInt64 = VK_TRUE;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &feats;
        VKCHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        // --- pipeline ------------------------------------------------------
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = kSha3SpirvWords * sizeof(uint32_t);
        smci.pCode = kSha3Spirv;
        VKCHECK(vkCreateShaderModule(dev_, &smci, nullptr, &shader_));

        VkDescriptorSetLayoutBinding binds[kBindings]{};
        for (uint32_t i = 0; i < kBindings; i++) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = kBindings;
        dslci.pBindings = binds;
        VKCHECK(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &setLayout_));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(Push);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VKCHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &pipeLayout_));

        const uint32_t specData[2] = {kLocalSize, kMaxSolutions};
        VkSpecializationMapEntry specEntries[2]{};
        specEntries[0].constantID = 0;
        specEntries[0].offset = 0;
        specEntries[0].size = sizeof(uint32_t);
        specEntries[1].constantID = 1;
        specEntries[1].offset = sizeof(uint32_t);
        specEntries[1].size = sizeof(uint32_t);
        VkSpecializationInfo spec{};
        spec.mapEntryCount = 2;
        spec.pMapEntries = specEntries;
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
        VKCHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                         &pipeline_));

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kBindings;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VKCHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &descPool_));

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &setLayout_;
        VKCHECK(vkAllocateDescriptorSets(dev_, &dsai, &descSet_));

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queueFamily_;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VKCHECK(vkCreateCommandPool(dev_, &cpi, nullptr, &cmdPool_));

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(dev_, &cbai, &cmd_));

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VKCHECK(vkCreateFence(dev_, &fci, nullptr, &fence_));

        // Both buffers are tiny and read back every batch, so both stay
        // host-visible. There is nothing device-local worth allocating here.
        if (!createBuffer(sizeof(uint64_t) * 5 * kMaxSolutions, &out_))
            return false;
        if (!createBuffer(sizeof(uint32_t), &cnt_)) return false;
        writeDescriptors();
        return true;
    }

    const char *deviceName() const { return devName_; }
    double deviceMemGB() const { return deviceLocalBytes_ / 1e9; }
    const char *driverVersion() const { return driver_; }

    /** Nothing to precompute: SHA3-256t has no dataset, no table and no epoch
     *  state. Every epoch-change path in the run loop collapses to nothing. */
    bool prepare(const Job &) override { return out_.buf != VK_NULL_HANDLE; }

    // ------------------------------------------------------------ search --
    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        s3::Header hdr;
        if (!loadJobHeader(job, &hdr)) {
            fprintf(stderr, "[sha3-256t] job carries no usable 80-byte header\n");
            return false;
        }
        if (out_.buf == VK_NULL_HANDLE) return false;

        // The run loop clamps to the 32-bit subspace (nonceBitsOwned() == 32);
        // this only guards --bench, which owns the whole space and passes
        // whatever it likes.
        uint64_t n = count;
        if (n > (1ULL << 32)) n = 1ULL << 32;

        *(uint32_t *)cnt_.mapped = 0;

        Push p{};
        for (int i = 0; i < 10; i++) p.h[i] = hdr.lane[i];
        for (int i = 0; i < 4; i++) p.t[i] = job.target[i];
        p.mode = kModeSearch;

        // Split into watchdog-sized dispatches. The counter is not reset
        // between them, so solutions from every slice accumulate in one buffer
        // exactly as a single dispatch would have left them.
        for (uint64_t done = 0; done < n; done += kMaxPerDispatch) {
            const uint32_t slice =
                (uint32_t)std::min<uint64_t>(kMaxPerDispatch, n - done);
            p.base = (uint32_t)(nonceBase + done);
            p.count = slice;
            if (!dispatch(p, (slice + kLocalSize - 1) / kLocalSize)) return false;
        }

        uint32_t found = *(uint32_t *)cnt_.mapped;
        if (found == 0) return true;
        if (found > kMaxSolutions) {
            fprintf(stderr,
                    "[sha3-256t] warning: %u solutions in one batch exceeds the "
                    "%u-solution buffer; %u dropped. Difficulty may be too low.\n",
                    found, kMaxSolutions, found - kMaxSolutions);
            found = kMaxSolutions;
        }

        const uint64_t *raw = (const uint64_t *)out_.mapped;
        for (uint32_t i = 0; i < found; i++) {
            Solution s;
            s.nonce = raw[i * 5];
            for (int l = 0; l < 4; l++) s.hit[l] = raw[i * 5 + 1 + l];
            out->push_back(s);
        }
        return true;
    }

    /**
     * Re-hash on the host, not on the device - see the file header. This is the
     * same independent check the CUDA backend makes, and on this backend it
     * carries a second job: it is a permanent host-vs-device comparison of the
     * GLSL transcription against sha3.h, running on every share the miner finds.
     */
    bool verify(const Job &job, const Solution &sol) const override {
        s3::Header hdr;
        if (!loadJobHeader(job, &hdr)) return false;
        uint64_t hit[4];
        s3::hash(hdr, (uint32_t)sol.nonce, hit);
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        return s3::underTarget(hit, job.target);
    }

    void release() override {}

    ~Sha3_256tVK() override {
        if (dev_) vkDeviceWaitIdle(dev_);
        destroyBuffer(&out_);
        destroyBuffer(&cnt_);
        if (fence_) vkDestroyFence(dev_, fence_, nullptr);
        if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        if (descPool_) vkDestroyDescriptorPool(dev_, descPool_, nullptr);
        if (pipeline_) vkDestroyPipeline(dev_, pipeline_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(dev_, pipeLayout_, nullptr);
        if (setLayout_) vkDestroyDescriptorSetLayout(dev_, setLayout_, nullptr);
        if (shader_) vkDestroyShaderModule(dev_, shader_, nullptr);
        if (dev_) vkDestroyDevice(dev_, nullptr);
        if (inst_) vkDestroyInstance(inst_, nullptr);
    }

   private:
    static const uint32_t kBindings = 2;
    static const uint32_t kModeSearch = 0;

    /**
     * The 80-byte header the job was built around. Identical to the CUDA
     * backend's, deliberately: both must agree on what a benchmark job means.
     *
     * --bench is the one case where a job arrives without a header, because
     * run.cpp fills in msg, epoch and a zero target and nothing else - Ergo
     * needs nothing more. Synthesising one here is safe precisely because bench
     * sets an all-zero target, so no synthetic header can produce a reportable
     * share, and Keccak's cost does not depend on its input so the rate is
     * identical either way.
     */
    static bool loadJobHeader(const Job &job, s3::Header *hdr) {
        uint8_t bytes[kBtcHeaderBytes];
        if (btcJobHeader(job.extra, bytes)) {
            s3::loadHeader(bytes, hdr);
            return true;
        }
        if (!job.extra.empty()) return false;
        memset(bytes, 0, sizeof(bytes));
        bytes[0] = 0x00; bytes[1] = 0x10; bytes[2] = 0x00; bytes[3] = 0x20;
        memcpy(bytes + 36, job.msg, 32);
        s3::loadHeader(bytes, hdr);
        return true;
    }

    uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++)
            if ((bits & (1u << i)) &&
                (memProps_.memoryTypes[i].propertyFlags & want) == want)
                return i;
        return UINT32_MAX;
    }

    bool createBuffer(VkDeviceSize size, Buffer *b) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKCHECK(vkCreateBuffer(dev_, &bci, nullptr, &b->buf));

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b->buf, &req);
        const uint32_t type = findMemType(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) {
            fprintf(stderr, "no host-visible memory type\n");
            return false;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &b->mem));
        VKCHECK(vkBindBufferMemory(dev_, b->buf, b->mem, 0));
        b->size = size;
        VKCHECK(vkMapMemory(dev_, b->mem, 0, size, 0, &b->mapped));
        return true;
    }

    void destroyBuffer(Buffer *b) {
        if (b->mapped) { vkUnmapMemory(dev_, b->mem); b->mapped = nullptr; }
        if (b->buf) { vkDestroyBuffer(dev_, b->buf, nullptr); b->buf = VK_NULL_HANDLE; }
        if (b->mem) { vkFreeMemory(dev_, b->mem, nullptr); b->mem = VK_NULL_HANDLE; }
        b->size = 0;
    }

    void writeDescriptors() {
        VkDescriptorBufferInfo info[kBindings]{};
        info[0].buffer = out_.buf; info[0].range = VK_WHOLE_SIZE;
        info[1].buffer = cnt_.buf; info[1].range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w[kBindings]{};
        for (uint32_t i = 0; i < kBindings; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = descSet_;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(dev_, kBindings, w, 0, nullptr);
    }

    bool dispatch(const Push &p, uint32_t groups) {
        if (groups == 0) groups = 1;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd_, &bi));

        // The fence between dispatches guarantees the previous one finished
        // executing, which is not the same thing as a shader-write ->
        // shader-read memory dependency. Slices of one search share the
        // solution buffer and the counter, so the barrier is load bearing.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cmd_, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(Push), &p);
        vkCmdDispatch(cmd_, groups, 1, 1);
        VKCHECK(vkEndCommandBuffer(cmd_));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VKCHECK(vkResetFences(dev_, 1, &fence_));
        VKCHECK(vkQueueSubmit(queue_, 1, &si, fence_));
        VKCHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        return true;
    }

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
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

    Buffer out_, cnt_;
    VkDeviceSize deviceLocalBytes_ = 0;
    char devName_[256] = {0};
    char driver_[64] = {0};
};

}  // namespace

Algorithm *makeSha3_256tVK(int deviceIndex) {
    auto *a = new Sha3_256tVK();
    if (!a->init(deviceIndex)) {
        delete a;
        return nullptr;
    }
    vkSetDeviceInfo(a->deviceName(), a->deviceMemGB(), a->driverVersion());
    return a;
}

}  // namespace om
