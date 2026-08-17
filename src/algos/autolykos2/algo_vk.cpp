// Autolykos v2 - Vulkan compute backend (AMD, NVIDIA, Intel).
//
// Chosen over OpenCL for availability, not speed: every gaming-oriented AMD
// distribution ships a working Vulkan driver, while AMD OpenCL needs ROCm,
// amdgpu-pro or Mesa rusticl and is often simply absent. The workload is
// memory-bandwidth bound, so the API makes no difference to throughput.
//
// The dataset is split across four storage buffers. Vulkan only guarantees
// maxStorageBufferRange per binding - 4 GB - 1 even on a 24 GB RTX 4090 -
// against a 7.27 GB dataset, so one binding cannot hold it.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/algo.h"

extern const uint32_t kAutolykosSpirv[];
extern const size_t kAutolykosSpirvWords;

namespace om {

const char *driverTypeName(int t);

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

/** Must match the push_constant block in kernel.comp exactly. */
struct Push {
    uint64_t msg0, msg1, msg2, msg3;
    uint64_t t0, t1, t2, t3;
    uint64_t baseNonce;
    uint32_t N, height, chunkShift, mode;
    uint32_t buildFirst, buildCount, buildChunk, maxOut;
};
static_assert(sizeof(Push) <= 128, "push constants must fit the guaranteed 128 bytes");

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

/** Workgroup size. Fed to the shader as a specialization constant. */
static const uint32_t kLocalSize = 256;

/**
 * Wave/subgroup size. 0 lets the driver choose (wave64 on RDNA2).
 * Overridable with SOAT_SUBGROUP=32|64 for A/B testing: this workload is
 * memory-latency bound, and narrower waves can keep more independent memory
 * requests in flight.
 */
static uint32_t subgroupSizeOverride() {
    const char *e = getenv("SOAT_SUBGROUP");
    return e ? (uint32_t)atoi(e) : 0u;
}

class Autolykos2VK : public Algorithm {
   public:
    const char *name() const override { return "autolykos2"; }

    size_t memoryBytes(const Job &job) const override {
        return (size_t)calcN(job.epoch) * 32ULL;
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

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(inst_, &count, nullptr);
        if (count == 0) {
            fprintf(stderr, "no Vulkan device found\n");
            return false;
        }
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(inst_, &count, devs.data());

        // Only real GPUs. llvmpipe/lavapipe advertise themselves as Vulkan
        // devices of type CPU and would "work" at about 0.1 MH/s.
        std::vector<VkPhysicalDevice> usable;
        for (auto d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                usable.push_back(d);
        }
        if (usable.empty()) {
            fprintf(stderr,
                    "no Vulkan GPU found (only CPU/software devices).\n"
                    "  install your GPU vendor's Vulkan driver, then check "
                    "with: vulkaninfo --summary\n");
            return false;
        }

        size_t pick = 0;
        if (requestedIndex >= 0) {
            if ((size_t)requestedIndex >= usable.size()) {
                fprintf(stderr, "device %d requested but only %zu GPU(s) found\n",
                        requestedIndex, usable.size());
                return false;
            }
            pick = (size_t)requestedIndex;
        } else {
            // Largest device-local heap wins: the right answer when an iGPU
            // sits alongside a discrete card.
            VkDeviceSize best = 0;
            for (size_t i = 0; i < usable.size(); i++) {
                VkPhysicalDeviceMemoryProperties mp{};
                vkGetPhysicalDeviceMemoryProperties(usable[i], &mp);
                VkDeviceSize local = 0;
                for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
                    if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                        local = std::max(local, mp.memoryHeaps[h].size);
                if (local > best) { best = local; pick = i; }
            }
        }
        phys_ = usable[pick];

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        snprintf(devName_, sizeof(devName_), "%s", props.deviceName);
        maxStorageRange_ = props.limits.maxStorageBufferRange;
        snprintf(driver_, sizeof(driver_), "Vulkan %u.%u.%u",
                 VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
                 VK_VERSION_PATCH(props.apiVersion));

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        memProps_ = mp;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                deviceLocalBytes_ = std::max(deviceLocalBytes_, mp.memoryHeaps[h].size);

        // --- queue ---------------------------------------------------------
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, qs.data());
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < qcount; i++)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; break; }
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

        // Blake2b is entirely 64-bit; the byte handling needs int8.
        VkPhysicalDeviceFeatures feats{};
        feats.shaderInt64 = VK_TRUE;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.shaderInt8 = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;

        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        f13.subgroupSizeControl = VK_TRUE;
        f13.computeFullSubgroups = VK_TRUE;
        f12.pNext = &f13;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &feats;
        dci.pNext = &f12;
        VKCHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        // --- pipeline ------------------------------------------------------
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = kAutolykosSpirvWords * sizeof(uint32_t);
        smci.pCode = kAutolykosSpirv;
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

        // Specialization constant 0 = local_size_x, so host and shader can
        // never disagree about the workgroup size.
        // Only local_size is specialized. N was tried as a spec constant too:
        // it gave no measurable gain (the search path is memory bound, not
        // arithmetic bound) and it baked N into the pipeline at init, which
        // would silently break at the next epoch where N grows.
        VkSpecializationMapEntry specEntry{};
        specEntry.constantID = 0;
        specEntry.offset = 0;
        specEntry.size = sizeof(uint32_t);
        const uint32_t localSize = kLocalSize;
        VkSpecializationInfo spec{};
        spec.mapEntryCount = 1;
        spec.pMapEntries = &specEntry;
        spec.dataSize = sizeof(uint32_t);
        spec.pData = &localSize;

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader_;
        stage.pName = "main";
        stage.pSpecializationInfo = &spec;

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgInfo{};
        const uint32_t wantSg = subgroupSizeOverride();
        if (wantSg == 32u || wantSg == 64u) {
            sgInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
            sgInfo.requiredSubgroupSize = wantSg;
            stage.pNext = &sgInfo;
            fprintf(stderr, "[vulkan] forcing subgroup size %u\n", wantSg);
        }

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

        // Output and counter stay host-visible: they are tiny and read back
        // every batch.
        if (!createBuffer(sizeof(uint64_t) * 5 * kMaxSolutions, true, &out_)) return false;
        if (!createBuffer(sizeof(uint32_t), true, &cnt_)) return false;
        return true;
    }

    const char *deviceName() const { return devName_; }
    double deviceMemGB() const { return deviceLocalBytes_ / 1e9; }
    const char *driverVersion() const { return driver_; }

    // ----------------------------------------------------------- dataset --
    bool prepare(const Job &job) override {
        const uint32_t N = calcN(job.epoch);
        const uint64_t total = (uint64_t)N * 32ULL;

        // Power-of-two chunking so the shader can use shift+mask instead of an
        // integer division on every one of its 33 lookups per nonce.
        uint32_t shift = 26;  // 2^26 elements = 2.15 GB, under the 4 GB limit
        while (((uint64_t)1 << shift) * 32ULL > (uint64_t)maxStorageRange_) shift--;
        const uint32_t elemsPerChunk = 1u << shift;
        uint32_t chunks = (uint32_t)((N + elemsPerChunk - 1) / elemsPerChunk);
        chunkShift_ = shift;
        if (chunks > 4) {
            fprintf(stderr,
                    "dataset needs %u buffers but the shader binds 4 "
                    "(maxStorageBufferRange %.2f GB)\n",
                    chunks, maxStorageRange_ / 1e9);
            return false;
        }

        if (N != n_ || chunks != chunks_) {
            release();
            chunkElems_ = elemsPerChunk;
            for (uint32_t c = 0; c < 4; c++) {
                const uint32_t count =
                    (c < chunks) ? ((c == chunks - 1) ? (N - c * chunkElems_)
                                                      : chunkElems_)
                                 : 1;
                if (!createBuffer((VkDeviceSize)count * 32ULL, false, &ds_[c])) {
                    fprintf(stderr, "chunk %u alloc failed (%.2f GB)\n", c,
                            count * 32.0 / 1e9);
                    return false;
                }
            }
            n_ = N;
            chunks_ = chunks;
            writeDescriptors();
        }

        for (uint32_t c = 0; c < chunks_; c++) {
            Push p{};
            p.mode = 0;
            p.height = (uint32_t)job.epoch;
            p.N = N;
            p.chunkShift = chunkShift_;
            p.buildChunk = c;
            p.buildFirst = c * chunkElems_;
            p.buildCount = (c == chunks_ - 1) ? (N - p.buildFirst) : chunkElems_;
            if (!dispatch(p, (p.buildCount + kLocalSize - 1) / kLocalSize)) return false;
        }
        height_ = (uint32_t)job.epoch;
        return true;
    }

    // ------------------------------------------------------------ search --
    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (ds_[0].buf == VK_NULL_HANDLE) return false;
        *(uint32_t *)cnt_.mapped = 0;

        Push p = pushFor(job);
        p.mode = 1;
        p.baseNonce = nonceBase;
        p.maxOut = kMaxSolutions;
        if (!dispatch(p, (uint32_t)(count / kLocalSize))) return false;

        uint32_t found = *(uint32_t *)cnt_.mapped;
        if (found == 0) return true;
        if (found > kMaxSolutions) found = kMaxSolutions;

        const uint64_t *raw = (const uint64_t *)out_.mapped;
        for (uint32_t i = 0; i < found; i++) {
            Solution s;
            s.nonce = raw[i * 5];
            for (int l = 0; l < 4; l++) s.hit[l] = raw[i * 5 + 1 + l];
            out->push_back(s);
        }
        return true;
    }

    bool verify(const Job &job, const Solution &sol) const override {
        if (ds_[0].buf == VK_NULL_HANDLE) return false;
        Push p = pushFor(job);
        p.mode = 2;
        p.baseNonce = sol.nonce;
        if (!const_cast<Autolykos2VK *>(this)->dispatch(p, 1)) return false;

        const uint64_t *raw = (const uint64_t *)out_.mapped;
        uint64_t hit[4];
        for (int l = 0; l < 4; l++) hit[l] = raw[l];
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        for (int i = 3; i >= 0; i--)
            if (hit[i] != job.target[i]) return hit[i] < job.target[i];
        return false;
    }

    void release() override {
        for (int c = 0; c < 4; c++) destroyBuffer(&ds_[c]);
        n_ = 0;
        chunks_ = 0;
    }

    ~Autolykos2VK() override {
        if (dev_) vkDeviceWaitIdle(dev_);
        release();
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
    static const uint32_t kMaxSolutions = 32;
    static const uint32_t kBindings = 6;

    Push pushFor(const Job &job) const {
        Push p{};
        memcpy(&p.msg0, job.msg + 0, 8);
        memcpy(&p.msg1, job.msg + 8, 8);
        memcpy(&p.msg2, job.msg + 16, 8);
        memcpy(&p.msg3, job.msg + 24, 8);
        p.t0 = job.target[0]; p.t1 = job.target[1];
        p.t2 = job.target[2]; p.t3 = job.target[3];
        p.N = n_;
        p.height = height_;
        p.chunkShift = chunkShift_;
        p.maxOut = kMaxSolutions;
        return p;
    }

    uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++)
            if ((bits & (1u << i)) &&
                (memProps_.memoryTypes[i].propertyFlags & want) == want)
                return i;
        return UINT32_MAX;
    }

    bool createBuffer(VkDeviceSize size, bool hostVisible, Buffer *b) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKCHECK(vkCreateBuffer(dev_, &bci, nullptr, &b->buf));

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b->buf, &req);

        const VkMemoryPropertyFlags want =
            hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const uint32_t type = findMemType(req.memoryTypeBits, want);
        if (type == UINT32_MAX) {
            fprintf(stderr, "no suitable memory type\n");
            return false;
        }

        // Dedicated allocation. This one flag is worth 72x on NVIDIA.
        //
        // The Vulkan backend used to collapse as the dataset grew - on a 4090,
        // same code, only the height changing: 119 MH/s at 2.15 GB, 5.4 at
        // 3.86, 2.9 at 7.27, while CUDA stayed flat near 218 across all three.
        // Two plausible causes were both wrong. It was not the four-way
        // if/else that picks a chunk buffer on every gather: removing that
        // branch entirely changed nothing. It was not register spilling
        // either: an RTX 5080 with essentially no spilling (48 bytes, against
        // 336-448 on the 4090) was just as slow.
        //
        // It is TLB reach. Confining the gathers to the first 2.15 GB of a
        // full 7.27 GB allocation restored the fast number exactly (88.5 vs
        // 89.3 MH/s), which rules out every code path and leaves only how far
        // the random gathers span. Without this flag the driver does not back
        // the dataset with large pages, and 33 random gathers per nonce across
        // several GB miss the TLB nearly every time - the card sat at 100%
        // "utilisation" with 4% memory utilisation and half its power budget
        // unused, which is a stalled kernel, not a saturated one.
        //
        // Measured at 7.27 GB with the flag: 4090 2.1 -> 153 MH/s,
        // 5080 5.7 -> 267 MH/s, memory utilisation 4% -> 99%. CUDA never
        // showed this because cudaMalloc gets large pages already.
        VkMemoryDedicatedAllocateInfo mdai{};
        mdai.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        mdai.buffer = b->buf;

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        mai.pNext = &mdai;
        VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &b->mem));
        VKCHECK(vkBindBufferMemory(dev_, b->buf, b->mem, 0));
        b->size = size;
        if (hostVisible) VKCHECK(vkMapMemory(dev_, b->mem, 0, size, 0, &b->mapped));
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
        for (int c = 0; c < 4; c++) {
            info[c].buffer = ds_[c].buf;
            info[c].range = VK_WHOLE_SIZE;
        }
        info[4].buffer = out_.buf; info[4].range = VK_WHOLE_SIZE;
        info[5].buffer = cnt_.buf; info[5].range = VK_WHOLE_SIZE;

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
        // Make storage-buffer writes from any earlier dispatch (the dataset
        // build) visible to this dispatch's shader reads. The fence wait between
        // dispatches guarantees the prior one finished executing, but execution
        // completion is not a shader-write -> shader-read memory dependency; that
        // needs a barrier. Global and cheap, since the fence already serialises
        // the dispatches. In submission order this depends on the prior submit.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0,
                                1, &descSet_, 0, nullptr);
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

    Buffer ds_[4], out_, cnt_;
    VkDeviceSize deviceLocalBytes_ = 0;
    double maxStorageRange_ = 0;
    char devName_[256] = {0};
    char driver_[64] = {0};
    uint32_t n_ = 0, chunks_ = 0, chunkElems_ = 0, chunkShift_ = 0, height_ = 0;
};

Autolykos2VK *g_instance = nullptr;

}  // namespace

Algorithm *makeAutolykos2VK(int deviceIndex) {
    auto *a = new Autolykos2VK();
    if (!a->init(deviceIndex)) {
        delete a;
        return nullptr;
    }
    g_instance = a;
    return a;
}

const char *vkDeviceName() { return g_instance ? g_instance->deviceName() : "unknown"; }
double vkDeviceMemGB() { return g_instance ? g_instance->deviceMemGB() : 0.0; }
const char *vkDriverVersion() { return g_instance ? g_instance->driverVersion() : ""; }

/** Prints every Vulkan GPU, for --list-devices. */
void vkListDevices() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        printf("cannot create a Vulkan instance - is a driver installed?\n");
        return;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    int idx = 0;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        const bool gpu = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                         p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(d, &mp);
        VkDeviceSize local = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                local = std::max(local, mp.memoryHeaps[h].size);
        if (gpu) {
            printf("  [%d] %-38s %5.1f GB  max buffer %4.1f GB  (%s)\n", idx++,
                   p.deviceName, local / 1e9,
                   p.limits.maxStorageBufferRange / 1e9, driverTypeName(p.deviceType));
        } else {
            printf("   -  %-38s %5.1f GB  (skipped: %s)\n", p.deviceName, local / 1e9,
                   driverTypeName(p.deviceType));
        }
    }
    if (idx == 0) printf("no usable Vulkan GPU found\n");
    vkDestroyInstance(inst, nullptr);
}

const char *driverTypeName(int t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU / software";
        default: return "other";
    }
}

}  // namespace om
