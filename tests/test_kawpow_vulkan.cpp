// The Vulkan DAG shader against the host reference.
//
// This is the check the shader is worthless without. NVIDIA's Vulkan compiler
// once miscompiled the Autolykos kernel in this repository and the miner mined
// happily while every share was silently rejected, so a shader that has been
// compiled is not a shader that has been verified.
//
// How it exercises the four-buffer split without allocating 6 GB
// --------------------------------------------------------------
// The split exists because a real 6.11 GB DAG fits in neither one Vulkan
// storage buffer nor, on AMD, one allocation. But the interesting part of that
// design is the index arithmetic, not the size, so the test makes the buffers
// deliberately small and builds only the first slice of the DAG. Four buffers
// of 4 MB each, 262144 items, and the sampled indices are chosen to land in
// every one of the four - including the last item of a buffer and the first of
// the next, which is where an off-by-one in the straddle arithmetic lives.
//
// That keeps the test runnable on any Vulkan GPU, including the 12 GB cards,
// and it runs in seconds rather than minutes.
//
//   ./tests/test_kawpow_vulkan build/dag.spv [deviceIndex]

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "algos/progpow/ethash.h"

using namespace om::progpow;

namespace {

int g_fail = 0;

void check(bool ok, const std::string &what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) g_fail++;
}

#define VKOK(expr, what)                                              \
    do {                                                              \
        const VkResult r__ = (expr);                                  \
        if (r__ != VK_SUCCESS) {                                      \
            fprintf(stderr, "  FAIL  %s: VkResult %d\n", what, r__);  \
            return false;                                             \
        }                                                             \
    } while (0)

std::vector<uint32_t> readSpirv(const char *path) {
    std::vector<uint32_t> out;
    FILE *f = fopen(path, "rb");
    if (!f) return out;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 0 && (n % 4) == 0) {
        out.resize((size_t)n / 4);
        if (fread(out.data(), 1, (size_t)n, f) != (size_t)n) out.clear();
    }
    fclose(f);
    return out;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

class Harness {
   public:
    // Deliberately small: the split arithmetic is what is under test, not the
    // allocator. 1<<20 words is 4 MB and a multiple of 64 words, which is the
    // property that stops a 16-word item straddling a boundary.
    static const uint32_t kWordsPerBuf = 1u << 20;
    static const uint32_t kNumBufs = 4;
    static const uint32_t kItems512 = (kWordsPerBuf * kNumBufs) / 16;

    bool init(int deviceIndex, const std::vector<uint32_t> &spirv) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VKOK(vkCreateInstance(&ici, nullptr, &inst_), "vkCreateInstance");

        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst_, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst_, &n, devs.data());
        std::vector<VkPhysicalDevice> usable;
        for (auto d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                usable.push_back(d);
        }
        if (usable.empty()) {
            fprintf(stderr, "  SKIP  no Vulkan GPU (software devices ignored)\n");
            return false;
        }
        phys_ = usable[deviceIndex < 0 ? 0 : (size_t)deviceIndex % usable.size()];
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        printf("  %s\n", props.deviceName);

        // shaderInt64 is required, not optional: Keccak-f[1600] is 64-bit
        // lanes and a 32-bit fallback would be dead code no test ever covers.
        VkPhysicalDeviceFeatures feats{};
        vkGetPhysicalDeviceFeatures(phys_, &feats);
        if (!feats.shaderInt64) {
            fprintf(stderr, "  SKIP  this device has no shaderInt64\n");
            return false;
        }

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qs.data());
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; break; }
        if (queueFamily_ == UINT32_MAX) return false;

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures want{};
        want.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &want;
        VKOK(vkCreateDevice(phys_, &dci, nullptr, &dev_), "vkCreateDevice");
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = queueFamily_;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VKOK(vkCreateCommandPool(dev_, &cpci, nullptr, &pool_), "command pool");

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * 4;
        smci.pCode = spirv.data();
        VKOK(vkCreateShaderModule(dev_, &smci, nullptr, &module_), "shader module");
        return true;
    }

    bool buildPipeline() {
        VkDescriptorSetLayoutBinding binds[5]{};
        for (int i = 0; i < 5; i++) {
            binds[i].binding = (uint32_t)i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 5;
        dslci.pBindings = binds;
        VKOK(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &setLayout_),
             "set layout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 16;   // four uints
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VKOK(vkCreatePipelineLayout(dev_, &plci, nullptr, &pipeLayout_),
             "pipeline layout");

        // Workgroup size is a specialization constant so the host dispatch and
        // the shader can never disagree about it - that drifted once already
        // in this repo and silently ran half the work.
        const uint32_t local = 64;
        VkSpecializationMapEntry sme{};
        sme.constantID = 0;
        sme.offset = 0;
        sme.size = sizeof(uint32_t);
        VkSpecializationInfo si{};
        si.mapEntryCount = 1;
        si.pMapEntries = &sme;
        si.dataSize = sizeof(uint32_t);
        si.pData = &local;

        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = module_;
        ssci.pName = "main";
        ssci.pSpecializationInfo = &si;
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = ssci;
        cpci.layout = pipeLayout_;
        VKOK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                      &pipe_),
             "compute pipeline");
        localSize_ = local;

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 5;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VKOK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &descPool_), "desc pool");
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &setLayout_;
        VKOK(vkAllocateDescriptorSets(dev_, &dsai, &descSet_), "desc set");
        return true;
    }

    uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags want) const {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & want) == want)
                return i;
        return UINT32_MAX;
    }

    bool createBuffer(VkDeviceSize size, VkMemoryPropertyFlags props, Buffer *b) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKOK(vkCreateBuffer(dev_, &bci, nullptr, &b->buf), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b->buf, &req);
        const uint32_t type = findMemType(req.memoryTypeBits, props);
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        VKOK(vkAllocateMemory(dev_, &mai, nullptr, &b->mem), "vkAllocateMemory");
        VKOK(vkBindBufferMemory(dev_, b->buf, b->mem, 0), "vkBindBufferMemory");
        b->size = size;
        return true;
    }

    bool upload(Buffer *b, const void *src, size_t bytes) {
        void *p = nullptr;
        VKOK(vkMapMemory(dev_, b->mem, 0, bytes, 0, &p), "vkMapMemory");
        memcpy(p, src, bytes);
        vkUnmapMemory(dev_, b->mem);
        return true;
    }

    bool download(Buffer *b, size_t offset, void *dst, size_t bytes) {
        void *p = nullptr;
        VKOK(vkMapMemory(dev_, b->mem, offset, bytes, 0, &p), "vkMapMemory");
        memcpy(dst, p, bytes);
        vkUnmapMemory(dev_, b->mem);
        return true;
    }

    void writeDescriptors(const Buffer &cache, Buffer dag[kNumBufs]) {
        VkDescriptorBufferInfo info[5]{};
        info[0].buffer = cache.buf;
        info[0].range = cache.size;
        for (uint32_t i = 0; i < kNumBufs; i++) {
            info[1 + i].buffer = dag[i].buf;
            info[1 + i].range = dag[i].size;
        }
        VkWriteDescriptorSet w[5]{};
        for (int i = 0; i < 5; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = descSet_;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(dev_, 5, w, 0, nullptr);
    }

    bool dispatch(const uint32_t push[4], uint32_t groups) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VKOK(vkAllocateCommandBuffers(dev_, &cbai, &cb), "alloc cmd buffer");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKOK(vkBeginCommandBuffer(cb, &bi), "begin cmd buffer");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                           push);
        vkCmdDispatch(cb, groups, 1, 1);
        VKOK(vkEndCommandBuffer(cb), "end cmd buffer");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        VKOK(vkCreateFence(dev_, &fci, nullptr, &fence), "create fence");
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VKOK(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
        const VkResult wr = vkWaitForFences(dev_, 1, &fence, VK_TRUE, 120ull * 1000000000ull);
        vkDestroyFence(dev_, fence, nullptr);
        vkFreeCommandBuffers(dev_, pool_, 1, &cb);
        if (wr != VK_SUCCESS) {
            fprintf(stderr, "  FAIL  dispatch did not complete (VkResult %d)\n", wr);
            return false;
        }
        return true;
    }

    uint32_t localSize() const { return localSize_; }
    VkDevice dev() const { return dev_; }

   private:
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    uint32_t localSize_ = 64;
};

std::string wordsToHex(const uint32_t *w, int words) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < words; i++)
        for (int b = 0; b < 4; b++) {
            const uint8_t v = (uint8_t)(w[i] >> (b * 8));
            s += d[v >> 4];
            s += d[v & 15];
        }
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    const char *spvPath = argc > 1 ? argv[1] : "build/dag.spv";
    const int deviceIndex = argc > 2 ? atoi(argv[2]) : -1;

    const std::vector<uint32_t> spirv = readSpirv(spvPath);
    if (spirv.empty()) {
        fprintf(stderr, "cannot read SPIR-V from %s\n", spvPath);
        return 2;
    }

    Harness h;
    if (!h.init(deviceIndex, spirv)) return 0;   // skip, politely
    if (!h.buildPipeline()) return 1;

    // Epoch 0's light cache: 16 MB, and the smallest real one there is.
    const int epoch = 0;
    const int numCacheItems = lightCacheNumItems(epoch);
    uint32_t seed[8];
    epochSeed(epoch, seed);
    std::vector<uint32_t> cache((size_t)numCacheItems * 16, 0);
    buildLightCache(cache.data(), numCacheItems, seed);

    Buffer cacheBuf, dag[Harness::kNumBufs];
    const VkMemoryPropertyFlags hostVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!h.createBuffer(cache.size() * 4, hostVisible, &cacheBuf)) return 1;
    if (!h.upload(&cacheBuf, cache.data(), cache.size() * 4)) return 1;
    for (uint32_t i = 0; i < Harness::kNumBufs; i++)
        if (!h.createBuffer((VkDeviceSize)Harness::kWordsPerBuf * 4, hostVisible,
                            &dag[i]))
            return 1;
    h.writeDescriptors(cacheBuf, dag);

    const uint32_t items = Harness::kItems512;
    printf("  building %u dataset items into %u buffers of %u words\n", items,
           Harness::kNumBufs, Harness::kWordsPerBuf);

    const uint32_t push[4] = {0u, items, (uint32_t)numCacheItems,
                              Harness::kWordsPerBuf};
    if (!h.dispatch(push, (items + h.localSize() - 1) / h.localSize())) return 1;

    // Sample across every buffer, and specifically the last item of each and
    // the first of the next - which is where the straddle arithmetic breaks.
    const uint32_t perBuf = Harness::kWordsPerBuf / 16;
    std::vector<uint32_t> probes = {0, 1, 17, perBuf - 1, perBuf,
                                    perBuf + 1, 2 * perBuf - 1, 2 * perBuf,
                                    3 * perBuf - 1, 3 * perBuf,
                                    items - 1};
    int mismatched = 0;
    for (uint32_t idx : probes) {
        uint32_t host[16];
        datasetItem512(cache.data(), numCacheItems, idx, host);

        const uint32_t wordIndex = idx * 16u;
        const uint32_t buf = wordIndex / Harness::kWordsPerBuf;
        const uint32_t off = wordIndex - buf * Harness::kWordsPerBuf;
        uint32_t got[16];
        if (!h.download(&dag[buf], (size_t)off * 4, got, sizeof(got))) return 1;

        if (memcmp(host, got, sizeof(got)) != 0) {
            if (mismatched < 3) {
                printf("  FAIL  item %u (buffer %u, word %u)\n", idx, buf, off);
                printf("        host %s\n        gpu  %s\n",
                       wordsToHex(host, 8).c_str(), wordsToHex(got, 8).c_str());
            }
            mismatched++;
        }
    }
    check(mismatched == 0,
          "every sampled DAG item matches the host, across all four buffers");

    // A shader that wrote nothing would match a host that also computed
    // nothing, so make sure the data is not simply zero.
    uint32_t first[16];
    h.download(&dag[0], 0, first, sizeof(first));
    bool allZero = true;
    for (int i = 0; i < 16; i++)
        if (first[i] != 0) allZero = false;
    check(!allZero, "the shader actually wrote something");

    if (g_fail == 0) printf("KawPow Vulkan DAG: matches the host reference\n");
    return g_fail == 0 ? 0 : 1;
}
