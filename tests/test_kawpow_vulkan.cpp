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

#include "algos/kawpow/kawpow.h"
#include "algos/progpow/ethash.h"
#include "algos/progpow/progpow.h"

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

    bool init(int deviceIndex) {
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

        return true;
    }

    /** One compute pipeline plus the descriptor set it owns. Two shaders means
     *  two of these, and keeping them separate is simpler than one set layout
     *  that has to satisfy both. */
    struct Pipeline {
        VkShaderModule module = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        uint32_t bindings = 0;
        uint32_t pushBytes = 0;
        uint32_t local = 64;
    };

    bool buildPipeline(const std::vector<uint32_t> &spirv, uint32_t nBindings,
                       uint32_t pushBytes, uint32_t local, Pipeline *out) {
        out->bindings = nBindings;
        out->pushBytes = pushBytes;
        out->local = local;

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * 4;
        smci.pCode = spirv.data();
        VKOK(vkCreateShaderModule(dev_, &smci, nullptr, &out->module),
             "shader module");

        std::vector<VkDescriptorSetLayoutBinding> binds(nBindings);
        for (uint32_t i = 0; i < nBindings; i++) {
            binds[i] = {};
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = nBindings;
        dslci.pBindings = binds.data();
        VKOK(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &out->setLayout),
             "set layout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = pushBytes;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &out->setLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VKOK(vkCreatePipelineLayout(dev_, &plci, nullptr, &out->layout),
             "pipeline layout");

        // Workgroup size is a specialization constant so the host dispatch and
        // the shader can never disagree about it - that drifted once already
        // in this repo and silently ran half the work while crediting all of it.
        const uint32_t spec[2] = {local, 256u};   // local size, MAX_OUT
        VkSpecializationMapEntry sme[2]{};
        sme[0].constantID = 0;
        sme[0].offset = 0;
        sme[0].size = sizeof(uint32_t);
        sme[1].constantID = 1;
        sme[1].offset = sizeof(uint32_t);
        sme[1].size = sizeof(uint32_t);
        VkSpecializationInfo si{};
        si.mapEntryCount = nBindings > 5 ? 2 : 1;   // the DAG shader has no MAX_OUT
        si.pMapEntries = sme;
        si.dataSize = si.mapEntryCount * sizeof(uint32_t);
        si.pData = spec;

        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = out->module;
        ssci.pName = "main";
        ssci.pSpecializationInfo = &si;
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = ssci;
        cpci.layout = out->layout;
        VKOK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                      &out->pipe),
             "compute pipeline");

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = nBindings;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VKOK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &out->pool), "desc pool");
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = out->pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &out->setLayout;
        VKOK(vkAllocateDescriptorSets(dev_, &dsai, &out->set), "desc set");
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

    void writeDescriptors(const Pipeline &p, const std::vector<Buffer *> &bufs) {
        std::vector<VkDescriptorBufferInfo> info(bufs.size());
        std::vector<VkWriteDescriptorSet> w(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            info[i] = {};
            info[i].buffer = bufs[i]->buf;
            info[i].range = bufs[i]->size;
            w[i] = {};
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = p.set;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(dev_, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    bool dispatch(const Pipeline &p, const void *push, uint32_t groups) {
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
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                                1, &p.set, 0, nullptr);
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           p.pushBytes, push);
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

    VkDevice dev() const { return dev_; }

   private:
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
    VkCommandPool pool_ = VK_NULL_HANDLE;

};

void hexToWords(const char *hex, uint32_t *out, int words) {
    auto hv = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return c - 'A' + 10;
    };
    for (int i = 0; i < words * 4; i++) {
        const uint8_t b = (uint8_t)((hv(hex[i * 2]) << 4) | hv(hex[i * 2 + 1]));
        if (i % 4 == 0) out[i / 4] = 0;
        out[i / 4] |= (uint32_t)b << ((i % 4) * 8);
    }
}

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
    const char *dagSpv = argc > 1 ? argv[1] : "build/dag.spv";
    const char *mixSpv = argc > 2 ? argv[2] : "build/kawpow.spv";
    const int deviceIndex = argc > 3 ? atoi(argv[3]) : -1;

    const std::vector<uint32_t> dagCode = readSpirv(dagSpv);
    const std::vector<uint32_t> mixCode = readSpirv(mixSpv);
    if (dagCode.empty() || mixCode.empty()) {
        fprintf(stderr, "cannot read SPIR-V from %s / %s\n", dagSpv, mixSpv);
        return 2;
    }

    Harness h;
    if (!h.init(deviceIndex)) return 0;   // skip, politely

    Harness::Pipeline dagPipe, mixPipe;
    if (!h.buildPipeline(dagCode, 5, 16, 64, &dagPipe)) return 1;
    if (!h.buildPipeline(mixCode, 7, 92, 64, &mixPipe)) return 1;

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

    h.writeDescriptors(dagPipe, {&cacheBuf, &dag[0], &dag[1], &dag[2], &dag[3]});

    const uint32_t items = Harness::kItems512;
    printf("  building %u dataset items into %u buffers of %u words\n", items,
           Harness::kNumBufs, Harness::kWordsPerBuf);
    const uint32_t push[4] = {0u, items, (uint32_t)numCacheItems,
                              Harness::kWordsPerBuf};
    if (!h.dispatch(dagPipe, push, (items + 63) / 64)) return 1;

    // --- 1. the DAG shader vs the host --------------------------------------
    const uint32_t perBuf = Harness::kWordsPerBuf / 16;
    std::vector<uint32_t> probes = {0,      1,           17,
                                    perBuf - 1, perBuf,  perBuf + 1,
                                    2 * perBuf - 1, 2 * perBuf,
                                    3 * perBuf - 1, 3 * perBuf, items - 1};
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

    uint32_t first[16];
    h.download(&dag[0], 0, first, sizeof(first));
    bool allZero = true;
    for (int i = 0; i < 16; i++)
        if (first[i] != 0) allZero = false;
    check(!allZero, "the DAG shader actually wrote something");
    if (mismatched || allZero) return 1;   // the mix result would be meaningless

    // --- 2. the mix shader vs the host --------------------------------------
    //
    // The DAG here is a small slice, so numItems2048 is set to what was
    // actually built and the HOST is told the same number. The hashes then
    // correspond to no real block, which does not matter: what is under test is
    // whether two implementations of the same function agree, and pointing both
    // at the same truncated dataset is what makes that comparison exact.
    const uint32_t numItems2048 = items / 4;
    const int height = 4501000;

    const om::progpow::Program prog =
        om::progpow::buildProgram(height, om::kawpow::params());
    Buffer progBuf, outBuf, cntBuf;
    // The shader's Prog block is a std430 run of plain uint arrays, which packs
    // exactly as the host struct's tables do - but only the tables, so they are
    // copied field by field rather than memcpy'd over the whole struct, whose
    // first three ints are counts the shader does not have.
    std::vector<uint32_t> progWords;
    auto push_n = [&](const uint32_t *a, int n) {
        for (int i = 0; i < n; i++) progWords.push_back(a[i]);
    };
    push_n(prog.cacheSrc, 11);
    push_n(prog.cacheDst, 11);
    push_n(prog.cacheSel, 11);
    push_n(prog.mathSrc1, 18);
    push_n(prog.mathSrc2, 18);
    push_n(prog.mathSel1, 18);
    push_n(prog.mathDst, 18);
    push_n(prog.mathSel2, 18);
    push_n(prog.dagDst, 4);
    push_n(prog.dagSel, 4);

    const uint32_t kNonces = 16;
    if (!h.createBuffer(progWords.size() * 4, hostVisible, &progBuf)) return 1;
    if (!h.upload(&progBuf, progWords.data(), progWords.size() * 4)) return 1;
    if (!h.createBuffer(kNonces * 18 * 4, hostVisible, &outBuf)) return 1;
    if (!h.createBuffer(4, hostVisible, &cntBuf)) return 1;
    const uint32_t zero = 0;
    h.upload(&cntBuf, &zero, 4);

    h.writeDescriptors(mixPipe, {&progBuf, &dag[0], &dag[1], &dag[2], &dag[3],
                                 &outBuf, &cntBuf});

    uint32_t mp[23]{};
    uint32_t hdr[8];
    hexToWords("8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7",
               hdr, 8);
    for (int i = 0; i < 8; i++) mp[i] = hdr[i];
    const uint64_t base = 0x329dd53ca526c2c0ull;
    mp[8] = (uint32_t)base;
    mp[9] = (uint32_t)(base >> 32);
    mp[10] = kNonces;
    mp[11] = 0;
    for (int i = 12; i < 20; i++) mp[i] = 0;   // target, unused in dump mode
    mp[20] = numItems2048;
    mp[21] = Harness::kWordsPerBuf;
    mp[22] = 1;   // dump every hash

    if (!h.dispatch(mixPipe, mp, (kNonces + 3) / 4)) return 1;

    std::vector<uint32_t> got(kNonces * 18);
    if (!h.download(&outBuf, 0, got.data(), got.size() * 4)) return 1;

    std::vector<uint32_t> l1(kL1CacheWords, 0);
    for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
        datasetItem2048(cache.data(), numCacheItems, i, l1.data() + i * 64);

    int mixBad = 0;
    for (uint32_t i = 0; i < kNonces; i++) {
        uint32_t hFinal[8], hMix[8];
        om::kawpow::hashLight(cache.data(), numCacheItems, l1.data(),
                              numItems2048, height, hdr, base + i, hFinal, hMix);
        const uint32_t *gFinal = &got[i * 18 + 2];
        const uint32_t *gMix = &got[i * 18 + 10];
        if (memcmp(hFinal, gFinal, 32) != 0 || memcmp(hMix, gMix, 32) != 0) {
            if (mixBad < 3) {
                printf("  FAIL  nonce %016llx\n        host %s\n        gpu  %s\n",
                       (unsigned long long)(base + i),
                       wordsToHex(hFinal, 8).c_str(),
                       wordsToHex(gFinal, 8).c_str());
            }
            mixBad++;
        }
    }
    check(mixBad == 0,
          std::to_string(kNonces) +
              " consecutive nonces hash identically on the shader and the host");

    if (g_fail == 0) printf("KawPow Vulkan: DAG and mix both match the host\n");
    return g_fail == 0 ? 0 : 1;
}
