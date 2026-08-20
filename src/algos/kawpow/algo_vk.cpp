// KawPow on Vulkan - the Algorithm implementation.
//
// The shaders are src/algos/progpow/dag.comp and src/algos/kawpow/kernel.comp,
// both verified against the host reference by tests/test_kawpow_vulkan. This
// file is the host half, and on this algorithm the host half is the larger
// one, for a reason that is worth stating up front.
//
// Why this is not just "the CUDA file with vkCmdDispatch"
// -------------------------------------------------------
// A Ravencoin DAG at epoch 600 is 6.11 GB, and Vulkan cannot put it in one
// buffer. Measured on both target vendors:
//
//   RTX 4090                  maxStorageBufferRange   4294967295
//   RX 7900 XT (RADV NAVI31)  maxStorageBufferRange   4294967295
//                             maxMemoryAllocationSize 0xfffffffc
//
// So the DAG is split across up to four buffers and four allocations, every
// one of them a multiple of 256 bytes so a dataset item never straddles a
// boundary, and both shaders take the split size as a parameter. CUDA needed
// one cudaMalloc.
//
// The second cost is that there are two pipelines where CUDA has two kernels,
// and each carries a descriptor set layout, a pipeline layout, a descriptor
// pool, a descriptor set and a push-constant range.

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <string>
#include <vector>

#include "../../core/algo.h"
#include "../../core/vk_common.h"
#include "../progpow/ethash.h"
#include "../progpow/progpow.h"
#include "job.h"
#include "kawpow.h"

// Declared exactly as scripts/embed_spirv.py emits them - size_t, not
// unsigned. A mismatched type here links on one toolchain and not another.
extern const uint32_t kProgpowDagSpirv[];
extern const size_t kProgpowDagSpirvWords;
extern const uint32_t kKawpowSpirv[];
extern const size_t kKawpowSpirvWords;
extern const uint32_t kKawpowSpecSpirv[];
extern const size_t kKawpowSpecSpirvWords;

namespace om {
namespace {

using namespace om::kawpow;

/**
 * Eight buffers of under 2 GiB each, which covers a 16 GiB DAG.
 *
 * Eight rather than four is not about capacity. A storage buffer of 2 GiB or
 * more measured 2x slower on device-local memory than the same data in a 1 GiB
 * buffer, and 2 GiB is exactly the signed-32-bit boundary. Four buffers forced
 * 2 GiB pieces for a 6 GB DAG; eight lets every piece stay under it.
 */
const uint32_t kMaxDagBufs = 8;
const uint32_t kMaxSolutions = 256;
/** 18 words per solution: nonce (2), final hash (8), mix hash (8). */
const uint32_t kSolutionWords = 18;
/** Lanes per hash, fixed by the algorithm, and the workgroup is a multiple. */
const uint32_t kLanes = 16;
const uint32_t kLocalSize = 256;   // 16 hashes per workgroup, not 4

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct Pipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t pushBytes = 0;
};

/** Push constants for dag.comp: start, numItems512, numCacheItems, wordsPerBuf. */
struct DagPush {
    uint32_t start, numItems512, numCacheItems, wordsPerBufShift;
};

/** Push constants for kernel.comp. 92 bytes, inside the 128 every device has. */
struct MixPush {
    uint32_t h[8];
    uint32_t baseLo, baseHi;
    uint32_t countLo, countHi;
    uint32_t t[8];
    uint32_t numItems2048;
    uint32_t wordsPerBufShift;
    uint32_t dumpMode;
    uint32_t spreadBufs;
    uint32_t spreadSlice;
};
static_assert(sizeof(MixPush) == 100, "the shader's push block must match");

class KawPowVK : public Algorithm {
   public:
    const char *name() const override { return "kawpow"; }

    size_t memoryBytes(const Job &job) const override {
        const int height = jobHeight(job.extra, job.epoch);
        return (size_t)fullDatasetBytes(epochOf(height));
    }

    bool init(int requestedIndex) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "soat-miner";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &inst_) != VK_SUCCESS) {
            fprintf(stderr, "cannot create a Vulkan instance - is a driver "
                            "installed?\n");
            return false;
        }
        if (!vkPickPhysicalDevice(inst_, requestedIndex, &phys_)) return false;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        snprintf(devName_, sizeof(devName_), "%s", props.deviceName);
        snprintf(driver_, sizeof(driver_), "Vulkan %u.%u",
                 VK_VERSION_MAJOR(props.apiVersion),
                 VK_VERSION_MINOR(props.apiVersion));
        maxStorageRange_ = props.limits.maxStorageBufferRange;

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                if (mp.memoryHeaps[i].size > deviceLocalBytes_)
                    deviceLocalBytes_ = mp.memoryHeaps[i].size;

        // Keccak-f[1600] is 64-bit lanes. Ask honestly rather than producing
        // wrong hashes on a device that cannot do it.
        VkPhysicalDeviceFeatures have{};
        vkGetPhysicalDeviceFeatures(phys_, &have);
        if (!have.shaderInt64) {
            fprintf(stderr,
                    "%s reports no shaderInt64, which KawPow's Keccak needs.\n",
                    devName_);
            return false;
        }

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qs.data());
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; break; }
        if (queueFamily_ == UINT32_MAX) {
            fprintf(stderr, "%s has no compute queue\n", devName_);
            return false;
        }

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
        if (vkCreateDevice(phys_, &dci, nullptr, &dev_) != VK_SUCCESS) {
            fprintf(stderr, "vkCreateDevice failed on %s\n", devName_);
            return false;
        }
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = queueFamily_;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(dev_, &cpci, nullptr, &cmdPool_) != VK_SUCCESS)
            return false;

        if (!makePipeline(kProgpowDagSpirv, kProgpowDagSpirvWords, 1 + kMaxDagBufs,
                          sizeof(DagPush), {}, {}, &dagPipe_))
            return false;
        // The mix pipeline is NOT built here. It carries the period's program
        // as specialisation constants, so it is rebuilt when the period turns
        // - see makeMixPipeline().
        return true;
    }

    const char *deviceName() const { return devName_; }
    double deviceMemGB() const { return deviceLocalBytes_ / 1e9; }
    const char *driverVersion() const { return driver_; }

    /**
     * Build the DAG for this job's epoch, if it is not already built.
     *
     * Guarded on the epoch rather than on being called: the run loop calls
     * prepare() whenever Job::epoch changes and the benchmark path moves it on
     * a timer, and rebuilding six gigabytes because a height moved by one would
     * be a self-inflicted stall every block.
     */
    bool prepare(const Job &job) override {
        const int height = jobHeight(job.extra, job.epoch);
        const int epoch = epochOf(height);
        if (epoch == builtEpoch_ && dagBufs_ > 0) return true;

        releaseDag();

        numCacheItems_ = lightCacheNumItems(epoch);
        numItems2048_ = (uint32_t)(fullDatasetNumItems(epoch) / 2);
        const uint64_t dagBytes = fullDatasetBytes(epoch);
        const uint32_t numItems512 = (uint32_t)fullDatasetNumItems(epoch) * 2;

        if (dagBytes + 256ull * 1024 * 1024 > deviceLocalBytes_) {
            fprintf(stderr,
                    "[kawpow] epoch %d needs a %.2f GB DAG and this device has "
                    "%.1f GB. KawPow cannot run on it.\n",
                    epoch, dagBytes / 1e9, deviceLocalBytes_ / 1e9);
            return false;
        }

        // Split so every buffer is inside the device's own limit, and size it
        // to a POWER OF TWO number of words.
        //
        // The power of two is not tidiness. The shader locates a word with
        // wordIndex >> shift, and a runtime divide there was measured holding
        // the whole mix at 1.67 MH/s: four words per lane per round, sixty-four
        // rounds, sixteen lanes is 4096 integer divisions per hash, and a GPU
        // has no integer divide unit. Neither specialisation constants nor
        // removing the per-round barriers moved that number at all.
        //
        // A power of two is also a multiple of 64 words, which is separately
        // required so a 2048-bit dataset item never straddles two buffers.
        // Cap a buffer at 1 GiB of words. Not a device limit - a measured
        // one: at or above 2 GiB the same data reads half as fast.
        uint64_t capWords = maxStorageRange_ / 4;
        if (capWords > (1ull << 28)) capWords = 1ull << 28;
        const uint64_t totalWords = (dagBytes + 3) / 4;
        uint32_t shift = 16;
        while ((1ull << shift) < ((totalWords + kMaxDagBufs - 1) / kMaxDagBufs))
            shift++;
        while ((1ull << shift) > capWords && shift > 16) shift--;
        if ((1ull << shift) * kMaxDagBufs < totalWords) {
            fprintf(stderr,
                    "[kawpow] a %.2f GB DAG will not fit in %u buffers of at "
                    "most %.2f GB on this device.\n",
                    dagBytes / 1e9, kMaxDagBufs, capWords * 4 / 1e9);
            return false;
        }
        // Debug knob for the buffer-size investigation. Not a tuning option:
        // it exists so the split can be varied without a rebuild.
        if (const char *ov = getenv("SOAT_DAG_SHIFT")) {
            const int v = atoi(ov);
            if (v >= 16 && v <= 30) shift = (uint32_t)v;
        }
        wordsPerBufShift_ = shift;
        wordsPerBuf_ = 1u << shift;
        dagBufs_ = (uint32_t)((totalWords + wordsPerBuf_ - 1) / wordsPerBuf_);
        if (dagBufs_ < 1) dagBufs_ = 1;
        const uint64_t bytesPerBuf = (uint64_t)wordsPerBuf_ * 4;

        // The light cache goes on the device: the DAG build reads a random
        // 64-byte item 512 times per output item, and doing that over PCIe
        // would take hours rather than seconds.
        cache_.assign((size_t)numCacheItems_ * 16, 0);
        uint32_t seed[8];
        epochSeed(epoch, seed);
        buildLightCache(cache_.data(), numCacheItems_, seed);

        l1_.assign(kL1CacheWords, 0);
        for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
            datasetItem2048(cache_.data(), numCacheItems_, i, l1_.data() + i * 64);

        Buffer staging;
        if (!createBuffer(cache_.size() * 4, hostVisible(), &staging)) return false;
        if (!upload(&staging, cache_.data(), cache_.size() * 4)) return false;
        if (!createBuffer(cache_.size() * 4, deviceLocal(), &cacheBuf_)) {
            destroyBuffer(&staging);
            return false;
        }
        if (!copyBuffer(staging, cacheBuf_, cache_.size() * 4)) {
            destroyBuffer(&staging);
            return false;
        }
        destroyBuffer(&staging);

        // Two ways to back the split, because which one the driver maps well
        // is a question about the driver, not about us.
        //
        // SOAT_DAG_ONEALLOC=1 takes ONE VkDeviceMemory for the whole DAG and
        // suballocates the buffers into it at offsets, so the six buffers are
        // one contiguous virtual range. wF:p1's reasoning for trying it: a
        // hardware page-walk ceiling would bind CUDA equally over the same
        // 6.11 GB and the same random gathers, and CUDA runs at 60.3 MH/s, so
        // the cliff is how the VULKAN allocation is mapped rather than the GPU.
        // If large-page promotion is keyed to allocation size, six 1 GiB
        // allocations may each be getting small pages.
        const bool oneAlloc = getenv("SOAT_DAG_ONEALLOC") != nullptr;
        if (oneAlloc) {
            if (!allocDagContiguous(bytesPerBuf)) return false;
        } else {
            for (uint32_t i = 0; i < dagBufs_; i++)
                if (!createBuffer(bytesPerBuf, deviceLocal(), &dag_[i])) {
                    fprintf(stderr,
                            "[kawpow] cannot allocate DAG buffer %u of %u (%.2f "
                            "GB each)\n", i + 1, dagBufs_, bytesPerBuf / 1e9);
                    return false;
                }
        }
        // Unused bindings still need a real buffer bound, so they alias the
        // first one. The shader never indexes them: wordsPerBuf * dagBufs_
        // covers the whole DAG by construction.
        for (uint32_t i = dagBufs_; i < kMaxDagBufs; i++) dag_[i] = dag_[0];

        {
            std::vector<Buffer *> bufs{&cacheBuf_};
            for (uint32_t i = 0; i < kMaxDagBufs; i++) bufs.push_back(&dag_[i]);
            writeDescriptors(dagPipe_, bufs);
        }

        // Sliced: one dispatch over 95 million items would sit on the card long
        // enough to trip a display driver's watchdog.
        const uint32_t slice = 1u << 21;
        for (uint32_t start = 0; start < numItems512; start += slice) {
            const uint32_t n =
                (numItems512 - start) < slice ? (numItems512 - start) : slice;
            DagPush p{start, numItems512, (uint32_t)numCacheItems_,
                      wordsPerBufShift_};
            if (!dispatch(dagPipe_, &p, (n + kLocalSize - 1) / kLocalSize))
                return false;
        }

        if (!ensureScratch()) return false;
        builtEpoch_ = epoch;
        builtPeriod_ = -1;   // force the program to be re-uploaded
        return true;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (dagBufs_ == 0) return false;
        const int height = jobHeight(job.extra, job.epoch);

        const int period = height / kPeriodLength;
        if (period != builtPeriod_) {
            if (!makeMixPipeline(height)) return false;
            builtPeriod_ = period;
        }

        const uint32_t zero = 0;
        if (!upload(&cntBuf_, &zero, sizeof(zero))) return false;

        MixPush p{};
        for (int i = 0; i < 8; i++) memcpy(&p.h[i], job.msg + i * 4, 4);
        p.baseLo = (uint32_t)nonceBase;
        p.baseHi = (uint32_t)(nonceBase >> 32);
        p.countLo = (uint32_t)count;
        p.countHi = (uint32_t)(count >> 32);
        for (int i = 0; i < 4; i++) {
            p.t[2 * i] = (uint32_t)job.target[i];
            p.t[2 * i + 1] = (uint32_t)(job.target[i] >> 32);
        }
        // Diagnostic override: shrink the RANDOMLY TOUCHED region without
        // changing the allocation, the buffers or anything else. Results are
        // wrong when set. It is what separates "footprint" from "geometry".
        p.numItems2048 = numItems2048_;
        if (const char *ov = getenv("SOAT_ITEMS_CLAMP")) {
            const uint32_t v = (uint32_t)strtoul(ov, nullptr, 10);
            if (v > 0 && v < numItems2048_) p.numItems2048 = v;
        }
        p.wordsPerBufShift = wordsPerBufShift_;
        p.dumpMode = 0;
        p.spreadBufs = 0;
        p.spreadSlice = 0;
        if (const char *sb = getenv("SOAT_SPREAD_BUFS")) {
            const uint32_t n = (uint32_t)strtoul(sb, nullptr, 10);
            const char *sl = getenv("SOAT_SPREAD_SLICE");
            if (n > 0 && sl) {
                p.spreadBufs = n;
                p.spreadSlice = (uint32_t)strtoul(sl, nullptr, 10);
            }
        }

        const uint32_t hashesPerGroup = kLocalSize / kLanes;
        uint64_t groups = (count + hashesPerGroup - 1) / hashesPerGroup;
        if (groups == 0) groups = 1;
        if (!dispatch(mixPipe_, &p, (uint32_t)groups)) return false;

        uint32_t found = 0;
        if (!download(&cntBuf_, 0, &found, sizeof(found))) return false;
        if (found == 0) return true;
        if (found > kMaxSolutions) {
            fprintf(stderr,
                    "[kawpow] warning: %u solutions in one batch exceeds the "
                    "%u-solution buffer; %u dropped.\n",
                    found, kMaxSolutions, found - kMaxSolutions);
            found = kMaxSolutions;
        }

        std::vector<uint32_t> words((size_t)found * kSolutionWords);
        if (!download(&outBuf_, 0, words.data(), words.size() * 4)) return false;
        for (uint32_t i = 0; i < found; i++) {
            const uint32_t *w = &words[(size_t)i * kSolutionWords];
            Solution s;
            s.nonce = (uint64_t)w[0] | ((uint64_t)w[1] << 32);
            hashToLimbs(&w[2], s.hit);
            s.extra = toHex(&w[10]);
            out->push_back(s);
        }
        return true;
    }

    /** Re-derive on the HOST, from the light cache. A card pushed into
     *  instability confirms its own bad answer if asked to check itself. */
    bool verify(const Job &job, const Solution &sol) const override {
        if (cache_.empty()) return false;
        const int height = jobHeight(job.extra, job.epoch);
        uint32_t hdr[8];
        for (int i = 0; i < 8; i++) memcpy(&hdr[i], job.msg + i * 4, 4);
        uint32_t final32[8], mix[8];
        hashLight(cache_.data(), numCacheItems_, l1_.data(), numItems2048_,
                  height, hdr, sol.nonce, final32, mix);
        uint64_t hit[4];
        hashToLimbs(final32, hit);
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        if (!sol.extra.empty() && sol.extra != toHex(mix)) return false;
        return underTarget(hit, job.target);
    }

    void release() override { releaseDag(); }

    ~KawPowVK() override {
        releaseDag();
        destroyBuffer(&outBuf_);
        destroyBuffer(&cntBuf_);
        destroyPipeline(&dagPipe_);
        destroyPipeline(&mixPipe_);
        if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        if (dev_) vkDestroyDevice(dev_, nullptr);
        if (inst_) vkDestroyInstance(inst_, nullptr);
    }

   private:
    static std::string toHex(const uint32_t w[8]) {
        static const char *d = "0123456789abcdef";
        std::string s;
        s.reserve(64);
        for (int i = 0; i < 8; i++)
            for (int b = 0; b < 4; b++) {
                const uint8_t v = (uint8_t)(w[i] >> (b * 8));
                s += d[v >> 4];
                s += d[v & 15];
            }
        return s;
    }

    static VkMemoryPropertyFlags hostVisible() {
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    static VkMemoryPropertyFlags deviceLocal() {
        return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
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
        if (vkCreateBuffer(dev_, &bci, nullptr, &b->buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b->buf, &req);
        const uint32_t type = findMemType(req.memoryTypeBits, props);
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev_, &mai, nullptr, &b->mem) != VK_SUCCESS)
            return false;
        if (vkBindBufferMemory(dev_, b->buf, b->mem, 0) != VK_SUCCESS) return false;
        b->size = size;
        return true;
    }

    void destroyBuffer(Buffer *b) {
        if (b->buf) vkDestroyBuffer(dev_, b->buf, nullptr);
        if (b->mem) vkFreeMemory(dev_, b->mem, nullptr);
        b->buf = VK_NULL_HANDLE;
        b->mem = VK_NULL_HANDLE;
        b->size = 0;
    }

    bool upload(Buffer *b, const void *src, size_t bytes) {
        void *p = nullptr;
        if (vkMapMemory(dev_, b->mem, 0, bytes, 0, &p) != VK_SUCCESS) return false;
        memcpy(p, src, bytes);
        vkUnmapMemory(dev_, b->mem);
        return true;
    }

    bool download(Buffer *b, size_t offset, void *dst, size_t bytes) {
        void *p = nullptr;
        if (vkMapMemory(dev_, b->mem, offset, bytes, 0, &p) != VK_SUCCESS)
            return false;
        memcpy(dst, p, bytes);
        vkUnmapMemory(dev_, b->mem);
        return true;
    }

    bool copyBuffer(const Buffer &src, const Buffer &dst, VkDeviceSize bytes) {
        VkCommandBuffer cb = beginCommands();
        if (!cb) return false;
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &region);
        return endCommands(cb);
    }

    bool makePipeline(const uint32_t *code, size_t words, uint32_t nBindings,
                      uint32_t pushBytes, const std::vector<uint32_t> &specIds,
                      const std::vector<uint32_t> &specVals, Pipeline *out) {
        out->pushBytes = pushBytes;
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = words * 4;
        smci.pCode = code;
        if (vkCreateShaderModule(dev_, &smci, nullptr, &out->module) != VK_SUCCESS)
            return false;

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
        if (vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &out->setLayout) !=
            VK_SUCCESS)
            return false;

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = pushBytes;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &out->setLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev_, &plci, nullptr, &out->layout) !=
            VK_SUCCESS)
            return false;

        // Workgroup size is always a specialization constant so the host
        // dispatch and the shader can never disagree - that drifted once in
        // this repo and silently ran half the work while crediting all of it.
        // The mix pipeline adds 131 more: the period's whole program.
        std::vector<uint32_t> ids{0};
        std::vector<uint32_t> vals{kLocalSize};
        if (!specIds.empty()) {
            ids.push_back(1);
            vals.push_back(kMaxSolutions);
            ids.insert(ids.end(), specIds.begin(), specIds.end());
            vals.insert(vals.end(), specVals.begin(), specVals.end());
        }
        std::vector<VkSpecializationMapEntry> sme(ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            sme[i] = {};
            sme[i].constantID = ids[i];
            sme[i].offset = (uint32_t)(i * sizeof(uint32_t));
            sme[i].size = sizeof(uint32_t);
        }
        VkSpecializationInfo si{};
        si.mapEntryCount = (uint32_t)sme.size();
        si.pMapEntries = sme.data();
        si.dataSize = vals.size() * sizeof(uint32_t);
        si.pData = vals.data();

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
        if (vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                     &out->pipe) != VK_SUCCESS)
            return false;

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = nBindings;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev_, &dpci, nullptr, &out->pool) != VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = out->pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &out->setLayout;
        return vkAllocateDescriptorSets(dev_, &dsai, &out->set) == VK_SUCCESS;
    }

    void destroyPipeline(Pipeline *p) {
        if (!dev_) return;
        if (p->pipe) vkDestroyPipeline(dev_, p->pipe, nullptr);
        if (p->layout) vkDestroyPipelineLayout(dev_, p->layout, nullptr);
        if (p->setLayout) vkDestroyDescriptorSetLayout(dev_, p->setLayout, nullptr);
        if (p->pool) vkDestroyDescriptorPool(dev_, p->pool, nullptr);
        if (p->module) vkDestroyShaderModule(dev_, p->module, nullptr);
        *p = Pipeline{};
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

    VkCommandBuffer beginCommands() {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(dev_, &cbai, &cb) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
        return cb;
    }

    bool endCommands(VkCommandBuffer cb) {
        if (vkEndCommandBuffer(cb) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(dev_, &fci, nullptr, &fence) != VK_SUCCESS) return false;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        bool ok = vkQueueSubmit(queue_, 1, &si, fence) == VK_SUCCESS;
        if (ok)
            ok = vkWaitForFences(dev_, 1, &fence, VK_TRUE,
                                 300ull * 1000000000ull) == VK_SUCCESS;
        vkDestroyFence(dev_, fence, nullptr);
        vkFreeCommandBuffers(dev_, cmdPool_, 1, &cb);
        return ok;
    }

    bool dispatch(const Pipeline &p, const void *push, uint32_t groups) {
        VkCommandBuffer cb = beginCommands();
        if (!cb) return false;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                                1, &p.set, 0, nullptr);
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           p.pushBytes, push);
        vkCmdDispatch(cb, groups, 1, 1);
        return endCommands(cb);
    }

    bool ensureScratch() {
        if (outBuf_.buf == VK_NULL_HANDLE &&
            !createBuffer((VkDeviceSize)kMaxSolutions * kSolutionWords * 4,
                          hostVisible(), &outBuf_))
            return false;
        if (cntBuf_.buf == VK_NULL_HANDLE &&
            !createBuffer(4, hostVisible(), &cntBuf_))
            return false;
        return true;
    }

    /**
     * Rebuild the mix pipeline with this period's program folded in.
     *
     * The program is 131 specialisation constants, which the driver resolves
     * at pipeline-creation time. That is what turns mix[dst] from a dynamic
     * index into a local array - the GLSL trap that costs this repository a
     * 75x slowdown - into a register. The cost is one shader compile every
     * three blocks, which on Ravencoin is about three minutes.
     *
     * The value order here must match scripts/gen_kawpow_spec.py's table
     * order, which is also src/algos/progpow/progpow.h's declaration order.
     */
    bool makeMixPipeline(int height) {
        const om::progpow::Program prog =
            om::progpow::buildProgram(height, params());
        std::vector<uint32_t> vals;
        vals.reserve(kProgramWords);
        auto put = [&vals](const uint32_t *a, int n) {
            for (int i = 0; i < n; i++) vals.push_back(a[i]);
        };
        put(prog.cacheSrc, 11);
        put(prog.cacheDst, 11);
        put(prog.cacheSel, 11);
        put(prog.mathSrc1, 18);
        put(prog.mathSrc2, 18);
        put(prog.mathSel1, 18);
        put(prog.mathDst, 18);
        put(prog.mathSel2, 18);
        put(prog.dagDst, 4);
        put(prog.dagSel, 4);
        if (vals.size() != kProgramWords) return false;

        std::vector<uint32_t> ids(vals.size());
        for (size_t i = 0; i < ids.size(); i++) ids[i] = (uint32_t)(10 + i);

        // The old pipeline is finished with: every dispatch waits on its own
        // fence before returning, so nothing is in flight here.
        destroyPipeline(&mixPipe_);
        if (!makePipeline(kKawpowSpecSpirv, kKawpowSpecSpirvWords,
                          2 + kMaxDagBufs, sizeof(MixPush), ids, vals, &mixPipe_))
            return false;

        std::vector<Buffer *> bufs;
        for (uint32_t i = 0; i < kMaxDagBufs; i++) bufs.push_back(&dag_[i]);
        bufs.push_back(&outBuf_);
        bufs.push_back(&cntBuf_);
        writeDescriptors(mixPipe_, bufs);
        return true;
    }

    /** One allocation, six buffers bound into it at offsets. */
    bool allocDagContiguous(VkDeviceSize bytesPerBuf) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytesPerBuf;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkMemoryRequirements req{};
        {
            VkBuffer probe = VK_NULL_HANDLE;
            if (vkCreateBuffer(dev_, &bci, nullptr, &probe) != VK_SUCCESS)
                return false;
            vkGetBufferMemoryRequirements(dev_, probe, &req);
            vkDestroyBuffer(dev_, probe, nullptr);
        }
        // Round the stride up to the required alignment; 1 GiB already is.
        VkDeviceSize stride = bytesPerBuf;
        if (req.alignment && (stride % req.alignment))
            stride += req.alignment - (stride % req.alignment);

        const uint32_t type = findMemType(req.memoryTypeBits, deviceLocal());
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = stride * dagBufs_;
        mai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev_, &mai, nullptr, &dagMem_) != VK_SUCCESS) {
            fprintf(stderr, "[kawpow] one-allocation DAG of %.2f GB refused\n",
                    (double)mai.allocationSize / 1e9);
            dagMem_ = VK_NULL_HANDLE;
            return false;
        }
        for (uint32_t i = 0; i < dagBufs_; i++) {
            if (vkCreateBuffer(dev_, &bci, nullptr, &dag_[i].buf) != VK_SUCCESS)
                return false;
            if (vkBindBufferMemory(dev_, dag_[i].buf, dagMem_, stride * i) !=
                VK_SUCCESS)
                return false;
            dag_[i].mem = VK_NULL_HANDLE;   // owned by dagMem_, not per-buffer
            dag_[i].size = bytesPerBuf;
        }
        return true;
    }

    void releaseDag() {
        // The unused bindings alias buffer 0, so only the real ones are freed.
        for (uint32_t i = 0; i < dagBufs_ && i < kMaxDagBufs; i++)
            destroyBuffer(&dag_[i]);
        if (dagMem_) {
            vkFreeMemory(dev_, dagMem_, nullptr);
            dagMem_ = VK_NULL_HANDLE;
        }
        for (uint32_t i = 0; i < kMaxDagBufs; i++) dag_[i] = Buffer{};
        destroyBuffer(&cacheBuf_);
        dagBufs_ = 0;
        builtEpoch_ = -1;
        builtPeriod_ = -1;
        cache_.clear();
        cache_.shrink_to_fit();
        l1_.clear();
    }

    static const uint32_t kProgramWords = 11 * 3 + 18 * 5 + 4 * 2;

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    Pipeline dagPipe_, mixPipe_;
    Buffer cacheBuf_, dag_[kMaxDagBufs], outBuf_, cntBuf_;
    VkDeviceMemory dagMem_ = VK_NULL_HANDLE;   // set only in one-alloc mode
    uint32_t dagBufs_ = 0;
    uint32_t wordsPerBuf_ = 0;
    uint32_t wordsPerBufShift_ = 0;

    std::vector<uint32_t> cache_, l1_;
    int numCacheItems_ = 0;
    uint32_t numItems2048_ = 0;
    int builtEpoch_ = -1;
    int builtPeriod_ = -1;

    char devName_[256] = "unknown";
    char driver_[64] = "";
    VkDeviceSize deviceLocalBytes_ = 0;
    uint32_t maxStorageRange_ = 0;
};

}  // namespace

Algorithm *makeKawpowVK(int deviceIndex) {
    auto *a = new KawPowVK();
    if (!a->init(deviceIndex)) {
        delete a;
        return nullptr;
    }
    vkSetDeviceInfo(a->deviceName(), a->deviceMemGB(), a->driverVersion());
    return a;
}

}  // namespace om
