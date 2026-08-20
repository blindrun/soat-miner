// Pearl NoisyGEMM on Vulkan.
//
// SKELETON. It stands up the device and every pipeline we have a shader for,
// and prepare() then refuses honestly, naming what is still missing. It is
// here now rather than at the end because the cost of Vulkan pipeline plumbing
// was the piece this project was mis-costed on twice, and the only way to stop
// estimating it was to write it.
//
// The registry line in vk_registry.cpp stays commented until prepare() really
// works: registering early makes `--algo pearl-pow` on the Vulkan binary start
// and then fail, which is worse for a user than the current clear "Pearl is
// CUDA only" message.
//
// What exists, and what does not:
//
// Every shader Pearl needs now exists and every one is byte-identical to the
// CUDA reference on a real card of each vendor - an RTX 4090 (Ada, coopmat
// K32, subgroup 32) and an RX 7900 XT (RDNA3, K16, subgroup 64):
//
//   noisy GEMM + transcript fold   kernel.comp          byte-identical
//   blake3 chunk CVs               merkle_chunk.comp    15/15
//   blake3 tree reduction          merkle_reduce.comp   15/15
//   blake3 root                    merkle_root.comp     15/15
//   matrix generation              genmatrix.comp       8/8
//   transposes                     transpose.comp       9/9
//   commitment derivation          commitments.comp     11/11
//   noise draws                    noise.comp           10/10
//   noising                        apply_noise.comp     5/5
//   powScan                        powscan.comp         13/13
//
// What is left is the driver loop: allocating for a job, running the stages in
// order, and reading the hits back. Ten stages standing up is not ten stages
// wired together, and this file does not yet claim to be the latter.
//
// Buffers are DEVICE_LOCAL. A host-visible allocation on a discrete GPU
// usually lands in system RAM across PCIe, and a kernel streaming a large
// buffer through it measures the bus - that cost another lane 27x on the
// blake3 harness before it was found. Staging is the only host-visible
// allocation here, which is what staging is for.

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "../../core/vk_common.h"
#include "job.h"

extern const uint32_t kPearlGemmSpirv[];
extern const size_t kPearlGemmSpirvWords;
extern const uint32_t kPearlMerkleChunkSpirv[];
extern const size_t kPearlMerkleChunkSpirvWords;
extern const uint32_t kPearlMerkleReduceSpirv[];
extern const size_t kPearlMerkleReduceSpirvWords;
extern const uint32_t kPearlMerkleRootSpirv[];
extern const size_t kPearlMerkleRootSpirvWords;
extern const uint32_t kPearlGenMatrixSpirv[];
extern const size_t kPearlGenMatrixSpirvWords;
extern const uint32_t kPearlTransposeSpirv[];
extern const size_t kPearlTransposeSpirvWords;
extern const uint32_t kPearlCommitmentsSpirv[];
extern const size_t kPearlCommitmentsSpirvWords;
extern const uint32_t kPearlNoiseSpirv[];
extern const size_t kPearlNoiseSpirvWords;
extern const uint32_t kPearlApplyNoiseSpirv[];
extern const size_t kPearlApplyNoiseSpirvWords;
extern const uint32_t kPearlPowScanSpirv[];
extern const size_t kPearlPowScanSpirvWords;

namespace om {
namespace {

#define VKCHECK(x)                                                          \
    do {                                                                    \
        VkResult r_ = (x);                                                  \
        if (r_ != VK_SUCCESS) {                                             \
            fprintf(stderr, "pearl-vk: %s failed (%d) at %s:%d\n", #x, r_,  \
                    __FILE__, __LINE__);                                    \
            return false;                                                   \
        }                                                                   \
    } while (0)

/**
 * One compute pipeline and everything it owns.
 *
 * Pearl needs about nine of these where sha3-256t needed one, and the
 * unfactored cost is roughly seventy lines each. That is the whole reason this
 * struct and makePipe() exist: without them the file is six hundred lines of
 * near-identical plumbing and every future algorithm pays it again.
 */
struct Pipe {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // ONE SET PER CALL SITE, not one per pipeline.
    //
    // Several stages run more than once per attempt against different buffers:
    // genMatrix builds A and B, noise draws both fields, applyNoise runs for A
    // and for B, and the Merkle reduction ping-pongs for as many rounds as the
    // tree is deep. A descriptor set records which buffers a dispatch reads,
    // and rewriting one that an already-recorded dispatch still refers to is
    // undefined - in practice every dispatch in the batch silently reads the
    // last binding written. That produces a miner that runs at full speed and
    // computes the wrong thing, which is this project's defining failure mode.
    // Call sites take a fixed index instead, so no set is ever rewritten
    // inside a submission.
    std::vector<VkDescriptorSet> sets;
    uint32_t bindings = 0;
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;     // staging only; everything else is device-local
};

class PearlPowVK : public Algorithm {
   public:
    const char *name() const override { return "pearl-pow"; }

    size_t memoryBytes(const Job &) const override { return 0; }

    /** A candidate is a 16x16 tile whose every element is a length-k dot
     *  product. The cooperative-matrix K is deliberately absent: a tile takes
     *  k/K instructions, so it cancels. */
    double macsPerUnit() const override { return 16.0 * 16.0 * 2048.0; }

    bool init(int requestedIndex) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "soat-miner";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VKCHECK(vkCreateInstance(&ici, nullptr, &inst_));

        if (!vkPickPhysicalDevice(inst_, requestedIndex, &phys_)) {
            fprintf(stderr, "pearl-vk: no usable Vulkan device\n");
            return false;
        }

        // Extension list, then feature bit, then the configuration list.
        // vk_common owns that order because the obvious one is wrong: an
        // RX 6700 XT enumerates int8 configurations and supports none of them.
        if (!vkInt8CooperativeMatrix(inst_, phys_, &coopK_)) {
            fprintf(stderr,
                    "pearl-vk: this device has no int8 cooperative matrix, "
                    "which Pearl's GEMM requires. RDNA2 needs a separate "
                    "dot-product shader that does not exist yet.\n");
            return false;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        VkPhysicalDeviceSubgroupProperties sgp{};
        sgp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &sgp;
        vkGetPhysicalDeviceProperties2(phys_, &p2);
        subgroup_ = sgp.subgroupSize;
        snprintf(devName_, sizeof(devName_), "%s", props.deviceName);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, qfs.data());
        queueFamily_ = UINT32_MAX;
        for (uint32_t i = 0; i < nq; i++)
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; break; }
        if (queueFamily_ == UINT32_MAX) {
            fprintf(stderr, "pearl-vk: no compute queue\n");
            return false;
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        // Cooperative matrix has to be asked for at device creation, not just
        // queried. Without both the extension and the feature chained in here
        // the GEMM pipeline compiles and then produces nothing.
        const char *devExts[] = {"VK_KHR_cooperative_matrix"};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmF{};
        cmF.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        cmF.cooperativeMatrix = VK_TRUE;
        VkPhysicalDeviceVulkan13Features v13{};
        v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        v13.pNext = &cmF;
        VkPhysicalDeviceVulkan12Features v12{};
        v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.shaderInt8 = VK_TRUE;
        v12.storageBuffer8BitAccess = VK_TRUE;
        v12.pNext = &v13;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &v12;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = devExts;
        VKCHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queueFamily_;
        VKCHECK(vkCreateCommandPool(dev_, &cpi, nullptr, &cmdPool_));

        // Four pipelines, four call sites. The GEMM takes both specialisation
        // constants because the two vendors disagree on both: subgroup 32 and
        // K 32 on Ada, 64 and 16 on RDNA3.
        const uint32_t gemmSpec[2] = {subgroup_, coopK_};
        if (!makePipe(kPearlGemmSpirv, kPearlGemmSpirvWords, 4, 20, gemmSpec, 2,
                      &gemm_))
            return false;
        if (!makePipe(kPearlMerkleChunkSpirv, kPearlMerkleChunkSpirvWords, 3, 4,
                      nullptr, 0, &merkleChunk_, 2))
            return false;
        if (!makePipe(kPearlMerkleReduceSpirv, kPearlMerkleReduceSpirvWords, 3, 4,
                      nullptr, 0, &merkleReduce_, kMaxReduceRounds))
            return false;
        if (!makePipe(kPearlMerkleRootSpirv, kPearlMerkleRootSpirvWords, 3, 4,
                      nullptr, 0, &merkleRoot_, 2))
            return false;

        // The remaining six. Binding counts and push-constant sizes come from
        // the shaders themselves - see each .comp's layout block - because a
        // mismatch here is not a compile error: the pipeline builds and the
        // dispatch reads whatever happens to be at that offset.
        if (!makePipe(kPearlGenMatrixSpirv, kPearlGenMatrixSpirvWords, 1, 16,
                      nullptr, 0, &genMatrix_, 2))       // seed(u64) + count
            return false;
        if (!makePipe(kPearlTransposeSpirv, kPearlTransposeSpirvWords, 2, 12,
                      nullptr, 0, &transpose_, 2))       // mode, n, inner
            return false;
        if (!makePipe(kPearlCommitmentsSpirv, kPearlCommitmentsSpirvWords, 2, 12,
                      nullptr, 0, &commitments_, 2))     // m, n, salted
            return false;
        if (!makePipe(kPearlNoiseSpirv, kPearlNoiseSpirvWords, 3, 20,
                      nullptr, 0, &noise_, 4))           // mode, count, mask, shift, rank
            return false;
        if (!makePipe(kPearlApplyNoiseSpirv, kPearlApplyNoiseSpirvWords, 5, 16,
                      nullptr, 0, &applyNoise_, 2))      // mode, rows, k, rank
            return false;
        if (!makePipe(kPearlPowScanSpirv, kPearlPowScanSpirvWords, 6, 8,
                      nullptr, 0, &powScan_))         // count, maxHits
            return false;

        fprintf(stderr,
                "[pearl-vk] %s, int8 coopmat M16 N16 K%u, subgroup %u, "
                "10 of 10 stages present\n",
                devName_, coopK_, subgroup_);
        return true;
    }

    bool prepare(const Job &) override {
        // Honest refusal, naming what is missing rather than failing opaquely
        // or - worse - mining something wrong. Every stage below has a CUDA
        // kernel and no GLSL.
        fprintf(stderr,
                "[pearl-vk] Pearl's Vulkan backend is incomplete. All ten "
                "shader stages exist and each is byte-identical to the CUDA "
                "reference on both an Ada and an RDNA3 card, but the driver "
                "loop that runs them in order for a job is not written yet. "
                "Use the CUDA binary for Pearl.\n");
        return false;
    }

    bool search(const Job &, uint64_t, uint64_t, std::vector<Solution> *) override {
        return false;
    }

    bool verify(const Job &, const Solution &) const override { return false; }

    void release() override {
        for (Pipe *p : {&gemm_, &merkleChunk_, &merkleReduce_, &merkleRoot_,
                        &genMatrix_, &transpose_, &commitments_, &noise_,
                        &applyNoise_, &powScan_})
            destroyPipe(p);
        if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        if (dev_) vkDestroyDevice(dev_, nullptr);
        if (inst_) vkDestroyInstance(inst_, nullptr);
        dev_ = VK_NULL_HANDLE;
        inst_ = VK_NULL_HANDLE;
    }

    const char *deviceName() const { return devName_; }
    double deviceMemGB() const { return 0.0; }
    const char *driverVersion() const { return "vulkan"; }

   private:
    uint32_t tiles() const {
        return (shape_.m / kTileSide) * (shape_.n / kTileSide);
    }

    // ------------------------------------------------------------ buffers --
    //
    // EVERY ROLE GETS ITS OWN BUFFER. The first draft of this aliased three
    // different uses onto one scratch buffer to save memory, and that is the
    // shape of bug this project exists to avoid: it runs, it is fast, and the
    // shares are wrong. Twenty-odd small allocations cost nothing beside the
    // matrices.
    struct Bufs {
        Buf a, an;                       // A, and A with its noise
        Buf bt, bn, bnt;                 // B^t, noised B k-major, noised B^t
        Buf eAL, eBR;                    // the two uniform noise fields
        Buf arF, arS, blF, blS;          // the two permutations, +1 and -1
        Buf transcripts;
        Buf hitIndex, hitDigest, hitCount;
        Buf scratch;                     // Merkle chaining values, ping-ponged
        Buf roots;                       // aRoot | btRoot
        Buf commitIn, commitOut;         // packed inputs | commitA | commitB
        Buf noiseKeyA, noiseKeyB;        // commitA|seedA and commitB|seedB
        Buf jobKey, target;
        Buf stage;                       // the only host-visible allocation
    } buf_;

    bool makeBuf(VkDeviceSize size, bool hostVisible, Buf *out) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size ? size : 4;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VKCHECK(vkCreateBuffer(dev_, &bci, nullptr, &out->buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev_, out->buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        const VkMemoryPropertyFlags want =
            hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & want) == want) { type = i; break; }
        if (type == UINT32_MAX) {
            fprintf(stderr, "[pearl-vk] no %s memory type for %llu bytes\n",
                    hostVisible ? "host-visible" : "device-local",
                    (unsigned long long)bci.size);
            return false;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = type;
        VKCHECK(vkAllocateMemory(dev_, &mai, nullptr, &out->mem));
        VKCHECK(vkBindBufferMemory(dev_, out->buf, out->mem, 0));
        out->size = bci.size;
        if (hostVisible)
            VKCHECK(vkMapMemory(dev_, out->mem, 0, bci.size, 0, &out->mapped));
        owned_.push_back(*out);
        return true;
    }

    bool allocate() {
        if (allocated_) return true;
        const size_t aB = (size_t)shape_.m * kK;
        const size_t bB = (size_t)shape_.n * kK;
        const size_t chunks = (aB > bB ? aB : bB) / pearl::kChunkLen;

        const bool ok =
            makeBuf(aB, false, &buf_.a)   && makeBuf(aB, false, &buf_.an) &&
            makeBuf(bB, false, &buf_.bt)  && makeBuf(bB, false, &buf_.bn) &&
            makeBuf(bB, false, &buf_.bnt) &&
            makeBuf((size_t)shape_.m * kRank, false, &buf_.eAL) &&
            makeBuf((size_t)kRank * shape_.n, false, &buf_.eBR) &&
            makeBuf(kK * 2, false, &buf_.arF) && makeBuf(kK * 2, false, &buf_.arS) &&
            makeBuf(kK * 2, false, &buf_.blF) && makeBuf(kK * 2, false, &buf_.blS) &&
            makeBuf((size_t)tiles() * 16 * 4, false, &buf_.transcripts) &&
            makeBuf(kMaxHits * 4, false, &buf_.hitIndex) &&
            makeBuf(kMaxHits * 32, false, &buf_.hitDigest) &&
            makeBuf(4, false, &buf_.hitCount) &&
            // Two halves: the reduction ping-pongs between them.
            makeBuf(chunks * 32 * 2, false, &buf_.scratch) &&
            makeBuf(64, false, &buf_.roots) &&
            makeBuf(160, false, &buf_.commitIn) &&
            makeBuf(64, false, &buf_.commitOut) &&
            makeBuf(64, false, &buf_.noiseKeyA) &&
            makeBuf(64, false, &buf_.noiseKeyB) &&
            makeBuf(32, false, &buf_.jobKey) && makeBuf(32, false, &buf_.target) &&
            makeBuf(bB > aB ? bB : aB, true, &buf_.stage);
        if (!ok) return false;

        allocated_ = true;
        fprintf(stderr, "[pearl-vk] %ux%u k=%u, about %.0f MB device-local\n",
                shape_.m, shape_.n, kK,
                (double)(aB * 2 + bB * 3 + (size_t)tiles() * 64) / 1e6);
        return true;
    }


    /**
     * A whole pipeline in one call. Everything that differs between Pearl's
     * shaders is a parameter: the SPIR-V, how many storage buffers it binds,
     * its push-constant size, and its specialisation constants.
     */
    bool makePipe(const uint32_t *spv, size_t words, uint32_t bindings,
                  uint32_t pushSize, const uint32_t *spec, uint32_t specCount,
                  Pipe *out, uint32_t maxSets = 1) {
        out->bindings = bindings;
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = words * sizeof(uint32_t);
        smci.pCode = spv;
        VKCHECK(vkCreateShaderModule(dev_, &smci, nullptr, &out->shader));

        std::vector<VkDescriptorSetLayoutBinding> binds(bindings);
        for (uint32_t i = 0; i < bindings; i++) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = bindings;
        dslci.pBindings = binds.data();
        VKCHECK(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &out->setLayout));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = pushSize;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &out->setLayout;
        plci.pushConstantRangeCount = pushSize ? 1 : 0;
        plci.pPushConstantRanges = pushSize ? &pcr : nullptr;
        VKCHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &out->layout));

        std::vector<VkSpecializationMapEntry> entries(specCount);
        for (uint32_t i = 0; i < specCount; i++) {
            entries[i].constantID = i;
            entries[i].offset = i * sizeof(uint32_t);
            entries[i].size = sizeof(uint32_t);
        }
        VkSpecializationInfo si{};
        si.mapEntryCount = specCount;
        si.pMapEntries = entries.data();
        si.dataSize = specCount * sizeof(uint32_t);
        si.pData = spec;

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = out->shader;
        stage.pName = "main";
        if (specCount) stage.pSpecializationInfo = &si;

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = out->layout;
        VKCHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                         &out->pipeline));

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = bindings * maxSets;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = maxSets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VKCHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &out->pool));

        std::vector<VkDescriptorSetLayout> layouts(maxSets, out->setLayout);
        out->sets.resize(maxSets);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = out->pool;
        dsai.descriptorSetCount = maxSets;
        dsai.pSetLayouts = layouts.data();
        VKCHECK(vkAllocateDescriptorSets(dev_, &dsai, out->sets.data()));
        return true;
    }

    // ---------------------------------------------------- recording work --
    //
    // One command buffer per phase, every dispatch separated by a full
    // memory barrier. Pearl's stages are a strict chain - the GEMM reads what
    // applyNoise wrote, powScan reads what the GEMM wrote - so there is no
    // overlap to win and a missing barrier here is a wrong share, not a
    // visible failure. Correctness first; a per-stage barrier can be relaxed
    // later against a measurement, not against a guess.

    bool beginCmd() {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(dev_, &cbai, &cmd_));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd_, &bi));
        return true;
    }

    void barrier() {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    /** Bind one call site's buffers and dispatch it.
     *
     * `setIndex` names the call site, never the order of the call. Two call
     * sites sharing an index rewrite each other's bindings; see Pipe::sets. */
    bool dispatch(Pipe &pipe, uint32_t setIndex,
                  std::initializer_list<const Buf *> bufs,
                  const void *push, uint32_t pushSize, uint32_t groups) {
        if (setIndex >= pipe.sets.size()) {
            fprintf(stderr, "[pearl-vk] descriptor set %u out of range (%zu)\n",
                    setIndex, pipe.sets.size());
            return false;
        }
        if (bufs.size() != pipe.bindings) {
            fprintf(stderr,
                    "[pearl-vk] this stage binds %u buffers, %zu given\n",
                    pipe.bindings, bufs.size());
            return false;
        }
        if (!groups) return true;   // nothing to do is not an error

        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(bufs.size());
        writes.reserve(bufs.size());
        uint32_t i = 0;
        for (const Buf *b : bufs) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = b->buf;
            bi.offset = 0;
            bi.range = VK_WHOLE_SIZE;
            infos.push_back(bi);
            i++;
        }
        i = 0;
        for (const Buf *b : bufs) {
            (void)b;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = pipe.sets[setIndex];
            w.dstBinding = i;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &infos[i];
            writes.push_back(w);
            i++;
        }
        vkUpdateDescriptorSets(dev_, (uint32_t)writes.size(), writes.data(), 0,
                               nullptr);

        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.layout, 0, 1, &pipe.sets[setIndex], 0,
                                nullptr);
        if (pushSize)
            vkCmdPushConstants(cmd_, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               pushSize, push);
        vkCmdDispatch(cmd_, groups, 1, 1);
        barrier();
        return true;
    }

    /** Submit what has been recorded and wait for it.
     *
     * A fence rather than vkQueueWaitIdle: this queue is ours, but waiting on
     * the device hides which submission actually hung when one does. */
    bool submitAndWait() {
        VKCHECK(vkEndCommandBuffer(cmd_));
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        VKCHECK(vkCreateFence(dev_, &fci, nullptr, &fence));
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VkResult r = vkQueueSubmit(queue_, 1, &si, fence);
        if (r == VK_SUCCESS)
            r = vkWaitForFences(dev_, 1, &fence, VK_TRUE, 30ULL * 1000000000ULL);
        vkDestroyFence(dev_, fence, nullptr);
        vkFreeCommandBuffers(dev_, cmdPool_, 1, &cmd_);
        cmd_ = VK_NULL_HANDLE;
        if (r == VK_TIMEOUT) {
            fprintf(stderr, "[pearl-vk] a submission did not finish in 30s\n");
            return false;
        }
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[pearl-vk] submission failed (%d)\n", r);
            return false;
        }
        return true;
    }

    /** Host bytes into a device-local buffer, through the one staging map. */
    bool upload(const void *src, size_t bytes, const Buf &dst) {
        if (bytes > buf_.stage.size) {
            fprintf(stderr, "[pearl-vk] staging is %llu bytes, upload is %zu\n",
                    (unsigned long long)buf_.stage.size, bytes);
            return false;
        }
        memcpy(buf_.stage.mapped, src, bytes);
        VkBufferCopy c{};
        c.size = bytes;
        vkCmdCopyBuffer(cmd_, buf_.stage.buf, dst.buf, 1, &c);
        barrier();
        return true;
    }

    /** Device-local bytes back to the host. Records, submits and waits: a
     *  readback is a synchronisation point by nature. */
    bool download(const Buf &src, size_t bytes, void *dst, size_t srcOffset = 0) {
        if (bytes > buf_.stage.size) {
            fprintf(stderr, "[pearl-vk] staging is %llu bytes, download is %zu\n",
                    (unsigned long long)buf_.stage.size, bytes);
            return false;
        }
        if (!beginCmd()) return false;
        VkBufferCopy c{};
        c.srcOffset = srcOffset;
        c.size = bytes;
        vkCmdCopyBuffer(cmd_, src.buf, buf_.stage.buf, 1, &c);
        if (!submitAndWait()) return false;
        memcpy(dst, buf_.stage.mapped, bytes);
        return true;
    }

    void destroyPipe(Pipe *p) {
        if (!dev_) return;
        if (p->pipeline) vkDestroyPipeline(dev_, p->pipeline, nullptr);
        if (p->layout) vkDestroyPipelineLayout(dev_, p->layout, nullptr);
        if (p->setLayout) vkDestroyDescriptorSetLayout(dev_, p->setLayout, nullptr);
        if (p->pool) vkDestroyDescriptorPool(dev_, p->pool, nullptr);
        if (p->shader) vkDestroyShaderModule(dev_, p->shader, nullptr);
        *p = Pipe{};
    }

    // The mining shape. Only one for now: the CUDA path measures a tuner
    // sweep to pick between several, and copying that before the Vulkan miner
    // runs at all would be optimising something unproven. One shape that fits
    // every card in the fleet, chosen the same way the CUDA default was.
    struct Shape { uint32_t m, n; };
    static constexpr Shape kShape{4096, 16384};

    static constexpr uint32_t kK = 2048;
    static constexpr int kRank = 128;
    static constexpr int kNoiseMask = 63, kNoiseShift = 32;
    static constexpr uint32_t kMaxHits = 64;
    static constexpr int kTileSide = 16;

    // The blake3 tree halves each round, so the deepest job needs
    // log2(bytes/kChunkLen) reductions. n*k = 16384*2048 gives 32768 chunks,
    // which is 15. Sized with room rather than computed, because running out
    // of descriptor sets mid-reduction is a wrong root, not a crash.
    static constexpr uint32_t kMaxReduceRounds = 24;

    struct Win {
        uint64_t nonce = 0;
        bool verified = false;
        pearl::U256 bound;
        pearl::U256 digest;
    };

    Shape shape_ = kShape;
    bool allocated_ = false;
    bool haveJob_ = false;
    uint8_t header_[76] = {};
    int cert_ = 3;
    pearl::MiningConfig cfg_;
    pearl::U256 bound_;
    uint8_t jobKey_[32] = {};
    uint8_t commitA_[32] = {}, commitB_[32] = {};
    uint64_t jobSalt_ = 0, btSeed_ = 0;
    std::vector<Win> wins_;
    std::vector<Buf> owned_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
    uint32_t coopK_ = 0;
    uint32_t subgroup_ = 0;
    char devName_[256] = {0};
    Pipe gemm_, merkleChunk_, merkleReduce_, merkleRoot_;
    Pipe genMatrix_, transpose_, commitments_, noise_, applyNoise_, powScan_;
};

}  // namespace

Algorithm *makePearlPowVK(int deviceIndex) {
    auto *a = new PearlPowVK();
    if (!a->init(deviceIndex)) {
        delete a;
        return nullptr;
    }
    vkSetDeviceInfo(a->deviceName(), a->deviceMemGB(), a->driverVersion());
    return a;
}

}  // namespace om
