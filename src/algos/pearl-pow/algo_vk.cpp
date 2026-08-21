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

#include "../../core/pearl_gateway.h"   // unpackPearlExtra
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
extern const uint32_t kPearlDpSpirv[];
extern const size_t kPearlDpSpirvWords;

namespace om {
namespace {

// THE SEAM FOR RDNA2.
//
// kernel_dp.comp is wF:p7's, landed and byte-identical to the CUDA reference
// on llvmpipe (product 0/65536 wrong, transcripts 0/4096 wrong). Only the GEMM
// forks; nothing else in this file differs between the two paths.
//
// The contract that makes that true is in scratchpad/handoffs/pearl-vulkan.md:
// same four bindings, same 20-byte push block, same one-workgroup-per-tile
// dispatch, and the same transcript index - which is block-major over
// rank-sized blocks and then (hi, wi) inside each, NOT row-major over tiles.
#define SOAT_PEARL_HAVE_DP_GEMM 1
constexpr bool kHaveDotProductGemm = SOAT_PEARL_HAVE_DP_GEMM != 0;

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
        // THE FORK POINT. Everything else in this file is shared between the
        // two paths; only the GEMM pipeline and the device features differ.
        //
        // Selected on CAPABILITY, never on vendor or card name. The naive
        // check lies on a real box in this fleet: llvmpipe advertises
        // cooperative matrix at every level including an int8 configuration,
        // next to an RX 6700 XT that has none. vkInt8CooperativeMatrix()
        // attributes per device and requires the M16 N16 sint8 x sint8 ->
        // sint32 shape this GEMM is written against.
        haveCoopmat_ = vkInt8CooperativeMatrix(inst_, phys_, &coopK_);

        if (!haveCoopmat_ && !kHaveDotProductGemm) {
            // NAME THE DEVICE. A box can have three Vulkan devices - an RX
            // 6700 XT, a Ryzen iGPU and llvmpipe were all enumerated on the
            // machine this was checked on - and "this device" tells the user
            // nothing about which one was tried. Fetched here rather than
            // relying on devName_, which is not filled in until below.
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(phys_, &p);
            fprintf(stderr,
                    "pearl-vk: %s has no int8 cooperative matrix, which "
                    "Pearl's GEMM requires.\n"
                    "  Pearl needs an NVIDIA card, or an AMD card from RDNA3 "
                    "onwards (RX 7000 series). RDNA2 and older would need a "
                    "separate dot-product shader, which does not exist.\n"
                    "  Other algorithms still work on this card: try "
                    "--algo sha3-256t or --algo autolykos2.\n",
                    p.deviceName);
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
        //
        // ASKED FOR ONLY WHEN THE DEVICE HAS IT. Requesting it on a card that
        // does not - an RX 6700 XT, say - makes vkCreateDevice fail outright
        // with VK_ERROR_FEATURE_NOT_PRESENT, so the dot-product path would die
        // at device creation before it ever reached its own shader.
        const char *devExts[] = {"VK_KHR_cooperative_matrix"};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmF{};
        cmF.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        cmF.cooperativeMatrix = VK_TRUE;
        VkPhysicalDeviceVulkan13Features v13{};
        v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        v13.pNext = haveCoopmat_ ? (void *)&cmF : nullptr;
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
        dci.enabledExtensionCount = haveCoopmat_ ? 1u : 0u;
        dci.ppEnabledExtensionNames = haveCoopmat_ ? devExts : nullptr;
        VKCHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, queueFamily_, 0, &queue_);

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queueFamily_;
        VKCHECK(vkCreateCommandPool(dev_, &cpi, nullptr, &cmdPool_));

        // Four pipelines, four call sites. The GEMM takes both specialisation
        // constants because the two vendors disagree on both: subgroup 32 and
        // K 32 on Ada, 64 and 16 on RDNA3.
        // The one pipeline that forks. Ada wants subgroup 32 / K 32, RDNA3
        // subgroup 64 / K 16, and the dot-product variant has no K at all -
        // so it takes one specialisation constant where the coopmat one takes
        // two.
        const uint32_t gemmSpec[2] = {subgroup_, coopK_};
        if (haveCoopmat_) {
            if (!makePipe(kPearlGemmSpirv, kPearlGemmSpirvWords, 4, 20, gemmSpec,
                          2, &gemm_))
                return false;
        } else if (!makeDotProductGemm(gemmSpec[0])) {
            return false;
        }
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
                "[pearl-vk] %s, %s, subgroup %u, 10 of 10 stages present\n",
                devName_, gemmVariant(), subgroup_);
        return true;
    }

    bool prepare(const Job &job) override {
        if (!allocated_ && !allocate()) return false;
        if (checked_) return true;
        if (!selfCheck(job)) {
            fprintf(stderr,
                    "[pearl-vk] REFUSING TO MINE: this GPU or driver does not "
                    "compute the chain correctly. Every share would be "
                    "rejected.\n");
            return false;
        }
        checked_ = true;
        return true;
    }

    /**
     * Prove the whole chain on THIS card before mining anything.
     *
     * The nine device gates prove each stage against the host in isolation.
     * That is not the same as proving they are wired together correctly: every
     * stage can be right and the driver still feed one the wrong buffer, and
     * the result is a miner that runs at full speed and is rejected by every
     * pool. So this runs one real attempt and checks the device's own
     * intermediate values against job.h, stage by stage.
     *
     * It exists for the same reason algo.cu's selfCheck does: offline testing
     * can only ever cover the cards we own.
     */
    bool selfCheck(const Job &job) {
        uint8_t header[76];
        int cert = 0;
        if (!unpackPearlExtra(job.extra, header, &cert)) {
            fprintf(stderr, "[pearl-vk] self-check needs a Pearl job\n");
            return false;
        }
        if (!beginJob(job, header, cert)) return false;
        if (!runAttempt(0)) return false;

        const size_t aBytes = (size_t)shape_.m * kK;
        const size_t bBytes = (size_t)shape_.n * kK;
        bool ok = true;
        auto check = [&](const char *what, bool good) {
            if (!good) {
                fprintf(stderr, "[pearl-vk] self-check FAILED: %s\n", what);
                ok = false;
            }
            return good;
        };

        // 1. genMatrix, against fillMatrix - the matrices ARE the nonce, so a
        //    divergence here is not a slow miner, it is every share rejected.
        std::vector<int8_t> a(aBytes), want(aBytes);
        if (!download(buf_.a, aBytes, a.data())) return false;
        pearl::fillMatrix(want.data(), aBytes, pearl::matrixSeed(0 ^ jobSalt_, false));
        check("A does not match fillMatrix", memcmp(a.data(), want.data(), aBytes) == 0);

        // 2. the Merkle root of that same A, against job.h's blake3.
        uint8_t rootA[32], wantRoot[32];
        if (!download(buf_.rootA, 32, rootA)) return false;
        pearl::b3::hash(jobKey_, (const uint8_t *)a.data(), aBytes, wantRoot);
        check("A's Merkle root does not match blake3",
              memcmp(rootA, wantRoot, 32) == 0);

        // 3. the commitment chain, from the device's own two roots.
        uint8_t rootBt[32], commit[64], wantA[32], wantB[32];
        if (!download(buf_.rootBt, 32, rootBt)) return false;
        if (!download(buf_.commitOut, 64, commit)) return false;
        pearl::commitments(rootA, rootBt, jobKey_, shape_.m, shape_.n,
                           cert_ >= 3, wantA, wantB);
        check("commitA does not match job.h", memcmp(commit, wantA, 32) == 0);
        check("commitB does not match job.h", memcmp(commit + 32, wantB, 32) == 0);

        // 4. the GEMM and its transcript fold, from the device's own noised
        //    matrices - so this isolates the multiply from the noising above.
        std::vector<int8_t> an(aBytes), bn(bBytes);
        if (!download(buf_.an, aBytes, an.data())) return false;
        if (!download(buf_.bn, bBytes, bn.data())) return false;
        std::vector<uint32_t> got(16);
        if (!download(buf_.transcripts, 16 * 4, got.data())) return false;
        uint32_t wantT[pearl::kTranscriptWords];
        pearl::tileTranscript(an.data(), bn.data(), kK, shape_.n, kRank,
                              /*tRow=*/0, /*tCol=*/0, kTileSide, kTileSide,
                              wantT);
        check("the tile 0 transcript does not match job.h",
              memcmp(got.data(), wantT, sizeof(wantT)) == 0);

        if (ok)
            fprintf(stderr,
                    "[pearl-vk] self-check passed: matrix, Merkle root, "
                    "commitment chain and transcript all match job.h\n");
        return ok;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!allocated_) return false;

        uint8_t header[76];
        int cert = 0;
        if (!unpackPearlExtra(job.extra, header, &cert)) {
            fprintf(stderr, "[pearl-vk] job carries no Pearl header\n");
            return false;
        }
        if (!beginJob(job, header, cert)) return false;

        // `count` is a candidate budget and one attempt yields a whole tile
        // grid, so always at least one attempt or a small batch setting spins
        // without doing any work.
        const uint32_t perAttempt = tiles();
        uint64_t attempts = count / perAttempt;
        if (attempts == 0) attempts = 1;

        // Only this call's wins matter to the verify() that follows it, and
        // keeping older ones risks an old entry shadowing a repeated nonce.
        wins_.clear();

        for (uint64_t i = 0; i < attempts; i++) {
            const uint64_t nonce = nonceBase + i;
            if (!runAttempt(nonce)) return false;

            uint32_t hits = 0;
            if (!download(buf_.hitCount, 4, &hits)) return false;
            if (hits == 0) continue;
            if (hits > kMaxHits) hits = kMaxHits;

            // A and B are overwritten by the next attempt, so a win has to be
            // opened now.
            if (!openWin(job, nonce, hits, out)) return false;
        }
        return true;
    }

    /**
     * Invert the transcript index, open the tile, and build the submission.
     *
     * Ported from algo.cu deliberately unchanged, including the index
     * inversion: the transcript layout is NOT row-major over tiles. It is
     * block-major over rank-sized blocks and then (hi, wi) inside each, to
     * match what the Python reference emits. Getting this wrong opens the
     * wrong sixteen rows, which verifies locally against nothing and is
     * rejected by the node with the same sentence as every other failure.
     */
    bool openWin(const Job &job, uint64_t nonce, uint32_t hits,
                 std::vector<Solution> *out) {
        (void)job;
        std::vector<uint32_t> index(hits), digest(hits * 8);
        if (!download(buf_.hitIndex, hits * 4, index.data())) return false;
        if (!download(buf_.hitDigest, hits * 32, digest.data())) return false;

        const size_t aBytes = (size_t)shape_.m * kK;
        const size_t bBytes = (size_t)shape_.n * kK;
        std::vector<int8_t> A(aBytes), Bt(bBytes);
        if (!download(buf_.a, aBytes, A.data())) return false;
        if (!download(buf_.bt, bBytes, Bt.data())) return false;
        uint8_t commit[64];
        if (!download(buf_.commitOut, 64, commit)) return false;
        const uint8_t *commitA = commit, *commitB = commit + 32;

        // Only the first win is submitted. A second tile from the same attempt
        // is a second solution to the SAME block, so it can never be accepted
        // once the first is.
        // WHICH win, when an attempt has several: the LOWEST tile index, not
        // the first one the scan happened to record.
        //
        // powScan appends hits through an atomic counter, so the order is
        // whatever the hardware produced - it differs between two runs on one
        // card and between CUDA and Vulkan on the same job and nonce. Any of
        // them is a valid solution to the same block, so nothing was wrong;
        // but it made the miner's output nondeterministic, which means two
        // runs of one job cannot be diffed and the two backends cannot be
        // compared at all. Taking the minimum costs a scan of at most 64
        // entries, once per win, and makes the share a function of the job and
        // the nonce.
        uint32_t sel = 0;
        for (uint32_t i = 1; i < hits; i++)
            if (index[i] < index[sel]) sel = i;
        const uint32_t flat = index[sel];
        const uint32_t tilesPerSide = kRank / kTileSide;
        const uint32_t blocksPerRow = shape_.n / kRank;
        const uint32_t wi = flat % tilesPerSide;
        const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
        const uint32_t block = flat / (tilesPerSide * tilesPerSide);
        const uint32_t jIdx = block % blocksPerRow;
        const uint32_t iIdx = block / blocksPerRow;
        const uint32_t tRow = iIdx * kRank + hi * kTileSide;
        const uint32_t tCol = jIdx * kRank + wi * kTileSide;

        Win win;
        win.nonce = nonce;
        win.bound = bound_;
        win.verified = recheck(A, Bt, commitA, commitB, tRow, tCol, &win.digest);
        wins_.push_back(win);
        if (!win.verified) {
            // Reported, not submitted: the core prints it as a host rejection.
            Solution sol;
            sol.nonce = nonce;
            out->push_back(sol);
            return true;
        }

        pearl::MerkleProof aProof, btProof;
        if (!pearl::buildProof((const uint8_t *)A.data(), A.size(), jobKey_,
                               pearl::leafIndicesFromRows(tRow, kTileSide, kK),
                               &aProof) ||
            !pearl::buildProof((const uint8_t *)Bt.data(), Bt.size(), jobKey_,
                               pearl::leafIndicesFromRows(tCol, kTileSide, kK),
                               &btProof))
            return false;

        // The link nothing else can catch, and the reason this is not a
        // comment. recheck() and digestFromProof() both take the commitments
        // FROM THE DEVICE, so a wrong Merkle root makes the noise, transcript
        // and digest all wrong together and every local check agrees - they
        // are downstream of the same bad value. The pool is the first party to
        // recompute the root from the opened leaves, and all it can say when
        // it disagrees is "hash does not meet difficulty target".
        uint8_t devRootA[32], devRootBt[32];
        if (!download(buf_.rootA, 32, devRootA)) return false;
        if (!download(buf_.rootBt, 32, devRootBt)) return false;
        const bool aRootOk = memcmp(devRootA, aProof.root, 32) == 0;
        const bool bRootOk = memcmp(devRootBt, btProof.root, 32) == 0;
        if (!aRootOk || !bRootOk) {
            fprintf(stderr,
                    "[pearl-vk] the device's Merkle root for %s disagrees with "
                    "the host's over the same bytes - the commitments this proof "
                    "was mined under are not the ones it opens, so it is not "
                    "submitted. This is a GPU-side defect, not a pool problem.\n",
                    !aRootOk ? (bRootOk ? "A" : "A and B") : "B");
            return true;
        }

        // The second half of the same blind spot.
        uint8_t hostCommitA[32], hostCommitB[32];
        pearl::commitments(aProof.root, btProof.root, jobKey_, shape_.m,
                           shape_.n, cert_ >= 3, hostCommitA, hostCommitB);
        if (memcmp(hostCommitA, commitA, 32) != 0 ||
            memcmp(hostCommitB, commitB, 32) != 0) {
            fprintf(stderr,
                    "[pearl-vk] the device's commitments disagree with the "
                    "host's over the same Merkle roots - this proof is bound to "
                    "a key the pool will not derive, so it is not submitted. "
                    "This is a GPU-side defect, not a pool problem.\n");
            return true;
        }

        // The last thing that can still be wrong is the proof itself.
        // recheck() agreed with the device because it was handed the device's
        // own buffers; this reads back only what was serialised, which is what
        // the pool reconstructs from.
        pearl::U256 proofDigest;
        if (!pearl::digestFromProof(aProof, tRow, btProof, tCol, commitA,
                                    commitB, shape_.m, shape_.n, kK, kRank,
                                    kTileSide, &proofDigest)) {
            fprintf(stderr,
                    "[pearl-vk] the proof does not open the rows this tile "
                    "needs - not submitted\n");
            return true;
        }
        if (memcmp(proofDigest.v, win.digest.v, sizeof(proofDigest.v)) != 0) {
            fprintf(stderr,
                    "[pearl-vk] proof digest disagrees with the mined digest "
                    "at tile (%u,%u) - not submitted\n", tRow, tCol);
            return true;
        }

        Solution sol;
        sol.nonce = nonce;
        // The digest of the tile we SELECTED, not of hit 0. These were the
        // same expression while the selection was always hit 0, and taking the
        // lowest index without moving this made the miner report one tile's
        // digest for another tile's proof. verify() caught it immediately,
        // which is exactly what it is for.
        for (int i = 0; i < 4; i++)
            sol.hit[i] = (uint64_t)digest[sel * 8 + i * 2] |
                         ((uint64_t)digest[sel * 8 + i * 2 + 1] << 32);
        sol.extra = pearl::base64(pearl::encodePlainProof(
            shape_.m, shape_.n, kK, kRank, aProof, tRow, kTileSide, btProof,
            tCol, kTileSide));
        out->push_back(sol);
        return true;
    }

    /** The node's own check, on the host, from the two matrices. */
    bool recheck(const std::vector<int8_t> &A, const std::vector<int8_t> &Bt,
                 const uint8_t commitA[32], const uint8_t commitB[32],
                 uint32_t tRow, uint32_t tCol, pearl::U256 *digestOut) const {
        pearl::Noise noise;
        noise.generate(commitA, commitB, shape_.m, shape_.n, kK, kRank);

        // Only the winning strips, not the whole product.
        std::vector<int8_t> aStrip((size_t)kTileSide * kK);
        std::vector<int8_t> bStrip((size_t)kK * kTileSide);
        for (int i = 0; i < kTileSide; i++)
            for (size_t col = 0; col < kK; col++) {
                const int32_t e =
                    (int32_t)noise.eAL[(size_t)(tRow + i) * kRank + noise.ar.first[col]] -
                    (int32_t)noise.eAL[(size_t)(tRow + i) * kRank + noise.ar.second[col]];
                aStrip[(size_t)i * kK + col] =
                    (int8_t)(A[(size_t)(tRow + i) * kK + col] + e);
            }
        for (size_t p = 0; p < kK; p++)
            for (int j = 0; j < kTileSide; j++) {
                const int32_t e =
                    (int32_t)noise.eBR[(size_t)noise.bl.first[p] * shape_.n + tCol + j] -
                    (int32_t)noise.eBR[(size_t)noise.bl.second[p] * shape_.n + tCol + j];
                bStrip[p * kTileSide + j] =
                    (int8_t)(Bt[(size_t)(tCol + j) * kK + p] + e);
            }

        uint32_t transcript[pearl::kTranscriptWords];
        pearl::tileTranscript(aStrip.data(), bStrip.data(), kK, kTileSide, kRank,
                              0, 0, kTileSide, kTileSide, transcript);
        uint8_t d[32];
        pearl::powDigest(transcript, commitA, d);
        const pearl::U256 digest = pearl::U256::fromBytesLE(d);
        if (digestOut) *digestOut = digest;
        return digest.le(bound_);
    }

    bool verify(const Job &, const Solution &sol) const override {
        // Newest first: if a nonce ever repeats, the current attempt's answer
        // is the right one.
        for (auto it = wins_.rbegin(); it != wins_.rend(); ++it)
            if (it->nonce == sol.nonce)
                return it->verified &&
                       memcmp(sol.hit, it->digest.v, sizeof(sol.hit)) == 0 &&
                       pearl::U256::fromLimbs(sol.hit).le(it->bound);
        return false;
    }

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
        Buf eBRflat;                     // eBR before the transpose
        // noise.comp binds two outputs and writes only the first in uniform
        // mode. Binding eBRflat to both would be a buffer bound twice as
        // writeonly, which is the aliasing shape this file exists to avoid, so
        // the unused binding gets its own allocation - sized to what a bug
        // would write rather than to what a correct run does.
        Buf noiseUnused;
        Buf arF, arS, blF, blS;          // the two permutations, +1 and -1
        Buf transcripts;
        // kernel.comp binds the int32 product C and writes it only when
        // writeC != 0. Mining never does: the full C at the mining shape is
        // m*n*4 = 268 MB, which is why the transcript fold exists at all. This
        // is a placeholder so the binding is a real buffer rather than an
        // alias of something live. writeC MUST stay 0 outside the GEMM gate -
        // a nonzero one here writes out of bounds by design.
        Buf gemmC;
        Buf hitIndex, hitDigest, hitCount;
        Buf scratchA, scratchB;          // Merkle chaining values, ping-ponged
        Buf rootA, rootBt;               // one buffer each: a descriptor binds a
                                         // whole buffer, so two roots at two
                                         // offsets of one buffer cannot both be
                                         // bound as binding 2
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
            makeBuf((size_t)kRank * shape_.n, false, &buf_.eBRflat) &&
            makeBuf((size_t)kRank * shape_.n, false, &buf_.noiseUnused) &&
            makeBuf(kK * 2, false, &buf_.arF) && makeBuf(kK * 2, false, &buf_.arS) &&
            makeBuf(kK * 2, false, &buf_.blF) && makeBuf(kK * 2, false, &buf_.blS) &&
            makeBuf((size_t)tiles() * 16 * 4, false, &buf_.transcripts) &&
            makeBuf(4096, false, &buf_.gemmC) &&
            makeBuf(kMaxHits * 4, false, &buf_.hitIndex) &&
            makeBuf(kMaxHits * 32, false, &buf_.hitDigest) &&
            makeBuf(4, false, &buf_.hitCount) &&
            // The reduction ping-pongs between these two.
            makeBuf(chunks * 32, false, &buf_.scratchA) &&
            makeBuf(chunks * 32, false, &buf_.scratchB) &&
            makeBuf(32, false, &buf_.rootA) && makeBuf(32, false, &buf_.rootBt) &&
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


    // ------------------------------------------------------------- driver --

    /** blake3 Merkle root of `chunks` 1 KB leaves, recorded into cmd_.
     *
     * The reduction ping-pongs between two scratch buffers, so level L+1 reads
     * what level L wrote and writes what level L-1 read. The barrier between
     * levels therefore has to cover write-after-read as well as
     * read-after-write; barrier() is a full memory barrier for that reason.
     *
     * `per` and the group count are the merkle gate's, which were checked
     * against the CUDA's own `threads = 256; while (threads > 1 && 2*threads >
     * count/2) threads >>= 1;` at 4, 8, 1024 and 131072 chunks. Do not
     * re-derive them.
     */
    bool merkleRootInto(const Buf &data, uint32_t chunks, const Buf &key,
                        const Buf &rootOut) {
        if (!chunks) return false;
        if (!dispatch(merkleChunk_, 0, {&data, &key, &buf_.scratchA}, &chunks, 4,
                      (chunks + 255) / 256))
            return false;

        uint32_t count = chunks;
        int src = 0;
        uint32_t round = 0;
        while (count > 2) {
            uint32_t per = count / 2;
            if (per > 512) per = 512;
            const uint32_t groups = count / per;
            const Buf &in = src ? buf_.scratchB : buf_.scratchA;
            const Buf &out = src ? buf_.scratchA : buf_.scratchB;
            if (round >= kMaxReduceRounds) {
                fprintf(stderr, "[pearl-vk] merkle tree deeper than %u rounds\n",
                        kMaxReduceRounds);
                return false;
            }
            if (!dispatch(merkleReduce_, round, {&in, &key, &out}, &per, 4, groups))
                return false;
            count = groups;
            src ^= 1;
            round++;
        }
        const Buf &last = src ? buf_.scratchB : buf_.scratchA;
        return dispatch(merkleRoot_, 0, {&last, &key, &rootOut}, nullptr, 0, 1);
    }

    struct GenPush { uint64_t seed; uint32_t count; };
    struct CommitPush { uint32_t m, n, salted; };
    struct NoisePush { uint32_t mode, count, mask, shift, rank; };
    struct ApplyPush { uint32_t mode, rows, k, rank; };
    struct TransposePush { uint32_t mode, n, inner; };
    struct ScanPush { uint32_t count, maxHits; };

    /** Copy one device buffer into another at an offset, inside cmd_. */
    void copyInto(const Buf &src, const Buf &dst, VkDeviceSize dstOffset,
                  VkDeviceSize bytes, VkDeviceSize srcOffset = 0) {
        VkBufferCopy c{};
        c.srcOffset = srcOffset;
        c.dstOffset = dstOffset;
        c.size = bytes;
        vkCmdCopyBuffer(cmd_, src.buf, dst.buf, 1, &c);
    }

    /** Everything that depends on the job but not on the nonce.
     *
     * B is rolled once per job, so a long run does not reuse the same columns
     * forever, and its whole noise chain is built here rather than per attempt.
     */
    bool beginJob(const Job &job, const uint8_t header[76], int cert) {
        if (haveJob_ && memcmp(header, header_, 76) == 0 && cert == cert_ &&
            memcmp(job.target, target_, sizeof(target_)) == 0)
            return true;

        memcpy(header_, header, 76);
        memcpy(target_, job.target, sizeof(target_));
        cert_ = cert;
        cfg_.commonDim = kK;
        cfg_.rank = kRank;

        // The dot-product GEMM's shared tiles are sized from a compile-time
        // MAX_RANK. Refuse up front rather than write past them - a job whose
        // rank exceeds it must never reach a dispatch.
        if (!haveCoopmat_ && cfg_.rank > (int)kDpMaxRank) {
            fprintf(stderr,
                    "[pearl-vk] this job wants rank %d and the dot-product "
                    "GEMM is built for at most %u. Refusing rather than "
                    "writing past its shared tiles.\n",
                    (int)cfg_.rank, kDpMaxRank);
            return false;
        }

        const std::string bad = cfg_.check(shape_.m, shape_.n);
        if (!bad.empty()) {
            fprintf(stderr, "[pearl-vk] illegal mining configuration: %s\n",
                    bad.c_str());
            return false;
        }
        // The same single scaling the CUDA path does. See algo.cu's comment and
        // scratchpad/handoffs/pearl.md: a proof built against the unscaled
        // target is one the pool will never ask for.
        if (!cfg_.penalizedTarget(pearl::U256::fromLimbs(job.target), &bound_)) {
            fprintf(stderr,
                    "[pearl-vk] this target does not scale into 256 bits for "
                    "the chosen configuration\n");
            return false;
        }
        if (getenv("SOAT_PEARL_HARD_TARGET")) {
            bound_ = pearl::U256{};
            fprintf(stderr, "[pearl-vk] benchmark hard target enabled (no shares)\n");
        }

        pearl::jobKey(header_, cfg_, jobKey_);
        memcpy(&jobSalt_, jobKey_, sizeof(jobSalt_));
        pearl::seedForA(seedA_);
        pearl::seedForB(seedB_);
        bound_.toBytesLE(targetBytes_);
        btSeed_ = pearl::matrixSeed(jobSalt_, true);

        if (!beginCmd()) return false;

        // The static inputs, once per job.
        if (!upload(jobKey_, 32, buf_.jobKey)) return false;
        if (!upload(targetBytes_, 32, buf_.target)) return false;

        // commitIn is aRoot | btRoot | jobKey | saltA | saltB. The two salts
        // and the job key never change within a job.
        if (!upload(pearl::seedSaltA(), 32, buf_.commitIn, 96)) return false;
        if (!upload(pearl::seedSaltB(), 32, buf_.commitIn, 128)) return false;
        copyInto(buf_.jobKey, buf_.commitIn, 64, 32);
        // The seeds sit beside their commitments in the noise keys.
        if (!upload(seedA_, 32, buf_.noiseKeyA, 32)) return false;
        if (!upload(seedB_, 32, buf_.noiseKeyB, 32)) return false;
        barrier();

        // B, and its Merkle root.
        const size_t bBytes = (size_t)shape_.n * kK;
        GenPush gp{btSeed_, (uint32_t)bBytes};
        if (!dispatch(genMatrix_, 1, {&buf_.bt}, &gp, sizeof(gp),
                      (uint32_t)((bBytes / 8 + 255) / 256)))
            return false;
        if (!merkleRootInto(buf_.bt, (uint32_t)(bBytes / pearl::kChunkLen),
                            buf_.jobKey, buf_.rootBt))
            return false;
        copyInto(buf_.rootBt, buf_.commitIn, 32, 32);
        barrier();

        // commitB, which depends only on the job key and B's root. This shader
        // derives commitA and commitB together, and commitA is garbage here
        // because aRoot is not written until an attempt runs - deliberately
        // unused. Three blake3 compressions; not worth a second pipeline.
        CommitPush cp{shape_.m, shape_.n, (uint32_t)(cert_ >= 3)};
        if (!dispatch(commitments_, 1, {&buf_.commitIn, &buf_.commitOut}, &cp,
                      sizeof(cp), 1))
            return false;
        // noiseKeyB = commitB | seedB.
        copyInto(buf_.commitOut, buf_.noiseKeyB, 0, 32, 32);   // commitB: words 8-15
        barrier();

        // B's noise: the uniform field over n*rank, transposed into eBR, and
        // the +1/-1 permutation pair over k.
        const uint32_t brCount = shape_.n * kRank;
        NoisePush un{0, brCount, (uint32_t)kNoiseMask, (uint32_t)kNoiseShift, 0};
        if (!dispatch(noise_, 0, {&buf_.noiseKeyB, &buf_.eBRflat, &buf_.noiseUnused},
                      &un, sizeof(un), ((brCount + 31) / 32 + 255) / 256))
            return false;
        TransposePush tp{1, shape_.n, (uint32_t)kRank};
        if (!dispatch(transpose_, 0, {&buf_.eBRflat, &buf_.eBR}, &tp, sizeof(tp),
                      (brCount + 255) / 256))
            return false;
        NoisePush pm{1, kK, 0, 0, (uint32_t)kRank};
        if (!dispatch(noise_, 1, {&buf_.noiseKeyB, &buf_.blF, &buf_.blS}, &pm,
                      sizeof(pm), ((kK + 7) / 8 + 255) / 256))
            return false;

        // Only the layout the GEMM reads. kernel.comp takes B k-major, which is
        // applyNoise mode 2; mode 1 (Bt) exists for the CUDA's n-major kernels
        // and building both would cost a second full-size buffer nothing reads.
        const uint32_t bElems = (uint32_t)bBytes;
        ApplyPush ab{2, shape_.n, kK, (uint32_t)kRank};
        if (!dispatch(applyNoise_, 1,
                      {&buf_.bt, &buf_.eBR, &buf_.blF, &buf_.blS, &buf_.bn}, &ab,
                      sizeof(ab), (bElems + 255) / 256))
            return false;

        if (!submitAndWait()) return false;
        haveJob_ = true;
        wins_.clear();
        return true;
    }

    /** One nonce: build A, noise it, multiply, scan.
     *
     * The order matches algo.cu exactly, because the chain is what the network
     * verifies. A's commitment covers B's, so commitments runs after A's root
     * is known and produces both again - three compressions, not worth a
     * second pipeline that would only ever compute half of it.
     */
    bool runAttempt(uint64_t nonce) {
        const size_t aBytes = (size_t)shape_.m * kK;
        if (!beginCmd()) return false;

        const uint32_t zero = 0;
        if (!upload(&zero, 4, buf_.hitCount)) return false;
        barrier();

        GenPush gp{pearl::matrixSeed(nonce ^ jobSalt_, false), (uint32_t)aBytes};
        if (!dispatch(genMatrix_, 0, {&buf_.a}, &gp, sizeof(gp),
                      (uint32_t)((aBytes / 8 + 255) / 256)))
            return false;
        if (!merkleRootInto(buf_.a, (uint32_t)(aBytes / pearl::kChunkLen),
                            buf_.jobKey, buf_.rootA))
            return false;
        copyInto(buf_.rootA, buf_.commitIn, 0, 32);
        barrier();

        CommitPush cp{shape_.m, shape_.n, (uint32_t)(cert_ >= 3)};
        if (!dispatch(commitments_, 0, {&buf_.commitIn, &buf_.commitOut}, &cp,
                      sizeof(cp), 1))
            return false;
        copyInto(buf_.commitOut, buf_.noiseKeyA, 0, 32);   // commitA: words 0-7
        barrier();

        const uint32_t alCount = shape_.m * kRank;
        NoisePush un{0, alCount, (uint32_t)kNoiseMask, (uint32_t)kNoiseShift, 0};
        if (!dispatch(noise_, 2, {&buf_.noiseKeyA, &buf_.eAL, &buf_.noiseUnused},
                      &un, sizeof(un), ((alCount + 31) / 32 + 255) / 256))
            return false;
        NoisePush pm{1, kK, 0, 0, (uint32_t)kRank};
        if (!dispatch(noise_, 3, {&buf_.noiseKeyA, &buf_.arF, &buf_.arS}, &pm,
                      sizeof(pm), ((kK + 7) / 8 + 255) / 256))
            return false;
        ApplyPush aa{0, shape_.m, kK, (uint32_t)kRank};
        if (!dispatch(applyNoise_, 0,
                      {&buf_.a, &buf_.eAL, &buf_.arF, &buf_.arS, &buf_.an}, &aa,
                      sizeof(aa), (uint32_t)((aBytes + 255) / 256)))
            return false;

        // The GEMM. Its push block is m, n, k, rank and a tile grid of
        // (n/16) x (m/16) candidates; see kernel.comp.
        struct GemmPush { int32_t m, n, k, rank, writeC; };
        GemmPush gem{(int32_t)shape_.m, (int32_t)shape_.n, (int32_t)kK, kRank, 0};
        if (!dispatch(gemm_, 0,
                      {&buf_.an, &buf_.bn, &buf_.gemmC, &buf_.transcripts}, &gem,
                      sizeof(gem), tiles()))
            return false;

        ScanPush sp{tiles(), kMaxHits};
        if (!dispatch(powScan_, 0,
                      {&buf_.transcripts, &buf_.commitOut, &buf_.target,
                       &buf_.hitIndex, &buf_.hitDigest, &buf_.hitCount},
                      &sp, sizeof(sp), (tiles() + 255) / 256))
            return false;

        return submitAndWait();
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
        stageUsed_ = 0;
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

    /** Host bytes into a device-local buffer, through the staging map.
     *
     * EACH UPLOAD IN A COMMAND BUFFER NEEDS ITS OWN STAGING REGION. The copies
     * are recorded now and run at submit, so two uploads writing the same
     * staging bytes both deliver whatever the second one wrote. `stageUsed_`
     * hands out a fresh region per upload and beginCmd() resets it; the
     * alternative - a copy per submission - would be four extra round trips
     * per job for 32 bytes each. */
    bool upload(const void *src, size_t bytes, const Buf &dst,
                VkDeviceSize dstOffset = 0) {
        const VkDeviceSize at = stageUsed_;
        if (at + bytes > buf_.stage.size) {
            fprintf(stderr,
                    "[pearl-vk] staging is %llu bytes; %llu used and %zu more "
                    "wanted\n", (unsigned long long)buf_.stage.size,
                    (unsigned long long)at, bytes);
            return false;
        }
        memcpy((uint8_t *)buf_.stage.mapped + at, src, bytes);
        VkBufferCopy c{};
        c.srcOffset = at;
        c.dstOffset = dstOffset;
        c.size = bytes;
        vkCmdCopyBuffer(cmd_, buf_.stage.buf, dst.buf, 1, &c);
        // 256 keeps every region aligned for any copy the driver may widen.
        stageUsed_ = (at + bytes + 255) & ~(VkDeviceSize)255;
        // A transfer write needs a barrier before ANYTHING reads it, including
        // another transfer. Dropping this while adding staging regions was the
        // first self-check failure: commitIn's job key was copied out of a
        // buffer whose upload had not necessarily landed, so the commitment
        // chain came out wrong while the matrix and its Merkle root - which do
        // not read an uploaded buffer - were both correct.
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

    /** Build the GEMM from the dot-product shader, for a device with no
     *  cooperative matrix. One specialisation constant (the subgroup size);
     *  the coopmat K is meaningless here. */
    bool makeDotProductGemm(uint32_t subgroupSize) {
#if SOAT_PEARL_HAVE_DP_GEMM
        // TWO specialisation constants, not one. constant_id 1 is K_DIM in
        // kernel.comp and coopmat-only; kernel_dp.comp reuses the slot for
        // MAX_RANK, which sizes its two shared staging tiles. Shared memory
        // cannot be sized at dispatch time and `rank` arrives in a push
        // constant, so it has to be a compile-time bound - and a rank above it
        // writes past the arrays rather than failing.
        static_assert(kRank <= (int)kDpMaxRank,
                      "kernel_dp.comp sizes its shared tiles from MAX_RANK; a "
                      "larger rank writes past them");
        const uint32_t spec[2] = {subgroupSize, kDpMaxRank};
        return makePipe(kPearlDpSpirv, kPearlDpSpirvWords, 4, 20, spec, 2, &gemm_);
#else
        (void)subgroupSize;
        fprintf(stderr, "[pearl-vk] no dot-product GEMM in this build\n");
        return false;
#endif
    }

    /** What the banner should say about which GEMM is running. Two paths must
     *  never be indistinguishable in a log: a user reporting a wrong share has
     *  to be able to tell us which one produced it. */
    const char *gemmVariant() const {
        static char buf[48];
        if (haveCoopmat_)
            snprintf(buf, sizeof(buf), "int8 coopmat M16 N16 K%u", coopK_);
        else
            snprintf(buf, sizeof(buf), "int8 dot-product GEMM (no coopmat)");
        return buf;
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

    // MAX_RANK for kernel_dp.comp. Not a tuning knob: it sizes shared memory,
    // and a rank above it is an out-of-bounds write rather than a slow path.
    // 128 covers the rank-128 and rank-64 vector sets.
    static constexpr uint32_t kDpMaxRank = 128;

    struct Win {
        uint64_t nonce = 0;
        bool verified = false;
        pearl::U256 bound;
        pearl::U256 digest;
    };

    Shape shape_ = kShape;
    bool allocated_ = false;
    bool checked_ = false;
    bool haveJob_ = false;
    uint8_t header_[76] = {};
    int cert_ = 3;
    pearl::MiningConfig cfg_;
    pearl::U256 bound_;
    uint8_t jobKey_[32] = {};
    uint64_t target_[4] = {};
    uint8_t seedA_[32] = {}, seedB_[32] = {}, targetBytes_[32] = {};
    uint8_t commitA_[32] = {}, commitB_[32] = {};
    uint64_t jobSalt_ = 0, btSeed_ = 0;
    std::vector<Win> wins_;
    std::vector<Buf> owned_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkDeviceSize stageUsed_ = 0;

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = UINT32_MAX;
    uint32_t coopK_ = 0;
    bool haveCoopmat_ = false;
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
