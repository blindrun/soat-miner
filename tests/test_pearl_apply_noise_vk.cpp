// Adding the noise to A and B on Vulkan, against job.h's Noise.
//
// usage: test_pearl_apply_noise_vk <apply_noise.spv> [device_index]
//
// Host side builds the noised matrices from job.h's Noise, the same structure
// recheck() uses when it decides whether a solution is real.
//
// All three modes compute the same element value and differ only in how they
// walk it. B READS transposed and writes k-major while Bt does neither, so
// swapping them gives a full buffer of plausible numbers in the wrong order.

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "vk_claim_guard.h"
#include "vk_spv.h"

#include "../src/algos/pearl-pow/job.h"

using namespace om::pearl;

static int gPass = 0, gFail = 0, gChecks = 0;

static void check(const char *name, bool ok, const char *detail = "") {
    gChecks++;
    if (ok) { gPass++; printf("  [ok] %s %s\n", name, detail); }
    else { gFail++; printf("  [FAIL] %s %s\n", name, detail); }
}

#define VKREQ(x)                                                             \
    do {                                                                     \
        VkResult r_ = (x);                                                   \
        if (r_ != VK_SUCCESS) {                                              \
            fprintf(stderr, "%s failed (%d) at line %d\n", #x, r_, __LINE__); \
            return 2;                                                        \
        }                                                                    \
    } while (0)

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits,
                        VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

int main(int argc, char **argv) {
    const char *spvPath = argc > 1 ? argv[1] : "build/genmatrix.spv";
    const int wantDev = argc > 2 ? atoi(argv[2]) : 0;

    std::vector<uint32_t> spv;
    if (!om::loadSpirvWords(spvPath, &spv)) return 2;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VKREQ(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> pds(nd);
    vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    if (wantDev >= (int)nd) { fprintf(stderr, "no device %d\n", wantDev); return 2; }
    VkPhysicalDevice pd = pds[wantDev];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("device: %s\n", props.deviceName);
    om::requireGpuClaim(props.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfs.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 2; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderBufferInt64Atomics = VK_FALSE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderInt64 = VK_TRUE;
    f2.pNext = &v12;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VKREQ(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    // Device-local output with a host-visible staging readback. Host-visible
    // for the data buffer would work and would measure the bus if this ever
    // grows a timing mode, which is how another lane lost 27x.
    const size_t kMaxBytes = 1u << 20;
    auto mkBuf = [&](VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props_, VkBuffer *b,
                     VkDeviceMemory *m) -> VkResult {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size; bci.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bci, nullptr, b);
        if (r != VK_SUCCESS) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memType(pd, mr.memoryTypeBits, props_);
        r = vkAllocateMemory(dev, &mai, nullptr, m);
        if (r != VK_SUCCESS) return r;
        return vkBindBufferMemory(dev, *b, *m, 0);
    };
    VkBuffer srcB, eB, fB, sB, outB, stageB;
    VkDeviceMemory srcM, eM, fM, sM, outM, stageM;
    const VkBufferUsageFlags kIn = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKREQ(mkBuf(kMaxBytes, kIn, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &srcB, &srcM));
    VKREQ(mkBuf(kMaxBytes, kIn, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &eB, &eM));
    VKREQ(mkBuf(kMaxBytes, kIn, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &fB, &fM));
    VKREQ(mkBuf(kMaxBytes, kIn, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &sB, &sM));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outB, &outM));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stageB, &stageM));
    void *staged = nullptr;
    VKREQ(vkMapMemory(dev, stageM, 0, kMaxBytes, 0, &staged));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4; smci.pCode = spv.data();
    VkShaderModule sm; VKREQ(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkDescriptorSetLayoutBinding bind[5]{};
    for (int i = 0; i < 5; i++) {
        bind[i].binding = i; bind[i].descriptorCount = 1;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 5; dslci.pBindings = bind;
    VkDescriptorSetLayout dsl; VKREQ(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    struct Push { uint32_t mode, rows, k, rank; } push{};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VKREQ(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT; st.module = sm; st.pName = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = st; cpci.layout = pl;
    VkPipeline pipe; VKREQ(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VKREQ(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VKREQ(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[5] = {{srcB, 0, VK_WHOLE_SIZE}, {eB, 0, VK_WHOLE_SIZE},
                                     {fB, 0, VK_WHOLE_SIZE},   {sB, 0, VK_WHOLE_SIZE},
                                     {outB, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[5]{};
    for (int i = 0; i < 5; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 5, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cp; VKREQ(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VKREQ(vkAllocateCommandBuffers(dev, &cbai, &cb));

    auto upload = [&](VkBuffer dst, const void *data, size_t bytes) {
        memcpy(staged, data, bytes);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy c{0, 0, bytes};
        vkCmdCopyBuffer(cb, stageB, dst, 1, &c);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
    };

    auto run = [&](uint32_t mode, uint32_t rows, uint32_t k, uint32_t rank,
                   std::vector<int8_t> *got) {
        const uint32_t total = rows * k;
        push.mode = mode; push.rows = rows; push.k = k; push.rank = rank;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cb, (total + 255) / 256, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy d{0, 0, total};
        vkCmdCopyBuffer(cb, outB, stageB, 1, &d);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        got->assign((const int8_t *)staged, (const int8_t *)staged + total);
    };

    // One noise field, shared by every mode, built the way the miner builds it.
    const uint32_t m = 256, n = 512, k = 1024;
    const int rank = 128;
    uint8_t commitA[32], commitB[32];
    for (int i = 0; i < 32; i++) { commitA[i] = (uint8_t)(i * 5 + 1); commitB[i] = (uint8_t)(i * 9 + 2); }
    Noise noise;
    noise.generate(commitA, commitB, m, n, k, rank);

    // The real generator, not a synthetic fill: fillMatrix is what genmatrix
    // .comp reproduces, so this gate consumes the same bytes the miner will.
    std::vector<int8_t> A((size_t)m * k), Bt((size_t)n * k);
    fillMatrix(A.data(), A.size(), 0x51ED0021ull);
    fillMatrix(Bt.data(), Bt.size(), 0x51ED0037ull);

    printf("1. applyNoiseA\n");
    {
        upload(srcB, A.data(), A.size());
        upload(eB, noise.eAL.data(), noise.eAL.size());
        upload(fB, noise.ar.first.data(), noise.ar.first.size() * 2);
        upload(sB, noise.ar.second.data(), noise.ar.second.size() * 2);
        std::vector<int8_t> got, want((size_t)m * k);
        for (uint32_t row = 0; row < m; row++)
            for (uint32_t col = 0; col < k; col++) {
                const int e = (int)noise.eAL[(size_t)row * rank + noise.ar.first[col]] -
                              (int)noise.eAL[(size_t)row * rank + noise.ar.second[col]];
                want[(size_t)row * k + col] = (int8_t)(A[(size_t)row * k + col] + e);
            }
        run(0, m, k, (uint32_t)rank, &got);
        check("A_noised matches the host", got == want);
    }

    printf("2. applyNoiseBt, k-major in and out\n");
    {
        upload(srcB, Bt.data(), Bt.size());
        upload(eB, noise.eBR.data(), noise.eBR.size());
        upload(fB, noise.bl.first.data(), noise.bl.first.size() * 2);
        upload(sB, noise.bl.second.data(), noise.bl.second.size() * 2);
        std::vector<int8_t> got, want((size_t)n * k);
        for (uint32_t j = 0; j < n; j++)
            for (uint32_t p = 0; p < k; p++) {
                const int e = (int)noise.eBR[(size_t)noise.bl.first[p] * n + j] -
                              (int)noise.eBR[(size_t)noise.bl.second[p] * n + j];
                want[(size_t)j * k + p] = (int8_t)(Bt[(size_t)j * k + p] + e);
            }
        run(1, n, k, (uint32_t)rank, &got);
        check("Bt_noised matches the host", got == want);
    }

    printf("3. applyNoiseB, which reads transposed\n");
    {
        std::vector<int8_t> got, want((size_t)k * n);
        for (uint32_t p = 0; p < k; p++)
            for (uint32_t j = 0; j < n; j++) {
                const int e = (int)noise.eBR[(size_t)noise.bl.first[p] * n + j] -
                              (int)noise.eBR[(size_t)noise.bl.second[p] * n + j];
                want[(size_t)p * n + j] = (int8_t)(Bt[(size_t)j * k + p] + e);
            }
        run(2, n, k, (uint32_t)rank, &got);
        check("B_noised matches the host", got == want);
    }

    printf("4. B and Bt are not the same buffer\n");
    {
        std::vector<int8_t> b, bt;
        run(1, n, k, (uint32_t)rank, &bt);
        run(2, n, k, (uint32_t)rank, &b);
        check("the two B modes produce different arrangements", b != bt);
    }

    printf("5. the gate can fail\n");
    {
        std::vector<int8_t> got;
        run(0, m, k, (uint32_t)rank, &got);
        check("noised A is not raw A", got != A);
    }

    if (gChecks == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, gChecks, gFail);
    return gFail ? 1 : 0;
}
