// Pearl's commitment chain on Vulkan, against job.h's commitments().
//
// usage: test_pearl_commitments_vk <commitments.spv> [device_index]
//
// Unlike the transpose gate, the host side here is NOT a restatement: it is
// job.h's commitments(), the same function the CUDA path and the proof builder
// use, which pearl_oracle.py has checked against Pearl's own Rust verifier. So
// agreeing with it means agreeing with the network.
//
// The chain is ordered: commitB binds the job key to B's root, commitA binds
// commitB to A's root, so A's commitment transitively covers B. Getting the
// order wrong produces two plausible 32-byte values the network rejects.

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
    VkBuffer inB, outB, stageB; VkDeviceMemory inM, outM, stageM;
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &inB, &inM));
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

    VkDescriptorSetLayoutBinding bind[2]{};
    for (int i = 0; i < 2; i++) {
        bind[i].binding = i; bind[i].descriptorCount = 1;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = bind;
    VkDescriptorSetLayout dsl; VKREQ(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    struct Push { uint32_t m, n, salted; } push{};
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

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VKREQ(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VKREQ(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[2] = {{inB, 0, VK_WHOLE_SIZE}, {outB, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[2]{};
    for (int i = 0; i < 2; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cp; VKREQ(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VKREQ(vkAllocateCommandBuffers(dev, &cbai, &cb));

    auto run = [&](const uint8_t aRoot[32], const uint8_t btRoot[32],
                   const uint8_t key[32], uint32_t m, uint32_t n, bool salted,
                   uint8_t gotA[32], uint8_t gotB[32]) {
        uint8_t in[160];
        memcpy(in +   0, aRoot, 32);
        memcpy(in +  32, btRoot, 32);
        memcpy(in +  64, key, 32);
        memcpy(in +  96, seedSaltA(), 32);
        memcpy(in + 128, seedSaltB(), 32);
        memcpy(staged, in, sizeof(in));
        push.m = m; push.n = n; push.salted = salted ? 1u : 0u;

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy up{0, 0, sizeof(in)};
        vkCmdCopyBuffer(cb, stageB, inB, 1, &up);
        VkMemoryBarrier m1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        m1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m1, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cb, 1, 1, 1);
        VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m2, 0, nullptr, 0, nullptr);
        VkBufferCopy down{0, 0, 64};
        vkCmdCopyBuffer(cb, outB, stageB, 1, &down);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        memcpy(gotA, staged, 32);
        memcpy(gotB, (const uint8_t *)staged + 32, 32);
    };

    auto synth32 = [](uint8_t *out, int salt) {
        for (int i = 0; i < 32; i++) out[i] = (uint8_t)(i * 37 + salt * 11 + 3);
    };

    printf("1. against job.h's commitments(), salted\n");
    for (auto dims : {std::pair<uint32_t,uint32_t>{128, 128},
                      {256, 4096}, {4096, 65536}}) {
        uint8_t aRoot[32], btRoot[32], key[32];
        synth32(aRoot, 1); synth32(btRoot, 2); synth32(key, 3);
        uint8_t gotA[32], gotB[32], wantA[32], wantB[32];
        run(aRoot, btRoot, key, dims.first, dims.second, true, gotA, gotB);
        commitments(aRoot, btRoot, key, dims.first, dims.second, true, wantA, wantB);
        char d[64]; snprintf(d, sizeof(d), "- m=%u n=%u", dims.first, dims.second);
        check("commitment A matches", memcmp(gotA, wantA, 32) == 0, d);
        check("commitment B matches", memcmp(gotB, wantB, 32) == 0, d);
    }

    printf("2. the unsalted path, which is a different derivation\n");
    {
        uint8_t aRoot[32], btRoot[32], key[32];
        synth32(aRoot, 4); synth32(btRoot, 5); synth32(key, 6);
        uint8_t gotA[32], gotB[32], wantA[32], wantB[32];
        run(aRoot, btRoot, key, 256, 256, false, gotA, gotB);
        commitments(aRoot, btRoot, key, 256, 256, false, wantA, wantB);
        check("unsalted A matches", memcmp(gotA, wantA, 32) == 0);
        check("unsalted B matches", memcmp(gotB, wantB, 32) == 0);
        uint8_t sA[32], sB[32];
        run(aRoot, btRoot, key, 256, 256, true, sA, sB);
        check("and salted differs from unsalted", memcmp(gotA, sA, 32) != 0);
    }

    printf("3. the dimensions are bound in, not decoration\n");
    {
        uint8_t aRoot[32], btRoot[32], key[32];
        synth32(aRoot, 7); synth32(btRoot, 8); synth32(key, 9);
        uint8_t a1[32], b1[32], a2[32], b2[32];
        run(aRoot, btRoot, key, 128, 256, true, a1, b1);
        run(aRoot, btRoot, key, 256, 128, true, a2, b2);
        check("swapping m and n changes both commitments",
              memcmp(a1, a2, 32) != 0 && memcmp(b1, b2, 32) != 0);
    }

    printf("4. the gate can fail\n");
    {
        uint8_t aRoot[32], btRoot[32], key[32];
        synth32(aRoot, 10); synth32(btRoot, 11); synth32(key, 12);
        uint8_t gotA[32], gotB[32], wantA[32], wantB[32];
        run(aRoot, btRoot, key, 512, 512, true, gotA, gotB);
        synth32(key, 13);   // a different job key
        commitments(aRoot, btRoot, key, 512, 512, true, wantA, wantB);
        check("a different job key does NOT match", memcmp(gotA, wantA, 32) != 0);
    }

    if (gChecks == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, gChecks, gFail);
    return gFail ? 1 : 0;
}
