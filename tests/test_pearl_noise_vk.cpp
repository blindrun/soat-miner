// Pearl's two noise draws on Vulkan, against job.h.
//
// usage: test_pearl_noise_vk <noise.spv> [device_index]
//
// Host side is job.h's uniformNoise() and permutationIndices() themselves, so
// agreeing with them means agreeing with what the proof builder and the CUDA
// path use.
//
// The two draws differ in ONE byte of the hashed block - the counter sits at
// offset 0 for the uniform draw and offset 4 for the permutation. That is the
// entire domain separation. Get it wrong and you produce the other draw's
// noise, which looks perfectly uniform and is silently wrong.

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "vk_claim_guard.h"

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
    {
        FILE *f = fopen(spvPath, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", spvPath); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        spv.resize((sz + 3) / 4);
        if (fread(spv.data(), 1, sz, f) != (size_t)sz) { fclose(f); return 2; }
        fclose(f);
    }

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
    VkBuffer inB, outB, out1B, stageB; VkDeviceMemory inM, outM, out1M, stageM;
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &inB, &inM));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outB, &outM));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &out1B, &out1M));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stageB, &stageM));
    void *staged = nullptr;
    VKREQ(vkMapMemory(dev, stageM, 0, kMaxBytes, 0, &staged));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4; smci.pCode = spv.data();
    VkShaderModule sm; VKREQ(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkDescriptorSetLayoutBinding bind[3]{};
    for (int i = 0; i < 3; i++) {
        bind[i].binding = i; bind[i].descriptorCount = 1;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3; dslci.pBindings = bind;
    VkDescriptorSetLayout dsl; VKREQ(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    struct Push { uint32_t mode, count, mask, shift, rank; } push{};
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

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VKREQ(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VKREQ(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[3] = {{inB, 0, VK_WHOLE_SIZE},
                                     {outB, 0, VK_WHOLE_SIZE},
                                     {out1B, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cp; VKREQ(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VKREQ(vkAllocateCommandBuffers(dev, &cbai, &cb));

    auto run = [&](uint32_t mode, const uint8_t key[32], const uint8_t seed[32],
                   uint32_t count, uint32_t mask, uint32_t shift, uint32_t rank,
                   std::vector<uint8_t> *got0, std::vector<uint8_t> *got1) {
        uint8_t in[64];
        memcpy(in, key, 32);
        memcpy(in + 32, seed, 32);
        memcpy(staged, in, sizeof(in));
        push.mode = mode; push.count = count;
        push.mask = mask; push.shift = shift; push.rank = rank;

        // Bytes out: uniform writes `count` int8; perm writes `count` uint16
        // into each of two buffers.
        const uint32_t bytes0 = (mode == 0) ? ((count + 3) / 4) * 4
                                            : ((count + 1) / 2) * 4;

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
        const uint32_t per = (mode == 0) ? 32u : 8u;
        vkCmdDispatch(cb, ((count + per - 1) / per + 255) / 256, 1, 1);
        VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m2, 0, nullptr, 0, nullptr);
        VkBufferCopy d0{0, 0, bytes0};
        vkCmdCopyBuffer(cb, outB, stageB, 1, &d0);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        got0->assign((const uint8_t *)staged, (const uint8_t *)staged + bytes0);

        if (mode != 0) {
            vkBeginCommandBuffer(cb, &bi);
            VkBufferCopy d1{0, 0, bytes0};
            vkCmdCopyBuffer(cb, out1B, stageB, 1, &d1);
            vkEndCommandBuffer(cb);
            vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            got1->assign((const uint8_t *)staged, (const uint8_t *)staged + bytes0);
        }
    };

    uint8_t key[32], seed[32];
    for (int i = 0; i < 32; i++) { key[i] = (uint8_t)(i * 7 + 1); seed[i] = (uint8_t)(i * 13 + 5); }

    printf("1. the uniform draw against job.h's uniformNoise()\n");
    for (auto rc : {std::pair<uint32_t,int>{128, 128}, {4096, 128}, {256, 64}}) {
        const uint32_t rows = rc.first; const int rank = rc.second;
        const std::vector<int8_t> want = uniformNoise(key, seed, rows, rank);
        std::vector<uint8_t> got, unused;
        run(0, key, seed, (uint32_t)want.size(), 63, 32, (uint32_t)rank, &got, &unused);
        bool ok = got.size() >= want.size();
        for (size_t i = 0; ok && i < want.size(); i++)
            if ((int8_t)got[i] != want[i]) ok = false;
        char d[80]; snprintf(d, sizeof(d), "- rows=%u rank=%d, %zu values", rows, rank, want.size());
        check("uniform noise matches", ok, d);
    }

    printf("2. the permutation draw against job.h's permutationIndices()\n");
    for (auto cr : {std::pair<uint32_t,int>{2048, 128}, {512, 64}, {100, 128}}) {
        const uint32_t count = cr.first; const int rank = cr.second;
        const PermIndices want = permutationIndices(key, seed, count, rank);
        std::vector<uint8_t> gotF, gotS;
        run(1, key, seed, count, 0, 0, (uint32_t)rank, &gotF, &gotS);
        const uint16_t *f = (const uint16_t *)gotF.data();
        const uint16_t *sd = (const uint16_t *)gotS.data();
        bool ok = gotF.size() >= count * 2 && gotS.size() >= count * 2;
        for (uint32_t i = 0; ok && i < count; i++)
            if (f[i] != want.first[i] || sd[i] != want.second[i]) ok = false;
        char d[80]; snprintf(d, sizeof(d), "- count=%u rank=%d", count, rank);
        check("permutation indices match", ok, d);
    }

    printf("3. the +1/-1 invariant the chain depends on\n");
    {
        const uint32_t count = 2048;
        std::vector<uint8_t> gotF, gotS;
        run(1, key, seed, count, 0, 0, 128, &gotF, &gotS);
        const uint16_t *f = (const uint16_t *)gotF.data();
        const uint16_t *sd = (const uint16_t *)gotS.data();
        bool distinct = true, inRange = true;
        for (uint32_t i = 0; i < count; i++) {
            if (f[i] == sd[i]) distinct = false;
            if (f[i] >= 128 || sd[i] >= 128) inRange = false;
        }
        check("every line has a distinct +1 and -1", distinct);
        check("and both indices stay inside rank", inRange);
    }

    printf("4. the domain separation is real\n");
    {
        // Same key, same seed, same block except which word holds the counter.
        // If that offset were wrong the two draws would agree here.
        std::vector<uint8_t> u, pf, ps;
        run(0, key, seed, 256, 63, 32, 128, &u, &pf);
        run(1, key, seed, 64, 0, 0, 128, &pf, &ps);
        check("uniform and permutation draws differ", memcmp(u.data(), pf.data(), 64) != 0);
    }

    printf("5. the gate can fail\n");
    {
        std::vector<uint8_t> got, unused;
        run(0, key, seed, 256, 63, 32, 128, &got, &unused);
        uint8_t other[32];
        for (int i = 0; i < 32; i++) other[i] = (uint8_t)(i * 13 + 6);
        const std::vector<int8_t> want = uniformNoise(key, other, 2, 128);
        bool same = true;
        for (size_t i = 0; i < 32 && i < want.size(); i++)
            if ((int8_t)got[i] != want[i]) same = false;
        check("a different seed does NOT match", !same);
    }

    if (gChecks == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, gChecks, gFail);
    return gFail ? 1 : 0;
}
