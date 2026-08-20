// Pearl matrix generation on Vulkan, against the host reference.
//
// usage: test_pearl_genmatrix_vk <genmatrix.spv> [device_index]
//
// The matrices ARE the nonce. If this shader disagrees with job.h's
// fillMatrix by one byte, the miner does not run slow - every share it finds
// is rejected, and nothing in the mining loop notices. So this compares every
// byte, at sizes that cross the workgroup boundary, and it carries a negative
// control so a gate that cannot fail is visible as one.

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
    VkBuffer outB, stageB; VkDeviceMemory outM, stageM;
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outB, &outM));
    VKREQ(mkBuf(kMaxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stageB, &stageM));
    void *staged = nullptr;
    VKREQ(vkMapMemory(dev, stageM, 0, kMaxBytes, 0, &staged));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4; smci.pCode = spv.data();
    VkShaderModule sm; VKREQ(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0; bind.descriptorCount = 1;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 1; dslci.pBindings = &bind;
    VkDescriptorSetLayout dsl; VKREQ(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    struct Push { uint64_t seed; uint32_t count; } push{};
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

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VKREQ(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VKREQ(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi{outB, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cp; VKREQ(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VKREQ(vkAllocateCommandBuffers(dev, &cbai, &cb));

    auto run = [&](uint64_t seed, uint32_t count, std::vector<int8_t> *got) {
        push.seed = seed; push.count = count;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        const uint32_t groups = ((count / 8) + 255) / 256;
        vkCmdDispatch(cb, groups, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy bc{0, 0, count};
        vkCmdCopyBuffer(cb, outB, stageB, 1, &bc);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        got->assign((const int8_t *)staged, (const int8_t *)staged + count);
    };

    printf("1. device output against job.h's fillMatrix\n");
    // Sizes that cross the 256-invocation workgroup boundary in both
    // directions, plus one that is not a multiple of it.
    const uint32_t sizes[] = {8, 64, 2048, 2048 * 8, 4096, 100000};
    for (uint32_t n : sizes) {
        std::vector<int8_t> got, want(n);
        run(0x51ED0000ull + n, n, &got);
        fillMatrix(want.data(), n, 0x51ED0000ull + n);
        char d[64]; snprintf(d, sizeof(d), "- %u bytes", n);
        check("device matches host", got == want, d);
    }

    printf("2. the range the chain requires\n");
    {
        std::vector<int8_t> got;
        run(12345, 65536, &got);
        int8_t lo = 127, hi = -128;
        for (int8_t v : got) { if (v < lo) lo = v; if (v > hi) hi = v; }
        char d[64]; snprintf(d, sizeof(d), "- %d..%d", lo, hi);
        check("values stay in [-64, 63], leaving room for the noise",
              lo >= -64 && hi <= 63, d);
    }

    printf("3. the gate can fail\n");
    {
        std::vector<int8_t> got, want(4096);
        run(999, 4096, &got);
        fillMatrix(want.data(), 4096, 1000);   // deliberately the wrong seed
        check("a different seed does NOT match", got != want);
    }

    if (gChecks == 0) {
        printf("\n%s: ZERO CHECKS RAN - failing\n", props.deviceName);
        return 1;
    }
    printf("\n%s: %d check(s), %d failure(s)\n", props.deviceName, gChecks, gFail);
    return gFail ? 1 : 0;
}
