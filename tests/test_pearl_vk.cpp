// Pearl NoisyGEMM on Vulkan, against the same reference vectors as the CUDA
// test. Every backend must produce byte-identical transcripts.
//
// This carries its own Vulkan setup rather than reusing algo_vk.cpp's, which
// keeps its boilerplate private. It is deliberately minimal: one queue, one
// pipeline, host-visible buffers, no staging. Correctness first.
//
// usage: test_pearl_vk <vectors.bin> <shader.spv> [device_index]
//
// NOT WIRED INTO `make test`, AND CANNOT BE UNTIL THERE IS A SHADER. There is
// no Vulkan Pearl kernel yet, so there is no .spv to hand it. It compiles and
// it is kept deliberately: the day someone writes the shader, the gate that
// proves it byte-identical to CUDA already exists. Build it on its own with
// `make tests/test_pearl_vk`.
//
// Feasibility for that shader was measured 2026-08-18 on an RX 6700 XT and is
// written up in ~/RESUME-pearl-pow-miner.md: RDNA2 has no matrix cores, so the
// path is dotPacked4x8AccSatEXT (hardware V_DOT4_I32_I8), and a tuned probe
// reached 3.06 T MAC/s against WildRig's 8.1. Every AMD card loses money on
// Pearl at 11.4 c/kWh even at the field's best rate, so this is a
// parity feature and never an economic one.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define VKCHECK(x)                                                        \
    do {                                                                  \
        VkResult r_ = (x);                                                \
        if (r_ != VK_SUCCESS) {                                           \
            fprintf(stderr, "%s:%d %s -> VkResult %d\n", __FILE__,        \
                    __LINE__, #x, (int)r_);                               \
            return 2;                                                     \
        }                                                                 \
    } while (0)

namespace {

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *mapped = nullptr;
};

uint32_t findMemType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Reader {
    const uint8_t *p, *end;
    bool ok = true;
    template <typename T>
    const T *take(size_t n) {
        if (p + n * sizeof(T) > end) { ok = false; return nullptr; }
        const T *r = reinterpret_cast<const T *>(p);
        p += n * sizeof(T);
        return r;
    }
};

}  // namespace

int main(int argc, char **argv) {
    const char *vecPath = argc > 1 ? argv[1] : "/tmp/pearl_vectors.bin";
    const char *spvPath = argc > 2 ? argv[2] : "/tmp/pearl.spv";
    const int wantDevice = argc > 3 ? atoi(argv[3]) : 0;

    // ---- vectors ----
    std::vector<uint8_t> vec;
    {
        FILE *f = fopen(vecPath, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", vecPath); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        vec.resize(sz);
        if (fread(vec.data(), 1, sz, f) != (size_t)sz) { fclose(f); return 2; }
        fclose(f);
    }
    if (vec.size() < 32 || memcmp(vec.data(), "PRLV0002", 8) != 0) {
        fprintf(stderr, "bad vector magic\n"); return 2;
    }
    Reader rd{vec.data() + 8, vec.data() + vec.size()};
    const int32_t *dims = rd.take<int32_t>(6);
    const int m = dims[0], n = dims[1], k = dims[2], rank = dims[3];
    const int numT = dims[5];
    rd.take<int8_t>((size_t)m * k); rd.take<int8_t>((size_t)k * n);
    rd.take<int8_t>((size_t)m * rank); rd.take<int8_t>((size_t)rank * k);
    rd.take<int8_t>((size_t)k * rank); rd.take<int8_t>((size_t)rank * n);
    const int8_t *aN = rd.take<int8_t>((size_t)m * k);
    const int8_t *bN = rd.take<int8_t>((size_t)k * n);
    const int32_t *cExp = rd.take<int32_t>((size_t)m * n);
    rd.take<int32_t>((size_t)m * n);
    const uint32_t *tExp = rd.take<uint32_t>((size_t)numT * 16);
    if (!rd.ok) { fprintf(stderr, "vector file truncated\n"); return 2; }
    printf("vectors: m=%d n=%d k=%d rank=%d transcripts=%d\n", m, n, k, rank, numT);

    // ---- spirv ----
    std::vector<uint32_t> spv;
    {
        FILE *f = fopen(spvPath, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", spvPath); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        spv.resize((sz + 3) / 4);
        if (fread(spv.data(), 1, sz, f) != (size_t)sz) { fclose(f); return 2; }
        fclose(f);
    }

    // ---- instance ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "test_pearl_vk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> pds(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, pds.data());
    if (wantDevice >= (int)ndev) { fprintf(stderr, "no device %d\n", wantDevice); return 2; }
    VkPhysicalDevice pd = pds[wantDevice];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);

    // ---- pick an int8 coopmat config; skip rather than fail if absent ----
    auto getCoop = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    int kDim = 0;
    if (getCoop) {
        uint32_t c = 0;
        getCoop(pd, &c, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cps(c);
        for (auto &p : cps) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        if (c) getCoop(pd, &c, cps.data());
        for (auto &p : cps)
            if (p.AType == VK_COMPONENT_TYPE_SINT8_KHR && p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                p.CType == VK_COMPONENT_TYPE_SINT32_KHR && p.MSize == 16 && p.NSize == 16) {
                kDim = (int)p.KSize;
                break;
            }
    }
    if (!kDim) {
        // Not a failure. A box without int8 cooperative matrix must not take
        // the rest of the suite down with it.
        printf("pearl-pow: no M16 N16 int8 coopmat on %s, skipping device test\n",
               props.deviceName);
        vkDestroyInstance(inst, nullptr);
        return 0;
    }
    printf("device: %s  (int8 coopmat M16 N16 K%d)\n", props.deviceName, kDim);

    VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sgp;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    const uint32_t subgroupSize = sgp.subgroupSize;
    printf("subgroup size: %u\n", subgroupSize);

    // ---- queue ----
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfs.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 2; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    // Ask the driver what it has, then request exactly that subset, rather
    // than demanding a fixed list. RADV and the NVIDIA driver do not expose
    // the same optional features, and vkCreateDevice fails the whole call with
    // VK_ERROR_FEATURE_NOT_PRESENT if any single requested feature is absent -
    // with no indication of which one.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopF{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &coopF;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.pNext = &v12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &v13;
    vkGetPhysicalDeviceFeatures2(pd, &f2);

    struct Need {
        const char *name;
        VkBool32 *flag;
        bool essential;
    } needs[] = {
        {"cooperativeMatrix", &coopF.cooperativeMatrix, true},
        {"shaderInt8", &v12.shaderInt8, true},
        {"storageBuffer8BitAccess", &v12.storageBuffer8BitAccess, true},
        {"vulkanMemoryModel", &v12.vulkanMemoryModel, true},
        {"vulkanMemoryModelDeviceScope", &v12.vulkanMemoryModelDeviceScope, false},
        {"subgroupSizeControl", &v13.subgroupSizeControl, false},
        {"computeFullSubgroups", &v13.computeFullSubgroups, false},
    };
    bool missingEssential = false;
    bool haveSizeControl = false;
    for (const Need &nd : needs) {
        if (!*nd.flag) {
            printf("  feature %s: NOT supported%s\n", nd.name, nd.essential ? " (essential)" : "");
            if (nd.essential) missingEssential = true;
        }
        if (!strcmp(nd.name, "subgroupSizeControl")) haveSizeControl = *nd.flag;
    }
    if (missingEssential) {
        printf("pearl-pow: %s lacks an essential feature, skipping device test\n",
               props.deviceName);
        vkDestroyInstance(inst, nullptr);
        return 0;
    }
    // Everything still set in the structs is what the driver reported, so the
    // chain can be passed straight to vkCreateDevice. Clear the rest of the
    // base features to avoid asking for anything incidental.
    memset(&f2.features, 0, sizeof(f2.features));

    const char *devExt[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = devExt;
    VkDevice dev;
    VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    // ---- buffers ----
    auto mkBuf = [&](VkDeviceSize size, Buf *out) -> VkResult {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r = vkCreateBuffer(dev, &bci, nullptr, &out->buf);
        if (r != VK_SUCCESS) return r;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, out->buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        r = vkAllocateMemory(dev, &mai, nullptr, &out->mem);
        if (r != VK_SUCCESS) return r;
        vkBindBufferMemory(dev, out->buf, out->mem, 0);
        return vkMapMemory(dev, out->mem, 0, size, 0, &out->mapped);
    };

    Buf bA, bB, bC, bT;
    VKCHECK(mkBuf((VkDeviceSize)m * k, &bA));
    VKCHECK(mkBuf((VkDeviceSize)k * n, &bB));
    VKCHECK(mkBuf((VkDeviceSize)m * n * 4, &bC));
    VKCHECK(mkBuf((VkDeviceSize)numT * 16 * 4, &bT));
    memcpy(bA.mapped, aN, (size_t)m * k);
    memcpy(bB.mapped, bN, (size_t)k * n);
    memset(bC.mapped, 0, (size_t)m * n * 4);
    memset(bT.mapped, 0, (size_t)numT * 16 * 4);

    // ---- descriptors ----
    VkDescriptorSetLayoutBinding binds[4]{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 4; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VkDescriptorPool dp;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

    VkDescriptorBufferInfo dbi[4] = {{bA.buf, 0, VK_WHOLE_SIZE},
                                     {bB.buf, 0, VK_WHOLE_SIZE},
                                     {bC.buf, 0, VK_WHOLE_SIZE},
                                     {bT.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[4]{};
    for (int i = 0; i < 4; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);

    // ---- pipeline ----
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4; smci.pCode = spv.data();
    VkShaderModule sm;
    VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    struct Spec { int32_t subgroup; int32_t k; } spec{(int32_t)subgroupSize, kDim};
    VkSpecializationMapEntry sme[2] = {{0, offsetof(Spec, subgroup), sizeof(int32_t)},
                                       {1, offsetof(Spec, k), sizeof(int32_t)}};
    VkSpecializationInfo si{2, sme, sizeof(spec), &spec};

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 5};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    // Pin the subgroup size so the shader's one-workgroup-is-one-subgroup
    // assumption holds on wave64 parts as well as wave32.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = subgroupSize;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // Only pin the subgroup size when the driver supports doing so. Without
    // it the shader still runs at the device's reported subgroup size, which
    // is what the spec constant was set from anyway.
    cpci.stage.pNext = haveSizeControl ? (const void *)&rss : nullptr;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &si;
    cpci.layout = pl;
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    // ---- dispatch ----
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfi;
    VkCommandPool cp;
    VKCHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKCHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    int32_t push[5] = {m, n, k, rank, 1};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    const uint32_t groups = (uint32_t)((m / 16) * (n / 16));
    vkCmdDispatch(cb, groups, 1, 1);
    VKCHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo subm{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    subm.commandBufferCount = 1; subm.pCommandBuffers = &cb;
    VKCHECK(vkQueueSubmit(queue, 1, &subm, VK_NULL_HANDLE));
    VKCHECK(vkQueueWaitIdle(queue));

    // ---- compare ----
    const int32_t *cGot = (const int32_t *)bC.mapped;
    const uint32_t *tGot = (const uint32_t *)bT.mapped;
    int badC = 0, badT = 0;
    for (size_t i = 0; i < (size_t)m * n; ++i)
        if (cGot[i] != cExp[i]) {
            if (badC < 3) fprintf(stderr, "  C[%zu] device %d != reference %d\n", i, cGot[i], cExp[i]);
            ++badC;
        }
    for (size_t i = 0; i < (size_t)numT * 16; ++i)
        if (tGot[i] != tExp[i]) {
            if (badT < 3)
                fprintf(stderr, "  transcript[%zu/%zu] device %08x != reference %08x\n", i / 16,
                        i % 16, tGot[i], tExp[i]);
            ++badT;
        }
    printf("  vulkan: product %s (%d/%d)   transcripts %s (%d/%d)\n", badC ? "FAIL" : "ok", badC,
           m * n, badT ? "FAIL" : "ok", badT, numT * 16);

    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    for (Buf *b : {&bA, &bB, &bC, &bT}) {
        vkUnmapMemory(dev, b->mem);
        vkDestroyBuffer(dev, b->buf, nullptr);
        vkFreeMemory(dev, b->mem, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return (badC || badT) ? 1 : 0;
}
