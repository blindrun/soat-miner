// Shared Vulkan device handling: device selection, enumeration, and the
// backend's device readout. See vk_common.h for why this is split out of
// the algorithms. The algorithm registry itself lives in vk_registry.cpp,
// so that linking the device helpers does not drag in every backend.

#include "vk_common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace om {

namespace {

char g_devName[256] = "unknown";
char g_driver[64] = "";
double g_memGB = 0.0;

}  // namespace

void vkSetDeviceInfo(const char *name, double memGB, const char *driver) {
    snprintf(g_devName, sizeof(g_devName), "%s", name ? name : "unknown");
    snprintf(g_driver, sizeof(g_driver), "%s", driver ? driver : "");
    g_memGB = memGB;
}

const char *vkDeviceName() { return g_devName; }
double vkDeviceMemGB() { return g_memGB; }
const char *vkDriverVersion() { return g_driver; }

bool vkPickPhysicalDevice(VkInstance inst, int requestedIndex,
                          VkPhysicalDevice *out) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    if (count == 0) {
        fprintf(stderr, "no Vulkan device found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    // Only real GPUs. llvmpipe/lavapipe advertise themselves as Vulkan devices
    // of type CPU and would "work" at about 0.1 MH/s.
    std::vector<VkPhysicalDevice> usable;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            usable.push_back(d);
    }
    if (usable.empty()) {
        fprintf(stderr,
                "no Vulkan GPU found (only CPU/software devices).\n"
                "  install your GPU vendor's Vulkan driver, then check "
                "with: vulkaninfo --summary\n");
        return false;
    }

    size_t pick = 0;
    if (requestedIndex >= 0) {
        if ((size_t)requestedIndex >= usable.size()) {
            fprintf(stderr, "device %d requested but only %zu GPU(s) found\n",
                    requestedIndex, usable.size());
            return false;
        }
        pick = (size_t)requestedIndex;
    } else {
        // Largest device-local heap wins: the right answer when an iGPU sits
        // alongside a discrete card.
        VkDeviceSize best = 0;
        for (size_t i = 0; i < usable.size(); i++) {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(usable[i], &mp);
            VkDeviceSize local = 0;
            for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
                if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    local = std::max(local, mp.memoryHeaps[h].size);
            if (local > best) { best = local; pick = i; }
        }
    }
    *out = usable[pick];
    return true;
}

void vkListDevices() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    // 1.3, not 1.2: the cooperative-matrix query below needs it, and this
    // listing is the only place a user can see WHICH device has it.
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        printf("cannot create a Vulkan instance - is a driver installed?\n");
        return;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    int idx = 0;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        const bool gpu = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                         p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(d, &mp);
        VkDeviceSize local = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                local = std::max(local, mp.memoryHeaps[h].size);
        // Report the capability PER DEVICE, because that is exactly what a
        // whole-machine check gets wrong. Measured on this fleet:
        //
        //   RTX 4090   extension YES  feature YES  15 configs,  2 int8
        //   llvmpipe   extension YES  feature YES   4 configs,  1 int8
        //   6700 XT    extension no
        //
        // So `vulkaninfo | grep cooperative` finds llvmpipe's support at every
        // level INCLUDING an int8 configuration, on a box whose actual GPU has
        // none. What llvmpipe does not have is the M16 N16 sint8 x sint8 ->
        // sint32 shape Pearl's GEMM is written against, which is what
        // vkInt8CooperativeMatrix() requires - hence the label naming the
        // shape rather than claiming the device has no int8 coopmat at all.
        uint32_t k = 0;
        const bool coop = vkInt8CooperativeMatrix(inst, d, &k);
        char cm[56];
        if (coop) snprintf(cm, sizeof(cm), "int8 coopmat M16N16K%u", k);
        else snprintf(cm, sizeof(cm), "no M16N16 int8 coopmat");

        if (gpu) {
            printf("  [%d] %-38s %5.1f GB  max buffer %4.1f GB  (%s, %s)\n",
                   idx++, p.deviceName, local / 1e9,
                   p.limits.maxStorageBufferRange / 1e9,
                   driverTypeName(p.deviceType), cm);
        } else {
            printf("   -  %-38s %5.1f GB  (skipped: %s, %s)\n", p.deviceName,
                   local / 1e9, driverTypeName(p.deviceType), cm);
        }
    }
    if (idx == 0) printf("no usable Vulkan GPU found\n");
    vkDestroyInstance(inst, nullptr);
}

bool vkInt8CooperativeMatrix(VkInstance inst, VkPhysicalDevice dev,
                             uint32_t *kOut) {
    if (kOut) *kOut = 0;

    // 1. THE DEVICE EXTENSION LIST. Not the configuration enumeration, which
    //    answers for devices that do not support this at all - the loader
    //    resolves the entry point if ANY device on the box has the extension,
    //    and then happily returns 14 configurations for an RX 6700 XT that
    //    supports none of them. Measured on a box where the extension and the
    //    feature bit both belonged to llvmpipe and neither to the 6700 XT.
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> exts(n);
    if (n && vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data()) != VK_SUCCESS)
        return false;
    bool haveExt = false;
    for (const auto &e : exts)
        if (!strcmp(e.extensionName, "VK_KHR_cooperative_matrix")) { haveExt = true; break; }
    if (!haveExt) return false;

    // 2. THE FEATURE BIT. Present-but-false is a real state; without this
    //    vkCreateDevice fails later with VK_ERROR_FEATURE_NOT_PRESENT, which
    //    is a hard failure where a clean skip was wanted.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{};
    cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &cm;
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    if (!cm.cooperativeMatrix) return false;

    // 3. ONLY NOW the configuration list, for the K this device wants. It is
    //    32 on Ada and 16 on RDNA3, which is why the shader takes it as a
    //    specialisation constant rather than baking one in.
    auto getCoop = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!getCoop) return false;
    uint32_t c = 0;
    getCoop(dev, &c, nullptr);
    if (!c) return false;
    std::vector<VkCooperativeMatrixPropertiesKHR> cps(c);
    for (auto &p : cps) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    getCoop(dev, &c, cps.data());
    for (const auto &p : cps)
        if (p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
            p.MSize == 16 && p.NSize == 16) {
            if (kOut) *kOut = p.KSize;
            return true;
        }
    return false;
}

const char *driverTypeName(int t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU / software";
        default: return "other";
    }
}

}  // namespace om
