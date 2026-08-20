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
    app.apiVersion = VK_API_VERSION_1_2;
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
        if (gpu) {
            printf("  [%d] %-38s %5.1f GB  max buffer %4.1f GB  (%s)\n", idx++,
                   p.deviceName, local / 1e9,
                   p.limits.maxStorageBufferRange / 1e9,
                   driverTypeName(p.deviceType));
        } else {
            printf("   -  %-38s %5.1f GB  (skipped: %s)\n", p.deviceName,
                   local / 1e9, driverTypeName(p.deviceType));
        }
    }
    if (idx == 0) printf("no usable Vulkan GPU found\n");
    vkDestroyInstance(inst, nullptr);
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
